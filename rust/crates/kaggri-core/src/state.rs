//! Game state: `Config`, `GameState`, `Farm`, `Tile`, `Private`, `Market`, `Town`.
//!
//! Layout notes that matter for bit-exactness:
//!
//! * **Unit inventories are insertion-ordered** ([`Inv`] is an ordered `Vec` of pairs).
//!   `DROP` and `_drop_inventories_to_shed` iterate `list(inv.items())` and stop filling
//!   at `shedCapacity`, discarding the rest — so the order a unit picked things up in
//!   decides what survives a full shed. `_inv_take` *deletes* a key when it reaches 0,
//!   so a later re-add lands at the **back**; [`Inv::take`] reproduces that.
//! * **The shed is a fixed 12-slot table** ([`Shed`]). `_new_private()` pre-seeds every
//!   `PRODUCTS + list(ANIMALS)` key with 0 and no code path in the environment ever
//!   inserts or deletes a shed key (checked empirically across full episodes), so key
//!   order is frozen at construction and `sum(shed.values())` is order-independent.
//! * **Money is `f64`** — `_new_farm` stores `float(starting_money)` and integer prices
//!   are added to / subtracted from it. Every value stays integral and well inside
//!   2^53, so the comparisons against integer prices are exact.

use crate::consts::*;

/// `[x, y]`, y growing downward. Always in bounds once stored.
pub type Pos = [i64; 2];

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Episode configuration.
///
/// [`Config::default`] reproduces the defaults in `kaggriculture.json` exactly.
/// `seed` has no JSON default (it is `null`, meaning "draw a random 31-bit int");
/// the port requires an explicit seed, and 0 is used when none is given.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub episode_steps: u32,
    pub board_size: usize,
    pub starting_money: f64,
    pub max_market_orders_per_turn: usize,
    pub turns_per_day: u32,
    pub shed_capacity: usize,
    pub weed_spawn_chance: f64,
    pub town_shop_unlock_interval: u32,
    pub town_center_sell_interval: u32,
    pub town_shop_sell_interval: u32,
    pub farm_hand_cost_mult: i64,
    pub seed: i64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            episode_steps: 720,
            board_size: 10,
            starting_money: 3000.0,
            max_market_orders_per_turn: 10,
            turns_per_day: 24,
            shed_capacity: 100,
            weed_spawn_chance: 0.005,
            town_shop_unlock_interval: 3,
            town_center_sell_interval: 24,
            town_shop_sell_interval: 4,
            farm_hand_cost_mult: FARM_HAND_COST_MULT,
            seed: 0,
        }
    }
}

impl Config {
    /// `max(1, int(get(cfg, "turnsPerDay", 24)))`.
    #[inline]
    pub fn turns_per_day_eff(&self) -> i64 {
        (self.turns_per_day as i64).max(1)
    }

    /// `max(1, int(get(cfg, "maxMarketOrdersPerTurn", 10)))`.
    #[inline]
    pub fn max_orders_eff(&self) -> usize {
        self.max_market_orders_per_turn.max(1)
    }

    /// `max(1, int(get(cfg, "townShopSellInterval", 4)))`.
    #[inline]
    pub fn shop_sell_interval_eff(&self) -> i64 {
        (self.town_shop_sell_interval as i64).max(1)
    }

    /// `max(1, int(get(cfg, "townCenterSellInterval", 24)))`.
    #[inline]
    pub fn center_sell_interval_eff(&self) -> i64 {
        (self.town_center_sell_interval as i64).max(1)
    }

    /// `max(1, int(get(cfg, "townShopUnlockInterval", 3)))`.
    #[inline]
    pub fn shop_unlock_interval_eff(&self) -> i64 {
        (self.town_shop_unlock_interval as i64).max(1)
    }

    /// `int(get(cfg, "shedCapacity", 100))`.
    #[inline]
    pub fn shed_capacity_i64(&self) -> i64 {
        self.shed_capacity as i64
    }
}

// ---------------------------------------------------------------------------
// Inv — an insertion-ordered unit inventory (a Python dict)
// ---------------------------------------------------------------------------

/// One unit's carried inventory: an insertion-ordered `{item: count}` map.
///
/// Backed by a `Vec` of at most 12 pairs, which is both faster than a hash map at
/// this size and preserves the observable insertion order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Inv {
    items: Vec<(ShedItem, i64)>,
}

impl Inv {
    #[inline]
    pub fn new() -> Self {
        Inv { items: Vec::new() }
    }

    #[inline]
    fn find(&self, item: ShedItem) -> Option<usize> {
        self.items.iter().position(|(k, _)| *k == item)
    }

    /// `inv.get(item, 0)`.
    #[inline]
    pub fn get(&self, item: ShedItem) -> i64 {
        match self.find(item) {
            Some(i) => self.items[i].1,
            None => 0,
        }
    }

    /// `_inv_add(inv, item, n)` — `inv[item] = inv.get(item, 0) + n`.
    ///
    /// A brand new key is appended at the back, which is where Python puts it.
    #[inline]
    pub fn add(&mut self, item: ShedItem, n: i64) {
        match self.find(item) {
            Some(i) => self.items[i].1 += n,
            None => self.items.push((item, n)),
        }
    }

    /// `_inv_take(inv, item, n)`.
    ///
    /// Returns false and changes nothing when the unit holds less than `n`. On
    /// success, decrements and **deletes the key when it reaches 0** — a later
    /// re-add therefore appends at the back of the order.
    ///
    /// The environment only ever calls `_inv_take` with `n == 1` (`FERTILIZE`, `FEED`,
    /// animal `PLACE`). Python with `n <= 0` and a missing key would fall past the
    /// `inv.get(item, 0) < n` guard and raise `KeyError`, so that case is asserted
    /// against rather than invented.
    #[inline]
    pub fn take(&mut self, item: ShedItem, n: i64) -> bool {
        debug_assert!(n > 0, "_inv_take is only ever called with n == 1");
        match self.find(item) {
            Some(i) if self.items[i].1 >= n => {
                self.items[i].1 -= n;
                if self.items[i].1 == 0 {
                    self.items.remove(i);
                }
                true
            }
            _ => false,
        }
    }

    /// The `PLACE`-into-shed tail: `inv[item] -= n; if inv[item] == 0: del inv[item]`.
    ///
    /// Only called when the key exists and `0 < n <= inv[item]`.
    #[inline]
    pub fn sub_existing(&mut self, item: ShedItem, n: i64) {
        if let Some(i) = self.find(item) {
            self.items[i].1 -= n;
            if self.items[i].1 == 0 {
                self.items.remove(i);
            }
        }
    }

    /// Iterate in insertion order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (ShedItem, i64)> + '_ {
        self.items.iter().copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Take the whole ordered contents out, leaving the inventory empty.
    ///
    /// `DROP` / `_drop_inventories_to_shed` snapshot `list(inv.items())` and `del`
    /// every key they visit, so the inventory always ends up empty.
    #[inline]
    pub fn drain_ordered(&mut self) -> Vec<(ShedItem, i64)> {
        core::mem::take(&mut self.items)
    }

    /// Total item count (only used for diagnostics; the shed cap is on the shed).
    #[inline]
    pub fn total(&self) -> i64 {
        self.items.iter().map(|(_, n)| *n).sum()
    }
}

// ---------------------------------------------------------------------------
// Shed
// ---------------------------------------------------------------------------

/// `private["shed"]` — a fixed 12-slot table in [`SHED_ITEMS`] order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shed {
    pub counts: [i64; N_SHED_ITEMS],
}

impl Default for Shed {
    #[inline]
    fn default() -> Self {
        Shed {
            counts: [0; N_SHED_ITEMS],
        }
    }
}

impl Shed {
    #[inline]
    pub fn get(&self, item: ShedItem) -> i64 {
        self.counts[item as usize]
    }

    #[inline]
    pub fn set(&mut self, item: ShedItem, n: i64) {
        self.counts[item as usize] = n;
    }

    #[inline]
    pub fn add(&mut self, item: ShedItem, n: i64) {
        self.counts[item as usize] += n;
    }

    /// `sum(shed.values())`.
    #[inline]
    pub fn total(&self) -> i64 {
        let mut t = 0;
        for c in self.counts {
            t += c;
        }
        t
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (ShedItem, i64)> + '_ {
        SHED_ITEMS.iter().map(|i| (*i, self.counts[*i as usize]))
    }
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

/// `_new_private()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Private {
    pub shed: Shed,
    /// `private["seeds"]`, indexed by [`Crop`] (fixed key set, fixed order).
    pub seeds: [i64; N_CROPS],
    /// `private["inventories"]` — `[main_farmer, *hands]`.
    pub inventories: Vec<Inv>,
}

impl Default for Private {
    fn default() -> Self {
        Private {
            shed: Shed::default(),
            seeds: [0; N_CROPS],
            inventories: vec![Inv::new()],
        }
    }
}

impl Private {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn seed(&self, crop: Crop) -> i64 {
        self.seeds[crop as usize]
    }

    /// `_farmer_inventory(private, idx)` — grows the list if `idx` is past the end.
    #[inline]
    pub fn farmer_inventory(&mut self, idx: usize) -> &mut Inv {
        while self.inventories.len() <= idx {
            self.inventories.push(Inv::new());
        }
        &mut self.inventories[idx]
    }

    /// `private["inventories"] = [{}]`.
    #[inline]
    pub fn reset_inventories(&mut self) {
        self.inventories.clear();
        self.inventories.push(Inv::new());
    }
}

// ---------------------------------------------------------------------------
// Tiles
// ---------------------------------------------------------------------------

/// A `{"kind": "PLANT", ...}` tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PlantTile {
    pub crop: Crop,
    pub planted_day: i64,
    pub watered_today: bool,
    pub consecutive_unwatered: i64,
    pub yield_units: i64,
    pub max_lifespan_step: i64,
    pub fertilized_until_day: i64,
}

/// A `{"kind": "COOP"|"PASTURE", "animal": ..., ...}` tile (an occupied structure).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AnimalTile {
    pub animal: Animal,
    pub placed_day: i64,
    pub yield_units: i64,
    pub consecutive_unfed: i64,
    pub fed_today: bool,
    pub cared_today: bool,
    pub fertilizer_available: bool,
    pub pending_care_bonus: i64,
}

/// `farm["tiles"][y][x]`.
///
/// Python encodes this as `None` | `"LOCKED"` | a dict. The dict cases are
/// `{"kind": "WEED"}`, a plant, an empty `{"kind": "COOP"|"PASTURE"}`, or an
/// occupied structure (a dict that additionally has an `"animal"` key).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Tile {
    /// `None` — empty and unlocked.
    #[default]
    Empty,
    /// `"LOCKED"`.
    Locked,
    /// `{"kind": "WEED"}`.
    Weed,
    /// `{"kind": "COOP"}` with no animal.
    Coop,
    /// `{"kind": "PASTURE"}` with no animal.
    Pasture,
    Plant(PlantTile),
    Animal(AnimalTile),
}

impl Tile {
    /// True for the tile values Python represents as a `dict`.
    #[inline]
    pub fn is_dict(self) -> bool {
        !matches!(self, Tile::Empty | Tile::Locked)
    }

    /// `isinstance(tile, dict) and tile.get("kind") == "PLANT"`.
    #[inline]
    pub fn as_plant(self) -> Option<PlantTile> {
        match self {
            Tile::Plant(p) => Some(p),
            _ => None,
        }
    }

    /// `isinstance(tile, dict) and "animal" in tile`.
    #[inline]
    pub fn as_animal(self) -> Option<AnimalTile> {
        match self {
            Tile::Animal(a) => Some(a),
            _ => None,
        }
    }

    /// `tile.get("yield_units", 0)`.
    #[inline]
    pub fn yield_units(self) -> i64 {
        match self {
            Tile::Plant(p) => p.yield_units,
            Tile::Animal(a) => a.yield_units,
            _ => 0,
        }
    }

    /// The empty structure a given animal needs (`ANIMALS[item]["structure"]`).
    #[inline]
    pub fn empty_structure(structure: Structure) -> Tile {
        match structure {
            Structure::Coop => Tile::Coop,
            Structure::Pasture => Tile::Pasture,
        }
    }

    /// `_render_tile(tile)`.
    pub fn render(self) -> char {
        match self {
            Tile::Empty => '.',
            Tile::Locked => '#',
            Tile::Weed => 'x',
            Tile::Coop => 'C',
            Tile::Pasture => 'P',
            Tile::Plant(p) => p.crop.as_str().as_bytes()[0].to_ascii_lowercase() as char,
            Tile::Animal(a) => a.animal.as_str().as_bytes()[0] as char,
        }
    }
}

// ---------------------------------------------------------------------------
// Farm
// ---------------------------------------------------------------------------

/// `_new_farm(board_size, starting_money)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Farm {
    pub money: f64,
    pub board_size: usize,
    /// Row-major `tiles[y * board_size + x]`.
    pub tiles: Vec<Tile>,
    pub farmer: Pos,
    pub hands: Vec<Pos>,
    pub unlocked_quadrants: Vec<Quadrant>,
    pub hires_today: i64,
}

impl Farm {
    /// `_new_farm`.
    pub fn new(board_size: usize, starting_money: f64) -> Self {
        let mut tiles = Vec::with_capacity(board_size * board_size);
        for y in 0..board_size as i64 {
            for x in 0..board_size as i64 {
                // `_initial_tile`: None inside NW, "LOCKED" everywhere else.
                tiles.push(if quadrant_of(x, y, board_size) == Quadrant::NW {
                    Tile::Empty
                } else {
                    Tile::Locked
                });
            }
        }
        Farm {
            money: starting_money,
            board_size,
            tiles,
            farmer: default_spawn(board_size),
            hands: Vec::new(),
            unlocked_quadrants: vec![Quadrant::NW],
            hires_today: 0,
        }
    }

    #[inline]
    pub fn tile(&self, x: i64, y: i64) -> Tile {
        self.tiles[y as usize * self.board_size + x as usize]
    }

    #[inline]
    pub fn tile_mut(&mut self, x: i64, y: i64) -> &mut Tile {
        &mut self.tiles[y as usize * self.board_size + x as usize]
    }

    #[inline]
    pub fn set_tile(&mut self, x: i64, y: i64, t: Tile) {
        self.tiles[y as usize * self.board_size + x as usize] = t;
    }

    /// `_farmer_position(farm, idx)` — `None` when the hand does not exist.
    #[inline]
    pub fn unit_pos(&self, idx: usize) -> Option<Pos> {
        if idx == 0 {
            Some(self.farmer)
        } else {
            self.hands.get(idx - 1).copied()
        }
    }

    /// `_set_farmer_position(farm, idx, pos)`.
    #[inline]
    pub fn set_unit_pos(&mut self, idx: usize, pos: Pos) {
        if idx == 0 {
            self.farmer = pos;
        } else {
            self.hands[idx - 1] = pos;
        }
    }
}

// ---------------------------------------------------------------------------
// Market / Town
// ---------------------------------------------------------------------------

/// `_new_market(params)` — fixed key sets in [`PRODUCTS`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct Market {
    pub inventory: [i64; N_PRODUCTS],
    pub prices: [i64; N_PRODUCTS],
    /// Resolved `MARKET_PARAMS` (plus any `configuration["marketParams"]` overrides).
    pub params: MarketParams,
}

impl Default for Market {
    fn default() -> Self {
        Self::new(MARKET_PARAMS)
    }
}

impl Market {
    pub fn new(params: MarketParams) -> Self {
        let mut inventory = [0i64; N_PRODUCTS];
        let mut prices = [0i64; N_PRODUCTS];
        for p in PRODUCTS {
            inventory[p.idx()] = params.get(p).i0;
            // `_new_market` seeds prices with `base`, *not* with `market_price(I0)`.
            prices[p.idx()] = params.get(p).base as i64;
        }
        Market {
            inventory,
            prices,
            params,
        }
    }

    #[inline]
    pub fn inv(&self, item: Product) -> i64 {
        self.inventory[item.idx()]
    }

    #[inline]
    pub fn price(&self, item: Product) -> i64 {
        self.prices[item.idx()]
    }
}

/// `_new_town()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Town {
    pub unlocked_shops: Vec<Shop>,
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

/// The whole episode state for a 2-player game.
#[derive(Clone, Debug, PartialEq)]
pub struct GameState {
    pub farms: [Farm; 2],
    pub privates: [Private; 2],
    pub market: Market,
    pub town: Town,
    /// The framework's step counter: the value the Python interpreter reads on entry.
    pub step: i64,
    /// `obs.day` — `next_step // turns_per_day` after the last step.
    pub day: i64,
    /// `obs.hour` — `next_step % turns_per_day` after the last step.
    pub hour: i64,
    /// `env.info["seed"]`.
    pub seed: i64,
    pub done: bool,
    /// `s.reward` once DONE fires (0.0 before that, as the framework default).
    pub rewards: [f64; 2],
}

impl GameState {
    /// `_initialize(state, env)`.
    pub fn new(cfg: &Config) -> Self {
        // `_new_farm` stores `float(int(startingMoney))`.
        let money = cfg.starting_money.trunc();
        GameState {
            farms: [
                Farm::new(cfg.board_size, money),
                Farm::new(cfg.board_size, money),
            ],
            privates: [Private::new(), Private::new()],
            market: Market::new(MARKET_PARAMS),
            town: Town::default(),
            step: 0,
            day: 0,
            hour: 0,
            seed: cfg.seed,
            done: false,
            rewards: [0.0, 0.0],
        }
    }

    /// Same as [`GameState::new`] but with resolved market-curve overrides
    /// (`configuration["marketParams"]`, i.e. `_resolve_market_params`).
    pub fn new_with_params(cfg: &Config, params: MarketParams) -> Self {
        let mut gs = Self::new(cfg);
        gs.market = Market::new(params);
        gs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_match_json() {
        let c = Config::default();
        assert_eq!(c.episode_steps, 720);
        assert_eq!(c.board_size, 10);
        assert_eq!(c.starting_money, 3000.0);
        assert_eq!(c.max_market_orders_per_turn, 10);
        assert_eq!(c.turns_per_day, 24);
        assert_eq!(c.shed_capacity, 100);
        assert_eq!(c.weed_spawn_chance, 0.005);
        assert_eq!(c.town_shop_unlock_interval, 3);
        assert_eq!(c.town_shop_sell_interval, 4);
        assert_eq!(c.town_center_sell_interval, 24);
        assert_eq!(c.farm_hand_cost_mult, 1);
    }

    #[test]
    fn initial_state_matches_python() {
        let gs = GameState::new(&Config::default());
        for f in &gs.farms {
            assert_eq!(f.money, 3000.0);
            assert_eq!(f.farmer, [4, 4]);
            assert!(f.hands.is_empty());
            assert_eq!(f.unlocked_quadrants, vec![Quadrant::NW]);
            assert_eq!(f.hires_today, 0);
            // 25 unlocked NW tiles, 75 locked.
            let empty = f.tiles.iter().filter(|t| **t == Tile::Empty).count();
            let locked = f.tiles.iter().filter(|t| **t == Tile::Locked).count();
            assert_eq!((empty, locked), (25, 75));
            assert_eq!(f.tile(0, 0), Tile::Empty);
            assert_eq!(f.tile(4, 4), Tile::Empty);
            assert_eq!(f.tile(5, 4), Tile::Locked);
            assert_eq!(f.tile(4, 5), Tile::Locked);
            assert_eq!(f.tile(5, 5), Tile::Locked);
        }
        for p in &gs.privates {
            assert_eq!(p.shed.total(), 0);
            assert_eq!(p.seeds, [0; 5]);
            assert_eq!(p.inventories.len(), 1);
        }
        assert_eq!(gs.market.inventory, [10_000; 9]);
        assert_eq!(gs.market.prices, [25, 35, 60, 120, 250, 50, 160, 200, 100]);
        assert!(gs.town.unlocked_shops.is_empty());
    }

    #[test]
    fn inv_is_insertion_ordered_and_deletes_at_zero() {
        let mut inv = Inv::new();
        inv.add(ShedItem::Melon, 5);
        inv.add(ShedItem::Wheat, 50);
        assert_eq!(
            inv.iter().collect::<Vec<_>>(),
            vec![(ShedItem::Melon, 5), (ShedItem::Wheat, 50)]
        );
        // Taking a key to exactly zero deletes it; re-adding appends at the back.
        assert!(inv.take(ShedItem::Melon, 5));
        assert_eq!(inv.iter().collect::<Vec<_>>(), vec![(ShedItem::Wheat, 50)]);
        inv.add(ShedItem::Melon, 1);
        assert_eq!(
            inv.iter().collect::<Vec<_>>(),
            vec![(ShedItem::Wheat, 50), (ShedItem::Melon, 1)]
        );
        // A partial take keeps the position.
        assert!(inv.take(ShedItem::Wheat, 10));
        assert_eq!(
            inv.iter().collect::<Vec<_>>(),
            vec![(ShedItem::Wheat, 40), (ShedItem::Melon, 1)]
        );
        // Over-take fails and changes nothing.
        assert!(!inv.take(ShedItem::Wheat, 41));
        assert_eq!(inv.get(ShedItem::Wheat), 40);
        assert!(!inv.take(ShedItem::Egg, 1));
    }
}
