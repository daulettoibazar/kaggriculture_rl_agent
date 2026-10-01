//! Target-tile macro action space and the market-order packer.
//!
//! The policy never emits raw movement. Each unit selects a **(target tile, verb)** pair
//! and this module resolves that pair into the one concrete [`UnitOp`] the unit should
//! execute *this* turn: a step along the shortest path while it is not standing on the
//! target, the verb once it is. Orders are re-issued every turn, so a macro is a soft
//! intention rather than a commitment — nothing here carries state across turns.
//!
//! Three properties are load-bearing and are what make this layer safe:
//!
//! 1. **Illegal actions in this environment are silent no-ops** (`_apply_unit_action`
//!    returns without mutating anything on every failed precondition), so unmasked
//!    exploration cannot corrupt a game. The resolver therefore never has to refuse an
//!    order to keep the episode valid — it only refuses the handful of orders that would
//!    be *destructive against intent*.
//! 2. **Movement has no side effects.** `FARMER_MOVES` only writes the position; locked
//!    tiles are passable and off-board moves are no-ops. Every shortest path between two
//!    tiles is therefore strategically identical, which is why a fixed BFS tie-break is
//!    sound (see [`first_step`]).
//! 3. **Two rules in the environment punish naive per-unit action selection**, and both
//!    are handled here rather than left to the policy:
//!    * atomic `PLANT` — if the summed `PLANT` demand for a crop across `[farmer, *hands]`
//!      exceeds `seeds[crop]`, the interpreter voids **every** `PLANT` for that crop this
//!      turn. [`resolve_turn`] hands out a per-crop seed budget in unit order so the rule
//!      can never fire.
//!    * `DROP` with a full shed `del`etes the remainder of the inventory. [`resolve_turn`]
//!      only emits `DROP` when the shed has room for the whole inventory; otherwise the
//!      policy should be using `Deposit`, which clamps instead of discarding.
//!
//! See `OBS_LAYOUT.md` for the action-head layout the training code is written against.

use std::sync::LazyLock;

use crate::actions::{MarketOrder, PlayerAction, UnitOp};
use crate::consts::*;
use crate::sim::hire_cost;
use crate::state::{Config, GameState, Inv, Shed, Tile};

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

/// Number of target tiles in the per-unit target head (a 10x10 board).
pub const N_TARGETS: usize = 100;

/// Number of verbs in the per-unit verb head.
pub const N_VERBS: usize = 38;

/// Farmer + at most [`MAX_HANDS`] hands. Every unit gets an observation token and a
/// macro order, so hiring is clamped to keep the whole crew addressable.
pub const MAX_UNITS: usize = 17;

/// Hard cap on hands alive at once, enforced by [`pack_market`].
///
/// The measured useful range is 2–6 hands and the fibonacci wall makes >13 unaffordable
/// (`research/logistics-action-budget.md` §1.3, §2), so 16 is comfortably above anything
/// a sane policy wants while keeping the unit-token block a fixed 17.
pub const MAX_HANDS: usize = MAX_UNITS - 1;

/// Items worth carrying. `FEED` consumes `WHEAT`, `FERTILIZE` consumes `FERTILIZER`, and
/// `PLACE` consumes an animal; nothing else in the environment reads a unit's inventory.
///
/// Harvested produce is deliberately absent: produce is sold out of the *shed*, so moving
/// it back into an inventory can only lose it to the end-of-day overflow. That drops 7 of
/// the 12 shed items from the pickup head.
pub const PICKUP_ITEMS: [ShedItem; 5] = [
    ShedItem::Wheat,
    ShedItem::Fertilizer,
    ShedItem::Goose,
    ShedItem::Cow,
    ShedItem::Sheep,
];

/// Pickup batch sizes. `1` for "collect exactly one animal to place"; `8` covers a unit's
/// whole feeding/fertilizing round (a hand has at most 23 actions in a day and has to walk
/// between tiles, so more than 8 consumables per trip is unreachable).
///
/// `_apply_unit_action` clamps `n` to what the shed actually holds, so an 8 against a
/// shed holding 3 takes 3 and is never an error.
pub const PICKUP_QTYS: [i64; 2] = [1, 8];

/// `PLACE <product> <n>` quantity used by [`Verb::Deposit`] — "as much as fits".
///
/// The environment clamps to `min(n, inv[item])` and then to the remaining shed room, and
/// leaves the remainder in the inventory, so an oversized `n` is the lossless deposit.
pub const DEPOSIT_QTY: i64 = 1_000_000;

// ---------------------------------------------------------------------------
// Verb
// ---------------------------------------------------------------------------

/// One macro verb. Flat head index is [`Verb::index`]; see `OBS_LAYOUT.md` §Action space.
///
/// Omissions, deliberate:
/// * **Raw `NORTH/SOUTH/EAST/WEST`** — the whole point of the macro layer. Pathing is
///   [`first_step`]'s job.
/// * **`PASS`** — folded into [`Verb::Idle`] (the resolver emits `UnitOp::Pass`).
/// * **`PICKUP` of the 7 non-consumable shed items** — see [`PICKUP_ITEMS`].
/// * **`PLACE <animal>` into the shed** — [`Verb::Deposit`] covers products only. A unit
///   standing on a matching empty structure with an animal in hand hits the
///   animal-placement branch of `PLACE`, so a "deposit this goose" verb would silently
///   become "place this goose" on exactly the tiles a deposit targets (a shed-access tile
///   can hold a coop). The unambiguous verb is [`Verb::PlaceAnimal`]; surplus animals go
///   back to the shed for free in the end-of-day drop.
/// * **A quantity head for `Deposit`** — an oversized `n` is already the optimal deposit
///   (see [`DEPOSIT_QTY`]).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Verb {
    /// `PASS`.
    Idle,
    Water,
    Harvest,
    Fertilize,
    Dig,
    BuildCoop,
    BuildPasture,
    Feed,
    Care,
    CollectFertilizer,
    /// `PLANT <crop>` on the target tile.
    Plant(Crop),
    /// `PLACE <animal>` onto a matching empty structure on the target tile.
    PlaceAnimal(Animal),
    /// `PICKUP <PICKUP_ITEMS[item_slot]> <PICKUP_QTYS[qty_slot]>` at a shed-access tile.
    Pickup { item_slot: u8, qty_slot: u8 },
    /// `PLACE <product> DEPOSIT_QTY` at a shed-access tile — lossless shed deposit.
    Deposit(Product),
    /// `DROP` at a shed-access tile — dumps the whole inventory in one action.
    Drop,
}

/// Every verb in flat head order.
pub const VERBS: [Verb; N_VERBS] = [
    Verb::Idle,
    Verb::Water,
    Verb::Harvest,
    Verb::Fertilize,
    Verb::Dig,
    Verb::BuildCoop,
    Verb::BuildPasture,
    Verb::Feed,
    Verb::Care,
    Verb::CollectFertilizer,
    Verb::Plant(Crop::Wheat),
    Verb::Plant(Crop::Carrot),
    Verb::Plant(Crop::Tomato),
    Verb::Plant(Crop::Strawberry),
    Verb::Plant(Crop::Melon),
    Verb::PlaceAnimal(Animal::Goose),
    Verb::PlaceAnimal(Animal::Cow),
    Verb::PlaceAnimal(Animal::Sheep),
    Verb::Pickup { item_slot: 0, qty_slot: 0 },
    Verb::Pickup { item_slot: 0, qty_slot: 1 },
    Verb::Pickup { item_slot: 1, qty_slot: 0 },
    Verb::Pickup { item_slot: 1, qty_slot: 1 },
    Verb::Pickup { item_slot: 2, qty_slot: 0 },
    Verb::Pickup { item_slot: 2, qty_slot: 1 },
    Verb::Pickup { item_slot: 3, qty_slot: 0 },
    Verb::Pickup { item_slot: 3, qty_slot: 1 },
    Verb::Pickup { item_slot: 4, qty_slot: 0 },
    Verb::Pickup { item_slot: 4, qty_slot: 1 },
    Verb::Deposit(Product::Wheat),
    Verb::Deposit(Product::Carrot),
    Verb::Deposit(Product::Tomato),
    Verb::Deposit(Product::Strawberry),
    Verb::Deposit(Product::Melon),
    Verb::Deposit(Product::Egg),
    Verb::Deposit(Product::Milk),
    Verb::Deposit(Product::Wool),
    Verb::Deposit(Product::Fertilizer),
    Verb::Drop,
];

impl Verb {
    /// Flat index into the verb head.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Verb::Idle => 0,
            Verb::Water => 1,
            Verb::Harvest => 2,
            Verb::Fertilize => 3,
            Verb::Dig => 4,
            Verb::BuildCoop => 5,
            Verb::BuildPasture => 6,
            Verb::Feed => 7,
            Verb::Care => 8,
            Verb::CollectFertilizer => 9,
            Verb::Plant(c) => 10 + c as usize,
            Verb::PlaceAnimal(a) => 15 + a as usize,
            Verb::Pickup { item_slot, qty_slot } => 18 + 2 * item_slot as usize + qty_slot as usize,
            Verb::Deposit(p) => 28 + p as usize,
            Verb::Drop => 37,
        }
    }

    #[inline]
    pub fn from_index(i: usize) -> Option<Verb> {
        VERBS.get(i).copied()
    }

    /// `PICKUP_ITEMS[item_slot]`.
    #[inline]
    pub fn pickup_item(self) -> Option<ShedItem> {
        match self {
            Verb::Pickup { item_slot, .. } => Some(PICKUP_ITEMS[item_slot as usize]),
            _ => None,
        }
    }

    /// `PICKUP_QTYS[qty_slot]`.
    #[inline]
    pub fn pickup_qty(self) -> Option<i64> {
        match self {
            Verb::Pickup { qty_slot, .. } => Some(PICKUP_QTYS[qty_slot as usize]),
            _ => None,
        }
    }

    /// True for the three shed verbs, whose target must be a shed-access tile.
    #[inline]
    pub const fn is_shed_verb(self) -> bool {
        matches!(self, Verb::Pickup { .. } | Verb::Deposit(_) | Verb::Drop)
    }
}

/// One unit's macro order: a target tile index (`y * board_size + x`) and a verb.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MacroOrder {
    pub target: u8,
    pub verb: Verb,
}

impl Default for MacroOrder {
    #[inline]
    fn default() -> Self {
        MacroOrder {
            target: 0,
            verb: Verb::Idle,
        }
    }
}

impl MacroOrder {
    /// Build from two head indices. `None` if either index is out of range.
    #[inline]
    pub fn from_indices(target: usize, verb: usize) -> Option<Self> {
        if target >= N_TARGETS {
            return None;
        }
        Verb::from_index(verb).map(|verb| MacroOrder {
            target: target as u8,
            verb,
        })
    }

    #[inline]
    pub fn idle() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Pathing
// ---------------------------------------------------------------------------

/// Board size the path table is precomputed for (`Config::default().board_size`).
pub const CACHED_BOARD: usize = 10;

/// `first_step[t][s]` = direction to move from `s` to get closer to `t`
/// (`NO_STEP` when `s == t`); `dist[t][s]` = shortest-path length.
struct PathTable {
    first_step: [[u8; N_TARGETS]; N_TARGETS],
    dist: [[u8; N_TARGETS]; N_TARGETS],
}

const NO_STEP: u8 = 255;
const UNREACHABLE: u8 = 255;

/// The four `FARMER_MOVES` in `Dir` enum order — also the BFS tie-break order.
const DIRS: [Dir; 4] = [Dir::North, Dir::South, Dir::East, Dir::West];

/// Breadth-first search from every target over the 4-neighbour grid.
///
/// **Every tile is passable.** `LOCKED` tiles block tile *operations*, never movement
/// (`_apply_unit_action` sets the new position before any tile lookup), so the graph has
/// no obstacles and BFS distance equals Manhattan distance. The search is kept anyway
/// because it is the definition that stays correct if a future rule adds obstacles, and
/// because it costs nothing: 100 BFS runs once per process, 20 KB of tables, O(1) lookup.
fn build_path_table() -> PathTable {
    let b = CACHED_BOARD as i64;
    let mut t = PathTable {
        first_step: [[NO_STEP; N_TARGETS]; N_TARGETS],
        dist: [[UNREACHABLE; N_TARGETS]; N_TARGETS],
    };
    let mut queue = [0u8; N_TARGETS];
    for target in 0..N_TARGETS {
        // --- BFS from `target` ---
        let mut head = 0usize;
        let mut tail = 0usize;
        t.dist[target][target] = 0;
        queue[tail] = target as u8;
        tail += 1;
        while head < tail {
            let cur = queue[head] as usize;
            head += 1;
            let d = t.dist[target][cur];
            let (cx, cy) = ((cur % CACHED_BOARD) as i64, (cur / CACHED_BOARD) as i64);
            for dir in DIRS {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (cx + dx, cy + dy);
                if nx < 0 || nx >= b || ny < 0 || ny >= b {
                    continue;
                }
                let n = (ny as usize) * CACHED_BOARD + nx as usize;
                if t.dist[target][n] == UNREACHABLE {
                    t.dist[target][n] = d + 1;
                    queue[tail] = n as u8;
                    tail += 1;
                }
            }
        }
        // --- first step: any neighbour strictly closer to `target`, first in Dir order ---
        for src in 0..N_TARGETS {
            if src == target {
                continue;
            }
            let (sx, sy) = ((src % CACHED_BOARD) as i64, (src / CACHED_BOARD) as i64);
            let here = t.dist[target][src];
            for dir in DIRS {
                let (dx, dy) = dir.delta();
                let (nx, ny) = (sx + dx, sy + dy);
                if nx < 0 || nx >= b || ny < 0 || ny >= b {
                    continue;
                }
                let n = (ny as usize) * CACHED_BOARD + nx as usize;
                if t.dist[target][n] + 1 == here {
                    t.first_step[target][src] = dir as u8;
                    break;
                }
            }
        }
    }
    t
}

static PATHS: LazyLock<PathTable> = LazyLock::new(build_path_table);

/// Closed form of the table's tie-break, for board sizes other than [`CACHED_BOARD`].
///
/// On an obstacle-free grid the set of distance-reducing directions is exactly
/// `{North if ty<sy, South if ty>sy, East if tx>sx, West if tx<sx}`, and the table picks
/// the first of those in `Dir` order. `paths_table_matches_closed_form` asserts the two
/// agree on all 10 000 ordered pairs.
#[inline]
fn closed_form_step(sx: i64, sy: i64, tx: i64, ty: i64) -> Option<Dir> {
    if ty < sy {
        Some(Dir::North)
    } else if ty > sy {
        Some(Dir::South)
    } else if tx > sx {
        Some(Dir::East)
    } else if tx < sx {
        Some(Dir::West)
    } else {
        None
    }
}

/// The move that takes a unit at `from` one step along a shortest path to `to`.
///
/// `None` when the unit is already on `to`. Uses the precomputed BFS table on a
/// [`CACHED_BOARD`]-sized board and the equivalent closed form otherwise.
#[inline]
pub fn first_step(from: [i64; 2], to: [i64; 2], board_size: usize) -> Option<Dir> {
    if from == to {
        return None;
    }
    if board_size == CACHED_BOARD {
        let s = from[1] as usize * CACHED_BOARD + from[0] as usize;
        let t = to[1] as usize * CACHED_BOARD + to[0] as usize;
        let d = PATHS.first_step[t][s];
        debug_assert!(d != NO_STEP);
        return DIRS.get(d as usize).copied();
    }
    closed_form_step(from[0], from[1], to[0], to[1])
}

/// Shortest-path length in moves.
#[inline]
pub fn path_len(from: [i64; 2], to: [i64; 2], board_size: usize) -> i64 {
    if board_size == CACHED_BOARD {
        let s = from[1] as usize * CACHED_BOARD + from[0] as usize;
        let t = to[1] as usize * CACHED_BOARD + to[0] as usize;
        return PATHS.dist[t][s] as i64;
    }
    (from[0] - to[0]).abs() + (from[1] - to[1]).abs()
}

/// Moves from `pos` to the nearest shed-access tile — the logistics feature the
/// observation exposes on every tile and unit token.
#[inline]
pub fn dist_to_shed(pos: [i64; 2], board_size: usize) -> i64 {
    let half = (board_size / 2) as i64;
    let dx = (pos[0] - (half - 1)).abs().min((pos[0] - half).abs());
    let dy = (pos[1] - (half - 1)).abs().min((pos[1] - half).abs());
    dx + dy
}

// ---------------------------------------------------------------------------
// Preconditions
// ---------------------------------------------------------------------------

/// Does the target tile's **current state** admit this verb?
///
/// This is the half of "would the verb do something" that depends only on the board, so
/// it is also the half that gates *walking*: a unit is sent towards a target only while
/// the target still looks like a place where the verb can land. A target that has become
/// invalid (plant turned to weed, animal escaped, tile already built on) collapses the
/// order to `PASS` on the very next turn instead of walking the unit somewhere useless.
///
/// `LOCKED` needs no special case: it is neither `Empty` nor a dict, so every tile-verb
/// predicate below already rejects it, while the three shed verbs only require the target
/// to be a shed-access tile — which is exactly the environment's own rule, since `DROP` /
/// `PICKUP` / `PLACE` resolve *before* the `LOCKED` guard and three of the four
/// shed-access tiles start locked.
pub fn target_plausible(verb: Verb, tile: Tile, day: i64, tx: i64, ty: i64, board_size: usize) -> bool {
    match verb {
        Verb::Idle => false,
        Verb::Water => matches!(tile.as_plant(), Some(p) if !p.watered_today),
        Verb::Harvest => match tile {
            Tile::Plant(p) => {
                p.yield_units > 0 && day - p.planted_day >= p.crop.data().first_yield_day
            }
            Tile::Animal(a) => a.yield_units > 0,
            _ => false,
        },
        Verb::Fertilize => tile.as_plant().is_some(),
        // `DIG` no-ops on `None`, is blocked on an occupied structure, and never reaches a
        // `LOCKED` tile. Everything else it clears.
        Verb::Dig => matches!(tile, Tile::Weed | Tile::Coop | Tile::Pasture | Tile::Plant(_)),
        Verb::BuildCoop | Verb::BuildPasture => tile == Tile::Empty,
        Verb::Feed => matches!(tile.as_animal(), Some(a) if !a.fed_today),
        Verb::Care => matches!(tile.as_animal(), Some(a) if !a.cared_today),
        Verb::CollectFertilizer => matches!(tile.as_animal(), Some(a) if a.fertilizer_available),
        Verb::Plant(_) => tile == Tile::Empty,
        Verb::PlaceAnimal(a) => tile == Tile::empty_structure(a.data().structure),
        Verb::Pickup { .. } | Verb::Deposit(_) | Verb::Drop => {
            is_shed_adjacent(tx, ty, board_size)
        }
    }
}

/// Does the unit hold / can the shed supply what the verb consumes?
///
/// Checked only at the moment of emission, never while walking: a unit can legitimately
/// be sent towards a tile it cannot yet act on, because `BUY_SEED` and `BUY_PRODUCT`
/// settle in the market phase and land *after* this turn's unit actions. Walking while
/// the seed is in flight is correct play.
///
/// `plant_budget` is the remaining per-crop seed allowance for this turn (see
/// [`resolve_turn`]); pass `private.seeds` for a stateless "is this verb live at all"
/// query, which is what [`crate::obs::legal_mask`] does.
pub fn resource_ok(
    verb: Verb,
    inv: &Inv,
    shed: &Shed,
    plant_budget: &[i64; N_CROPS],
    shed_capacity: i64,
) -> bool {
    match verb {
        Verb::Fertilize => inv.get(ShedItem::Fertilizer) >= 1,
        Verb::Feed => inv.get(ShedItem::Wheat) >= 1,
        Verb::Plant(c) => plant_budget[c as usize] >= 1,
        Verb::PlaceAnimal(a) => inv.get(a.shed()) >= 1,
        Verb::Pickup { item_slot, .. } => shed.get(PICKUP_ITEMS[item_slot as usize]) >= 1,
        Verb::Deposit(p) => inv.get(p.shed()) >= 1 && shed.total() < shed_capacity,
        // `DROP` deletes every visited key whether or not it fit, so it is only ever
        // emitted when the entire inventory is guaranteed to land.
        Verb::Drop => !inv.is_empty() && shed_capacity - shed.total() >= inv.total(),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Per-turn context
// ---------------------------------------------------------------------------

/// Verbs that can usefully be applied to a given tile at most once per turn. A second
/// unit issuing the same class at the same tile is either a wasted action (the flag is
/// already set, the yield is already taken) or, for `FERTILIZE`, a destroyed item.
///
/// Classes are per-verb rather than per-tile so that genuinely additive combinations
/// still work: `FERTILIZE` then `WATER` on one plant doubles the day's bonus, `FEED` and
/// `CARE` on one animal bank a care bonus, `DIG` then `BUILD_COOP` clears and rebuilds a
/// tile in a single turn.
const CLAIM_WATER: usize = 0;
const CLAIM_FEED: usize = 1;
const CLAIM_CARE: usize = 2;
const CLAIM_COLLECT: usize = 3;
const CLAIM_HARVEST: usize = 4;
const CLAIM_FERTILIZE: usize = 5;
const CLAIM_PLANT: usize = 6;
const N_CLAIM: usize = 7;

#[inline]
fn claim_class(verb: Verb) -> Option<usize> {
    Some(match verb {
        Verb::Water => CLAIM_WATER,
        Verb::Feed => CLAIM_FEED,
        Verb::Care => CLAIM_CARE,
        Verb::CollectFertilizer => CLAIM_COLLECT,
        Verb::Harvest => CLAIM_HARVEST,
        Verb::Fertilize => CLAIM_FERTILIZE,
        Verb::Plant(_) => CLAIM_PLANT,
        _ => return None,
    })
}

/// Scratch state shared by all of one player's units within one turn. Fixed size, stack
/// allocated, no heap traffic.
struct TurnCtx {
    /// Remaining `PLANT` allowance per crop — starts at `private.seeds`.
    plant_budget: [i64; N_CROPS],
    /// `claimed[class]` is a 100-bit set of target tiles already taken by that class.
    claimed: [[u64; 2]; N_CLAIM],
    /// Shed room not yet promised to an earlier unit's `Deposit` / `Drop` this turn.
    ///
    /// Without this, two units both passing the "room for my whole inventory" test would
    /// both `DROP` and the second one's overflow would be deleted — the exact loss the
    /// single-unit check exists to prevent.
    room_left: i64,
}

impl TurnCtx {
    #[inline]
    fn new(seeds: [i64; N_CROPS], room_left: i64) -> Self {
        TurnCtx {
            plant_budget: seeds,
            claimed: [[0; 2]; N_CLAIM],
            room_left,
        }
    }

    #[inline]
    fn is_claimed(&self, class: usize, target: usize) -> bool {
        self.claimed[class][target / 64] & (1u64 << (target % 64)) != 0
    }

    #[inline]
    fn claim(&mut self, class: usize, target: usize) {
        self.claimed[class][target / 64] |= 1u64 << (target % 64);
    }
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Resolve one unit's macro order into the op it should run this turn.
///
/// Order of checks — each one is a deliberate policy about what the wrapper refuses:
/// 1. unit does not exist / verb is `Idle` / target off-board  → `PASS`
/// 2. target tile state does not admit the verb                → `PASS` (target invalid)
/// 3. not standing on the target                               → one move along the path
/// 4. the verb would consume something the unit does not have  → `PASS`
/// 5. another unit already claimed this (tile, verb class)      → `PASS`
/// 6. otherwise                                                → the verb
///
/// Steps 2, 4 and 5 mean the emitted op is always *effective*: it is never a guaranteed
/// no-op and never destroys value the policy did not ask to destroy. Everything the
/// resolver refuses becomes `PASS`, which is exactly what the environment would have made
/// of the raw op anyway — so the policy is not being masked, only spared the pointless
/// walk. `DIG` on a plant that still holds yield is *not* refused: clearing a decaying
/// plant to reuse the tile is legitimate play.
fn resolve_one(
    gs: &GameState,
    player: usize,
    cfg: &Config,
    idx: usize,
    order: Option<MacroOrder>,
    ctx: &mut TurnCtx,
) -> UnitOp {
    let Some(order) = order else {
        return UnitOp::Pass;
    };
    if order.verb == Verb::Idle {
        return UnitOp::Pass;
    }
    let farm = &gs.farms[player];
    let Some(pos) = farm.unit_pos(idx) else {
        return UnitOp::Pass;
    };
    let b = cfg.board_size;
    let target = order.target as usize;
    if target >= b * b {
        return UnitOp::Pass;
    }
    let (tx, ty) = ((target % b) as i64, (target / b) as i64);
    let tile = farm.tile(tx, ty);

    if !target_plausible(order.verb, tile, gs.day, tx, ty, b) {
        return UnitOp::Pass;
    }
    if pos != [tx, ty] {
        return match first_step(pos, [tx, ty], b) {
            Some(d) => UnitOp::Move(d),
            None => UnitOp::Pass,
        };
    }

    let private = &gs.privates[player];
    // `Inv::new()` is an empty `Vec`, so this fallback does not allocate.
    let empty = Inv::new();
    let inv = private.inventories.get(idx).unwrap_or(&empty);
    if !resource_ok(
        order.verb,
        inv,
        &private.shed,
        &ctx.plant_budget,
        cfg.shed_capacity_i64(),
    ) {
        return UnitOp::Pass;
    }
    // Shed room already promised to an earlier unit this turn.
    match order.verb {
        Verb::Deposit(_) if ctx.room_left <= 0 => return UnitOp::Pass,
        Verb::Drop if ctx.room_left < inv.total() => return UnitOp::Pass,
        _ => {}
    }
    if let Some(class) = claim_class(order.verb) {
        if ctx.is_claimed(class, target) {
            return UnitOp::Pass;
        }
        ctx.claim(class, target);
    }
    match order.verb {
        Verb::Plant(c) => ctx.plant_budget[c as usize] -= 1,
        Verb::Deposit(p) => ctx.room_left -= inv.get(p.shed()).min(ctx.room_left),
        Verb::Drop => ctx.room_left -= inv.total(),
        _ => {}
    }

    match order.verb {
        Verb::Idle => UnitOp::Pass,
        Verb::Water => UnitOp::Water,
        Verb::Harvest => UnitOp::Harvest,
        Verb::Fertilize => UnitOp::Fertilize,
        Verb::Dig => UnitOp::Dig,
        Verb::BuildCoop => UnitOp::BuildCoop,
        Verb::BuildPasture => UnitOp::BuildPasture,
        Verb::Feed => UnitOp::Feed,
        Verb::Care => UnitOp::Care,
        Verb::CollectFertilizer => UnitOp::CollectFertilizer,
        Verb::Plant(c) => UnitOp::Plant(c),
        Verb::PlaceAnimal(a) => UnitOp::Place {
            item: a.shed(),
            n: 1,
        },
        Verb::Pickup {
            item_slot,
            qty_slot,
        } => UnitOp::Pickup {
            item: PICKUP_ITEMS[item_slot as usize],
            n: PICKUP_QTYS[qty_slot as usize],
        },
        Verb::Deposit(p) => UnitOp::Place {
            item: p.shed(),
            n: DEPOSIT_QTY,
        },
        Verb::Drop => UnitOp::Drop,
    }
}

/// Resolve a whole turn for one player: every unit's macro order plus the market queue.
///
/// `orders[0]` is the farmer and `orders[1 + h]` is hand `h`; a short slice leaves the
/// remaining units on `PASS`. `out` is cleared and refilled, so reusing one
/// [`PlayerAction`] across steps allocates nothing after the first turn.
///
/// `out.hands` is always exactly `farm.hands.len()` long — extra entries would still
/// count towards the atomic-`PLANT` demand in the interpreter, so emitting them would
/// reintroduce the very trap the seed budget exists to avoid.
pub fn resolve_turn(
    gs: &GameState,
    player: usize,
    cfg: &Config,
    orders: &[MacroOrder],
    market: &MarketHead,
    out: &mut PlayerAction,
) {
    let private = &gs.privates[player];
    let mut ctx = TurnCtx::new(
        private.seeds,
        (cfg.shed_capacity_i64() - private.shed.total()).max(0),
    );
    out.farmer = resolve_one(gs, player, cfg, 0, orders.first().copied(), &mut ctx);
    out.hands.clear();
    for h in 0..gs.farms[player].hands.len() {
        let op = resolve_one(gs, player, cfg, h + 1, orders.get(h + 1).copied(), &mut ctx);
        out.hands.push(op);
    }
    pack_market(gs, player, cfg, market, &mut out.market);
}

// ---------------------------------------------------------------------------
// Market action head
// ---------------------------------------------------------------------------

/// Quantity buckets shared by `BUY_SEED`, `BUY_ANIMAL` and `BUY_PRODUCT`.
///
/// One market order drains its whole `remaining` count inside a single order index (the
/// inner lockstep loop repeats until the quantity is exhausted or a commit fails), so a
/// quantity is *free* in slot terms — one bucket per item is all the packer ever needs.
/// `32` covers a whole 25-tile quadrant of seeds in one order.
pub const QTY_BUCKETS: [i64; 7] = [0, 1, 2, 4, 8, 16, 32];

/// Sell buckets, as a fraction of the shed holding for that product.
///
/// Fractions rather than absolute counts because holdings span three orders of magnitude
/// over an episode and "sell half of what I have" is the scale-free thing a policy wants
/// to say. A non-zero fraction of a non-zero holding always rounds up to at least one unit.
pub const SELL_FRACS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Number of categorical heads in the market action: hire, buy_land, 5 seeds, 3 animals,
/// 2 buy_product, 9 sell fractions.
pub const N_MARKET_HEADS: usize = 21;

/// Categories per market head, in [`MarketHead::from_choices`] order.
pub const MARKET_HEAD_SIZES: [usize; N_MARKET_HEADS] = [
    MAX_HIRE_PER_TURN + 1, // hire count 0..=10
    2,                     // buy_land
    7, 7, 7, 7, 7,         // BUY_SEED  x 5 crops
    7, 7, 7,               // BUY_ANIMAL x 3 animals
    7, 7,                  // BUY_PRODUCT WHEAT, FERTILIZER
    5, 5, 5, 5, 5, 5, 5, 5, 5, // SELL fraction x 9 products
];

/// Largest hire count one turn can express — the market queue is 10 long and each `HIRE`
/// costs one slot, so more than 10 is unrepresentable anyway.
pub const MAX_HIRE_PER_TURN: usize = 10;

/// One turn's market intent, straight off the policy's categorical heads.
///
/// Every field is a *request*; [`pack_market`] decides which requests fit in the
/// 10-order queue and in what order they execute.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MarketHead {
    /// Hands to hire this turn, `0..=MAX_HIRE_PER_TURN`.
    pub hire: u8,
    /// Buy the next quadrant.
    pub buy_land: bool,
    /// Bucket index into [`QTY_BUCKETS`] per crop, in [`CROP_LIST`] order.
    pub seed_qty: [u8; N_CROPS],
    /// Bucket index into [`QTY_BUCKETS`] per animal, in [`ANIMAL_LIST`] order.
    pub animal_qty: [u8; N_ANIMALS],
    /// Bucket index into [`QTY_BUCKETS`] for `[WHEAT, FERTILIZER]`.
    pub buy_product: [u8; 2],
    /// Bucket index into [`SELL_FRACS`] per product, in [`PRODUCTS`] order.
    pub sell_frac: [u8; N_PRODUCTS],
}

/// The two products `BUY_PRODUCT` accepts, in head order.
pub const BUYABLE: [Product; 2] = [Product::Wheat, Product::Fertilizer];

impl MarketHead {
    /// Assemble from one categorical choice per head. Out-of-range choices clamp to the
    /// last category of that head, so a malformed sample can never panic on the hot path.
    pub fn from_choices(choices: &[u8; N_MARKET_HEADS]) -> Self {
        let c = |i: usize| -> u8 {
            let hi = (MARKET_HEAD_SIZES[i] - 1) as u8;
            if choices[i] > hi {
                hi
            } else {
                choices[i]
            }
        };
        let mut h = MarketHead {
            hire: c(0),
            buy_land: c(1) != 0,
            ..Default::default()
        };
        for k in 0..N_CROPS {
            h.seed_qty[k] = c(2 + k);
        }
        for k in 0..N_ANIMALS {
            h.animal_qty[k] = c(7 + k);
        }
        for k in 0..2 {
            h.buy_product[k] = c(10 + k);
        }
        for k in 0..N_PRODUCTS {
            h.sell_frac[k] = c(12 + k);
        }
        h
    }
}

// Inclusion priorities: which requests win a slot when more than 10 are asked for.
const INCL_HIRE: u8 = 0;
const INCL_BUY_PRODUCT: u8 = 1;
const INCL_BUY_LAND: u8 = 2;
const INCL_SELL: u8 = 3;
const INCL_BUY_SEED: u8 = 4;
const INCL_BUY_ANIMAL: u8 = 5;

// Execution priorities: the order the surviving requests are written to the queue.
const EXEC_SELL: u8 = 0;
const EXEC_BUY_PRODUCT: u8 = 1;
const EXEC_BUY_SEED: u8 = 2;
const EXEC_BUY_ANIMAL: u8 = 3;
const EXEC_BUY_LAND: u8 = 4;
const EXEC_HIRE: u8 = 5;

#[derive(Copy, Clone)]
struct Cand {
    incl: u8,
    exec: u8,
    seq: u16,
    ord: MarketOrder,
}

/// Maximum candidates before truncation: 10 hires + 2 buys + 1 land + 9 sells + 5 seeds
/// + 3 animals.
const MAX_CANDS: usize = MAX_HIRE_PER_TURN + 2 + 1 + N_PRODUCTS + N_CROPS + N_ANIMALS;

/// Turn a [`MarketHead`] into an ordered queue of at most `maxMarketOrdersPerTurn` orders.
///
/// # Two different priorities, and why
///
/// Slot *inclusion* and queue *position* are ranked separately, because they are ranked by
/// different things.
///
/// **Inclusion — what deserves one of the 10 slots.** Ranked by what dropping the request
/// permanently costs, since anything dropped can simply be re-issued next turn:
///
/// | order | dropped-for-one-turn cost | rank |
/// |---|---|---|
/// | `HIRE` | a hand hired at hour `h` gets `23 - h` actions and every hand is wiped at midnight, so a deferred hire is permanently one action-day shorter | 1 |
/// | `BUY_PRODUCT` | wheat feeds animals; a missed buy on the second unfed day loses a $300–500 animal outright | 2 |
/// | `BUY_LAND` | ~25 tile-days, and there are only ever three of these in an episode | 3 |
/// | `SELL` | ≈0 — the goods sit in the shed and town consumption *raises* the price while they wait | 4 |
/// | `BUY_SEED` | one turn of planting delay | 5 |
/// | `BUY_ANIMAL` | one turn of production delay on the most expensive purchase in the game | 6 |
///
/// The consequence is deliberate and matches the measured advice in
/// `research/logistics-action-budget.md` §2.2: asking for 10 hands at hour 0 leaves no
/// slots for anything else that turn. The policy controls `hire` and can spend hour 0 on
/// hiring and hour 1 on selling.
///
/// **Position — where in the queue.** `_process_market` walks order index `i` across both
/// players, quotes both against the same pre-commit inventory, commits, then refreshes
/// prices. Two facts follow:
/// * an earlier index sells into a fresher market — measured at **+$644 (12.6%)** on a
///   100-carrot sale when the opponent sold at index 3 instead of index 0 (§7.2);
/// * atomic orders at index `i` see the money left by every commit at index `< i`, so a
///   `SELL` at index 0 *funds* a `BUY_LAND` or `HIRE` later in the same turn.
///
/// So `SELL` goes first, then the buys (which are also cheaper after a sell has pushed
/// market inventory up), then `BUY_LAND`, then `HIRE` last. `HIRE`'s value does not depend
/// on its index at all — a hand hired at index 0 and one hired at index 9 both start
/// acting next turn — which is exactly why it can afford to be last.
///
/// Sells are ordered among themselves by descending `units × current price`, so the
/// largest sale gets the freshest quote.
///
/// # Requests the packer drops as provably dead
///
/// Only cases that *cannot* succeed, so no slot is wasted on a guaranteed no-op:
/// * `SELL` of a product the shed does not hold;
/// * `BUY_PRODUCT` / `BUY_ANIMAL` when the shed will still be at capacity after this
///   turn's sells (`_commit_unit` refuses both at `sum(shed) >= shedCapacity`);
/// * `BUY_LAND` when all three quadrants are already unlocked;
/// * `HIRE` beyond [`MAX_HANDS`], so every unit stays addressable by a macro order;
/// * when the queue contains no sells — and therefore money can only fall during the turn
///   — any purchase the current balance cannot cover, and the unaffordable tail of a hire
///   run priced at `farmHandCostMult * fib(hires_today + j)`. With sells queued the packer
///   makes no affordability judgement at all, because money at execution time is strictly
///   higher than it is now.
pub fn pack_market(
    gs: &GameState,
    player: usize,
    cfg: &Config,
    head: &MarketHead,
    out: &mut Vec<Option<MarketOrder>>,
) {
    out.clear();
    let farm = &gs.farms[player];
    let private = &gs.privates[player];
    let max_orders = cfg.max_orders_eff();

    let mut cands = [Cand {
        incl: 0,
        exec: 0,
        seq: 0,
        ord: MarketOrder::Hire,
    }; MAX_CANDS];
    let mut n = 0usize;
    let mut seq = 0u16;
    let mut push = |c: &mut [Cand; MAX_CANDS], n: &mut usize, incl, exec, ord| {
        c[*n] = Cand {
            incl,
            exec,
            seq,
            ord,
        };
        *n += 1;
        seq += 1;
    };

    // --- sells, ranked by expected revenue ------------------------------------------
    // (item, units, revenue) for the products with a live request.
    let mut sells = [(Product::Wheat, 0i64, 0i64); N_PRODUCTS];
    let mut n_sells = 0usize;
    for item in PRODUCTS {
        let have = private.shed.get(item.shed());
        let frac = SELL_FRACS[head.sell_frac[item.idx()] as usize];
        if have <= 0 || frac <= 0.0 {
            continue;
        }
        let units = if frac >= 1.0 {
            have
        } else {
            (((have as f32) * frac).ceil() as i64).clamp(1, have)
        };
        sells[n_sells] = (item, units, units.saturating_mul(gs.market.price(item)));
        n_sells += 1;
    }
    // Descending revenue, product order as the tie-break; `sort_unstable` never allocates.
    sells[..n_sells].sort_unstable_by_key(|(item, _, rev)| (-*rev, item.idx()));
    let sold_units: i64 = sells[..n_sells].iter().map(|(_, u, _)| *u).sum();

    // With no sells queued nothing can add money this turn, so the current balance is a
    // hard upper bound on money at any order index and affordability is decidable.
    let strict_budget = if n_sells == 0 {
        Some(farm.money)
    } else {
        None
    };
    let affordable = |cost: i64| -> bool {
        match strict_budget {
            Some(m) => m >= cost as f64,
            None => true,
        }
    };

    // --- hires ---------------------------------------------------------------------
    let mut n_hire = (head.hire as usize)
        .min(MAX_HIRE_PER_TURN)
        .min(max_orders)
        .min(MAX_HANDS.saturating_sub(farm.hands.len()));
    if let Some(money) = strict_budget {
        let mut acc = 0f64;
        let mut k = 0usize;
        while k < n_hire {
            let c = hire_cost(farm.hires_today + k as i64, cfg.farm_hand_cost_mult) as f64;
            if acc + c > money {
                break;
            }
            acc += c;
            k += 1;
        }
        n_hire = k;
    }
    for _ in 0..n_hire {
        push(&mut cands, &mut n, INCL_HIRE, EXEC_HIRE, MarketOrder::Hire);
    }

    // --- BUY_PRODUCT ---------------------------------------------------------------
    // A sell at an earlier index frees exactly one shed slot per unit sold, so the shed
    // total the buy will see is `total - sold_units` (deposits by this turn's unit actions
    // can only add, which keeps this test conservative in the safe direction).
    let shed_after_sells = (private.shed.total() - sold_units).max(0);
    let shed_has_room = shed_after_sells < cfg.shed_capacity_i64();
    for (k, item) in BUYABLE.iter().enumerate() {
        let qty = QTY_BUCKETS[head.buy_product[k] as usize];
        if qty <= 0 || !shed_has_room {
            continue;
        }
        // Quoted at `inventory - 1`, which is the price a single unit will cost.
        let price = crate::sim::market_price(*item, gs.market.inv(*item) - 1, &gs.market.params);
        if !affordable(price) {
            continue;
        }
        push(
            &mut cands,
            &mut n,
            INCL_BUY_PRODUCT,
            EXEC_BUY_PRODUCT,
            MarketOrder::BuyProduct {
                item: *item,
                n: qty,
            },
        );
    }

    // --- BUY_LAND -------------------------------------------------------------------
    let extra_unlocked = farm.unlocked_quadrants.len() - 1;
    if head.buy_land && extra_unlocked < LAND_ORDER.len() && affordable(LAND_PRICES[extra_unlocked])
    {
        push(
            &mut cands,
            &mut n,
            INCL_BUY_LAND,
            EXEC_BUY_LAND,
            MarketOrder::BuyLand,
        );
    }

    // --- sells ----------------------------------------------------------------------
    for &(item, units, _) in &sells[..n_sells] {
        push(
            &mut cands,
            &mut n,
            INCL_SELL,
            EXEC_SELL,
            MarketOrder::Sell { item, n: units },
        );
    }

    // --- BUY_SEED (no shed check: seeds live outside the shed and are uncapped) ------
    for crop in CROP_LIST {
        let qty = QTY_BUCKETS[head.seed_qty[crop.idx()] as usize];
        if qty <= 0 || !affordable(crop.data().seed) {
            continue;
        }
        push(
            &mut cands,
            &mut n,
            INCL_BUY_SEED,
            EXEC_BUY_SEED,
            MarketOrder::BuySeed { crop, n: qty },
        );
    }

    // --- BUY_ANIMAL -----------------------------------------------------------------
    for animal in ANIMAL_LIST {
        let qty = QTY_BUCKETS[head.animal_qty[animal.idx()] as usize];
        if qty <= 0 || !shed_has_room || !affordable(animal.data().cost) {
            continue;
        }
        push(
            &mut cands,
            &mut n,
            INCL_BUY_ANIMAL,
            EXEC_BUY_ANIMAL,
            MarketOrder::BuyAnimal { animal, n: qty },
        );
    }

    // --- select by inclusion priority, then emit in execution order ------------------
    let keep = &mut cands[..n];
    keep.sort_unstable_by_key(|c| (c.incl, c.seq));
    let kept = n.min(max_orders);
    let kept_slice = &mut keep[..kept];
    kept_slice.sort_unstable_by_key(|c| (c.exec, c.seq));
    for c in kept_slice.iter() {
        out.push(Some(c.ord));
    }
    debug_assert!(out.len() <= max_orders);
}
