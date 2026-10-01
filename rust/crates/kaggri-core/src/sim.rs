//! The interpreter port: every helper in `kaggriculture.py`, in the same order, plus
//! [`step`] for the `interpreter()` body itself.
//!
//! Read this file next to the Python source; the function names match one-for-one
//! (`_shape` → [`shape`], `_apply_unit_action` → [`apply_unit_action`], …). The
//! comments call out the eleven confirmed parity traps where they bite.

use crate::actions::{MarketOrder, PlayerAction, UnitOp};
use crate::consts::*;
use crate::pyrng::{py_round, PyRandom};
use crate::state::*;

// ---------------------------------------------------------------------------
// Market price curve
// ---------------------------------------------------------------------------

/// `_shape(func, x, T)`.
///
/// `x = max(0.0, x)` first, then the shape. `hinge` degenerates to `linear` when
/// `T` is missing or non-positive (`if not T or T <= 0`).
#[inline]
pub fn shape(func: Shape, x: f64, t: f64) -> f64 {
    let x = x.max(0.0);
    match func {
        Shape::Linear => x,
        Shape::Sq => x * x,
        Shape::Sqrt => x.sqrt(),
        Shape::Log => (1.0 + x).ln(),
        Shape::Log10 => (1.0 + x).log10(),
        Shape::Hinge => {
            if !(t > 0.0) {
                return x;
            }
            let u = x / t;
            let over = (u - 1.0).max(0.0);
            // Python writes `HINGE_GAIN * max(0.0, u - 1.0) ** 2`; `x ** 2` and `x * x`
            // are both the correctly-rounded square, so they agree bit for bit.
            u + HINGE_GAIN * (over * over)
        }
    }
}

/// The unrounded `price` inside `market_price`.
///
/// Kept separate so the f64 arithmetic can be compared bit-for-bit against CPython.
/// The order of operations is exactly Python's: `amp = target * base / f(T)` and then
/// `base ± amp * f(|inv - I0|)`.
#[inline]
pub fn raw_market_price(p: &ItemParams, inventory: i64) -> f64 {
    if inventory < p.i0 {
        let amp = p.below_target * p.base / shape(p.below_func, p.t, p.t);
        p.base + amp * shape(p.below_func, (p.i0 - inventory) as f64, p.t)
    } else {
        let amp = p.above_target * p.base / shape(p.above_func, p.t, p.t);
        p.base - amp * shape(p.above_func, (inventory - p.i0) as f64, p.t)
    }
}

/// `market_price(item, inventory, params)` — `max(PRICE_FLOOR, int(round(price)))`.
///
/// **Trap 1**: `round` is Python's banker's rounding (half to even), not Rust's
/// `f64::round` (half away from zero). CARROT at inventory 9595 lands on exactly 66.5
/// and must come out 66, not 67. [`crate::pyrng::py_round`] is the only rounding
/// allowed here.
#[inline]
pub fn market_price(item: Product, inventory: i64, params: &MarketParams) -> i64 {
    let price = raw_market_price(params.get(item), inventory);
    let rounded = py_round(price);
    if PRICE_FLOOR > rounded {
        PRICE_FLOOR
    } else {
        rounded
    }
}

/// `_refresh_prices(market)`.
#[inline]
pub fn refresh_prices(market: &mut Market) {
    for item in PRODUCTS {
        market.prices[item.idx()] = market_price(item, market.inventory[item.idx()], &market.params);
    }
}

// ---------------------------------------------------------------------------
// Tile constructors
// ---------------------------------------------------------------------------

/// `_new_plant(crop, day, turns_per_day)`.
#[inline]
pub fn new_plant(crop: Crop, day: i64, turns_per_day: i64) -> PlantTile {
    let cd = crop.data();
    PlantTile {
        crop,
        planted_day: day,
        watered_today: false,
        // The planting day counts as unwatered.
        consecutive_unwatered: 1,
        yield_units: if cd.ongoing { 0 } else { 1 },
        max_lifespan_step: if cd.ongoing {
            -1
        } else {
            (day + cd.max_yield_day + 1) * turns_per_day
        },
        fertilized_until_day: -1,
    }
}

/// `_new_animal(animal, day)`.
#[inline]
pub fn new_animal(animal: Animal, day: i64) -> AnimalTile {
    AnimalTile {
        animal,
        placed_day: day,
        yield_units: 0,
        consecutive_unfed: 0,
        fed_today: false,
        cared_today: false,
        fertilizer_available: false,
        pending_care_bonus: 0,
    }
}

// ---------------------------------------------------------------------------
// Unit actions
// ---------------------------------------------------------------------------

/// The shared body of `DROP` and `_drop_inventories_to_shed`.
///
/// **Trap 2**: the shed room is recomputed per item and the remainder of an item that
/// does not fit is *discarded*, so insertion order decides what survives. Python
/// snapshots `list(inv.items())` and `del`s every key it visits, which always leaves
/// the inventory empty.
fn drop_inv_into_shed(shed: &mut Shed, inv: &mut Inv, capacity: i64) {
    for (item, n) in inv.drain_ordered() {
        if n <= 0 {
            continue;
        }
        let room = (capacity - shed.total()).max(0);
        let take = n.min(room);
        if take > 0 {
            shed.add(item, take);
        }
    }
}

/// `_apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)`.
///
/// **Trap 5**: the op order is movement → `PASS` → `DROP`/`PICKUP`/`PLACE` → the
/// `LOCKED` guard → everything else. Shed ops deliberately run before the guard
/// because three of the four shed-access tiles start locked.
pub fn apply_unit_action(
    farm: &mut Farm,
    private: &mut Private,
    idx: usize,
    op: UnitOp,
    board_size: usize,
    day: i64,
    turns_per_day: i64,
    shed_capacity: i64,
) {
    // `pos = _farmer_position(farm, idx); if pos is None: return`
    let Some(pos) = farm.unit_pos(idx) else {
        return;
    };
    let (fx, fy) = (pos[0], pos[1]);
    // `inv = _farmer_inventory(private, idx)` — grows `inventories` to cover idx. The
    // invariant `len(inventories) == 1 + len(hands)` makes this a no-op in practice.
    private.farmer_inventory(idx);

    match op {
        UnitOp::Move(dir) => {
            let (dx, dy) = dir.delta();
            let (nx, ny) = (fx + dx, fy + dy);
            let bs = board_size as i64;
            if !(nx >= 0 && nx < bs && ny >= 0 && ny < bs) {
                return;
            }
            // Movement onto LOCKED tiles is allowed.
            farm.set_unit_pos(idx, [nx, ny]);
            return;
        }
        // `PASS`, and every malformed/unknown op, falls straight through with no
        // side effects.
        UnitOp::Pass | UnitOp::Noop => return,
        _ => {}
    }

    let tile = farm.tile(fx, fy);

    // ---- shed operations: before the LOCKED guard -------------------------
    match op {
        UnitOp::Drop => {
            if !is_shed_adjacent(fx, fy, board_size) {
                return;
            }
            let Private {
                shed, inventories, ..
            } = private;
            drop_inv_into_shed(shed, &mut inventories[idx], shed_capacity);
            return;
        }
        UnitOp::Pickup { item, n } => {
            if !is_shed_adjacent(fx, fy, board_size) {
                return;
            }
            if n <= 0 {
                return;
            }
            // Seeds live in private["seeds"] and never pass through here.
            let available = private.shed.get(item);
            let n = n.min(available);
            if n <= 0 {
                return;
            }
            private.shed.add(item, -n);
            private.inventories[idx].add(item, n);
            return;
        }
        UnitOp::Place { item, n } => {
            // Animal placement: standing on a matching *unoccupied* structure. A
            // LOCKED tile is the string "LOCKED", never a dict, so it cannot match
            // and PLACE falls through to the shed path.
            if let Some(animal) = item.animal() {
                if tile == Tile::empty_structure(animal.data().structure) {
                    if private.inventories[idx].take(item, 1) {
                        farm.set_tile(fx, fy, Tile::Animal(new_animal(animal, day)));
                    }
                    return;
                }
            }
            // Shed drop.
            if is_shed_adjacent(fx, fy, board_size) {
                if n <= 0 {
                    return;
                }
                let mut n = n.min(private.inventories[idx].get(item));
                if n <= 0 {
                    return;
                }
                let current = private.shed.total();
                let room = (shed_capacity - current).max(0);
                n = n.min(room);
                if n <= 0 {
                    return;
                }
                private.inventories[idx].sub_existing(item, n);
                private.shed.add(item, n);
            }
            return;
        }
        _ => {}
    }

    // ---- everything below mutates the tile, so the tile must be owned ----
    if tile == Tile::Locked {
        return;
    }

    match op {
        UnitOp::Plant(crop) => {
            // `if tile is not None: return`
            if tile != Tile::Empty {
                return;
            }
            if private.seeds[crop.idx()] <= 0 {
                return;
            }
            private.seeds[crop.idx()] -= 1;
            farm.set_tile(fx, fy, Tile::Plant(new_plant(crop, day, turns_per_day)));
        }
        UnitOp::Water => {
            let Some(mut p) = tile.as_plant() else {
                return;
            };
            if p.watered_today {
                return;
            }
            p.watered_today = true;
            let cd = p.crop.data();
            if !cd.ongoing {
                let age_days = day - p.planted_day;
                let window_start = (cd.max_yield_day + 1).div_euclid(2);
                if window_start <= age_days && age_days <= cd.max_yield_day {
                    let bonus = if p.fertilized_until_day >= day { 2 } else { 1 };
                    p.yield_units = cd.max_yield.min(p.yield_units + bonus);
                }
            }
            farm.set_tile(fx, fy, Tile::Plant(p));
        }
        UnitOp::Harvest => {
            if !tile.is_dict() {
                return;
            }
            if tile.yield_units() <= 0 {
                return;
            }
            match tile {
                Tile::Plant(mut p) => {
                    let cd = p.crop.data();
                    if day - p.planted_day < cd.first_yield_day {
                        // Python prints a warning here for ongoing crops; it is
                        // unreachable (ongoing yield_units only appear on or after
                        // first_yield_day).
                        return;
                    }
                    let units = p.yield_units;
                    p.yield_units = 0;
                    private.inventories[idx].add(p.crop.shed(), units);
                    if cd.ongoing {
                        farm.set_tile(fx, fy, Tile::Plant(p));
                    } else {
                        farm.set_tile(fx, fy, Tile::Empty);
                    }
                }
                Tile::Animal(mut a) => {
                    let units = a.yield_units;
                    a.yield_units = 0;
                    private.inventories[idx].add(a.animal.data().product.shed(), units);
                    farm.set_tile(fx, fy, Tile::Animal(a));
                }
                // WEED / COOP / PASTURE have no yield_units and returned above.
                _ => {}
            }
        }
        UnitOp::Fertilize => {
            let Some(mut p) = tile.as_plant() else {
                return;
            };
            if !private.inventories[idx].take(ShedItem::Fertilizer, 1) {
                return;
            }
            // Active for day, day+1, day+2.
            p.fertilized_until_day = p.fertilized_until_day.max(day + 2);
            farm.set_tile(fx, fy, Tile::Plant(p));
        }
        UnitOp::Dig => {
            if tile == Tile::Empty {
                return;
            }
            // Does NOT remove a placed animal.
            if tile.as_animal().is_some() {
                return;
            }
            farm.set_tile(fx, fy, Tile::Empty);
        }
        UnitOp::BuildCoop => {
            if tile != Tile::Empty {
                return;
            }
            farm.set_tile(fx, fy, Tile::Coop);
        }
        UnitOp::BuildPasture => {
            if tile != Tile::Empty {
                return;
            }
            farm.set_tile(fx, fy, Tile::Pasture);
        }
        UnitOp::Feed => {
            let Some(mut a) = tile.as_animal() else {
                return;
            };
            if a.fed_today {
                return;
            }
            if !private.inventories[idx].take(ShedItem::Wheat, 1) {
                return;
            }
            a.fed_today = true;
            farm.set_tile(fx, fy, Tile::Animal(a));
        }
        UnitOp::CollectFertilizer => {
            let Some(mut a) = tile.as_animal() else {
                return;
            };
            if !a.fertilizer_available {
                return;
            }
            a.fertilizer_available = false;
            private.inventories[idx].add(ShedItem::Fertilizer, 1);
            farm.set_tile(fx, fy, Tile::Animal(a));
        }
        UnitOp::Care => {
            let Some(mut a) = tile.as_animal() else {
                return;
            };
            if a.cared_today {
                return;
            }
            a.cared_today = true;
            farm.set_tile(fx, fy, Tile::Animal(a));
        }
        // Handled above.
        UnitOp::Move(_)
        | UnitOp::Pass
        | UnitOp::Noop
        | UnitOp::Drop
        | UnitOp::Pickup { .. }
        | UnitOp::Place { .. } => {}
    }
}

// ---------------------------------------------------------------------------
// Hiring / land
// ---------------------------------------------------------------------------

/// `_spawn_hand(farm, board_size)` — first free shed-access tile in NWSE order, ties
/// broken by minimum occupancy.
pub fn spawn_hand(farm: &Farm, board_size: usize) -> Pos {
    let tiles = shed_access_tiles(board_size);
    let mut occupants = [0i64; 4];
    for pos in core::iter::once(farm.farmer).chain(farm.hands.iter().copied()) {
        for (i, t) in tiles.iter().enumerate() {
            if *t == pos {
                occupants[i] += 1;
            }
        }
    }
    // sorted(key=(count, nwse_index)) then take the first: a strict `<` scan keeps the
    // earliest NWSE tile on ties, which is what the stable sort does.
    let mut best = 0usize;
    for i in 1..4 {
        if occupants[i] < occupants[best] {
            best = i;
        }
    }
    tiles[best]
}

/// `_fib(n)` — indexed so `_fib(0) == 1`, `_fib(1) == 1`, `_fib(2) == 2`.
///
/// Python is arbitrary-precision; this saturates at `i64::MAX`, which is only ever
/// reached for `n >= 92` where the true cost already exceeds any reachable balance.
pub fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (1i64, 1i64);
    let mut i = 0;
    while i < n {
        let next = a.saturating_add(b);
        a = b;
        b = next;
        if a == i64::MAX {
            break;
        }
        i += 1;
    }
    a
}

/// `_hire_cost(n_already_today, mult)`.
pub fn hire_cost(n_already_today: i64, mult: i64) -> i64 {
    mult.saturating_mul(fib(n_already_today))
}

/// `_do_hire(farm, private, board_size, mult)`.
pub fn do_hire(farm: &mut Farm, private: &mut Private, board_size: usize, mult: i64) {
    let cost = hire_cost(farm.hires_today, mult);
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    farm.hires_today += 1;
    // Python evaluates `_spawn_hand(farm, board_size)` before appending.
    let pos = spawn_hand(farm, board_size);
    farm.hands.push(pos);
    private.inventories.push(Inv::new());
}

/// `_do_buy_land(farm, board_size)`.
pub fn do_buy_land(farm: &mut Farm, board_size: usize) {
    debug_assert_eq!(board_size, farm.board_size);
    // NW is always there.
    let n_unlocked_extra = farm.unlocked_quadrants.len() - 1;
    if n_unlocked_extra >= LAND_ORDER.len() {
        return;
    }
    let cost = LAND_PRICES[n_unlocked_extra];
    if farm.money < cost as f64 {
        return;
    }
    farm.money -= cost as f64;
    let quadrant = LAND_ORDER[n_unlocked_extra];
    farm.unlocked_quadrants.push(quadrant);
    for y in 0..board_size as i64 {
        for x in 0..board_size as i64 {
            if quadrant_of(x, y, board_size) == quadrant && farm.tile(x, y) == Tile::Locked {
                farm.set_tile(x, y, Tile::Empty);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Market processing
// ---------------------------------------------------------------------------

/// "Neither seat is a phantom" — the value that reproduces the pre-phantom code path
/// exactly, and the one every unqualified entry point passes.
pub const NO_PHANTOM: [bool; 2] = [false, false];

/// The four per-unit market operations, with their already-resolved item.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommitOp {
    Sell(Product),
    BuyProduct(Product),
    BuySeed(Crop),
    BuyAnimal(Animal),
}

/// The market half of a **phantom** seat's order: a pure exogenous market impulse.
///
/// A phantom seat is not a player. It has no shed, no bank and no units; it exists only
/// to push the shared market the way a real opponent's realised order stream would. So
/// its `SELL` always fills (no shed requirement, no proceeds) and its `BUY_PRODUCT`
/// always fills (no cash requirement, no shed space requirement) — the *market* side of
/// [`commit_unit`] runs, the *farm* side does not. `BUY_SEED` / `BUY_ANIMAL` never reach
/// here (they are dropped while the order is quoted, exactly as `HIRE`/`BUY_LAND` are).
///
/// **Trap 8 still holds**: a unit sold at the $1 floor does not increase supply, for the
/// phantom exactly as for a player. Losing that here would let a phantom stream refill
/// a glutted market for free and silently break the floor's infinite sink.
#[inline]
pub fn commit_unit_phantom(op: CommitOp, price: i64, market: &mut Market) -> bool {
    match op {
        CommitOp::Sell(item) => {
            if price > 1 {
                market.inventory[item.idx()] += 1;
            }
            true
        }
        CommitOp::BuyProduct(item) => {
            market.inventory[item.idx()] -= 1;
            true
        }
        // Unreachable: `process_market_with_phantom` clears these before quoting.
        CommitOp::BuySeed(_) | CommitOp::BuyAnimal(_) => false,
    }
}

/// `_commit_unit(op, item, price, farm, private, market, shed_capacity)`.
pub fn commit_unit(
    op: CommitOp,
    price: i64,
    farm: &mut Farm,
    private: &mut Private,
    market: &mut Market,
    shed_capacity: i64,
) -> bool {
    match op {
        CommitOp::Sell(item) => {
            let slot = item.shed();
            if private.shed.get(slot) <= 0 {
                return false;
            }
            private.shed.add(slot, -1);
            farm.money += price as f64;
            // **Trap 8**: sales at the $1 floor do not increase market supply, which
            // keeps the floor responsive (an infinite sink).
            if price > 1 {
                market.inventory[item.idx()] += 1;
            }
            true
        }
        CommitOp::BuyProduct(item) => {
            if farm.money < price as f64 {
                return false;
            }
            if private.shed.total() >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed.add(item.shed(), 1);
            market.inventory[item.idx()] -= 1;
            true
        }
        CommitOp::BuySeed(crop) => {
            if farm.money < price as f64 {
                return false;
            }
            farm.money -= price as f64;
            // Seeds live outside the shed and are uncapped.
            private.seeds[crop.idx()] += 1;
            true
        }
        CommitOp::BuyAnimal(animal) => {
            if farm.money < price as f64 {
                return false;
            }
            if private.shed.total() >= shed_capacity {
                return false;
            }
            farm.money -= price as f64;
            private.shed.add(animal.shed(), 1);
            true
        }
    }
}

/// `_process_market(state, env)`.
///
/// **Trap 8**: queues truncated to `maxMarketOrdersPerTurn`; order index `i` advanced
/// across both players together; `HIRE`/`BUY_LAND` handled once in player order and
/// then cleared; a per-unit loop that quotes both players from the same pre-commit
/// inventory and then commits in player order; `SELL` quotes at the current inventory
/// while `BUY_PRODUCT` quotes at `inventory - 1`; a failed commit aborts the rest of
/// that player's order; the loop breaks when nothing committed; `_refresh_prices`
/// after every order index; 100k-iteration escape hatch.
pub fn process_market(gs: &mut GameState, actions: &[PlayerAction; 2], cfg: &Config) {
    process_market_with_phantom(gs, actions, cfg, NO_PHANTOM);
}

/// [`process_market`] with a per-seat **phantom** flag.
///
/// `phantom[p]` turns seat `p` into an exogenous market-pressure stream: its `HIRE`,
/// `BUY_LAND`, `BUY_SEED` and `BUY_ANIMAL` orders are dropped, and its `SELL` /
/// `BUY_PRODUCT` orders commit through [`commit_unit_phantom`] instead of
/// [`commit_unit`], so they always fill and touch nothing but `market.inventory`.
/// Quoting, the order-index advance, the per-unit lockstep, the abort-on-failed-commit
/// rule, the break-when-nothing-committed rule and the `refresh_prices` cadence are
/// **unchanged** — the phantom is interleaved against the real seat exactly as a real
/// opponent would be.
///
/// `phantom == NO_PHANTOM` runs the identical code path as before this existed.
pub fn process_market_with_phantom(
    gs: &mut GameState,
    actions: &[PlayerAction; 2],
    cfg: &Config,
    phantom: [bool; 2],
) {
    let max_orders = cfg.max_orders_eff();
    let hire_mult = cfg.farm_hand_cost_mult;
    let shed_capacity = cfg.shed_capacity_i64();
    let board_size = cfg.board_size;

    let qlen = [
        actions[0].market.len().min(max_orders),
        actions[1].market.len().min(max_orders),
    ];
    let max_len = qlen[0].max(qlen[1]);

    for i in 0..max_len {
        // `_parse_order` already ran; `None` covers both "no order in this slot" and
        // "order whose item is invalid for its op" (see actions.rs).
        let mut order_states: [Option<(MarketOrder, i64)>; 2] = [None, None];
        for p in 0..2 {
            if i < qlen[p] {
                order_states[p] = actions[p].market[i].map(|o| (o, o.quantity()));
            }
        }

        // Atomic orders (HIRE, BUY_LAND): handle once, in player order.
        for p in 0..2 {
            match order_states[p] {
                Some((MarketOrder::Hire, _)) => {
                    // A phantom seat has no farm to hire onto: drop the order, exactly as
                    // the real path drops it after acting on it.
                    if !phantom[p] {
                        do_hire(&mut gs.farms[p], &mut gs.privates[p], board_size, hire_mult);
                    }
                    order_states[p] = None;
                }
                Some((MarketOrder::BuyLand, _)) => {
                    if !phantom[p] {
                        do_buy_land(&mut gs.farms[p], board_size);
                    }
                    order_states[p] = None;
                }
                // A phantom seat has no seed store and no shed, so these two are dropped
                // here rather than quoted -- the same shape as the atomic pass above, and
                // it keeps the per-unit loop's phantom commit total over its two arms.
                Some((MarketOrder::BuySeed { .. }, _)) | Some((MarketOrder::BuyAnimal { .. }, _))
                    if phantom[p] =>
                {
                    order_states[p] = None;
                }
                _ => {}
            }
        }

        // Per-unit lockstep loop for SELL / BUY_*.
        let mut idx_esc: u32 = 0;
        loop {
            idx_esc += 1;
            if idx_esc >= 100_000 {
                // Python prints a warning and breaks.
                break;
            }
            let mut quoted: [Option<(CommitOp, i64)>; 2] = [None, None];
            for p in 0..2 {
                let Some((order, remaining)) = order_states[p] else {
                    continue;
                };
                if remaining <= 0 {
                    continue;
                }
                quoted[p] = Some(match order {
                    MarketOrder::Sell { item, .. } => (
                        CommitOp::Sell(item),
                        market_price(item, gs.market.inv(item), &gs.market.params),
                    ),
                    // Quote at the post-buy inventory so a buy/sell round trip against
                    // an unchanged market nets zero.
                    MarketOrder::BuyProduct { item, .. } => (
                        CommitOp::BuyProduct(item),
                        market_price(item, gs.market.inv(item) - 1, &gs.market.params),
                    ),
                    MarketOrder::BuySeed { crop, .. } => {
                        (CommitOp::BuySeed(crop), crop.data().seed)
                    }
                    MarketOrder::BuyAnimal { animal, .. } => {
                        (CommitOp::BuyAnimal(animal), animal.data().cost)
                    }
                    // Unreachable: the atomic pass above cleared these. Python's
                    // `else` branch aborts the order, which is what this does.
                    MarketOrder::Hire | MarketOrder::BuyLand => {
                        order_states[p] = None;
                        continue;
                    }
                });
            }

            if quoted[0].is_none() && quoted[1].is_none() {
                break;
            }

            // Both players saw the same pre-commit inventory; now commit in order.
            let mut committed_any = false;
            for p in 0..2 {
                let Some((cop, price)) = quoted[p] else {
                    continue;
                };
                let ok = if phantom[p] {
                    commit_unit_phantom(cop, price, &mut gs.market)
                } else {
                    commit_unit(
                        cop,
                        price,
                        &mut gs.farms[p],
                        &mut gs.privates[p],
                        &mut gs.market,
                        shed_capacity,
                    )
                };
                if ok {
                    if let Some(st) = order_states[p].as_mut() {
                        st.1 -= 1;
                    }
                    committed_any = true;
                } else {
                    // Can't continue this order.
                    order_states[p] = None;
                }
            }

            if !committed_any {
                break;
            }
        }

        refresh_prices(&mut gs.market);
    }
}

// ---------------------------------------------------------------------------
// Town
// ---------------------------------------------------------------------------

/// `_town_consume(env, state, step)` — `step` is the **pre-increment** step.
pub fn town_consume(gs: &mut GameState, cfg: &Config, step: i64) {
    let shop_interval = cfg.shop_sell_interval_eff();
    let center_interval = cfg.center_sell_interval_eff();

    if step % shop_interval == 0 {
        // unlocked_shops may list the same shop more than once; each instance
        // consumes independently.
        for shop in &gs.town.unlocked_shops {
            let multiplier = shop.multiplier();
            for item in shop.products() {
                gs.market.inventory[item.idx()] -= multiplier;
            }
        }
    }

    if step % center_interval == 0 {
        for item in TOWN_CENTER_PRODUCTS {
            gs.market.inventory[item.idx()] -= 1;
        }
    }

    refresh_prices(&mut gs.market);
}

// ---------------------------------------------------------------------------
// Daily / per-step refreshes
// ---------------------------------------------------------------------------

/// `_decay_plants(farm, step)`.
///
/// **Trap 9**: called every step, but only acts once `step >= max_lifespan_step` and
/// then only on every other step (`(step - mls) % 2 == 0`).
pub fn decay_plants(farm: &mut Farm, step: i64) {
    // tiles are row-major, so a flat scan is `for y: for x:`.
    for i in 0..farm.tiles.len() {
        let Tile::Plant(p) = &mut farm.tiles[i] else {
            continue;
        };
        let mls = p.max_lifespan_step;
        if mls < 0 || step < mls {
            continue;
        }
        if (step - mls) % 2 != 0 {
            continue;
        }
        p.yield_units -= 1;
        if p.yield_units <= 0 {
            farm.tiles[i] = Tile::Weed;
        }
    }
}

/// `_daily_refresh_plants(farm, current_day, turns_per_day)`.
///
/// **Trap 11**: `was_watered` is captured before `watered_today` is reset; the weed
/// conversion at `consecutive_unwatered >= 2` happens *before* the production logic
/// (a plant that dies tonight produces nothing); the fertilizer bonus needs the plant
/// to have been watered that same day; hitting `max_yield` sets
/// `max_lifespan_step = (next_day + 1) * turns_per_day`.
pub fn daily_refresh_plants(farm: &mut Farm, current_day: i64, turns_per_day: i64) {
    let next_day = current_day + 1;
    for i in 0..farm.tiles.len() {
        let Tile::Plant(p) = &mut farm.tiles[i] else {
            continue;
        };
        let was_watered = p.watered_today;
        if was_watered {
            p.consecutive_unwatered = 0;
        } else {
            p.consecutive_unwatered += 1;
        }
        p.watered_today = false;
        if p.consecutive_unwatered >= 2 {
            farm.tiles[i] = Tile::Weed;
            continue;
        }
        let cd = p.crop.data();
        if !cd.ongoing {
            continue;
        }
        let days_since_first = next_day - p.planted_day - cd.first_yield_day;
        if days_since_first < 0 {
            continue;
        }
        let interval = cd.interval;
        debug_assert!(interval > 0, "ongoing crops always have interval >= 1");
        if days_since_first % interval != 0 {
            continue;
        }
        let production_count = days_since_first / interval + 1;
        if production_count > cd.max_yield {
            continue;
        }
        // Fertilizer bonus only applies on watered days (basic needs first).
        let fertilized = was_watered && p.fertilized_until_day >= current_day;
        p.yield_units = cd.max_yield.min(p.yield_units + if fertilized { 2 } else { 1 });
        if production_count == cd.max_yield {
            p.max_lifespan_step = (next_day + 1) * turns_per_day;
        }
    }
}

/// `_daily_refresh_animals(farm, day)`.
///
/// **Trap 10**: feed/escape → production (which consumes `pending_care_bonus`, and
/// only when `fed_today`) → *then* today's care increment → `fertilizer_available`
/// for every survivor → reset `fed_today`/`cared_today`. A day's care can never pay
/// out on that same day's production.
pub fn daily_refresh_animals(farm: &mut Farm, day: i64) {
    let next_day = day + 1;
    for i in 0..farm.tiles.len() {
        let Tile::Animal(a) = &mut farm.tiles[i] else {
            continue;
        };
        if a.fed_today {
            a.consecutive_unfed = 0;
        } else {
            a.consecutive_unfed += 1;
        }
        if a.consecutive_unfed >= 2 {
            // Animal escapes; structure remains.
            let structure = a.animal.data().structure;
            farm.tiles[i] = Tile::empty_structure(structure);
            continue;
        }
        let ad = a.animal.data();
        let days_since_first = next_day - a.placed_day - ad.first_yield_day;
        debug_assert!(ad.interval > 0);
        if days_since_first >= 0 && days_since_first % ad.interval == 0 {
            let base = 1;
            // `tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0`, and
            // then the bank is reset to 0 either way.
            let bonus = if a.fed_today { a.pending_care_bonus } else { 0 };
            a.yield_units = ad.max_held.min(a.yield_units + base + bonus);
            a.pending_care_bonus = 0;
        }
        if a.cared_today && a.fed_today {
            a.pending_care_bonus += 1;
        }
        a.fertilizer_available = true;
        a.fed_today = false;
        a.cared_today = false;
    }
}

/// `_spawn_weeds(farm, board_size, weed_chance, rng)`.
///
/// **Trap 3**: `rng.random()` is only drawn for tiles that are currently `None`, so
/// the number of draws consumed depends on the board. The iteration is `y` outer,
/// `x` inner.
pub fn spawn_weeds(farm: &mut Farm, board_size: usize, weed_chance: f64, rng: &mut PyRandom) {
    for y in 0..board_size {
        for x in 0..board_size {
            let i = y * farm.board_size + x;
            if farm.tiles[i] == Tile::Empty && rng.random() < weed_chance {
                farm.tiles[i] = Tile::Weed;
            }
        }
    }
}

/// `_drop_inventories_to_shed(private, capacity)`.
pub fn drop_inventories_to_shed(private: &mut Private, capacity: i64) {
    let Private {
        shed, inventories, ..
    } = private;
    for inv in inventories.iter_mut() {
        drop_inv_into_shed(shed, inv, capacity);
    }
}

/// `_end_of_day(state, env, day)`.
///
/// **Trap 3**: one `random.Random((seed * 1_000_003) ^ day)` is shared by, in order,
/// player 0's weed spawn, player 1's weed spawn, and then the shop draw.
pub fn end_of_day(gs: &mut GameState, cfg: &Config, day: i64) {
    let board_size = cfg.board_size;
    let turns_per_day = cfg.turns_per_day_eff();
    let weed_chance = cfg.weed_spawn_chance;
    let shed_cap = cfg.shed_capacity_i64();
    let shop_interval = cfg.shop_unlock_interval_eff();

    // Stable RNG keyed off env.info["seed"] + day so replays reproduce.
    let mut rng = PyRandom::for_day(gs.seed, day);

    for p in 0..2 {
        daily_refresh_plants(&mut gs.farms[p], day, turns_per_day);
        daily_refresh_animals(&mut gs.farms[p], day);
        spawn_weeds(&mut gs.farms[p], board_size, weed_chance, &mut rng);
        drop_inventories_to_shed(&mut gs.privates[p], shed_cap);
        gs.farms[p].farmer = default_spawn(board_size);
        gs.farms[p].hands.clear();
        gs.farms[p].hires_today = 0;
        gs.privates[p].reset_inventories();
    }

    let next_day = day + 1;
    if next_day > 0 && next_day % shop_interval == 0 {
        // Drawn with replacement: the same shop can unlock repeatedly.
        if gs.town.unlocked_shops.len() < MAX_SHOP_INSTANCES {
            let k = rng.choice_index(N_SHOPS);
            gs.town.unlocked_shops.push(SHOPS_SORTED[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// The interpreter
// ---------------------------------------------------------------------------

/// `interpreter(state, env)` for one step.
///
/// **Trap 6**: per player in index order, compute the atomic-PLANT blocking set, apply
/// the farmer action, then each hand action in order. Then `_process_market` for both
/// players, `_town_consume(step)`, `_decay_plants(step)` per farm, and finally
/// `_end_of_day(day)` when `(step + 1) % turnsPerDay == 0`. The `step` handed to
/// `_town_consume`/`_decay_plants` is the pre-increment step. DONE fires when
/// `step >= episodeSteps - 2`, setting `reward = money`.
pub fn step(gs: &mut GameState, actions: &[PlayerAction; 2], cfg: &Config) {
    step_with_phantom(gs, actions, cfg, NO_PHANTOM);
}

/// [`step`] with a per-seat **phantom** flag.
///
/// A phantom seat contributes *market pressure only*: its unit actions (farmer and every
/// hand) are not applied, and its `HIRE` / `BUY_LAND` / `BUY_SEED` / `BUY_ANIMAL` orders
/// are dropped, so its farm, shed and bank never move. Its `SELL` and `BUY_PRODUCT`
/// orders still run through the real per-unit lockstep against the real seat and still
/// move `market.inventory`, so the price curve the real seat trades into is exactly the
/// one this exogenous stream produces. See [`process_market_with_phantom`].
///
/// Everything downstream of the market -- `town_consume`, `decay_plants`, `end_of_day`,
/// the step/day/hour advance and the DONE rule -- is untouched, including for the phantom
/// seat's own (permanently empty) farm.
///
/// `phantom == NO_PHANTOM` is bit-identical to the pre-phantom [`step`].
pub fn step_with_phantom(
    gs: &mut GameState,
    actions: &[PlayerAction; 2],
    cfg: &Config,
    phantom: [bool; 2],
) {
    // `if env.done: return state`
    if gs.done {
        return;
    }

    let turns_per_day = cfg.turns_per_day_eff();
    let board_size = cfg.board_size;
    let shed_capacity = cfg.shed_capacity_i64();

    let step_no = gs.step;
    let day = step_no.div_euclid(turns_per_day);

    for p in 0..2 {
        // A phantom seat has no units to move. Skipping the whole block (rather than
        // rewriting the ops to PASS) also skips the atomic-PLANT bookkeeping, which is a
        // pure function of that seat's own seeds and so cannot affect the other seat.
        if phantom[p] {
            continue;
        }
        let act = &actions[p];

        // **Trap 7**: atomic PLANT validation. Sum PLANT requests per crop across
        // [farmer, *hands] — including hand slots with no hand behind them — and if
        // the demand for a crop exceeds the seeds held, *every* PLANT request for that
        // crop becomes PASS. Seeds are read before any action applies.
        let mut plant_demand = [0i64; N_CROPS];
        if let UnitOp::Plant(c) = act.farmer {
            plant_demand[c.idx()] += 1;
        }
        for h in &act.hands {
            if let UnitOp::Plant(c) = *h {
                plant_demand[c.idx()] += 1;
            }
        }
        let mut blocked = [false; N_CROPS];
        for c in 0..N_CROPS {
            if plant_demand[c] > gs.privates[p].seeds[c] {
                blocked[c] = true;
            }
        }
        let allowed = |op: UnitOp| -> UnitOp {
            match op {
                UnitOp::Plant(c) if blocked[c.idx()] => UnitOp::Pass,
                other => other,
            }
        };

        apply_unit_action(
            &mut gs.farms[p],
            &mut gs.privates[p],
            0,
            allowed(act.farmer),
            board_size,
            day,
            turns_per_day,
            shed_capacity,
        );
        for (h_idx, hand_op) in act.hands.iter().enumerate() {
            apply_unit_action(
                &mut gs.farms[p],
                &mut gs.privates[p],
                h_idx + 1,
                allowed(*hand_op),
                board_size,
                day,
                turns_per_day,
                shed_capacity,
            );
        }
    }

    process_market_with_phantom(gs, actions, cfg, phantom);
    town_consume(gs, cfg, step_no);
    for p in 0..2 {
        decay_plants(&mut gs.farms[p], step_no);
    }
    if (step_no + 1) % turns_per_day == 0 {
        end_of_day(gs, cfg, day);
    }

    let next_step = step_no + 1;
    gs.step = next_step;
    gs.day = next_step.div_euclid(turns_per_day);
    gs.hour = next_step.rem_euclid(turns_per_day);

    // `step` here is the previous step counter; -2 fires DONE on the final recorded
    // step.
    if step_no >= cfg.episode_steps as i64 - 2 {
        gs.done = true;
        gs.rewards = [gs.farms[0].money, gs.farms[1].money];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::MarketOrder as MO;

    fn cfg() -> Config {
        Config::default()
    }

    fn pa(farmer: UnitOp) -> PlayerAction {
        PlayerAction {
            farmer,
            hands: vec![],
            market: vec![],
        }
    }

    // -- market price -----------------------------------------------------

    #[test]
    fn market_price_at_i0_is_base() {
        let p = MARKET_PARAMS;
        for item in PRODUCTS {
            assert_eq!(
                market_price(item, MARKET_I0, &p),
                p.get(item).base as i64,
                "{}",
                item.as_str()
            );
        }
    }

    #[test]
    fn market_price_bankers_rounding_cases() {
        // Every exact-.5 raw price for inventories 9000..=11000 (all CARROT), taken
        // from CPython: K.market_price('CARROT', inv).
        let cases = [
            (9325i64, 158i64), // 157.5 -> 158
            (9595, 66),        // 66.5  -> 66   (Rust f64::round would give 67)
            (9685, 60),        // 59.5  -> 60
            (9775, 52),        // 52.5  -> 52
            (9865, 46),        // 45.5  -> 46
            (9955, 38),        // 38.5  -> 38
            (10450, 10),       // 10.5  -> 10
        ];
        for (inv, want) in cases {
            assert_eq!(
                market_price(Product::Carrot, inv, &MARKET_PARAMS),
                want,
                "CARROT @ {inv}"
            );
        }
    }

    #[test]
    fn market_price_floor() {
        // Straight from README's table: P(I0 + 2T) == $1 for CARROT.
        assert_eq!(market_price(Product::Carrot, 10_900, &MARKET_PARAMS), 1);
        assert_eq!(market_price(Product::Melon, 20_000, &MARKET_PARAMS), 1);
        assert_eq!(market_price(Product::Wool, 20_000, &MARKET_PARAMS), 1);
    }

    #[test]
    fn readme_price_table() {
        // P(I0-T), P(I0+T), P(I0+2T) from docs/README.md.
        let table: [(Product, i64, i64, i64, i64); 9] = [
            (Product::Wheat, 400, 45, 20, 19),
            (Product::Carrot, 450, 70, 10, 1),
            (Product::Tomato, 200, 84, 24, 9),
            (Product::Strawberry, 100, 204, 1, 1),
            (Product::Melon, 300, 300, 1, 1),
            (Product::Egg, 332, 70, 40, 39),
            (Product::Milk, 122, 256, 1, 1),
            (Product::Wool, 105, 240, 1, 1),
            (Product::Fertilizer, 200, 140, 60, 20),
        ];
        for (item, t, below, above, above2) in table {
            let p = &MARKET_PARAMS;
            assert_eq!(market_price(item, MARKET_I0 - t, p), below, "{} -T", item.as_str());
            assert_eq!(market_price(item, MARKET_I0 + t, p), above, "{} +T", item.as_str());
            assert_eq!(
                market_price(item, MARKET_I0 + 2 * t, p),
                above2,
                "{} +2T",
                item.as_str()
            );
        }
    }

    #[test]
    fn shape_dispatch() {
        assert_eq!(shape(Shape::Linear, 3.0, 10.0), 3.0);
        assert_eq!(shape(Shape::Sq, 3.0, 10.0), 9.0);
        assert_eq!(shape(Shape::Sqrt, 9.0, 10.0), 3.0);
        assert_eq!(shape(Shape::Log, 0.0, 10.0), 0.0);
        assert_eq!(shape(Shape::Log10, 9.0, 10.0), 1.0);
        // f(T) == 1 by construction for hinge.
        assert_eq!(shape(Shape::Hinge, 450.0, 450.0), 1.0);
        assert_eq!(shape(Shape::Hinge, 900.0, 450.0), 2.0 + 8.0);
        // Degenerate T falls back to linear.
        assert_eq!(shape(Shape::Hinge, 7.0, 0.0), 7.0);
        // max(0.0, x)
        assert_eq!(shape(Shape::Linear, -5.0, 10.0), 0.0);
    }

    // -- fib / hire -------------------------------------------------------

    #[test]
    fn fib_indexing_matches_python() {
        let want = [1i64, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        for (n, w) in want.iter().enumerate() {
            assert_eq!(fib(n as i64), *w, "fib({n})");
        }
        assert_eq!(fib(-3), 1);
        assert_eq!(hire_cost(4, 1), 5);
        assert_eq!(hire_cost(4, 3), 15);
        assert_eq!(hire_cost(0, 0), 0);
        // Saturation must never wrap into something affordable.
        assert!(hire_cost(200, 1) as f64 > 1e18);
    }

    #[test]
    fn hire_spawns_nwse_then_least_occupied() {
        let mut gs = GameState::new(&cfg());
        let c = cfg();
        // farmer sits on (4,4), so the first hire goes to (5,4) (locked, but spawn
        // placement ignores locking).
        do_hire(&mut gs.farms[0], &mut gs.privates[0], c.board_size, 1);
        assert_eq!(gs.farms[0].hands, vec![[5, 4]]);
        assert_eq!(gs.farms[0].hires_today, 1);
        assert_eq!(gs.farms[0].money, 3000.0 - 1.0);
        assert_eq!(gs.privates[0].inventories.len(), 2);
        do_hire(&mut gs.farms[0], &mut gs.privates[0], c.board_size, 1);
        assert_eq!(gs.farms[0].hands, vec![[5, 4], [4, 5]]);
        do_hire(&mut gs.farms[0], &mut gs.privates[0], c.board_size, 1);
        assert_eq!(gs.farms[0].hands, vec![[5, 4], [4, 5], [5, 5]]);
        // All four occupied once -> back to NWSE order.
        do_hire(&mut gs.farms[0], &mut gs.privates[0], c.board_size, 1);
        assert_eq!(gs.farms[0].hands, vec![[5, 4], [4, 5], [5, 5], [4, 4]]);
        // costs: 1 + 1 + 2 + 3
        assert_eq!(gs.farms[0].money, 3000.0 - 7.0);
    }

    #[test]
    fn hire_fails_when_broke() {
        let mut gs = GameState::new(&cfg());
        gs.farms[0].money = 0.0;
        gs.farms[0].hires_today = 3; // cost 3
        do_hire(&mut gs.farms[0], &mut gs.privates[0], 10, 1);
        assert!(gs.farms[0].hands.is_empty());
        assert_eq!(gs.farms[0].hires_today, 3);
    }

    // -- land -------------------------------------------------------------

    #[test]
    fn buy_land_unlocks_in_order() {
        let mut gs = GameState::new(&cfg());
        gs.farms[0].money = 10_000.0;
        do_buy_land(&mut gs.farms[0], 10);
        assert_eq!(
            gs.farms[0].unlocked_quadrants,
            vec![Quadrant::NW, Quadrant::NE]
        );
        assert_eq!(gs.farms[0].money, 9_000.0);
        assert_eq!(gs.farms[0].tile(5, 4), Tile::Empty);
        assert_eq!(gs.farms[0].tile(4, 5), Tile::Locked);
        do_buy_land(&mut gs.farms[0], 10);
        assert_eq!(gs.farms[0].money, 7_000.0);
        do_buy_land(&mut gs.farms[0], 10);
        assert_eq!(gs.farms[0].money, 3_000.0);
        assert!(gs.farms[0].tiles.iter().all(|t| *t != Tile::Locked));
        // Fourth purchase is a no-op.
        do_buy_land(&mut gs.farms[0], 10);
        assert_eq!(gs.farms[0].money, 3_000.0);
        assert_eq!(gs.farms[0].unlocked_quadrants.len(), 4);
    }

    // -- unit actions -----------------------------------------------------

    #[test]
    fn movement_is_clamped_and_locked_tiles_are_passable() {
        let mut gs = GameState::new(&cfg());
        let c = cfg();
        let f = &mut gs.farms[0];
        let pv = &mut gs.privates[0];
        // (4,4) -> east onto the locked (5,4)
        apply_unit_action(f, pv, 0, UnitOp::Move(Dir::East), c.board_size, 0, 24, 100);
        assert_eq!(f.farmer, [5, 4]);
        // walk to the east edge and try to step off
        for _ in 0..10 {
            apply_unit_action(f, pv, 0, UnitOp::Move(Dir::East), c.board_size, 0, 24, 100);
        }
        assert_eq!(f.farmer, [9, 4]);
        for _ in 0..10 {
            apply_unit_action(f, pv, 0, UnitOp::Move(Dir::North), c.board_size, 0, 24, 100);
        }
        assert_eq!(f.farmer, [9, 0]);
        // A hand that does not exist is a silent no-op.
        apply_unit_action(f, pv, 7, UnitOp::Move(Dir::South), c.board_size, 0, 24, 100);
        assert!(f.hands.is_empty());
    }

    #[test]
    fn shed_ops_work_from_a_locked_tile_but_tile_ops_do_not() {
        let mut gs = GameState::new(&cfg());
        let c = cfg();
        gs.privates[0].shed.set(ShedItem::Wheat, 5);
        gs.privates[0].seeds[Crop::Wheat.idx()] = 3;
        gs.farms[0].farmer = [5, 5]; // locked shed-access tile
        assert_eq!(gs.farms[0].tile(5, 5), Tile::Locked);

        // PICKUP resolves before the LOCKED guard.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Pickup {
                item: ShedItem::Wheat,
                n: 2,
            },
            c.board_size,
            0,
            24,
            100,
        );
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Wheat), 2);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Wheat), 3);

        // PLANT does not.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Plant(Crop::Wheat),
            c.board_size,
            0,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(5, 5), Tile::Locked);
        assert_eq!(gs.privates[0].seeds[Crop::Wheat.idx()], 3);

        // DROP resolves before the guard too.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Drop,
            c.board_size,
            0,
            24,
            100,
        );
        assert!(gs.privates[0].inventories[0].is_empty());
        assert_eq!(gs.privates[0].shed.get(ShedItem::Wheat), 5);
    }

    #[test]
    fn drop_overflow_respects_inventory_insertion_order() {
        // PORT_SPEC trap 2, verified against Python: shed at 96/100 and a unit
        // carrying {MELON: 5, WHEAT: 50}.
        for melon_first in [true, false] {
            let mut gs = GameState::new(&cfg());
            let c = cfg();
            gs.privates[0].shed.set(ShedItem::Carrot, 96);
            let inv = &mut gs.privates[0].inventories[0];
            if melon_first {
                inv.add(ShedItem::Melon, 5);
                inv.add(ShedItem::Wheat, 50);
            } else {
                inv.add(ShedItem::Wheat, 50);
                inv.add(ShedItem::Melon, 5);
            }
            apply_unit_action(
                &mut gs.farms[0],
                &mut gs.privates[0],
                0,
                UnitOp::Drop,
                c.board_size,
                0,
                24,
                100,
            );
            let shed = &gs.privates[0].shed;
            if melon_first {
                assert_eq!(shed.get(ShedItem::Melon), 4);
                assert_eq!(shed.get(ShedItem::Wheat), 0);
            } else {
                assert_eq!(shed.get(ShedItem::Melon), 0);
                assert_eq!(shed.get(ShedItem::Wheat), 4);
            }
            assert_eq!(shed.total(), 100);
            assert!(gs.privates[0].inventories[0].is_empty());
        }
    }

    #[test]
    fn place_animal_needs_matching_empty_structure() {
        let mut gs = GameState::new(&cfg());
        let c = cfg();
        gs.farms[0].set_tile(0, 0, Tile::Coop);
        gs.farms[0].farmer = [0, 0];
        gs.privates[0].inventories[0].add(ShedItem::Cow, 1);
        gs.privates[0].inventories[0].add(ShedItem::Goose, 1);
        // COW needs a PASTURE; the tile is a COOP -> falls through to the shed path,
        // which is not shed-adjacent here, so nothing happens.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Place {
                item: ShedItem::Cow,
                n: 1,
            },
            c.board_size,
            3,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(0, 0), Tile::Coop);
        // GOOSE matches.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Place {
                item: ShedItem::Goose,
                n: 1,
            },
            c.board_size,
            3,
            24,
            100,
        );
        assert_eq!(
            gs.farms[0].tile(0, 0),
            Tile::Animal(new_animal(Animal::Goose, 3))
        );
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Goose), 0);
        // A second goose cannot go onto an occupied coop, and the tile is not
        // shed-adjacent, so the shed path does nothing either.
        gs.privates[0].inventories[0].add(ShedItem::Goose, 1);
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Place {
                item: ShedItem::Goose,
                n: 1,
            },
            c.board_size,
            3,
            24,
            100,
        );
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Goose), 1);
    }

    #[test]
    fn water_bonus_window_and_fertilizer() {
        let c = cfg();
        // CARROT: max_yield_day 3 -> window_start = (3+1)//2 = 2, so ages 2..=3.
        let mut gs = GameState::new(&c);
        gs.privates[0].seeds[Crop::Carrot.idx()] = 1;
        gs.farms[0].farmer = [0, 0];
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Plant(Crop::Carrot),
            c.board_size,
            0,
            24,
            100,
        );
        let p = gs.farms[0].tile(0, 0).as_plant().unwrap();
        assert_eq!(p.yield_units, 1);
        assert_eq!(p.max_lifespan_step, (0 + 3 + 1) * 24);
        assert_eq!(p.consecutive_unwatered, 1);

        // Watering on day 0 and 1 (outside the window) adds nothing.
        for day in [0i64, 1] {
            apply_unit_action(
                &mut gs.farms[0],
                &mut gs.privates[0],
                0,
                UnitOp::Water,
                c.board_size,
                day,
                24,
                100,
            );
            let mut t = gs.farms[0].tile(0, 0).as_plant().unwrap();
            assert_eq!(t.yield_units, 1, "day {day}");
            t.watered_today = false; // pretend the day rolled over
            gs.farms[0].set_tile(0, 0, Tile::Plant(t));
        }
        // Day 2 is in the window: +1.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Water,
            c.board_size,
            2,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(0, 0).as_plant().unwrap().yield_units, 2);
        // A second watering the same day is a no-op.
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Water,
            c.board_size,
            2,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(0, 0).as_plant().unwrap().yield_units, 2);

        // Fertilize, then water on day 3: +2 (capped at max_yield 4).
        let mut t = gs.farms[0].tile(0, 0).as_plant().unwrap();
        t.watered_today = false;
        gs.farms[0].set_tile(0, 0, Tile::Plant(t));
        gs.privates[0].inventories[0].add(ShedItem::Fertilizer, 1);
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Fertilize,
            c.board_size,
            3,
            24,
            100,
        );
        assert_eq!(
            gs.farms[0]
                .tile(0, 0)
                .as_plant()
                .unwrap()
                .fertilized_until_day,
            5
        );
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Water,
            c.board_size,
            3,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(0, 0).as_plant().unwrap().yield_units, 4);
    }

    #[test]
    fn harvest_one_time_clears_tile_and_respects_first_yield_day() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.farms[0].farmer = [1, 1];
        gs.farms[0]
            .set_tile(1, 1, Tile::Plant(new_plant(Crop::Wheat, 0, 24)));
        // day 1 < first_yield_day 2 -> no-op
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Harvest,
            c.board_size,
            1,
            24,
            100,
        );
        assert!(gs.farms[0].tile(1, 1).as_plant().is_some());
        assert!(gs.privates[0].inventories[0].is_empty());
        // day 2 -> harvest, tile cleared
        apply_unit_action(
            &mut gs.farms[0],
            &mut gs.privates[0],
            0,
            UnitOp::Harvest,
            c.board_size,
            2,
            24,
            100,
        );
        assert_eq!(gs.farms[0].tile(1, 1), Tile::Empty);
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Wheat), 1);
    }

    #[test]
    fn dig_and_build() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.farms[0].farmer = [2, 2];
        let call = |gs: &mut GameState, op: UnitOp| {
            apply_unit_action(
                &mut gs.farms[0],
                &mut gs.privates[0],
                0,
                op,
                c.board_size,
                0,
                24,
                100,
            )
        };
        call(&mut gs, UnitOp::BuildCoop);
        assert_eq!(gs.farms[0].tile(2, 2), Tile::Coop);
        // Building on an occupied tile is a no-op.
        call(&mut gs, UnitOp::BuildPasture);
        assert_eq!(gs.farms[0].tile(2, 2), Tile::Coop);
        // An empty coop can be dug.
        call(&mut gs, UnitOp::Dig);
        assert_eq!(gs.farms[0].tile(2, 2), Tile::Empty);
        // DIG on empty is a no-op; a weed can be dug.
        call(&mut gs, UnitOp::Dig);
        gs.farms[0].set_tile(2, 2, Tile::Weed);
        call(&mut gs, UnitOp::Dig);
        assert_eq!(gs.farms[0].tile(2, 2), Tile::Empty);
        // An occupied structure cannot.
        gs.farms[0].set_tile(2, 2, Tile::Animal(new_animal(Animal::Sheep, 0)));
        call(&mut gs, UnitOp::Dig);
        assert!(gs.farms[0].tile(2, 2).as_animal().is_some());
    }

    #[test]
    fn feed_care_and_collect_fertilizer() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.farms[0].farmer = [3, 3];
        gs.farms[0].set_tile(3, 3, Tile::Animal(new_animal(Animal::Goose, 0)));
        let call = |gs: &mut GameState, op: UnitOp| {
            apply_unit_action(
                &mut gs.farms[0],
                &mut gs.privates[0],
                0,
                op,
                c.board_size,
                1,
                24,
                100,
            )
        };
        // No wheat -> FEED fails.
        call(&mut gs, UnitOp::Feed);
        assert!(!gs.farms[0].tile(3, 3).as_animal().unwrap().fed_today);
        gs.privates[0].inventories[0].add(ShedItem::Wheat, 2);
        call(&mut gs, UnitOp::Feed);
        assert!(gs.farms[0].tile(3, 3).as_animal().unwrap().fed_today);
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Wheat), 1);
        // Second FEED the same day is a no-op (no wheat consumed).
        call(&mut gs, UnitOp::Feed);
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Wheat), 1);
        // CARE once.
        call(&mut gs, UnitOp::Care);
        assert!(gs.farms[0].tile(3, 3).as_animal().unwrap().cared_today);
        // COLLECT_FERTILIZER needs fertilizer_available.
        call(&mut gs, UnitOp::CollectFertilizer);
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Fertilizer), 0);
        let mut a = gs.farms[0].tile(3, 3).as_animal().unwrap();
        a.fertilizer_available = true;
        gs.farms[0].set_tile(3, 3, Tile::Animal(a));
        call(&mut gs, UnitOp::CollectFertilizer);
        assert_eq!(gs.privates[0].inventories[0].get(ShedItem::Fertilizer), 1);
        assert!(!gs.farms[0].tile(3, 3).as_animal().unwrap().fertilizer_available);
    }

    // -- market -----------------------------------------------------------

    #[test]
    fn buy_then_sell_round_trip_nets_zero() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![
                    Some(MO::BuyProduct {
                        item: Product::Wheat,
                        n: 1,
                    }),
                    Some(MO::Sell {
                        item: Product::Wheat,
                        n: 1,
                    }),
                ],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.farms[0].money, 3000.0);
        assert_eq!(gs.market.inv(Product::Wheat), MARKET_I0);
        assert_eq!(gs.privates[0].shed.total(), 0);
    }

    #[test]
    fn lockstep_quotes_both_players_from_the_same_inventory() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        for p in 0..2 {
            gs.privates[p].shed.set(ShedItem::Carrot, 10);
        }
        let order = Some(MO::Sell {
            item: Product::Carrot,
            n: 2,
        });
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![order],
            },
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![order],
            },
        ];
        let p0 = market_price(Product::Carrot, MARKET_I0, &gs.market.params);
        let p1 = market_price(Product::Carrot, MARKET_I0 + 2, &gs.market.params);
        process_market(&mut gs, &actions, &c);
        // Both got the same price for unit 1, then both the same (updated) price for
        // unit 2.
        assert_eq!(gs.farms[0].money, 3000.0 + (p0 + p1) as f64);
        assert_eq!(gs.farms[1].money, 3000.0 + (p0 + p1) as f64);
        assert_eq!(gs.market.inv(Product::Carrot), MARKET_I0 + 4);
    }

    #[test]
    fn sell_at_floor_does_not_add_supply() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        // Push melon far above I0 so the price is pinned to the floor.
        gs.market.inventory[Product::Melon.idx()] = 20_000;
        refresh_prices(&mut gs.market);
        assert_eq!(gs.market.price(Product::Melon), 1);
        gs.privates[0].shed.set(ShedItem::Melon, 5);
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::Sell {
                    item: Product::Melon,
                    n: 5,
                })],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.farms[0].money, 3005.0);
        assert_eq!(gs.market.inv(Product::Melon), 20_000);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Melon), 0);
    }

    #[test]
    fn failed_commit_aborts_the_rest_of_the_order() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[0].shed.set(ShedItem::Wheat, 2);
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::Sell {
                    item: Product::Wheat,
                    n: 10,
                })],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Wheat), 0);
        // Only two units sold.
        assert_eq!(gs.market.inv(Product::Wheat), MARKET_I0 + 2);
    }

    #[test]
    fn buy_product_blocked_by_full_shed_but_seeds_are_not() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[0].shed.set(ShedItem::Carrot, 100);
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![
                    Some(MO::BuyProduct {
                        item: Product::Wheat,
                        n: 1,
                    }),
                    Some(MO::BuyAnimal {
                        animal: Animal::Goose,
                        n: 1,
                    }),
                    Some(MO::BuySeed {
                        crop: Crop::Melon,
                        n: 2,
                    }),
                ],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Wheat), 0);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Goose), 0);
        assert_eq!(gs.privates[0].seeds[Crop::Melon.idx()], 2);
        assert_eq!(gs.farms[0].money, 3000.0 - 160.0);
    }

    #[test]
    fn queue_is_truncated_to_max_orders() {
        let mut c = cfg();
        c.max_market_orders_per_turn = 2;
        let mut gs = GameState::new(&c);
        let buy = Some(MO::BuySeed {
            crop: Crop::Wheat,
            n: 1,
        });
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![buy, buy, buy, buy],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.privates[0].seeds[Crop::Wheat.idx()], 2);
    }

    #[test]
    fn atomic_orders_are_handled_once_per_index() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        let actions = [
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::Hire), Some(MO::Hire), Some(MO::BuyLand)],
            },
            PlayerAction::pass(),
        ];
        process_market(&mut gs, &actions, &c);
        assert_eq!(gs.farms[0].hands.len(), 2);
        assert_eq!(gs.farms[0].unlocked_quadrants.len(), 2);
        assert_eq!(gs.farms[0].money, 3000.0 - 1.0 - 1.0 - 1000.0);
    }

    // -- town -------------------------------------------------------------

    #[test]
    fn town_consume_intervals() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.town.unlocked_shops = vec![Shop::YarnStore, Shop::Bakery];
        // step 0: both the shops (step % 4 == 0) and the town center (step % 24 == 0).
        town_consume(&mut gs, &c, 0);
        assert_eq!(gs.market.inv(Product::Wool), MARKET_I0 - 2 - 1);
        assert_eq!(gs.market.inv(Product::Egg), MARKET_I0 - 1 - 1);
        assert_eq!(gs.market.inv(Product::Wheat), MARKET_I0 - 1 - 1);
        assert_eq!(gs.market.inv(Product::Fertilizer), MARKET_I0);
        // step 1: nothing.
        let before = gs.market.inventory;
        town_consume(&mut gs, &c, 1);
        assert_eq!(gs.market.inventory, before);
        // step 4: shops only.
        town_consume(&mut gs, &c, 4);
        assert_eq!(gs.market.inv(Product::Wool), MARKET_I0 - 5);
    }

    // -- decay / daily refresh --------------------------------------------

    #[test]
    fn decay_runs_every_other_step_from_max_lifespan() {
        let mut farm = Farm::new(10, 0.0);
        let mut p = new_plant(Crop::Wheat, 0, 24);
        p.yield_units = 3;
        p.max_lifespan_step = 100;
        farm.set_tile(0, 0, Tile::Plant(p));
        decay_plants(&mut farm, 99);
        assert_eq!(farm.tile(0, 0).yield_units(), 3);
        decay_plants(&mut farm, 100);
        assert_eq!(farm.tile(0, 0).yield_units(), 2);
        decay_plants(&mut farm, 101);
        assert_eq!(farm.tile(0, 0).yield_units(), 2);
        decay_plants(&mut farm, 102);
        assert_eq!(farm.tile(0, 0).yield_units(), 1);
        decay_plants(&mut farm, 103);
        decay_plants(&mut farm, 104);
        assert_eq!(farm.tile(0, 0), Tile::Weed);
        // Ongoing crops (mls == -1) never decay.
        farm.set_tile(1, 1, Tile::Plant(new_plant(Crop::Tomato, 0, 24)));
        decay_plants(&mut farm, 10_000);
        assert!(farm.tile(1, 1).as_plant().is_some());
    }

    #[test]
    fn unwatered_plants_become_weeds_and_produce_nothing() {
        let mut farm = Farm::new(10, 0.0);
        // A tomato planted on day 0 and never watered: consecutive_unwatered starts at
        // 1, hits 2 at the first refresh, so it dies that night.
        farm.set_tile(0, 0, Tile::Plant(new_plant(Crop::Tomato, 0, 24)));
        daily_refresh_plants(&mut farm, 0, 24);
        assert_eq!(farm.tile(0, 0), Tile::Weed);
    }

    #[test]
    fn ongoing_production_schedule_and_lifespan() {
        let mut farm = Farm::new(10, 0.0);
        // TOMATO: first_yield_day 8, interval 1, max_yield 4.
        farm.set_tile(0, 0, Tile::Plant(new_plant(Crop::Tomato, 0, 24)));
        for day in 0..12i64 {
            // keep it watered so it survives
            let mut t = farm.tile(0, 0).as_plant().unwrap();
            t.watered_today = true;
            farm.set_tile(0, 0, Tile::Plant(t));
            daily_refresh_plants(&mut farm, day, 24);
            let t = farm.tile(0, 0).as_plant().unwrap();
            let expected = match day {
                0..=6 => 0,          // next_day 1..7 < first_yield_day
                7 => 1,              // next_day 8 -> production 1
                8 => 2,
                9 => 3,
                10 => 4,             // production 4 == max_yield -> lifespan set
                _ => 4,
            };
            assert_eq!(t.yield_units, expected, "day {day}");
            if day >= 10 {
                assert_eq!(t.max_lifespan_step, (11 + 1) * 24);
            } else {
                assert_eq!(t.max_lifespan_step, -1, "day {day}");
            }
        }
    }

    #[test]
    fn fertilizer_doubles_ongoing_production_only_when_watered() {
        for watered in [true, false] {
            let mut farm = Farm::new(10, 0.0);
            let mut t = new_plant(Crop::Tomato, 0, 24);
            t.consecutive_unwatered = 0;
            t.fertilized_until_day = 100;
            t.watered_today = watered;
            farm.set_tile(0, 0, Tile::Plant(t));
            // next_day 8 == first_yield_day -> one production
            daily_refresh_plants(&mut farm, 7, 24);
            if watered {
                assert_eq!(farm.tile(0, 0).as_plant().unwrap().yield_units, 2);
            } else {
                // consecutive_unwatered goes 0 -> 1, survives, produces 1 (unfertilized)
                assert_eq!(farm.tile(0, 0).as_plant().unwrap().yield_units, 1);
            }
        }
    }

    #[test]
    fn animal_production_cannot_consume_the_same_days_care() {
        let mut farm = Farm::new(10, 0.0);
        // GOOSE: first_yield_day 4, interval 1, max_held 4.
        let mut a = new_animal(Animal::Goose, 0);
        a.fed_today = true;
        a.cared_today = true;
        farm.set_tile(0, 0, Tile::Animal(a));
        // day 3 -> next_day 4 == first_yield_day: production of 1 (the care bonus is
        // banked afterwards).
        daily_refresh_animals(&mut farm, 3);
        let a = farm.tile(0, 0).as_animal().unwrap();
        assert_eq!(a.yield_units, 1);
        assert_eq!(a.pending_care_bonus, 1);
        assert!(a.fertilizer_available);
        assert!(!a.fed_today && !a.cared_today);
        // Next day, fed but not cared: the bank pays out on production and resets.
        let mut a = farm.tile(0, 0).as_animal().unwrap();
        a.fed_today = true;
        farm.set_tile(0, 0, Tile::Animal(a));
        daily_refresh_animals(&mut farm, 4);
        let a = farm.tile(0, 0).as_animal().unwrap();
        assert_eq!(a.yield_units, 3); // 1 + base 1 + bonus 1
        assert_eq!(a.pending_care_bonus, 0);
    }

    #[test]
    fn unfed_production_day_drops_the_care_bank() {
        let mut farm = Farm::new(10, 0.0);
        let mut a = new_animal(Animal::Goose, 0);
        a.pending_care_bonus = 3;
        a.yield_units = 0;
        a.fed_today = false; // consecutive_unfed 0 -> 1, survives
        farm.set_tile(0, 0, Tile::Animal(a));
        daily_refresh_animals(&mut farm, 3);
        let a = farm.tile(0, 0).as_animal().unwrap();
        assert_eq!(a.yield_units, 1);
        assert_eq!(a.pending_care_bonus, 0);
    }

    #[test]
    fn animals_escape_after_two_unfed_days_leaving_the_structure() {
        let mut farm = Farm::new(10, 0.0);
        farm.set_tile(0, 0, Tile::Animal(new_animal(Animal::Cow, 0)));
        daily_refresh_animals(&mut farm, 0);
        assert_eq!(farm.tile(0, 0).as_animal().unwrap().consecutive_unfed, 1);
        daily_refresh_animals(&mut farm, 1);
        assert_eq!(farm.tile(0, 0), Tile::Pasture);
    }

    #[test]
    fn fertilizer_is_available_even_when_unfed_and_uncared() {
        let mut farm = Farm::new(10, 0.0);
        farm.set_tile(0, 0, Tile::Animal(new_animal(Animal::Sheep, 0)));
        daily_refresh_animals(&mut farm, 0);
        assert!(farm.tile(0, 0).as_animal().unwrap().fertilizer_available);
    }

    // -- end of day -------------------------------------------------------

    #[test]
    fn end_of_day_resets_units_and_drops_inventories() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.farms[0].hands = vec![[4, 4], [5, 4]];
        gs.farms[0].farmer = [9, 9];
        gs.farms[0].hires_today = 2;
        gs.privates[0].inventories = vec![Inv::new(), Inv::new(), Inv::new()];
        gs.privates[0].inventories[0].add(ShedItem::Wheat, 3);
        gs.privates[0].inventories[2].add(ShedItem::Egg, 4);
        end_of_day(&mut gs, &c, 0);
        assert_eq!(gs.farms[0].farmer, [4, 4]);
        assert!(gs.farms[0].hands.is_empty());
        assert_eq!(gs.farms[0].hires_today, 0);
        assert_eq!(gs.privates[0].inventories.len(), 1);
        assert!(gs.privates[0].inventories[0].is_empty());
        assert_eq!(gs.privates[0].shed.get(ShedItem::Wheat), 3);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Egg), 4);
    }

    #[test]
    fn shop_unlocks_only_on_the_interval_and_stops_at_the_cap() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        // day 0 -> next_day 1, 1 % 3 != 0 -> no unlock
        end_of_day(&mut gs, &c, 0);
        assert!(gs.town.unlocked_shops.is_empty());
        // day 2 -> next_day 3 -> unlock
        end_of_day(&mut gs, &c, 2);
        assert_eq!(gs.town.unlocked_shops.len(), 1);
        for day in [5, 8, 11, 14, 17, 20, 23, 26, 29] {
            end_of_day(&mut gs, &c, day);
        }
        assert_eq!(gs.town.unlocked_shops.len(), MAX_SHOP_INSTANCES);
    }

    // -- step -------------------------------------------------------------

    #[test]
    fn step_advances_step_day_hour_and_fires_done() {
        let mut c = cfg();
        c.episode_steps = 10;
        c.turns_per_day = 4;
        let mut gs = GameState::new(&c);
        let actions = [pa(UnitOp::Pass), pa(UnitOp::Pass)];
        for i in 1..=8 {
            step(&mut gs, &actions, &c);
            assert_eq!(gs.step, i);
            assert_eq!(gs.day, i / 4);
            assert_eq!(gs.hour, i % 4);
            assert_eq!(gs.done, i >= 9);
        }
        // step 8 (pre-increment) >= 10 - 2 -> DONE
        step(&mut gs, &actions, &c);
        assert_eq!(gs.step, 9);
        assert!(gs.done);
        assert_eq!(gs.rewards, [gs.farms[0].money, gs.farms[1].money]);
        // Further steps are no-ops.
        let before = gs.clone();
        step(&mut gs, &actions, &c);
        assert_eq!(gs, before);
    }

    #[test]
    fn atomic_plant_blocks_all_requests_for_the_contended_crop() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[0].seeds[Crop::Melon.idx()] = 1;
        gs.privates[0].seeds[Crop::Wheat.idx()] = 2;
        gs.farms[0].hands = vec![[0, 1]];
        gs.privates[0].inventories.push(Inv::new());
        gs.farms[0].farmer = [0, 0];
        let actions = [
            PlayerAction {
                farmer: UnitOp::Plant(Crop::Melon),
                hands: vec![UnitOp::Plant(Crop::Melon)],
                market: vec![],
            },
            PlayerAction::pass(),
        ];
        step(&mut gs, &actions, &c);
        // Two melon requests, one seed -> both dropped.
        assert_eq!(gs.privates[0].seeds[Crop::Melon.idx()], 1);
        assert_eq!(gs.farms[0].tile(0, 0), Tile::Empty);
        assert_eq!(gs.farms[0].tile(0, 1), Tile::Empty);

        // Same demand with two seeds succeeds for both.
        let mut gs = GameState::new(&c);
        gs.privates[0].seeds[Crop::Melon.idx()] = 2;
        gs.farms[0].hands = vec![[0, 1]];
        gs.privates[0].inventories.push(Inv::new());
        gs.farms[0].farmer = [0, 0];
        step(&mut gs, &actions, &c);
        assert_eq!(gs.privates[0].seeds[Crop::Melon.idx()], 0);
        assert!(gs.farms[0].tile(0, 0).as_plant().is_some());
        assert!(gs.farms[0].tile(0, 1).as_plant().is_some());
    }

    /// Rough single-core throughput, for sizing the RL data path.
    ///
    /// `cargo test -p kaggri-core --release -- --ignored --nocapture bench_`
    #[test]
    #[ignore = "benchmark"]
    fn bench_single_core_steps_per_second() {
        let c = cfg();
        let mk = |farmer: UnitOp, hands: usize, market: Vec<Option<MO>>| PlayerAction {
            farmer,
            hands: vec![UnitOp::Water; hands],
            market,
        };
        let episodes = 200;
        let t0 = std::time::Instant::now();
        let mut steps = 0u64;
        let mut checksum = 0.0f64;
        for _ in 0..episodes {
            let mut gs = GameState::new(&c);
            for s in 0..c.episode_steps as i64 {
                let market = if s % 4 == 0 {
                    vec![
                        Some(MO::Hire),
                        Some(MO::BuySeed {
                            crop: Crop::Wheat,
                            n: 2,
                        }),
                        Some(MO::Sell {
                            item: Product::Wheat,
                            n: 3,
                        }),
                    ]
                } else {
                    vec![]
                };
                let a0 = mk(
                    if s % 3 == 0 {
                        UnitOp::Plant(Crop::Wheat)
                    } else {
                        UnitOp::Water
                    },
                    gs.farms[0].hands.len(),
                    market.clone(),
                );
                let a1 = mk(UnitOp::Move(Dir::East), gs.farms[1].hands.len(), market);
                let actions = [a0, a1];
                step(&mut gs, &actions, &c);
                steps += 1;
            }
            checksum += gs.farms[0].money;
        }
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "bench: {steps} steps in {secs:.3}s = {:.0} steps/s single core ({} episodes, checksum {checksum})",
            steps as f64 / secs,
            episodes
        );
    }

    #[test]
    fn phantom_hand_slots_still_block_planting() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[0].seeds[Crop::Melon.idx()] = 1;
        gs.farms[0].farmer = [0, 0];
        // No hands hired, but the action list carries a hand PLANT anyway: Python
        // counts it in plant_demand before discovering the hand does not exist.
        let actions = [
            PlayerAction {
                farmer: UnitOp::Plant(Crop::Melon),
                hands: vec![UnitOp::Plant(Crop::Melon)],
                market: vec![],
            },
            PlayerAction::pass(),
        ];
        step(&mut gs, &actions, &c);
        assert_eq!(gs.privates[0].seeds[Crop::Melon.idx()], 1);
        assert_eq!(gs.farms[0].tile(0, 0), Tile::Empty);
    }

    // -- phantom seats ----------------------------------------------------
    //
    // A phantom seat is an exogenous market-pressure stream: no units, no farm, no bank,
    // just realised SELL / BUY_PRODUCT orders pushed through the real lockstep market.

    /// A deterministic 64-bit LCG, so the parity episode below is reproducible without
    /// pulling a dev-dependency in.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn phantom_seat_sell_adds_inventory_without_touching_shed_or_money() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        // The phantom's shed is empty: a real seat's SELL would abort on the first unit.
        assert_eq!(gs.privates[1].shed.total(), 0);
        let before_price = market_price(Product::Wheat, gs.market.inv(Product::Wheat), &gs.market.params);
        let actions = [
            PlayerAction::pass(),
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::Sell { item: Product::Wheat, n: 40 })],
            },
        ];
        process_market_with_phantom(&mut gs, &actions, &c, [false, true]);
        assert_eq!(gs.market.inv(Product::Wheat), MARKET_I0 + 40);
        assert_eq!(gs.privates[1].shed.total(), 0, "phantom shed must be untouched");
        assert_eq!(gs.farms[1].money, 3000.0, "phantom bank must be untouched");
        let after_price = market_price(Product::Wheat, gs.market.inv(Product::Wheat), &gs.market.params);
        assert!(after_price < before_price, "supply must depress the price");
        assert_eq!(gs.market.prices[Product::Wheat.idx()], after_price, "refresh_prices ran");
    }

    #[test]
    fn phantom_seat_buy_product_removes_inventory_without_money_or_shed_space() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        // Broke and shed-full: both real-seat guards would refuse every unit.
        gs.farms[1].money = 0.0;
        gs.privates[1].shed.set(ShedItem::Wool, c.shed_capacity_i64());
        let full = gs.privates[1].shed.total();
        let actions = [
            PlayerAction::pass(),
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::BuyProduct { item: Product::Wheat, n: 25 })],
            },
        ];
        process_market_with_phantom(&mut gs, &actions, &c, [false, true]);
        assert_eq!(gs.market.inv(Product::Wheat), MARKET_I0 - 25);
        assert_eq!(gs.farms[1].money, 0.0);
        assert_eq!(gs.privates[1].shed.total(), full);
    }

    #[test]
    fn phantom_seat_respects_the_dollar_floor() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        // Drive WHEAT well past the floor, then keep selling: floor sales are an
        // infinite sink and must NOT add supply, for a phantom exactly as for a player.
        gs.market.inventory[Product::Melon.idx()] = 20_000;
        refresh_prices(&mut gs.market);
        assert_eq!(gs.market.price(Product::Melon), 1);
        let actions = [
            PlayerAction::pass(),
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![Some(MO::Sell { item: Product::Melon, n: 50 })],
            },
        ];
        process_market_with_phantom(&mut gs, &actions, &c, [false, true]);
        assert_eq!(gs.market.inv(Product::Melon), 20_000, "floor sales add no supply");
        assert_eq!(gs.farms[1].money, 3000.0, "and the phantom banks nothing either");
    }

    #[test]
    fn phantom_seat_drops_hire_land_seed_and_animal() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        let hands_before = gs.farms[1].hands.len();
        let quads_before = gs.farms[1].unlocked_quadrants.len();
        let actions = [
            PlayerAction::pass(),
            PlayerAction {
                farmer: UnitOp::Pass,
                hands: vec![],
                market: vec![
                    Some(MO::Hire),
                    Some(MO::BuyLand),
                    Some(MO::BuySeed { crop: Crop::Wheat, n: 5 }),
                    Some(MO::BuyAnimal { animal: Animal::Cow, n: 1 }),
                ],
            },
        ];
        process_market_with_phantom(&mut gs, &actions, &c, [false, true]);
        assert_eq!(gs.farms[1].hands.len(), hands_before);
        assert_eq!(gs.farms[1].unlocked_quadrants.len(), quads_before);
        assert_eq!(gs.privates[1].seeds[Crop::Wheat.idx()], 0);
        assert_eq!(gs.privates[1].shed.total(), 0);
        assert_eq!(gs.farms[1].money, 3000.0);
    }

    #[test]
    fn phantom_seat_ignores_unit_actions() {
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[1].seeds[Crop::Wheat.idx()] = 5;
        let farm_before = gs.farms[1].clone();
        let actions = [
            PlayerAction::pass(),
            PlayerAction {
                farmer: UnitOp::Plant(Crop::Wheat),
                hands: vec![],
                market: vec![],
            },
        ];
        step_with_phantom(&mut gs, &actions, &c, [false, true]);
        assert_eq!(gs.privates[1].seeds[Crop::Wheat.idx()], 5, "no seed consumed");
        assert_eq!(gs.farms[1].tiles, farm_before.tiles, "no tile planted");
    }

    #[test]
    fn phantom_stream_interleaves_with_the_real_seat_in_lockstep() {
        // Both seats sell the same product in the same order slot: the two quotes must
        // come from the SAME pre-commit inventory, then commit in player order -- exactly
        // as `lockstep_quotes_both_players_from_the_same_inventory` asserts for two real
        // seats. The phantom must not short-circuit that.
        let c = cfg();
        let mut gs = GameState::new(&c);
        gs.privates[0].shed.set(ShedItem::Carrot, 10);
        let order = Some(MO::Sell { item: Product::Carrot, n: 2 });
        let actions = [
            PlayerAction { farmer: UnitOp::Pass, hands: vec![], market: vec![order] },
            PlayerAction { farmer: UnitOp::Pass, hands: vec![], market: vec![order] },
        ];
        // The reference arm gives seat 1 a real shed to sell out of; the phantom arm
        // leaves it empty, which is what a real seat could never sell from.
        let mut both_real = gs.clone();
        both_real.privates[1].shed.set(ShedItem::Carrot, 10);
        process_market(&mut both_real, &actions, &c);
        process_market_with_phantom(&mut gs, &actions, &c, [false, true]);
        // Four units cleared either way, so the real seat's proceeds are identical: the
        // phantom pays the same price pressure a real seller would have.
        assert_eq!(gs.market.inv(Product::Carrot), MARKET_I0 + 4);
        assert_eq!(gs.farms[0].money, both_real.farms[0].money);
        assert_eq!(gs.privates[0].shed.get(ShedItem::Carrot), 8);
        assert_eq!(gs.privates[1].shed.get(ShedItem::Carrot), 0, "phantom shed untouched");
    }

    #[test]
    fn no_phantom_is_bit_identical_on_a_random_action_episode() {
        // The parity gate for the whole feature: with NO_PHANTOM the new entry points
        // must reproduce the old ones exactly, state field for state field, over a long
        // pseudo-random episode that exercises units, seeds, animals and the market.
        let c = Config { episode_steps: 240, ..Config::default() };
        let mut a = GameState::new(&c);
        let mut b = GameState::new(&c);
        let mut rng = 0x5eed_1234_abcd_9876u64;
        for _ in 0..(c.episode_steps as usize - 1) {
            let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
            for p in 0..2 {
                let n_hands = a.farms[p].hands.len();
                actions[p].farmer = random_unit_op(&mut rng);
                actions[p].hands = (0..n_hands).map(|_| random_unit_op(&mut rng)).collect();
                let n_orders = (lcg(&mut rng) % 4) as usize;
                actions[p].market = (0..n_orders).map(|_| Some(random_order(&mut rng))).collect();
            }
            step(&mut a, &actions, &c);
            step_with_phantom(&mut b, &actions, &c, NO_PHANTOM);
            assert_eq!(a, b, "NO_PHANTOM diverged from step()");
        }
        // The episode has to be non-trivial for the equality to mean anything.
        assert!(a.farms[0].money != 3000.0 || a.market.inv(Product::Wheat) != MARKET_I0);
    }

    fn random_unit_op(rng: &mut u64) -> UnitOp {
        match lcg(rng) % 12 {
            0 => UnitOp::Move(Dir::North),
            1 => UnitOp::Move(Dir::South),
            2 => UnitOp::Move(Dir::East),
            3 => UnitOp::Move(Dir::West),
            4 => UnitOp::Plant(CROP_CYCLE[(lcg(rng) % 5) as usize]),
            5 => UnitOp::Water,
            6 => UnitOp::Harvest,
            7 => UnitOp::Dig,
            8 => UnitOp::BuildCoop,
            9 => UnitOp::BuildPasture,
            10 => UnitOp::Pickup { item: ShedItem::Wheat, n: 2 },
            _ => UnitOp::Pass,
        }
    }

    fn random_order(rng: &mut u64) -> MO {
        let n = 1 + (lcg(rng) % 6) as i64;
        match lcg(rng) % 6 {
            0 => MO::Hire,
            1 => MO::BuyLand,
            2 => MO::BuySeed { crop: CROP_CYCLE[(lcg(rng) % 5) as usize], n },
            3 => MO::BuyAnimal { animal: ANIMAL_CYCLE[(lcg(rng) % 3) as usize], n },
            4 => MO::BuyProduct { item: Product::Wheat, n },
            _ => MO::Sell { item: PRODUCT_CYCLE[(lcg(rng) % 9) as usize], n },
        }
    }

    const CROP_CYCLE: [Crop; 5] = [Crop::Wheat, Crop::Carrot, Crop::Tomato, Crop::Strawberry, Crop::Melon];
    const ANIMAL_CYCLE: [Animal; 3] = [Animal::Goose, Animal::Cow, Animal::Sheep];
    const PRODUCT_CYCLE: [Product; 9] = [
        Product::Wheat, Product::Carrot, Product::Tomato, Product::Strawberry, Product::Melon,
        Product::Egg, Product::Milk, Product::Wool, Product::Fertilizer,
    ];
}
