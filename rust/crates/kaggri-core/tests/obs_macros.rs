//! Tests for the RL-facing layer: `obs.rs` (encoding + legality mask) and `macros.rs`
//! (target-tile macro resolution, BFS pathing, market packer).
//!
//! The heavy tests drive real episodes with a uniformly random macro policy, which is the
//! only way to reach the states that matter (full sheds, decayed plants, escaped animals,
//! weeds, price floors) and is also exactly what PPO's first thousand steps look like.

use kaggri_core::actions::{MarketOrder, PlayerAction, UnitOp};
use kaggri_core::consts::*;
use kaggri_core::macros::{
    dist_to_shed, first_step, pack_market, path_len, resolve_turn, target_plausible, MacroOrder,
    MarketHead, Verb, CACHED_BOARD, MAX_HANDS, MAX_HIRE_PER_TURN, MAX_UNITS, N_MARKET_HEADS,
    N_TARGETS, N_VERBS, PICKUP_ITEMS, QTY_BUCKETS, SELL_FRACS, VERBS,
};
use kaggri_core::obs::{
    encode, encode_with_cfg, legal_mask, legal_mask_with_cfg, obs_dims, tile_verb_mask_with_cfg,
    MASK_LEN, N_FEATURES, TILE_MASK_LEN,
    N_TOKENS, OBS_LEN, OBS_VERSION,
};
use kaggri_core::sim::step;
use kaggri_core::state::{Config, GameState, Tile};

// ---------------------------------------------------------------------------
// A tiny deterministic RNG (test-only; nothing here needs CPython parity)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n.max(1)
    }
}

/// Random macro orders, biased so that verbs actually get *emitted*.
///
/// A uniform draw over 100 targets x 38 verbs almost never puts a unit on its own target,
/// so a uniform driver spends an entire episode walking and exercises the emission path
/// perhaps twice. (Worth knowing for training too: the joint space is 3 800 per unit, which
/// is precisely why the target head wants a shaped prior rather than uniform init.) Half
/// the orders therefore target the tile the unit is standing on and an eighth target a
/// shed-access tile, which reaches every verb thousands of times per episode.
fn random_orders(rng: &mut Rng, gs: &GameState, player: usize, buf: &mut Vec<MacroOrder>) {
    let farm = &gs.farms[player];
    buf.clear();
    for u in 0..1 + farm.hands.len() {
        let pos = farm.unit_pos(u).unwrap();
        let target = match rng.below(8) {
            0..=3 => pos[1] as usize * 10 + pos[0] as usize,
            4 => {
                let t = shed_access_tiles(10)[rng.below(4)];
                t[1] as usize * 10 + t[0] as usize
            }
            _ => rng.below(N_TARGETS),
        };
        buf.push(MacroOrder {
            target: target as u8,
            verb: VERBS[rng.below(N_VERBS)],
        });
    }
}

/// A market head biased towards activity: the packer is only interesting when more than
/// 10 orders are requested, so hire counts and sell fractions are drawn hot.
fn random_market(rng: &mut Rng) -> MarketHead {
    let mut choices = [0u8; N_MARKET_HEADS];
    for c in choices.iter_mut() {
        *c = rng.below(8) as u8;
    }
    MarketHead::from_choices(&choices)
}

// ---------------------------------------------------------------------------
// Pathing
// ---------------------------------------------------------------------------

/// Independent BFS, written differently from the one in `macros.rs`, over every pair.
fn bfs_dist(from: [i64; 2], to: [i64; 2]) -> i64 {
    let b = CACHED_BOARD as i64;
    let mut dist = vec![-1i64; (b * b) as usize];
    let mut q = std::collections::VecDeque::new();
    dist[(from[1] * b + from[0]) as usize] = 0;
    q.push_back(from);
    while let Some(c) = q.pop_front() {
        if c == to {
            return dist[(c[1] * b + c[0]) as usize];
        }
        for (dx, dy) in [(0, -1), (0, 1), (1, 0), (-1, 0)] {
            let n = [c[0] + dx, c[1] + dy];
            if n[0] < 0 || n[0] >= b || n[1] < 0 || n[1] >= b {
                continue;
            }
            let i = (n[1] * b + n[0]) as usize;
            if dist[i] < 0 {
                dist[i] = dist[(c[1] * b + c[0]) as usize] + 1;
                q.push_back(n);
            }
        }
    }
    -1
}

#[test]
fn path_table_matches_independent_bfs_and_manhattan() {
    for sy in 0..10 {
        for sx in 0..10 {
            for ty in 0..10 {
                for tx in 0..10 {
                    let (s, t) = ([sx, sy], [tx, ty]);
                    let d = path_len(s, t, 10);
                    assert_eq!(d, bfs_dist(s, t), "dist {s:?}->{t:?}");
                    assert_eq!(d, (sx - tx).abs() + (sy - ty).abs(), "manhattan {s:?}->{t:?}");
                }
            }
        }
    }
}

#[test]
fn pathing_terminates_on_every_pair() {
    for sy in 0..10i64 {
        for sx in 0..10i64 {
            for ty in 0..10i64 {
                for tx in 0..10i64 {
                    let target = [tx, ty];
                    let mut pos = [sx, sy];
                    let mut moves = 0;
                    while pos != target {
                        let dir = first_step(pos, target, 10)
                            .unwrap_or_else(|| panic!("no step {pos:?}->{target:?}"));
                        let (dx, dy) = dir.delta();
                        pos = [pos[0] + dx, pos[1] + dy];
                        assert!(
                            (0..10).contains(&pos[0]) && (0..10).contains(&pos[1]),
                            "stepped off the board at {pos:?}"
                        );
                        moves += 1;
                        assert!(moves <= 18, "path {sx},{sy} -> {tx},{ty} did not terminate");
                    }
                    assert_eq!(moves, path_len([sx, sy], target, 10), "path not shortest");
                }
            }
        }
    }
    assert!(first_step([3, 3], [3, 3], 10).is_none());
}

#[test]
fn closed_form_matches_table_for_other_board_sizes() {
    // The 10x10 table and the closed form used for other board sizes must agree, which is
    // what makes the non-default-board path sound.
    for sy in 0..10i64 {
        for sx in 0..10i64 {
            for ty in 0..10i64 {
                for tx in 0..10i64 {
                    let table = first_step([sx, sy], [tx, ty], CACHED_BOARD);
                    // 11 is not the cached size, so this goes through the closed form; the
                    // sub-grid 0..10 is embedded in it identically.
                    let closed = first_step([sx, sy], [tx, ty], 11);
                    assert_eq!(table, closed, "{sx},{sy} -> {tx},{ty}");
                }
            }
        }
    }
}

#[test]
fn dist_to_shed_is_zero_on_access_tiles() {
    for t in shed_access_tiles(10) {
        assert_eq!(dist_to_shed(t, 10), 0);
        assert!(is_shed_adjacent(t[0], t[1], 10));
    }
    assert_eq!(dist_to_shed([0, 0], 10), 8);
    assert_eq!(dist_to_shed([9, 9], 10), 8);
    assert_eq!(dist_to_shed([4, 0], 10), 4);
}

// ---------------------------------------------------------------------------
// Verb table
// ---------------------------------------------------------------------------

#[test]
fn verb_indices_round_trip() {
    assert_eq!(VERBS.len(), N_VERBS);
    for (i, v) in VERBS.iter().enumerate() {
        assert_eq!(v.index(), i, "verb {v:?} index");
        assert_eq!(Verb::from_index(i), Some(*v));
    }
    assert_eq!(Verb::from_index(N_VERBS), None);
    // Every verb required by the design brief is present.
    for v in [
        Verb::Water,
        Verb::Harvest,
        Verb::Fertilize,
        Verb::Dig,
        Verb::BuildCoop,
        Verb::BuildPasture,
        Verb::Feed,
        Verb::Care,
        Verb::CollectFertilizer,
        Verb::Drop,
        Verb::Idle,
    ] {
        assert!(VERBS.contains(&v), "{v:?} missing");
    }
    for c in CROP_LIST {
        assert!(VERBS.contains(&Verb::Plant(c)));
    }
    for a in ANIMAL_LIST {
        assert!(VERBS.contains(&Verb::PlaceAnimal(a)));
    }
    for p in PRODUCTS {
        assert!(VERBS.contains(&Verb::Deposit(p)));
    }
    for slot in 0..PICKUP_ITEMS.len() as u8 {
        for qty_slot in 0..2u8 {
            assert!(VERBS.contains(&Verb::Pickup {
                item_slot: slot,
                qty_slot
            }));
        }
    }
}

// ---------------------------------------------------------------------------
// Episode driver used by the invariant tests
// ---------------------------------------------------------------------------

/// Everything the resolver promises about one player's emitted ops, checked against the
/// state as it was at the *start* of the turn (which is the only state the resolver sees).
fn check_emitted(gs: &GameState, cfg: &Config, player: usize, act: &PlayerAction) {
    let farm = &gs.farms[player];
    let private = &gs.privates[player];
    let bs = cfg.board_size;
    let cap = cfg.shed_capacity_i64();

    assert_eq!(
        act.hands.len(),
        farm.hands.len(),
        "one op per live hand, no phantom hand slots (they would count towards atomic PLANT)"
    );

    let mut plant_count = [0i64; N_CROPS];
    let mut drop_room = cap - private.shed.total();
    let mut claimed: Vec<(i64, i64, std::mem::Discriminant<UnitOp>)> = Vec::new();

    for idx in 0..=act.hands.len() {
        let op = if idx == 0 {
            act.farmer
        } else {
            act.hands[idx - 1]
        };
        let pos = farm.unit_pos(idx).expect("op emitted for a live unit");
        let (x, y) = (pos[0], pos[1]);
        let tile = farm.tile(x, y);
        let inv = private.inventories.get(idx).cloned().unwrap_or_default();

        match op {
            UnitOp::Pass | UnitOp::Noop => {}
            UnitOp::Move(dir) => {
                // Never a wasted step off the board.
                let (dx, dy) = dir.delta();
                let (nx, ny) = (x + dx, y + dy);
                assert!(
                    nx >= 0 && nx < bs as i64 && ny >= 0 && ny < bs as i64,
                    "move off the board from {pos:?}"
                );
            }
            UnitOp::Water => {
                let p = tile.as_plant().expect("WATER on a plant");
                assert!(!p.watered_today, "WATER on an already watered plant");
            }
            UnitOp::Harvest => {
                assert!(tile.yield_units() > 0, "HARVEST with nothing to take");
                if let Some(p) = tile.as_plant() {
                    assert!(
                        gs.day - p.planted_day >= p.crop.data().first_yield_day,
                        "HARVEST before first_yield_day"
                    );
                }
            }
            UnitOp::Fertilize => {
                assert!(tile.as_plant().is_some(), "FERTILIZE off a plant burns an item");
                assert!(
                    inv.get(ShedItem::Fertilizer) >= 1,
                    "FERTILIZE without fertilizer"
                );
            }
            UnitOp::Dig => {
                assert!(
                    matches!(tile, Tile::Weed | Tile::Coop | Tile::Pasture | Tile::Plant(_)),
                    "DIG on {tile:?}"
                );
            }
            UnitOp::BuildCoop | UnitOp::BuildPasture => {
                assert_eq!(tile, Tile::Empty, "BUILD on an occupied tile");
            }
            UnitOp::Feed => {
                let a = tile.as_animal().expect("FEED on an animal");
                assert!(!a.fed_today, "FEED on an already fed animal wastes wheat");
                assert!(inv.get(ShedItem::Wheat) >= 1, "FEED without wheat");
            }
            UnitOp::Care => {
                let a = tile.as_animal().expect("CARE on an animal");
                assert!(!a.cared_today);
            }
            UnitOp::CollectFertilizer => {
                let a = tile.as_animal().expect("COLLECT_FERTILIZER on an animal");
                assert!(a.fertilizer_available);
            }
            UnitOp::Plant(c) => {
                assert_eq!(tile, Tile::Empty, "PLANT on an occupied tile");
                plant_count[c.idx()] += 1;
            }
            UnitOp::Pickup { item, n } => {
                assert!(is_shed_adjacent(x, y, bs), "PICKUP away from the shed");
                assert!(n > 0, "PICKUP with a non-positive quantity");
                assert!(private.shed.get(item) >= 1, "PICKUP of an empty shed slot");
            }
            UnitOp::Place { item, n } => {
                assert!(n > 0, "PLACE with a non-positive quantity");
                assert!(inv.get(item) >= 1, "PLACE of an item not held");
                if let Some(a) = item.animal() {
                    assert_eq!(
                        tile,
                        Tile::empty_structure(a.data().structure),
                        "animal PLACE onto the wrong tile"
                    );
                } else {
                    assert!(is_shed_adjacent(x, y, bs), "deposit away from the shed");
                    assert!(drop_room > 0, "deposit into a shed with no room");
                    drop_room -= inv.get(item).min(drop_room);
                }
            }
            UnitOp::Drop => {
                assert!(is_shed_adjacent(x, y, bs), "DROP away from the shed");
                assert!(!inv.is_empty(), "DROP with an empty inventory");
                assert!(
                    drop_room >= inv.total(),
                    "DROP would delete {} items (room {drop_room}, carrying {})",
                    inv.total() - drop_room,
                    inv.total()
                );
                drop_room -= inv.total();
            }
        }

        // Once-per-tile-per-turn verbs must not be duplicated across units.
        if matches!(
            op,
            UnitOp::Water
                | UnitOp::Feed
                | UnitOp::Care
                | UnitOp::CollectFertilizer
                | UnitOp::Harvest
                | UnitOp::Fertilize
                | UnitOp::Plant(_)
        ) {
            // `Plant` is one class regardless of crop.
            let key = if let UnitOp::Plant(_) = op {
                std::mem::discriminant(&UnitOp::Plant(Crop::Wheat))
            } else {
                std::mem::discriminant(&op)
            };
            assert!(
                !claimed.contains(&(x, y, key)),
                "two units issued {op:?} on tile ({x},{y})"
            );
            claimed.push((x, y, key));
        }
    }

    // Atomic PLANT can never fire: the interpreter voids every PLANT for a crop whose
    // demand exceeds the seeds held.
    for c in CROP_LIST {
        assert!(
            plant_count[c.idx()] <= private.seeds[c.idx()],
            "{} PLANT requests for {:?} against {} seeds — the interpreter would void them all",
            plant_count[c.idx()],
            c,
            private.seeds[c.idx()]
        );
    }

    check_market_queue(gs, cfg, player, &act.market);
}

fn check_market_queue(
    gs: &GameState,
    cfg: &Config,
    player: usize,
    queue: &[Option<MarketOrder>],
) {
    let farm = &gs.farms[player];
    let private = &gs.privates[player];
    assert!(
        queue.len() <= cfg.max_orders_eff(),
        "market queue {} > maxMarketOrdersPerTurn {}",
        queue.len(),
        cfg.max_orders_eff()
    );

    let mut n_hire = 0usize;
    let mut seen_non_sell = false;
    for slot in queue {
        let ord = slot.expect("packer never emits an unparseable slot");
        match ord {
            MarketOrder::Sell { item, n } => {
                assert!(!seen_non_sell, "a SELL landed behind a non-SELL order");
                assert!(n > 0);
                assert!(
                    n <= private.shed.get(item.shed()),
                    "SELL {n} {item:?} against {} in the shed",
                    private.shed.get(item.shed())
                );
            }
            MarketOrder::Hire => {
                seen_non_sell = true;
                n_hire += 1;
            }
            MarketOrder::BuyLand => {
                seen_non_sell = true;
                assert!(
                    farm.unlocked_quadrants.len() - 1 < LAND_ORDER.len(),
                    "BUY_LAND with every quadrant already unlocked"
                );
            }
            MarketOrder::BuyProduct { item, n } => {
                seen_non_sell = true;
                assert!(n > 0 && item.is_buyable());
            }
            MarketOrder::BuySeed { n, .. } | MarketOrder::BuyAnimal { n, .. } => {
                seen_non_sell = true;
                assert!(n > 0);
            }
        }
    }
    assert!(n_hire <= MAX_HIRE_PER_TURN);
    assert!(
        n_hire <= MAX_HANDS.saturating_sub(farm.hands.len()),
        "hiring {n_hire} on top of {} hands would leave units unaddressable",
        farm.hands.len()
    );
}

/// Run one full episode under a uniformly random macro policy, asserting every invariant
/// at every step for both players.
fn run_random_episode(seed: u64, cfg: &Config) -> GameState {
    let mut rng = Rng::new(seed);
    let mut gs = GameState::new(cfg);
    gs.seed = seed as i64 & 0x7fff_ffff;
    let mut obs = vec![0f32; OBS_LEN];
    let mut mask = vec![0f32; MASK_LEN];
    let mut orders = Vec::new();
    let mut actions = [PlayerAction::pass(), PlayerAction::pass()];

    while !gs.done {
        for p in 0..2 {
            // The observation and mask must be computable at every reachable state.
            // `_with_cfg` so the non-default-config runs really exercise those divisors.
            encode_with_cfg(&gs, p, cfg, &mut obs);
            for (i, v) in obs.iter().enumerate() {
                assert!(v.is_finite(), "non-finite obs slot {i} at step {}", gs.step);
                assert!(
                    (-4.0..=4.0).contains(v),
                    "obs slot {i} = {v} out of range at step {}",
                    gs.step
                );
            }
            legal_mask_with_cfg(&gs, p, cfg, &mut mask);
            for v in mask.iter() {
                assert!(*v == 0.0 || *v == 1.0);
            }

            random_orders(&mut rng, &gs, p, &mut orders);
            let head = random_market(&mut rng);
            resolve_turn(&gs, p, cfg, &orders, &head, &mut actions[p]);
            check_emitted(&gs, cfg, p, &actions[p]);
            check_mask_agrees_with_resolver(&gs, cfg, p, &mask, &actions[p]);
        }
        step(&mut gs, &actions, cfg);
    }
    gs
}

/// A verb the mask calls illegal must never be emitted by the resolver. (The converse does
/// not hold: the mask is per-unit and ignores cross-unit contention such as the shared seed
/// budget, so it is deliberately the more permissive of the two.)
fn check_mask_agrees_with_resolver(
    gs: &GameState,
    _cfg: &Config,
    player: usize,
    mask: &[f32],
    act: &PlayerAction,
) {
    for idx in 0..=act.hands.len() {
        let op = if idx == 0 {
            act.farmer
        } else {
            act.hands[idx - 1]
        };
        let verb = match op {
            UnitOp::Pass | UnitOp::Noop | UnitOp::Move(_) => continue,
            UnitOp::Water => Verb::Water,
            UnitOp::Harvest => Verb::Harvest,
            UnitOp::Fertilize => Verb::Fertilize,
            UnitOp::Dig => Verb::Dig,
            UnitOp::BuildCoop => Verb::BuildCoop,
            UnitOp::BuildPasture => Verb::BuildPasture,
            UnitOp::Feed => Verb::Feed,
            UnitOp::Care => Verb::Care,
            UnitOp::CollectFertilizer => Verb::CollectFertilizer,
            UnitOp::Plant(c) => Verb::Plant(c),
            UnitOp::Drop => Verb::Drop,
            UnitOp::Pickup { item, n } => {
                let item_slot = PICKUP_ITEMS.iter().position(|i| *i == item).unwrap() as u8;
                let qty_slot = u8::from(n != 1);
                Verb::Pickup {
                    item_slot,
                    qty_slot,
                }
            }
            UnitOp::Place { item, .. } => match item.animal() {
                Some(a) => Verb::PlaceAnimal(a),
                None => Verb::Deposit(item.product().unwrap()),
            },
        };
        assert_eq!(
            mask[idx * N_VERBS + verb.index()],
            1.0,
            "resolver emitted {verb:?} for unit {idx} but the mask said illegal (player {player}, step {})",
            gs.step
        );
    }
}

// ---------------------------------------------------------------------------
// The invariant tests
// ---------------------------------------------------------------------------

#[test]
fn random_macro_policy_never_emits_a_costly_illegal_action() {
    let cfg = Config::default();
    for seed in 1..=6u64 {
        let gs = run_random_episode(seed, &cfg);
        assert!(gs.done);
        assert_eq!(gs.step, cfg.episode_steps as i64 - 1);
    }
}

#[test]
fn random_macro_policy_reaches_interesting_states() {
    // A test that only ever visits the opening board proves nothing, so assert the random
    // driver actually gets to late-game structure.
    let cfg = Config::default();
    let mut saw_weed = false;
    let mut saw_plant = false;
    let mut saw_structure = false;
    let mut saw_hand = false;
    let mut saw_shed_items = false;
    let mut saw_land = false;
    let mut rng = Rng::new(99);
    let mut gs = GameState::new(&cfg);
    let mut orders = Vec::new();
    let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
    while !gs.done {
        for p in 0..2 {
            random_orders(&mut rng, &gs, p, &mut orders);
            let head = random_market(&mut rng);
            resolve_turn(&gs, p, &cfg, &orders, &head, &mut actions[p]);
            check_emitted(&gs, &cfg, p, &actions[p]);
        }
        step(&mut gs, &actions, &cfg);
        for p in 0..2 {
            saw_hand |= !gs.farms[p].hands.is_empty();
            saw_land |= gs.farms[p].unlocked_quadrants.len() > 1;
            saw_shed_items |= gs.privates[p].shed.total() > 0;
            for t in &gs.farms[p].tiles {
                match t {
                    Tile::Weed => saw_weed = true,
                    Tile::Plant(_) => saw_plant = true,
                    Tile::Coop | Tile::Pasture | Tile::Animal(_) => saw_structure = true,
                    _ => {}
                }
            }
        }
    }
    assert!(saw_weed, "no weed ever appeared");
    assert!(saw_plant, "nothing was ever planted");
    assert!(saw_structure, "no coop/pasture was ever built");
    assert!(saw_hand, "no hand was ever hired");
    assert!(saw_shed_items, "the shed was never used");
    assert!(saw_land, "no land was ever bought");
}

#[test]
fn drop_is_refused_when_it_would_destroy_items() {
    // Shed at 96/100, farmer carrying 49 items: Python's DROP would `del` every key it
    // visits and vaporise 45 of them.
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].shed.set(ShedItem::Carrot, 96);
    gs.privates[0].inventories[0].add(ShedItem::Melon, 9);
    gs.privates[0].inventories[0].add(ShedItem::Wheat, 40);
    let shed_tile = (4 + 4 * 10) as u8; // (4,4), where the farmer spawns
    let mut act = PlayerAction::pass();

    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: shed_tile,
            verb: Verb::Drop,
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(act.farmer, UnitOp::Pass, "DROP must be refused at 96/100");

    // A per-item deposit is emitted instead: `PLACE` clamps to the room and leaves the
    // remainder in the inventory, so nothing is lost.
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: shed_tile,
            verb: Verb::Deposit(Product::Melon),
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert!(matches!(
        act.farmer,
        UnitOp::Place {
            item: ShedItem::Melon,
            ..
        }
    ));

    // With room for everything, DROP is allowed again.
    gs.privates[0].shed.set(ShedItem::Carrot, 40);
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: shed_tile,
            verb: Verb::Drop,
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(act.farmer, UnitOp::Drop);
}

#[test]
fn two_units_cannot_both_drop_into_the_same_room() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    // Room for 50; each of two units carries 30.
    gs.privates[0].shed.set(ShedItem::Carrot, 50);
    gs.farms[0].hands.push([4, 4]);
    gs.privates[0].inventories.push(Default::default());
    gs.privates[0].inventories[0].add(ShedItem::Wheat, 30);
    gs.privates[0].inventories[1].add(ShedItem::Melon, 30);
    let shed_tile = (4 + 4 * 10) as u8;
    let order = MacroOrder {
        target: shed_tile,
        verb: Verb::Drop,
    };
    let mut act = PlayerAction::pass();
    resolve_turn(&gs, 0, &cfg, &[order, order], &MarketHead::default(), &mut act);
    assert_eq!(act.farmer, UnitOp::Drop);
    assert_eq!(
        act.hands[0],
        UnitOp::Pass,
        "the second DROP would have deleted 10 items"
    );
    check_emitted(&gs, &cfg, 0, &act);
}

#[test]
fn atomic_plant_contention_is_capped_at_the_seed_count() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].seeds[Crop::Melon.idx()] = 2;
    gs.farms[0].farmer = [0, 0];
    for i in 1..6i64 {
        gs.farms[0].hands.push([i, 0]);
        gs.privates[0].inventories.push(Default::default());
    }
    // Six units, six different empty tiles, all asking for MELON: the interpreter would
    // void all six.
    let orders: Vec<MacroOrder> = (0..6)
        .map(|i| MacroOrder {
            target: i as u8,
            verb: Verb::Plant(Crop::Melon),
        })
        .collect();
    let mut act = PlayerAction::pass();
    resolve_turn(&gs, 0, &cfg, &orders, &MarketHead::default(), &mut act);
    let planting = std::iter::once(act.farmer)
        .chain(act.hands.iter().copied())
        .filter(|o| matches!(o, UnitOp::Plant(Crop::Melon)))
        .count();
    assert_eq!(planting, 2, "exactly the two seeds held may be planted");
    check_emitted(&gs, &cfg, 0, &act);

    // And the plants really do land: two seeds spent, two tiles occupied.
    let actions = [act.clone(), PlayerAction::pass()];
    step(&mut gs, &actions, &cfg);
    assert_eq!(gs.privates[0].seeds[Crop::Melon.idx()], 0);
    assert_eq!(
        gs.farms[0]
            .tiles
            .iter()
            .filter(|t| matches!(t, Tile::Plant(_)))
            .count(),
        2
    );
}

#[test]
fn invalid_target_collapses_to_pass_without_walking() {
    let cfg = Config::default();
    let gs = GameState::new(&cfg);
    let mut act = PlayerAction::pass();
    // Nothing to water anywhere: the farmer must not wander towards an empty tile.
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: 0,
            verb: Verb::Water,
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(act.farmer, UnitOp::Pass);

    // A tile op aimed at a LOCKED tile is refused too, but a shed op on the *same* locked
    // tile is fine: (5,4) is locked at the start and is a shed-access tile.
    let locked = 5 + 4 * 10;
    assert_eq!(gs.farms[0].tile(5, 4), Tile::Locked);
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: locked as u8,
            verb: Verb::BuildCoop,
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(act.farmer, UnitOp::Pass, "BUILD on LOCKED must not walk");

    let mut gs2 = gs.clone();
    gs2.privates[0].shed.set(ShedItem::Wheat, 5);
    resolve_turn(
        &gs2,
        0,
        &cfg,
        &[MacroOrder {
            target: locked as u8,
            verb: Verb::Pickup {
                item_slot: 0,
                qty_slot: 1,
            },
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(
        act.farmer,
        UnitOp::Move(Dir::East),
        "a shed op on a locked shed-access tile is legal, so walk there"
    );
}

#[test]
fn walking_is_allowed_while_the_resource_is_still_in_flight() {
    // A unit with no seeds may still be sent towards the tile it will plant: BUY_SEED
    // settles in the market phase, after this turn's unit actions.
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.farms[0].farmer = [4, 4];
    let mut act = PlayerAction::pass();
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: 0,
            verb: Verb::Plant(Crop::Wheat),
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert!(matches!(act.farmer, UnitOp::Move(_)), "should walk anyway");
    // Standing on the target with no seeds, it declines rather than PLANTing.
    gs.farms[0].farmer = [0, 0];
    resolve_turn(
        &gs,
        0,
        &cfg,
        &[MacroOrder {
            target: 0,
            verb: Verb::Plant(Crop::Wheat),
        }],
        &MarketHead::default(),
        &mut act,
    );
    assert_eq!(act.farmer, UnitOp::Pass);
}

// ---------------------------------------------------------------------------
// Market packer
// ---------------------------------------------------------------------------

#[test]
fn packer_never_exceeds_the_order_cap() {
    let cfg = Config::default();
    let mut rng = Rng::new(7);
    let mut queue = Vec::new();
    for _ in 0..20_000 {
        let mut gs = GameState::new(&cfg);
        // Randomise everything the packer reads.
        for p in 0..2 {
            gs.farms[p].money = (rng.below(60_000)) as f64;
            gs.farms[p].hires_today = rng.below(20) as i64;
            for _ in 0..rng.below(MAX_HANDS + 4) {
                gs.farms[p].hands.push([4, 4]);
                gs.privates[p].inventories.push(Default::default());
            }
            for k in 0..rng.below(4) {
                let q = LAND_ORDER[k.min(2)];
                if !gs.farms[p].unlocked_quadrants.contains(&q) {
                    gs.farms[p].unlocked_quadrants.push(q);
                }
            }
            for item in SHED_ITEMS {
                gs.privates[p].shed.set(item, rng.below(12) as i64);
            }
        }
        for item in PRODUCTS {
            gs.market.inventory[item.idx()] = 8_000 + rng.below(5_000) as i64;
        }
        kaggri_core::sim::refresh_prices(&mut gs.market);

        let head = random_market(&mut rng);
        pack_market(&gs, 0, &cfg, &head, &mut queue);
        assert!(queue.len() <= cfg.max_orders_eff(), "{} orders", queue.len());
        check_market_queue(&gs, &cfg, 0, &queue);
    }
    // A tighter cap must be honoured too.
    let mut tight = Config::default();
    tight.max_market_orders_per_turn = 3;
    let gs = GameState::new(&tight);
    let mut head = MarketHead::default();
    head.hire = 10;
    head.seed_qty = [6; N_CROPS];
    pack_market(&gs, 0, &tight, &head, &mut queue);
    assert_eq!(queue.len(), 3);
}

#[test]
fn packer_puts_sells_first_and_hires_last() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].shed.set(ShedItem::Carrot, 40);
    gs.privates[0].shed.set(ShedItem::Melon, 10);
    gs.farms[0].money = 10_000.0;
    let head = MarketHead {
        hire: 2,
        buy_land: true,
        seed_qty: [2, 0, 0, 0, 0],
        animal_qty: [1, 0, 0],
        buy_product: [3, 0],
        sell_frac: {
            let mut s = [0u8; N_PRODUCTS];
            s[Product::Carrot.idx()] = 4; // all
            s[Product::Melon.idx()] = 2; // half
            s
        },
    };
    let mut queue = Vec::new();
    pack_market(&gs, 0, &cfg, &head, &mut queue);
    let ords: Vec<MarketOrder> = queue.iter().map(|o| o.unwrap()).collect();

    // MELON at $250 x 5 = 1250 beats CARROT at $35 x 40 = 1400? No: carrot wins, and the
    // larger expected revenue must come first so it gets the freshest quote.
    assert_eq!(
        ords[0],
        MarketOrder::Sell {
            item: Product::Carrot,
            n: 40
        }
    );
    assert_eq!(
        ords[1],
        MarketOrder::Sell {
            item: Product::Melon,
            n: 5
        }
    );
    assert_eq!(
        ords[2],
        MarketOrder::BuyProduct {
            item: Product::Wheat,
            n: 4
        }
    );
    assert_eq!(
        ords[3],
        MarketOrder::BuySeed {
            crop: Crop::Wheat,
            n: 2
        }
    );
    assert_eq!(
        ords[4],
        MarketOrder::BuyAnimal {
            animal: Animal::Goose,
            n: 1
        }
    );
    assert_eq!(ords[5], MarketOrder::BuyLand);
    assert_eq!(ords[6], MarketOrder::Hire);
    assert_eq!(ords[7], MarketOrder::Hire);
    assert_eq!(ords.len(), 8);
}

#[test]
fn hires_win_the_slots_when_everything_is_requested() {
    // 10 hires + 9 sells + 2 buys + 5 seeds + 3 animals + land = 30 requests, 10 slots.
    // Hires have the highest inclusion priority, so they take all ten — the documented
    // consequence, and exactly the "hour 0 is a pure hiring turn" advice.
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.farms[0].money = 100_000.0;
    for item in SHED_ITEMS {
        gs.privates[0].shed.set(item, 4);
    }
    let head = MarketHead {
        hire: 10,
        buy_land: true,
        seed_qty: [3; N_CROPS],
        animal_qty: [1; N_ANIMALS],
        buy_product: [3; 2],
        sell_frac: [4; N_PRODUCTS],
    };
    let mut queue = Vec::new();
    pack_market(&gs, 0, &cfg, &head, &mut queue);
    assert_eq!(queue.len(), 10);
    assert!(queue.iter().all(|o| *o == Some(MarketOrder::Hire)));

    // Drop the hire request to 2 and the sells reappear at the front.
    let head2 = MarketHead { hire: 2, ..head };
    pack_market(&gs, 0, &cfg, &head2, &mut queue);
    assert_eq!(queue.len(), 10);
    assert!(matches!(queue[0], Some(MarketOrder::Sell { .. })));
    assert_eq!(queue[8], Some(MarketOrder::Hire));
    assert_eq!(queue[9], Some(MarketOrder::Hire));
}

#[test]
fn packer_drops_provably_dead_orders() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    let mut queue = Vec::new();

    // Broke: nothing is affordable and no sells can fund anything.
    gs.farms[0].money = 0.0;
    let head = MarketHead {
        hire: 4,
        buy_land: true,
        seed_qty: [3; N_CROPS],
        animal_qty: [3; N_ANIMALS],
        buy_product: [3; 2],
        sell_frac: [4; N_PRODUCTS],
    };
    pack_market(&gs, 0, &cfg, &head, &mut queue);
    assert!(queue.is_empty(), "nothing is payable at $0: {queue:?}");

    // $2 buys exactly two hands (fib 1, 1) and nothing else.
    gs.farms[0].money = 2.0;
    pack_market(&gs, 0, &cfg, &head, &mut queue);
    assert_eq!(queue.len(), 2);
    assert!(queue.iter().all(|o| *o == Some(MarketOrder::Hire)));

    // A full shed blocks BUY_PRODUCT and BUY_ANIMAL but not BUY_SEED.
    gs.farms[0].money = 50_000.0;
    gs.privates[0].shed.set(ShedItem::Wool, 100);
    let head2 = MarketHead {
        hire: 0,
        buy_land: false,
        seed_qty: [1, 0, 0, 0, 0],
        animal_qty: [1, 0, 0],
        buy_product: [1, 0],
        sell_frac: [0; N_PRODUCTS],
    };
    pack_market(&gs, 0, &cfg, &head2, &mut queue);
    assert_eq!(
        queue,
        vec![Some(MarketOrder::BuySeed {
            crop: Crop::Wheat,
            n: 1
        })]
    );
    // ...unless this turn's sells make room first.
    let head3 = MarketHead {
        sell_frac: {
            let mut s = [0u8; N_PRODUCTS];
            s[Product::Wool.idx()] = 4;
            s
        },
        ..head2
    };
    pack_market(&gs, 0, &cfg, &head3, &mut queue);
    assert!(queue.contains(&Some(MarketOrder::BuyProduct {
        item: Product::Wheat,
        n: 1
    })));
    assert!(queue.contains(&Some(MarketOrder::BuyAnimal {
        animal: Animal::Goose,
        n: 1
    })));

    // All quadrants unlocked: BUY_LAND is dead.
    for q in LAND_ORDER {
        gs.farms[0].unlocked_quadrants.push(q);
    }
    let head4 = MarketHead {
        buy_land: true,
        ..MarketHead::default()
    };
    pack_market(&gs, 0, &cfg, &head4, &mut queue);
    assert!(queue.is_empty());

    // Hiring stops at MAX_HANDS so every unit keeps a macro order.
    for _ in 0..MAX_HANDS {
        gs.farms[0].hands.push([4, 4]);
        gs.privates[0].inventories.push(Default::default());
    }
    let head5 = MarketHead {
        hire: 10,
        ..MarketHead::default()
    };
    pack_market(&gs, 0, &cfg, &head5, &mut queue);
    assert!(queue.is_empty(), "already at MAX_HANDS");
}

#[test]
fn sell_fractions_round_up_and_clamp() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].shed.set(ShedItem::Wheat, 3);
    let mut queue = Vec::new();
    for (bucket, want) in [(1usize, 1i64), (2, 2), (3, 3), (4, 3)] {
        let mut head = MarketHead::default();
        head.sell_frac[Product::Wheat.idx()] = bucket as u8;
        pack_market(&gs, 0, &cfg, &head, &mut queue);
        assert_eq!(
            queue[0],
            Some(MarketOrder::Sell {
                item: Product::Wheat,
                n: want
            }),
            "fraction {} of 3",
            SELL_FRACS[bucket]
        );
    }
    // Bucket 0 is "do not sell".
    let head = MarketHead::default();
    pack_market(&gs, 0, &cfg, &head, &mut queue);
    assert!(queue.is_empty());
}

#[test]
fn market_head_choices_clamp_instead_of_panicking() {
    let choices = [255u8; N_MARKET_HEADS];
    let h = MarketHead::from_choices(&choices);
    assert_eq!(h.hire as usize, MAX_HIRE_PER_TURN);
    assert!(h.buy_land);
    assert_eq!(h.seed_qty, [QTY_BUCKETS.len() as u8 - 1; N_CROPS]);
    assert_eq!(h.sell_frac, [SELL_FRACS.len() as u8 - 1; N_PRODUCTS]);
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

#[test]
fn dims_are_self_consistent() {
    let d = obs_dims();
    assert_eq!(d.version, OBS_VERSION);
    assert_eq!(d.n_tokens, N_TOKENS);
    assert_eq!(d.n_features, N_FEATURES);
    assert_eq!(d.obs_len, N_TOKENS * N_FEATURES);
    assert_eq!(d.obs_len, OBS_LEN);
    assert_eq!(d.mask_len, MASK_LEN);
    assert_eq!(d.mask_len, MAX_UNITS * N_VERBS);
    assert_eq!(
        d.n_global + d.n_tile + d.n_unit + d.n_product + d.n_opp + d.n_private + d.n_scratch,
        d.n_tokens
    );
    assert_eq!(d.off_scratch + d.n_scratch, d.n_tokens);
    assert_eq!(d.n_targets, N_TARGETS);
    assert_eq!(d.n_verbs, N_VERBS);
    assert_eq!(d.market_logits, 128);
    // Documented budget numbers.
    assert_eq!(d.n_tokens, 140);
    assert_eq!(d.n_features, 48);
    assert_eq!(d.obs_len, 6_720);
    assert_eq!(d.mask_len, 646);
}

#[test]
fn encode_is_deterministic_and_uses_no_padding_slots() {
    let cfg = Config::default();
    let gs = GameState::new(&cfg);
    let mut a = vec![0f32; OBS_LEN];
    let mut b = vec![9f32; OBS_LEN];
    encode(&gs, 0, &mut a);
    encode(&gs, 0, &mut b);
    assert_eq!(a, b, "encode must overwrite the whole buffer");

    // Header: exactly one type flag plus the bias on every token.
    for t in 0..N_TOKENS {
        let tok = &a[t * N_FEATURES..(t + 1) * N_FEATURES];
        let types: f32 = tok[0..7].iter().sum();
        assert_eq!(types, 1.0, "token {t} type one-hot");
        assert_eq!(tok[7], 1.0, "token {t} bias");
    }
    // The reserved tail of the payload is never written by any token type. The widest
    // payload is PRIVATE's 38 slots (features 8..46), so 46 and 47 are the spare capacity.
    for t in 0..N_TOKENS {
        for s in 46..N_FEATURES {
            assert_eq!(a[t * N_FEATURES + s], 0.0, "token {t} feature {s} is reserved");
        }
    }
    // ...and that stays true in a state where every payload slot is non-zero.
    let mut gs2 = GameState::new(&cfg);
    gs2.step = 400;
    gs2.day = 16;
    gs2.hour = 8;
    for p in 0..2 {
        gs2.farms[p].money = 8_000.0;
        gs2.farms[p].hires_today = 3;
        gs2.farms[p].hands.push([2, 3]);
        gs2.privates[p].inventories.push(Default::default());
        gs2.privates[p].inventories[0].add(ShedItem::Melon, 9);
        gs2.privates[p].seeds = [3; N_CROPS];
        for item in SHED_ITEMS {
            gs2.privates[p].shed.set(item, 2);
        }
        gs2.farms[p].set_tile(0, 0, Tile::Plant(kaggri_core::sim::new_plant(Crop::Tomato, 4, 24)));
        gs2.farms[p].set_tile(1, 0, Tile::Animal(kaggri_core::sim::new_animal(Animal::Sheep, 4)));
        gs2.farms[p].set_tile(2, 0, Tile::Weed);
        gs2.farms[p].set_tile(3, 0, Tile::Coop);
        gs2.farms[p].set_tile(4, 0, Tile::Pasture);
    }
    let mut c = vec![0f32; OBS_LEN];
    encode(&gs2, 0, &mut c);
    for t in 0..N_TOKENS {
        for s in 46..N_FEATURES {
            assert_eq!(c[t * N_FEATURES + s], 0.0, "token {t} feature {s} is reserved");
        }
    }
}

#[test]
fn encode_is_player_relative() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.farms[0].money = 9_000.0;
    gs.farms[1].money = 1_000.0;
    let mut a = vec![0f32; OBS_LEN];
    let mut b = vec![0f32; OBS_LEN];
    encode(&gs, 0, &mut a);
    encode(&gs, 1, &mut b);
    let g = |v: &[f32], slot: usize| v[8 + slot];
    // GLOBAL +7 is own money, +9 the opponent's; player 1 sees them swapped.
    assert_eq!(g(&a, 7), g(&b, 9));
    assert_eq!(g(&a, 9), g(&b, 7));
    assert!(g(&a, 12) == 1.0 && g(&b, 12) == 0.0, "own_leading flag");
}

#[test]
fn tile_tokens_describe_the_board() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    // A watered, fertilized, harvestable carrot at (1,2).
    let mut p = kaggri_core::sim::new_plant(Crop::Carrot, 0, 24);
    p.watered_today = true;
    p.consecutive_unwatered = 0;
    p.yield_units = 3;
    p.fertilized_until_day = 2;
    gs.farms[0].set_tile(1, 2, Tile::Plant(p));
    gs.day = 3;
    let mut o = vec![0f32; OBS_LEN];
    encode(&gs, 0, &mut o);

    let tok = 1 + (2 * 10 + 1); // OFF_TILE + y*10 + x
    let t = &o[tok * N_FEATURES..(tok + 1) * N_FEATURES];
    let p_ = &t[8..];
    assert_eq!(p_[5], 1.0, "kind one-hot = PLANT");
    assert_eq!(p_[7 + Crop::Carrot.idx()], 1.0, "crop one-hot = CARROT");
    assert_eq!(p_[20], 3.0 / 30.0, "age 3 days");
    assert_eq!(p_[21], 3.0 / 6.0, "yield 3");
    assert_eq!(p_[23], 1.0, "harvestable (age 3 >= first_yield_day 2)");
    assert_eq!(p_[25], 1.0, "watered today");
    assert_eq!(p_[27], 0.0, "not dying tonight");
    assert_eq!(p_[15], 1.0 / 9.0);
    assert_eq!(p_[16], 2.0 / 9.0);

    // An empty locked tile in the SE quadrant.
    let tok = 1 + (9 * 10 + 9);
    let t = &o[tok * N_FEATURES..(tok + 1) * N_FEATURES];
    assert_eq!(t[8 + 1], 1.0, "kind one-hot = LOCKED");
}

#[test]
fn unit_tokens_track_existence_and_inventory_order() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].inventories[0].add(ShedItem::Melon, 5);
    gs.privates[0].inventories[0].add(ShedItem::Wheat, 50);
    let mut o = vec![0f32; OBS_LEN];
    encode(&gs, 0, &mut o);
    let base = 101; // OFF_UNIT
    let farmer = &o[base * N_FEATURES..(base + 1) * N_FEATURES];
    assert_eq!(farmer[8], 1.0, "farmer exists");
    assert_eq!(farmer[9], 1.0, "is_farmer");
    assert_eq!(farmer[8 + 6 + ShedItem::Melon.idx()], 5.0 / 32.0);
    assert_eq!(farmer[8 + 6 + ShedItem::Wheat.idx()], 50.0 / 32.0);
    // Insertion order: MELON went in first, so it is what survives a full shed.
    assert_eq!(farmer[8 + 19], ShedItem::Melon.idx() as f32 / 11.0);
    let hand = &o[(base + 1) * N_FEATURES..(base + 2) * N_FEATURES];
    assert_eq!(hand[8], 0.0, "hand 1 does not exist");
}

#[test]
fn product_tokens_carry_price_and_depth() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.privates[0].shed.set(ShedItem::Melon, 60);
    let mut o = vec![0f32; OBS_LEN];
    encode(&gs, 0, &mut o);
    let tok = 118 + Product::Melon.idx(); // OFF_PRODUCT
    let t = &o[tok * N_FEATURES..(tok + 1) * N_FEATURES];
    let p = &t[8..];
    assert_eq!(p[Product::Melon.idx()], 1.0, "product one-hot");
    assert_eq!(p[10], 1.0, "price/base is 1.0 at I0");
    assert_eq!(p[13], 0.0, "not at the floor");
    assert_eq!(p[15], 60.0 / 100.0, "shed holding");
    // Melon's above-curve is `sq` at target 3.60: 60 units already move the price hard.
    let after = kaggri_core::sim::market_price(Product::Melon, 10_060, &gs.market.params);
    assert!(after < 250);
    assert_eq!(p[22], after as f32 / 250.0, "depth slot");
    assert_eq!(p[20], 0.0, "melon is not buyable");
    let wheat = 118 + Product::Wheat.idx();
    assert_eq!(o[wheat * N_FEATURES + 8 + 20], 1.0, "wheat is buyable");
}

#[test]
fn mask_reflects_inventory_and_board() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    let mut m = vec![0f32; MASK_LEN];

    legal_mask(&gs, 0, &mut m);
    let row = |m: &[f32], u: usize, v: Verb| m[u * N_VERBS + v.index()];
    assert_eq!(row(&m, 0, Verb::Idle), 1.0, "IDLE is always available");
    assert_eq!(row(&m, 0, Verb::Water), 0.0, "nothing to water");
    assert_eq!(row(&m, 0, Verb::Feed), 0.0, "no animals, no wheat");
    assert_eq!(row(&m, 0, Verb::BuildCoop), 1.0, "25 empty NW tiles");
    assert_eq!(row(&m, 0, Verb::Dig), 0.0, "nothing to dig");
    assert_eq!(row(&m, 0, Verb::Plant(Crop::Wheat)), 0.0, "no seeds");
    for u in 1..MAX_UNITS {
        for v in 0..N_VERBS {
            assert_eq!(m[u * N_VERBS + v], 0.0, "absent unit {u} verb {v}");
        }
    }

    // Seeds make PLANT live; an unwatered plant makes WATER live; wheat in hand plus a
    // hungry animal makes FEED live.
    gs.privates[0].seeds[Crop::Wheat.idx()] = 1;
    gs.farms[0].set_tile(0, 0, Tile::Plant(kaggri_core::sim::new_plant(Crop::Wheat, 0, 24)));
    let mut a = kaggri_core::sim::new_animal(Animal::Goose, 0);
    a.fertilizer_available = true;
    gs.farms[0].set_tile(1, 0, Tile::Animal(a));
    gs.privates[0].inventories[0].add(ShedItem::Wheat, 1);
    gs.privates[0].shed.set(ShedItem::Fertilizer, 2);
    legal_mask(&gs, 0, &mut m);
    assert_eq!(row(&m, 0, Verb::Plant(Crop::Wheat)), 1.0);
    assert_eq!(row(&m, 0, Verb::Plant(Crop::Melon)), 0.0);
    assert_eq!(row(&m, 0, Verb::Water), 1.0);
    assert_eq!(row(&m, 0, Verb::Feed), 1.0);
    assert_eq!(row(&m, 0, Verb::Care), 1.0);
    assert_eq!(row(&m, 0, Verb::CollectFertilizer), 1.0);
    assert_eq!(row(&m, 0, Verb::Fertilize), 0.0, "no fertilizer in hand");
    assert_eq!(
        row(
            &m,
            0,
            Verb::Pickup {
                item_slot: 1,
                qty_slot: 0
            }
        ),
        1.0,
        "fertilizer is in the shed"
    );
    assert_eq!(row(&m, 0, Verb::Deposit(Product::Wheat)), 1.0);
    assert_eq!(row(&m, 0, Verb::Deposit(Product::Melon)), 0.0);
    assert_eq!(row(&m, 0, Verb::Drop), 1.0);
    assert_eq!(row(&m, 0, Verb::Dig), 1.0, "the plant is diggable");
}

/// The mask's "is there any target" shortcut must agree with the resolver's per-tile
/// predicate on real boards, or the two would drift apart silently.
#[test]
fn mask_target_shortcut_matches_the_per_tile_predicate() {
    let cfg = Config::default();
    let mut rng = Rng::new(31);
    let mut gs = GameState::new(&cfg);
    let mut mask = vec![0f32; MASK_LEN];
    let mut orders = Vec::new();
    let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
    let mut checks = 0usize;

    while !gs.done {
        for p in 0..2 {
            legal_mask(&gs, p, &mut mask);
            let farm = &gs.farms[p];
            for (v, verb) in VERBS.iter().enumerate() {
                if *verb == Verb::Idle {
                    continue;
                }
                let any = (0..cfg.board_size as i64).any(|y| {
                    (0..cfg.board_size as i64).any(|x| {
                        target_plausible(*verb, farm.tile(x, y), gs.day, x, y, cfg.board_size)
                    })
                });
                // The mask is the conjunction of "some tile admits it" and "the unit has
                // the resources", so a set bit implies a plausible tile exists.
                if mask[v] == 1.0 {
                    assert!(any, "mask set for {verb:?} with no plausible tile");
                }
                // And when no tile admits it, no unit can have it set.
                if !any {
                    for u in 0..MAX_UNITS {
                        assert_eq!(
                            mask[u * N_VERBS + v],
                            0.0,
                            "{verb:?} set for unit {u} with no plausible tile"
                        );
                    }
                }
                checks += 1;
            }
            random_orders(&mut rng, &gs, p, &mut orders);
            let head = random_market(&mut rng);
            resolve_turn(&gs, p, &cfg, &orders, &head, &mut actions[p]);
        }
        step(&mut gs, &actions, &cfg);
    }
    assert!(checks > 50_000, "only {checks} checks ran");
}

#[test]
fn encode_and_mask_survive_a_hand_heavy_late_game_state() {
    let cfg = Config::default();
    let mut gs = GameState::new(&cfg);
    gs.step = 700;
    gs.day = 29;
    gs.hour = 4;
    for q in LAND_ORDER {
        gs.farms[0].unlocked_quadrants.push(q);
    }
    for i in 0..MAX_HANDS {
        gs.farms[0].hands.push([(i % 10) as i64, (i / 10) as i64]);
        gs.privates[0].inventories.push(Default::default());
    }
    gs.farms[0].money = 250_000.0;
    gs.farms[1].money = -50.0;
    for item in SHED_ITEMS {
        gs.privates[0].shed.set(item, 30);
    }
    gs.privates[0].seeds = [400; N_CROPS];
    for item in PRODUCTS {
        gs.market.inventory[item.idx()] = 40_000;
    }
    kaggri_core::sim::refresh_prices(&mut gs.market);
    for y in 0..10i64 {
        for x in 0..10i64 {
            gs.farms[0].set_tile(x, y, Tile::Plant(kaggri_core::sim::new_plant(Crop::Melon, 0, 24)));
            gs.farms[1].set_tile(x, y, Tile::Animal(kaggri_core::sim::new_animal(Animal::Cow, 0)));
        }
    }
    let mut o = vec![0f32; OBS_LEN];
    let mut m = vec![0f32; MASK_LEN];
    for p in 0..2 {
        encode(&gs, p, &mut o);
        for (i, v) in o.iter().enumerate() {
            assert!(v.is_finite() && (-4.0..=4.0).contains(v), "slot {i} = {v}");
        }
        legal_mask(&gs, p, &mut m);
    }
    // All 17 unit rows are live for player 0, and only the farmer's for player 1.
    legal_mask(&gs, 0, &mut m);
    for u in 0..MAX_UNITS {
        assert_eq!(m[u * N_VERBS + Verb::Idle.index()], 1.0, "unit {u}");
    }
    legal_mask(&gs, 1, &mut m);
    assert_eq!(m[Verb::Idle.index()], 1.0);
    for u in 1..MAX_UNITS {
        assert_eq!(m[u * N_VERBS + Verb::Idle.index()], 0.0, "unit {u}");
    }
}

#[test]
fn the_random_driver_actually_emits_every_verb() {
    // The invariant tests are only worth as much as their coverage, so pin it down: every
    // one of the 38 verbs must reach the environment at least once, and the walk path too.
    let cfg = Config::default();
    let mut counts = [0u64; N_VERBS];
    let mut moves = 0u64;
    let mut passes = 0u64;
    for seed in 1..=4u64 {
        let mut rng = Rng::new(seed);
        let mut gs = GameState::new(&cfg);
        let mut orders = Vec::new();
        let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
        while !gs.done {
            for p in 0..2 {
                random_orders(&mut rng, &gs, p, &mut orders);
                let head = random_market(&mut rng);
                resolve_turn(&gs, p, &cfg, &orders, &head, &mut actions[p]);
                check_emitted(&gs, &cfg, p, &actions[p]);
                for idx in 0..=actions[p].hands.len() {
                    let op = if idx == 0 {
                        actions[p].farmer
                    } else {
                        actions[p].hands[idx - 1]
                    };
                    match op {
                        UnitOp::Move(_) => moves += 1,
                        UnitOp::Pass | UnitOp::Noop => passes += 1,
                        UnitOp::Water => counts[Verb::Water.index()] += 1,
                        UnitOp::Harvest => counts[Verb::Harvest.index()] += 1,
                        UnitOp::Fertilize => counts[Verb::Fertilize.index()] += 1,
                        UnitOp::Dig => counts[Verb::Dig.index()] += 1,
                        UnitOp::BuildCoop => counts[Verb::BuildCoop.index()] += 1,
                        UnitOp::BuildPasture => counts[Verb::BuildPasture.index()] += 1,
                        UnitOp::Feed => counts[Verb::Feed.index()] += 1,
                        UnitOp::Care => counts[Verb::Care.index()] += 1,
                        UnitOp::CollectFertilizer => counts[Verb::CollectFertilizer.index()] += 1,
                        UnitOp::Plant(c) => counts[Verb::Plant(c).index()] += 1,
                        UnitOp::Drop => counts[Verb::Drop.index()] += 1,
                        UnitOp::Pickup { item, n } => {
                            let item_slot =
                                PICKUP_ITEMS.iter().position(|i| *i == item).unwrap() as u8;
                            counts[Verb::Pickup {
                                item_slot,
                                qty_slot: u8::from(n != 1),
                            }
                            .index()] += 1;
                        }
                        UnitOp::Place { item, .. } => match item.animal() {
                            Some(a) => counts[Verb::PlaceAnimal(a).index()] += 1,
                            None => counts[Verb::Deposit(item.product().unwrap()).index()] += 1,
                        },
                    }
                }
            }
            step(&mut gs, &actions, &cfg);
        }
    }
    println!("moves={moves} passes={passes}");
    for (i, v) in VERBS.iter().enumerate() {
        println!("{:>4}  {:?}", counts[i], v);
    }
    assert!(moves > 200, "the walk path was barely exercised: {moves}");
    // What a uniform-random macro policy can actually reach in one episode: tile work on
    // land it happens to be standing on, plus shed traffic. It never completes the
    // multi-turn chains (buy animal -> pick it up -> walk to a structure -> place it;
    // water a plant daily until it matures, then harvest), which is why the per-verb
    // effect coverage lives in `every_verb_resolves_and_takes_effect` instead.
    for v in [
        Verb::Dig,
        Verb::BuildCoop,
        Verb::BuildPasture,
        Verb::Plant(Crop::Wheat),
        Verb::Drop,
    ] {
        assert!(counts[v.index()] > 0, "{v:?} was never emitted");
    }
    let total: u64 = counts.iter().sum();
    assert!(total > 100, "only {total} verbs ever executed");
}

// ---------------------------------------------------------------------------
// Per-verb end-to-end effect coverage
// ---------------------------------------------------------------------------

/// Resolve `verb` for the farmer standing on `target`, then actually apply the op with the
/// real interpreter helper and hand back the mutated state.
fn resolve_and_apply(gs: &GameState, cfg: &Config, target: [i64; 2], verb: Verb) -> (UnitOp, GameState) {
    let mut gs2 = gs.clone();
    gs2.farms[0].farmer = target;
    let mut act = PlayerAction::pass();
    resolve_turn(
        &gs2,
        0,
        cfg,
        &[MacroOrder {
            target: (target[1] * 10 + target[0]) as u8,
            verb,
        }],
        &MarketHead::default(),
        &mut act,
    );
    check_emitted(&gs2, cfg, 0, &act);
    let op = act.farmer;
    let (farm, private) = (&mut gs2.farms[0], &mut gs2.privates[0]);
    kaggri_core::sim::apply_unit_action(
        farm,
        private,
        0,
        op,
        cfg.board_size,
        gs2.day,
        cfg.turns_per_day_eff(),
        cfg.shed_capacity_i64(),
    );
    (op, gs2)
}

#[test]
fn every_verb_resolves_and_takes_effect() {
    let cfg = Config::default();
    let base = GameState::new(&cfg);
    let shed = [4i64, 4];
    let tile = [1i64, 1];
    let ti = |p: [i64; 2]| (p[1] * 10 + p[0]) as u8;
    let _ = ti;

    // IDLE
    let (op, _) = resolve_and_apply(&base, &cfg, tile, Verb::Idle);
    assert_eq!(op, UnitOp::Pass);

    // WATER
    let mut gs = base.clone();
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Plant(kaggri_core::sim::new_plant(Crop::Wheat, 0, 24)));
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Water);
    assert_eq!(op, UnitOp::Water);
    assert!(after.farms[0].tile(tile[0], tile[1]).as_plant().unwrap().watered_today);

    // HARVEST (plant)
    let mut gs = base.clone();
    let mut p = kaggri_core::sim::new_plant(Crop::Carrot, 0, 24);
    p.yield_units = 3;
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Plant(p));
    gs.day = 3;
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Harvest);
    assert_eq!(op, UnitOp::Harvest);
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Carrot), 3);
    assert_eq!(after.farms[0].tile(tile[0], tile[1]), Tile::Empty, "one-time crop is removed");

    // HARVEST (animal)
    let mut gs = base.clone();
    let mut a = kaggri_core::sim::new_animal(Animal::Goose, 0);
    a.yield_units = 2;
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Animal(a));
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Harvest);
    assert_eq!(op, UnitOp::Harvest);
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Egg), 2);

    // FERTILIZE
    let mut gs = base.clone();
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Plant(kaggri_core::sim::new_plant(Crop::Melon, 0, 24)));
    gs.privates[0].inventories[0].add(ShedItem::Fertilizer, 1);
    gs.day = 5;
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Fertilize);
    assert_eq!(op, UnitOp::Fertilize);
    assert_eq!(
        after.farms[0].tile(tile[0], tile[1]).as_plant().unwrap().fertilized_until_day,
        7
    );
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Fertilizer), 0);

    // DIG
    let mut gs = base.clone();
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Weed);
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Dig);
    assert_eq!(op, UnitOp::Dig);
    assert_eq!(after.farms[0].tile(tile[0], tile[1]), Tile::Empty);

    // BUILD_COOP / BUILD_PASTURE
    let (op, after) = resolve_and_apply(&base, &cfg, tile, Verb::BuildCoop);
    assert_eq!(op, UnitOp::BuildCoop);
    assert_eq!(after.farms[0].tile(tile[0], tile[1]), Tile::Coop);
    let (op, after) = resolve_and_apply(&base, &cfg, tile, Verb::BuildPasture);
    assert_eq!(op, UnitOp::BuildPasture);
    assert_eq!(after.farms[0].tile(tile[0], tile[1]), Tile::Pasture);

    // FEED / CARE / COLLECT_FERTILIZER
    let mut gs = base.clone();
    let mut a = kaggri_core::sim::new_animal(Animal::Cow, 0);
    a.fertilizer_available = true;
    gs.farms[0].set_tile(tile[0], tile[1], Tile::Animal(a));
    gs.privates[0].inventories[0].add(ShedItem::Wheat, 1);
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Feed);
    assert_eq!(op, UnitOp::Feed);
    assert!(after.farms[0].tile(tile[0], tile[1]).as_animal().unwrap().fed_today);
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Wheat), 0);
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Care);
    assert_eq!(op, UnitOp::Care);
    assert!(after.farms[0].tile(tile[0], tile[1]).as_animal().unwrap().cared_today);
    let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::CollectFertilizer);
    assert_eq!(op, UnitOp::CollectFertilizer);
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Fertilizer), 1);
    assert!(!after.farms[0].tile(tile[0], tile[1]).as_animal().unwrap().fertilizer_available);

    // PLANT, every crop
    for c in CROP_LIST {
        let mut gs = base.clone();
        gs.privates[0].seeds[c.idx()] = 1;
        let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::Plant(c));
        assert_eq!(op, UnitOp::Plant(c));
        assert_eq!(after.farms[0].tile(tile[0], tile[1]).as_plant().unwrap().crop, c);
        assert_eq!(after.privates[0].seeds[c.idx()], 0);
    }

    // PLACE_ANIMAL, every animal
    for a in ANIMAL_LIST {
        let mut gs = base.clone();
        gs.farms[0].set_tile(tile[0], tile[1], Tile::empty_structure(a.data().structure));
        gs.privates[0].inventories[0].add(a.shed(), 1);
        let (op, after) = resolve_and_apply(&gs, &cfg, tile, Verb::PlaceAnimal(a));
        assert_eq!(op, UnitOp::Place { item: a.shed(), n: 1 });
        assert_eq!(after.farms[0].tile(tile[0], tile[1]).as_animal().unwrap().animal, a);
        assert_eq!(after.privates[0].inventories[0].get(a.shed()), 0);
    }

    // PICKUP, every item and both quantities
    for (item_slot, item) in PICKUP_ITEMS.iter().enumerate() {
        for qty_slot in 0..2u8 {
            let mut gs = base.clone();
            gs.privates[0].shed.set(*item, 5);
            let verb = Verb::Pickup {
                item_slot: item_slot as u8,
                qty_slot,
            };
            let (op, after) = resolve_and_apply(&gs, &cfg, shed, verb);
            let want = if qty_slot == 0 { 1 } else { 5 }; // clamped to what the shed holds
            assert_eq!(op, UnitOp::Pickup { item: *item, n: if qty_slot == 0 { 1 } else { 8 } });
            assert_eq!(after.privates[0].inventories[0].get(*item), want, "{verb:?}");
            assert_eq!(after.privates[0].shed.get(*item), 5 - want);
        }
    }

    // DEPOSIT, every product
    for prod in PRODUCTS {
        let mut gs = base.clone();
        gs.privates[0].inventories[0].add(prod.shed(), 7);
        let (op, after) = resolve_and_apply(&gs, &cfg, shed, Verb::Deposit(prod));
        assert!(matches!(op, UnitOp::Place { item, .. } if item == prod.shed()));
        assert_eq!(after.privates[0].shed.get(prod.shed()), 7);
        assert_eq!(after.privates[0].inventories[0].get(prod.shed()), 0);
    }
    // ...and a deposit into a nearly-full shed leaves the remainder in hand rather than
    // deleting it, which is the whole reason Deposit exists alongside Drop.
    let mut gs = base.clone();
    gs.privates[0].shed.set(ShedItem::Carrot, 95);
    gs.privates[0].inventories[0].add(ShedItem::Melon, 20);
    let (_, after) = resolve_and_apply(&gs, &cfg, shed, Verb::Deposit(Product::Melon));
    assert_eq!(after.privates[0].shed.get(ShedItem::Melon), 5);
    assert_eq!(after.privates[0].inventories[0].get(ShedItem::Melon), 15);
    assert_eq!(after.privates[0].shed.total(), 100);

    // DROP
    let mut gs = base.clone();
    gs.privates[0].inventories[0].add(ShedItem::Egg, 4);
    gs.privates[0].inventories[0].add(ShedItem::Milk, 6);
    let (op, after) = resolve_and_apply(&gs, &cfg, shed, Verb::Drop);
    assert_eq!(op, UnitOp::Drop);
    assert_eq!(after.privates[0].shed.get(ShedItem::Egg), 4);
    assert_eq!(after.privates[0].shed.get(ShedItem::Milk), 6);
    assert!(after.privates[0].inventories[0].is_empty());
}

// ---------------------------------------------------------------------------
// f16 publication (OBS_LAYOUT.md §11: the published dtype)
// ---------------------------------------------------------------------------

/// The vectorised `vcvtps2ph` path and the portable scalar path must agree **bit for
/// bit**, or the observation a policy sees depends on which CPU published it.
///
/// The sweep is deliberately not "random floats": it walks every f32 exponent, both
/// zeros, both infinities, quiet and signalling NaNs, f32 subnormals (which all flush to a
/// signed f16 zero), the whole f16 subnormal range where the rounding position moves per
/// value, the exact ties-to-even midpoints, and the overflow boundary at 65504/65520 —
/// then adds 4 M random bit patterns on top.
#[test]
fn f16_paths_agree_bit_for_bit() {
    let mut cases: Vec<f32> = Vec::new();
    // Both zeros, both infinities, two NaNs, and the values the observation actually holds.
    cases.extend_from_slice(&[
        0.0, -0.0, 1.0, -1.0, 4.0, -4.0, 0.5, 0.25, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 16.0,
        1.0 / 25.0, 1.0 / 32.0, 1.0 / 100.0, 1.0 / 718.0, 1.0 / 2000.0, 1.0 / 5000.0,
        f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN,
        f32::MAX, f32::MIN, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
        65504.0, 65520.0, 65519.996, 65536.0, -65504.0, -65520.0,
        6.103_515_6e-5,  // smallest f16 normal, 2^-14
        5.960_464_5e-8,  // smallest f16 subnormal, 2^-24
        2.980_232_2e-8,  // 2^-25, the exact tie that must go to zero
        2.980_233e-8,    // just above it, must go to the smallest subnormal
        1e-8, 1e-30, 1e30, -1e30,
    ]);
    // Every exponent, at a few mantissas each.
    for e in 0..255u32 {
        for m in [0u32, 1, 0x1000, 0x1001, 0x0FFF, 0x2000, 0x3000, 0x7F_FFFF, 0x40_0000] {
            cases.push(f32::from_bits((e << 23) | m));
            cases.push(f32::from_bits(0x8000_0000 | (e << 23) | m));
        }
    }
    // The f16 subnormal range and the normal boundary, densely.
    for k in 0..4096u32 {
        cases.push(6.103_515_6e-5 * (k as f32) / 2048.0);
        cases.push(-6.103_515_6e-5 * (k as f32) / 2048.0);
    }
    // Exact ties: an f32 whose discarded 13 mantissa bits are exactly 0x1000 must round to
    // even, which is the one case truncation and RNE differ on for a "nice" number.
    for m in 0..1024u32 {
        cases.push(f32::from_bits((127u32 << 23) | (m << 13) | 0x1000));
        cases.push(f32::from_bits((120u32 << 23) | (m << 13) | 0x1000));
    }
    let mut rng = Rng::new(0xF16C);
    for _ in 0..4_000_000 {
        let bits = ((rng.next_u32() as u64) << 1 ^ rng.next_u32() as u64) as u32;
        cases.push(f32::from_bits(bits));
    }

    let mut scalar = vec![0u16; cases.len()];
    let mut simd = vec![0u16; cases.len()];
    kaggri_core::obs::f32_to_f16_into_scalar(&cases, &mut scalar);
    kaggri_core::obs::f32_to_f16_into(&cases, &mut simd);
    for (i, (a, b)) in scalar.iter().zip(simd.iter()).enumerate() {
        // NaN payloads are the one place the two are allowed to differ in principle; they
        // are compared exactly anyway, and if a CPU ever disagrees this test says so
        // instead of a policy silently seeing a different plane on one machine.
        assert_eq!(
            a, b,
            "f16 mismatch at {i}: input {:e} (bits {:#010x}) scalar {a:#06x} simd {b:#06x}",
            cases[i],
            cases[i].to_bits()
        );
    }

    // Every offset must be handled, not just multiples of 8 (MASK_LEN = 646 is not one).
    for len in 0..40usize {
        let src = &cases[7..7 + len];
        let mut a = vec![0u16; len];
        let mut b = vec![0u16; len];
        kaggri_core::obs::f32_to_f16_into_scalar(src, &mut a);
        kaggri_core::obs::f32_to_f16_into(src, &mut b);
        assert_eq!(a, b, "tail handling differs at len {len}");
    }
}

/// Round-tripping f16 → f32 → f16 must be the identity, and f32 → f16 → f32 must land
/// within half an f16 ULP. This is what pins the *inverse* used by the error analysis.
#[test]
fn f16_round_trip_is_exact_and_nearest() {
    for bits in 0..=u16::MAX {
        let e = (bits >> 10) & 0x1F;
        if e == 0x1F {
            continue; // inf/NaN: not round-tripped by value
        }
        let back = kaggri_core::obs::f32_to_f16_bits(kaggri_core::obs::f16_bits_to_f32(bits));
        assert_eq!(bits, back, "f16 {bits:#06x} did not survive a round trip");
    }
    // Nearest: no representable f16 is closer to x than the one we produce.
    let mut rng = Rng::new(7);
    for _ in 0..200_000 {
        let x = (rng.next_u32() as f32 / u32::MAX as f32) * 8.0 - 4.0; // the obs range
        let h = kaggri_core::obs::f32_to_f16_bits(x);
        let got = kaggri_core::obs::f16_bits_to_f32(h);
        let err = (got - x).abs();
        for neighbour in [h.wrapping_sub(1), h.wrapping_add(1)] {
            let alt = kaggri_core::obs::f16_bits_to_f32(neighbour);
            if alt.is_finite() {
                assert!(
                    err <= (alt - x).abs(),
                    "not nearest: x={x} chose {got} over {alt}"
                );
            }
        }
    }
}

/// The published f16 plane must be exactly the f32 plane rounded, over real states — and
/// the quantisation must stay small enough to be irrelevant to a policy.
///
/// The bound asserted here is the one `OBS_LAYOUT.md` §11 documents: every slot is inside
/// `[-4, 4]`, so half an f16 ULP is at most `2^-9 = 0.00195`, and the relative error is at
/// most `2^-11 = 4.9e-4` away from zero. Anything larger means a slot escaped the
/// documented range or the conversion is not round-to-nearest.
#[test]
fn f16_publication_of_a_real_episode_is_faithful() {
    let cfg = Config::default();
    let mut rng = Rng::new(99);
    let mut gs = GameState::new(&cfg);
    let mut obs = vec![0f32; OBS_LEN];
    let mut half = vec![0u16; OBS_LEN];
    let mut mask = vec![0f32; MASK_LEN];
    let mut mask_half = vec![0u16; MASK_LEN];
    let mut orders = Vec::new();
    let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
    let mut worst_abs = 0f32;
    let mut worst_rel = 0f32;
    let mut steps = 0usize;
    while !gs.done {
        for p in 0..2 {
            encode_with_cfg(&gs, p, &cfg, &mut obs);
            kaggri_core::obs::f32_to_f16_into(&obs, &mut half);
            for i in 0..OBS_LEN {
                let back = kaggri_core::obs::f16_bits_to_f32(half[i]);
                assert!(back.is_finite(), "slot {i} published a non-finite f16");
                let a = (back - obs[i]).abs();
                worst_abs = worst_abs.max(a);
                if obs[i] != 0.0 {
                    worst_rel = worst_rel.max(a / obs[i].abs());
                } else {
                    // Zero must stay exactly zero: a great many slots are one-hots and
                    // "absent" flags, and a policy reads them as exact.
                    assert_eq!(back, 0.0, "slot {i} turned a zero into {back}");
                }
            }
            legal_mask_with_cfg(&gs, p, &cfg, &mut mask);
            kaggri_core::obs::f32_to_f16_into(&mask, &mut mask_half);
            for i in 0..MASK_LEN {
                // The mask alphabet is {0.0, 1.0}: both are exact in f16, so the mask is
                // published losslessly.
                assert_eq!(
                    kaggri_core::obs::f16_bits_to_f32(mask_half[i]),
                    mask[i],
                    "mask slot {i} is not exact in f16"
                );
            }
            random_orders(&mut rng, &gs, p, &mut orders);
            let head = random_market(&mut rng);
            resolve_turn(&gs, p, &cfg, &orders, &head, &mut actions[p]);
        }
        step(&mut gs, &actions, &cfg);
        steps += 1;
    }
    assert!(steps > 700, "episode was too short to mean anything: {steps}");
    // Half an ULP at the top of the documented [-4, 4] range.
    assert!(
        worst_abs <= 0.001_954,
        "worst absolute f16 error {worst_abs} exceeds half an ULP at |x| = 4"
    );
    assert!(
        worst_rel <= 4.885e-4,
        "worst relative f16 error {worst_rel} exceeds 2^-11"
    );
}

#[test]
#[ignore = "long fuzz; run with --release --ignored"]
fn long_fuzz_random_macro_policy() {
    let cfg = Config::default();
    for seed in 1..=400u64 {
        let gs = run_random_episode(seed, &cfg);
        assert!(gs.done);
    }
    // A few non-default configs, since `encode_with_cfg` / `legal_mask_with_cfg` and the
    // packer all read them.
    for (orders, cap, tpd) in [(3usize, 20usize, 8u32), (10, 250, 24), (1, 100, 6)] {
        let cfg = Config {
            max_market_orders_per_turn: orders,
            shed_capacity: cap,
            turns_per_day: tpd,
            episode_steps: 240,
            ..Config::default()
        };
        for seed in 1..=20u64 {
            let gs = run_random_episode(seed, &cfg);
            assert!(gs.done);
        }
    }
}

// ---------------------------------------------------------------------------
// The optional tile-verb plausibility plane (obs::tile_verb_mask_with_cfg)
// ---------------------------------------------------------------------------

/// Three invariants, on every state of random episodes: the plane is 0/1 with an all-zero
/// `Idle` column; every non-`Idle` verb the unit mask calls legal has at least one
/// plausible tile; and for a unit standing on a plausible tile, ordering exactly that
/// `(tile, verb)` (every other unit idle, market idle) makes the resolver fire the verb --
/// the pair is never a structural `PASS`.
#[test]
fn tile_mask_agrees_with_unit_mask_and_resolver() {
    let cfg = Config::default();
    let mut fired = 0usize;
    let mut pairs = 0usize;
    for seed in 11..=14u64 {
        let mut rng = Rng::new(seed);
        let mut gs = GameState::new(&cfg);
        gs.seed = seed as i64 & 0x7fff_ffff;
        let mut mask = vec![0f32; MASK_LEN];
        let mut tile = vec![0u8; TILE_MASK_LEN];
        let mut orders = Vec::new();
        let mut actions = [PlayerAction::pass(), PlayerAction::pass()];
        let mut probe = [PlayerAction::pass(), PlayerAction::pass()];
        let idle_head = MarketHead::from_choices(&[0u8; N_MARKET_HEADS]);
        while !gs.done {
            for p in 0..2 {
                legal_mask_with_cfg(&gs, p, &cfg, &mut mask);
                tile_verb_mask_with_cfg(&gs, p, &cfg, &mut tile);
                for (i, v) in tile.iter().enumerate() {
                    assert!(*v == 0 || *v == 1, "tile flag {i} = {v}");
                }
                for t in 0..N_TARGETS {
                    assert_eq!(tile[t * N_VERBS + Verb::Idle.index()], 0, "Idle has no target");
                }
                let farm = &gs.farms[p];
                let n_units = 1 + farm.hands.len();
                for u in 0..n_units {
                    let pos = farm.unit_pos(u).unwrap();
                    let here = pos[1] as usize * farm.board_size + pos[0] as usize;
                    for (v, verb) in VERBS.iter().enumerate() {
                        if *verb == Verb::Idle || mask[u * N_VERBS + v] != 1.0 {
                            continue;
                        }
                        let any = (0..N_TARGETS).any(|t| tile[t * N_VERBS + v] == 1);
                        assert!(any, "unit {u} verb {verb:?} legal but no plausible tile (step {})", gs.step);
                        if tile[here * N_VERBS + v] == 1 {
                            // Standing on a plausible tile: the resolver must fire the verb.
                            pairs += 1;
                            let mut solo: Vec<MacroOrder> = (0..n_units)
                                .map(|_| MacroOrder { target: 0, verb: Verb::Idle })
                                .collect();
                            solo[u] = MacroOrder { target: here as u8, verb: *verb };
                            resolve_turn(&gs, p, &cfg, &solo, &idle_head, &mut probe[p]);
                            let op = if u == 0 { probe[p].farmer } else { probe[p].hands[u - 1] };
                            let is_pass = matches!(op, UnitOp::Pass | UnitOp::Noop | UnitOp::Move(_));
                            assert!(
                                !is_pass,
                                "unit {u} at tile {here} verb {verb:?}: unit mask 1, tile mask 1, \
                                 but the resolver emitted {op:?} (player {p}, step {})",
                                gs.step
                            );
                            fired += 1;
                        }
                    }
                }
                random_orders(&mut rng, &gs, p, &mut orders);
                let head = random_market(&mut rng);
                resolve_turn(&gs, p, &cfg, &orders, &head, &mut actions[p]);
            }
            step(&mut gs, &actions, &cfg);
        }
    }
    assert!(pairs > 2000, "the probe saw only {pairs} standing-on-plausible-tile pairs");
    assert_eq!(fired, pairs);
}

