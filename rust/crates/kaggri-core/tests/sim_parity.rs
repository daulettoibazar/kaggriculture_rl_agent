//! Full-episode lockstep parity against the Python interpreter.
//!
//! `tests/gen_sim_fixtures.py` drives `kaggriculture.interpreter` directly over a
//! recorded action trace and writes, for every step, the actions it fed in plus a
//! canonical digest of the **entire** state: every tile field, both sheds and seed
//! counts, every unit inventory *including key order*, market inventory and prices,
//! town shops, money as raw f64 bits, positions, `hires_today`, unlocked quadrants,
//! step/day/hour, done and the final rewards.
//!
//! This test replays those actions through [`kaggri_core::sim::step`] and asserts the
//! digest matches byte for byte after every single step. On failure it reports the
//! first divergent step and a field-level diff.
//!
//! Regenerate with:
//! ```sh
//! python3 crates/kaggri-core/tests/gen_sim_fixtures.py
//! ```

use kaggri_core::actions::{parse_market_order, parse_unit_op, MarketOrder, PlayerAction, Tok, UnitOp};
use kaggri_core::consts::*;
use kaggri_core::sim::step as sim_step;
use kaggri_core::state::*;

// ---------------------------------------------------------------------------
// fixture parsing
// ---------------------------------------------------------------------------

fn parse_cfg(line: &str) -> Config {
    let mut cfg = Config::default();
    for kv in line.split_whitespace().skip(1) {
        let (k, v) = kv.split_once('=').expect("CFG key=value");
        match k {
            "episode_steps" => cfg.episode_steps = v.parse().unwrap(),
            "board_size" => cfg.board_size = v.parse().unwrap(),
            "starting_money" => cfg.starting_money = v.parse::<f64>().unwrap(),
            "max_orders" => cfg.max_market_orders_per_turn = v.parse().unwrap(),
            "turns_per_day" => cfg.turns_per_day = v.parse().unwrap(),
            "shed_capacity" => cfg.shed_capacity = v.parse().unwrap(),
            "weed_spawn_chance" => cfg.weed_spawn_chance = v.parse().unwrap(),
            "town_shop_unlock_interval" => cfg.town_shop_unlock_interval = v.parse().unwrap(),
            "town_center_sell_interval" => cfg.town_center_sell_interval = v.parse().unwrap(),
            "town_shop_sell_interval" => cfg.town_shop_sell_interval = v.parse().unwrap(),
            "farm_hand_cost_mult" => cfg.farm_hand_cost_mult = v.parse().unwrap(),
            "seed" => cfg.seed = v.parse().unwrap(),
            other => panic!("unknown config key {other}"),
        }
    }
    cfg
}

fn parse_tok(s: &str) -> Tok {
    match s {
        "N" => Tok::Null,
        "O" => Tok::Other,
        _ => match s.split_at(2) {
            ("B:", v) => Tok::Bool(v == "1"),
            ("I:", v) => Tok::Int(v.parse().unwrap()),
            ("F:", v) => Tok::Float(v.parse().unwrap()),
            ("S:", v) => Tok::Str(v.to_string()),
            _ => panic!("bad token {s:?}"),
        },
    }
}

/// One encoded action: `X` = not a list, `-` = empty list, else space-separated tokens.
enum Enc {
    NotAList,
    Toks(Vec<Tok>),
}

fn parse_enc(s: &str) -> Enc {
    match s {
        "X" => Enc::NotAList,
        "-" => Enc::Toks(vec![]),
        _ => Enc::Toks(s.split(' ').map(parse_tok).collect()),
    }
}

fn parse_enc_list(field: &str) -> Option<Vec<Enc>> {
    if field == "X" {
        return None; // the whole value was not a list
    }
    if field.is_empty() {
        return Some(vec![]);
    }
    Some(field.split(',').map(parse_enc).collect())
}

fn parse_action_line(line: &str) -> PlayerAction {
    // "A0 farmer|hands|market"
    let body = line.splitn(2, ' ').nth(1).unwrap_or("");
    let mut fields = body.split('|');
    let farmer_f = fields.next().unwrap_or("");
    let hands_f = fields.next().unwrap_or("");
    let market_f = fields.next().unwrap_or("");

    let farmer = match parse_enc(farmer_f) {
        // A non-list farmer action returns immediately in `_apply_unit_action`.
        Enc::NotAList => UnitOp::Noop,
        Enc::Toks(t) => parse_unit_op(&t).expect("fixture must not contain raising actions"),
    };
    let hands = parse_enc_list(hands_f)
        .unwrap_or_default()
        .into_iter()
        .map(|e| match e {
            Enc::NotAList => UnitOp::Noop,
            Enc::Toks(t) => parse_unit_op(&t).expect("fixture must not contain raising actions"),
        })
        .collect();
    let market: Vec<Option<MarketOrder>> = parse_enc_list(market_f)
        .unwrap_or_default()
        .into_iter()
        .map(|e| match e {
            // `_parse_order` returns None for a non-list, but the slot still counts.
            Enc::NotAList => None,
            Enc::Toks(t) => parse_market_order(&t).unwrap(),
        })
        .collect();
    PlayerAction {
        farmer,
        hands,
        market,
    }
}

// ---------------------------------------------------------------------------
// the canonical digest — must match gen_sim_fixtures.py byte for byte
// ---------------------------------------------------------------------------

fn f64bits(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

fn tile_code(t: Tile) -> String {
    match t {
        Tile::Empty => ".".to_string(),
        Tile::Locked => "#".to_string(),
        Tile::Weed => "W".to_string(),
        Tile::Coop => "C".to_string(),
        Tile::Pasture => "S".to_string(),
        Tile::Plant(p) => format!(
            "P{}:{}:{}:{}:{}:{}:{}",
            p.crop.idx(),
            p.planted_day,
            p.watered_today as i32,
            p.consecutive_unwatered,
            p.yield_units,
            p.max_lifespan_step,
            p.fertilized_until_day
        ),
        Tile::Animal(a) => format!(
            "A{}:{}:{}:{}:{}:{}:{}:{}",
            a.animal.idx(),
            a.placed_day,
            a.yield_units,
            a.consecutive_unfed,
            a.fed_today as i32,
            a.cared_today as i32,
            a.fertilizer_available as i32,
            a.pending_care_bonus
        ),
    }
}

fn join<T: std::fmt::Display>(sep: &str, it: impl Iterator<Item = T>) -> String {
    it.map(|v| v.to_string()).collect::<Vec<_>>().join(sep)
}

fn digest(gs: &GameState) -> String {
    let mut out: Vec<String> = vec![
        format!("S{}", gs.step),
        format!("D{}", gs.day),
        format!("H{}", gs.hour),
        format!("Q{}", gs.done as i32),
        format!("R{},{}", f64bits(gs.rewards[0]), f64bits(gs.rewards[1])),
        format!("MI{}", join(",", PRODUCTS.iter().map(|p| gs.market.inv(*p)))),
        format!("MP{}", join(",", PRODUCTS.iter().map(|p| gs.market.price(*p)))),
        format!(
            "TS{}",
            join(",", gs.town.unlocked_shops.iter().map(|s| s.idx()))
        ),
    ];
    for i in 0..2 {
        let f = &gs.farms[i];
        out.push(format!(
            "F{} M{} P{},{} H{} Q{} N{} T{}",
            i,
            f64bits(f.money),
            f.farmer[0],
            f.farmer[1],
            join(";", f.hands.iter().map(|p| format!("{},{}", p[0], p[1]))),
            join(",", f.unlocked_quadrants.iter().map(|q| *q as usize)),
            f.hires_today,
            join(";", f.tiles.iter().map(|t| tile_code(*t))),
        ));
        let pv = &gs.privates[i];
        out.push(format!(
            "V{} SH{} SD{} IV{}",
            i,
            join(",", pv.shed.counts.iter()),
            join(",", pv.seeds.iter()),
            join(
                "/",
                pv.inventories
                    .iter()
                    .map(|inv| join(",", inv.iter().map(|(k, n)| format!("{}={}", k.idx(), n))))
            ),
        ));
    }
    out.join("|")
}

// ---------------------------------------------------------------------------
// failure reporting
// ---------------------------------------------------------------------------

fn explain(got: &str, want: &str) -> String {
    let g: Vec<&str> = got.split('|').collect();
    let w: Vec<&str> = want.split('|').collect();
    let mut msg = String::new();
    if g.len() != w.len() {
        msg.push_str(&format!("field count {} vs {}\n", g.len(), w.len()));
    }
    for (gi, wi) in g.iter().zip(w.iter()) {
        if gi == wi {
            continue;
        }
        // Tile lists are long; point at the first differing tile.
        if let (Some(gt), Some(wt)) = (gi.find(" T"), wi.find(" T")) {
            let (ghead, gtiles) = gi.split_at(gt);
            let (whead, wtiles) = wi.split_at(wt);
            if ghead != whead {
                msg.push_str(&format!("  rust: {ghead}\n  py  : {whead}\n"));
            }
            for (idx, (a, b)) in gtiles[2..]
                .split(';')
                .zip(wtiles[2..].split(';'))
                .enumerate()
            {
                if a != b {
                    msg.push_str(&format!(
                        "  tile[{}] = ({},{}) rust={a} py={b}\n",
                        idx,
                        idx % 10,
                        idx / 10
                    ));
                }
            }
            continue;
        }
        msg.push_str(&format!("  rust: {gi}\n  py  : {wi}\n"));
    }
    msg
}

// ---------------------------------------------------------------------------
// the replay
// ---------------------------------------------------------------------------

fn replay(name: &str, fixture: &str) {
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    let cfg_line = lines.next().expect("CFG line");
    let cfg = parse_cfg(cfg_line);
    let mut gs = GameState::new(&cfg);

    let mut pending: [Option<PlayerAction>; 2] = [None, None];
    let mut n_steps = 0usize;
    let mut cur_step: i64 = -1;

    for line in lines {
        if let Some(rest) = line.strip_prefix("STEP ") {
            cur_step = rest.trim().parse().unwrap();
            pending = [None, None];
        } else if line.starts_with("A0 ") {
            pending[0] = Some(parse_action_line(line));
        } else if line.starts_with("A1 ") {
            pending[1] = Some(parse_action_line(line));
        } else if let Some(want) = line.strip_prefix("D ") {
            let actions = [
                pending[0].take().expect("A0 before D"),
                pending[1].take().expect("A1 before D"),
            ];
            assert_eq!(gs.step, cur_step, "[{name}] step counter desync");
            sim_step(&mut gs, &actions, &cfg);
            let got = digest(&gs);
            if got != want {
                panic!(
                    "[{name}] state diverged after step {cur_step}\n{}",
                    explain(&got, want)
                );
            }
            n_steps += 1;
        } else if !line.trim().is_empty() {
            panic!("unexpected fixture line: {line:?}");
        }
    }
    assert!(n_steps > 50, "[{name}] fixture looks truncated ({n_steps})");
    println!("[{name}] {n_steps} steps replayed with zero divergence");
}

macro_rules! parity_test {
    ($fn_name:ident, $name:literal) => {
        #[test]
        fn $fn_name() {
            replay(
                $name,
                include_str!(concat!("data/sim_trace_", $name, ".txt")),
            );
        }
    };
}

parity_test!(trace_default, "default");
parity_test!(trace_bigseed, "bigseed");
parity_test!(trace_stress, "stress");
parity_test!(trace_rich, "rich");
parity_test!(trace_drain, "drain");
parity_test!(trace_husbandry, "husbandry");
parity_test!(trace_scripted, "scripted");
parity_test!(trace_floor, "floor");
parity_test!(trace_tiny, "tiny");
// Recorded through the real framework path (`env.run(["starter", "starter"])`) rather
// than by driving the interpreter directly: an independent check on the step/day/hour,
// DONE-timing and reward bookkeeping.
parity_test!(trace_envrun, "envrun");
