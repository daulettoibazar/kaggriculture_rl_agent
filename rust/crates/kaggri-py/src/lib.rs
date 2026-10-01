//! `kaggri` — the PyO3 surface of the Kaggriculture Rust port.
//!
//! Two classes, with deliberately different priorities:
//!
//! * [`PyGame`] — a single-episode handle for **parity and debugging**. It takes the
//!   action dicts the Kaggle environment takes, verbatim and including malformed ones,
//!   and hands back a plain-Python state dict that `parity/canon.py` can compare
//!   field-by-field against the Python environment's observation. Correctness and
//!   observability win over speed here; `state_dict()` allocates freely.
//!
//! * [`PyVecEnv`] — the **batched RL** interface: `n_envs` independent episodes stepped
//!   in parallel with rayon, numpy in and numpy out, and no per-step allocation of
//!   Python objects on the hot path (the returned arrays are env-owned buffers that are
//!   overwritten in place every step).
//!
//! ## Contract notes that matter for bit-exactness
//!
//! * **Unit inventories come out as ordered sequences.** `privates[p]["inventories"][i]`
//!   is emitted from `Inv`'s insertion-ordered `Vec`, and *additionally* as
//!   `inventories_ordered`, a list of `[item, count]` pairs, so a caller that
//!   accidentally re-sorts the dict can still check the real order. Trap 2 makes this
//!   order observable game state: `DROP` and `_drop_inventories_to_shed` fill the shed
//!   in iteration order and discard the remainder.
//! * **`money` is a Python `float`**, never an int (trap 4).
//! * **`seed` is the *resolved* episode seed** — the integer Python stores in
//!   `env.info["seed"]` after `resolve_episode_seed` scrubs `configuration["seed"]`.
//!   This crate does not port `resolve_episode_seed`: an absent or `None` seed becomes
//!   0 rather than a random draw, so a `PyGame` is always reproducible.
//! * Actions that CPython would raise on (a non-int-coercible `PICKUP`/`PLACE`
//!   quantity) raise `ValueError` here too, rather than inventing a no-op. There is no
//!   bit-exact behaviour to reproduce for those inputs.
//!
//! ## The RL layer — `OBS_VERSION = 1`, `ACTION_VERSION = 1`
//!
//! [`PyVecEnv`] exposes the shipped RL interface, normatively specified in
//! `rust/OBS_LAYOUT.md` and implemented in `kaggri_core::{obs, macros}`. This file only
//! plumbs it: it owns no observation slot, no verb and no market rule of its own.
//!
//! | direction | what crosses the boundary | shape | dtype |
//! |---|---|---|---|
//! | out | observation | `(n_envs, 2, 6720)` = `(n_envs, 2, 140 x 48)` | `float16` \* |
//! | out | legality mask | `(n_envs, 2, 646)` = `(n_envs, 2, 17 x 38)` | `float16` \* |
//! | out | rewards | `(n_envs, 2)` | `float32` |
//! | out | dones | `(n_envs,)` | `bool` |
//! | out | `info["final_money"]` | `(n_envs, 2)` | `float32` |
//! | out | `info["step"]` | `(n_envs,)` | `int64` |
//! | in | actions | `(n_envs, 2, 2 * units_per_player + 21)` | `int64` / `int32` |
//!
//! `reset()` hands back `(obs, mask)`; `step(actions)` hands back
//! `(obs, mask, rewards, dones, info)`.
//!
//! \* **The two input planes are published as `float16` by default** — the observation is
//! still *computed* in f32, and `PyVecEnv(..., obs_dtype="float32")` publishes it wide for
//! an A/B. This is a deliberate run-level decision, because it is the policy's input
//! dtype: a batched vec-step is ~85 % memcpy into the observation plane, so halving the
//! element width is the single largest lever on collection throughput, and it halves the
//! host→device transfer and the rollout buffer too. Numerically it is free where it
//! matters — every slot is normalised into `[-4, 4]`, one-hots, booleans and the mask's
//! `{0.0, 1.0}` alphabet are exact, and the conversion is round-to-nearest-ties-to-even so
//! the plane is bit-identical to `np.float16` of the f32 plane. `rewards` and
//! `info["final_money"]` stay f32: they carry money, which f16 could not hold.
//! See [`PyVecEnv::obs_dtype`].
//!
//! Axis 1 of every output is the player, and **every plane is player-relative**: slice
//! `[i, p]` is what player `p` of env `i` sees, with itself as "me". Self-play needs no
//! side-swapping.
//!
//! The mask is published as a **second input plane, not a constraint** — `resolve_turn`
//! does not consult it and nothing here applies it (`OBS_LAYOUT.md` §10). It is there to be
//! A/B'd as an extra input or an auxiliary loss target: every illegal action in this
//! environment is a silent no-op, so unmasked exploration costs a wasted turn and never a
//! corrupted game.
//!
//! **Every returned array is an env-owned buffer that `step` overwrites in place.** That
//! is what keeps the hot path free of Python allocation; the price is that a caller who
//! needs to keep a step's values must copy them (`np.array(obs)`, or write them straight
//! into a rollout buffer). Nothing may touch a returned array while another thread is
//! inside `step()` — `step` releases the GIL and writes these buffers from rayon workers.
//!
//! The action decode is deliberately total: an out-of-range target or verb becomes
//! `Idle` (a `PASS`), a negative market choice becomes category 0 (the do-nothing
//! category of every market head) and an over-range one is clamped by
//! `MarketHead::from_choices`. A malformed sample from an untrained policy can therefore
//! neither panic a rayon worker nor spend money the policy did not ask to spend.

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayDescr, PyArrayDescrMethods, PyArrayDyn,
    PyArrayMethods, PyUntypedArrayMethods,
};

use rayon::prelude::*;

use kaggri_core::actions::{
    parse_market_order, parse_unit_op, ActionParseError, MarketOrder, PlayerAction, Tok, UnitOp,
};
use kaggri_core::consts::{
    quadrant_of, Animal, Crop, ItemParams, ItemParamsPatch, MarketParams, Product, Quadrant, Shape,
    ShedItem, Shop, Structure, ANIMAL_LIST, CROP_LIST, MARKET_PARAMS, N_PRODUCTS, PRODUCTS,
    SHED_ITEMS,
};
use kaggri_core::macros::{
    self, MacroOrder, MarketHead, Verb, DEPOSIT_QTY, MARKET_HEAD_SIZES, MAX_HANDS,
    MAX_HIRE_PER_TURN, MAX_UNITS, N_MARKET_HEADS, N_TARGETS, N_VERBS, PICKUP_ITEMS, PICKUP_QTYS,
    QTY_BUCKETS, SELL_FRACS, VERBS,
};
use kaggri_core::raw::{
    raw_market_order, raw_unit_op, MARKET_ACTION_VERSION, RAW_ACTION_VERSION,
    RAW_MARKET_ACTIONS, RAW_MARKET_SLOTS, RAW_UNIT_ACTIONS,
};
use kaggri_core::obs::{
    encode_with_cfg, f32_to_f16_into, legal_mask_with_cfg, obs_dims, tile_verb_mask_with_cfg,
    MASK_LEN, N_FEATURES, N_TOKENS, OBS_LEN, OBS_VERSION, TILE_MASK_LEN,
};
use kaggri_core::sim;
use kaggri_core::state::{
    AnimalTile, Config, Farm, GameState, Inv, Market, PlantTile, Private, Tile, Town,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

// ===========================================================================
// small Python-semantics helpers
// ===========================================================================

/// `int(x)` for the numeric configuration values, mirroring `int(get(cfg, k, d))`.
///
/// Accepts Python ints, floats (truncated toward zero, as `int()` does) and bools.
fn py_int_arg(v: &Bound<'_, PyAny>, what: &str) -> PyResult<i64> {
    if v.is_instance_of::<PyBool>() {
        return Ok(v.extract::<bool>()? as i64);
    }
    if v.is_instance_of::<PyInt>() {
        return v.extract::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "config key {what:?}: integer does not fit in i64 ({})",
                v.str().map(|s| s.to_string()).unwrap_or_default()
            ))
        });
    }
    if v.is_instance_of::<PyFloat>() {
        let f = v.extract::<f64>()?;
        if !f.is_finite() {
            return Err(PyValueError::new_err(format!(
                "config key {what:?}: cannot convert non-finite float to int"
            )));
        }
        return Ok(f.trunc() as i64);
    }
    Err(PyTypeError::new_err(format!(
        "config key {what:?}: expected a number, got {}",
        v.get_type().name()?
    )))
}

/// `float(x)`.
fn py_float_arg(v: &Bound<'_, PyAny>, what: &str) -> PyResult<f64> {
    if v.is_instance_of::<PyBool>() {
        return Ok(v.extract::<bool>()? as i64 as f64);
    }
    if v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>() {
        return v.extract::<f64>();
    }
    Err(PyTypeError::new_err(format!(
        "config key {what:?}: expected a number, got {}",
        v.get_type().name()?
    )))
}

/// One element of a raw action list, as the Python environment would see it.
///
/// The `isinstance` ladder matters: `bool` is a subclass of `int` in Python, so it has
/// to be tested first, and a non-string is never `==` to an op or item name.
fn tok_from_py(v: &Bound<'_, PyAny>) -> Tok {
    if v.is_none() {
        return Tok::Null;
    }
    if v.is_instance_of::<PyBool>() {
        return match v.extract::<bool>() {
            Ok(b) => Tok::Bool(b),
            Err(_) => Tok::Other,
        };
    }
    if let Ok(s) = v.cast::<PyString>() {
        return match s.to_str() {
            Ok(t) => Tok::Str(t.to_string()),
            // A surrogate-containing str cannot equal any op name we care about.
            Err(_) => Tok::Other,
        };
    }
    if v.is_instance_of::<PyInt>() {
        return match v.extract::<i64>() {
            Ok(i) => Tok::Int(i),
            // A Python int outside i64. Saturating keeps the *behaviour* identical for
            // every reachable use (quantities are clamped against inventory or hit the
            // market escape hatch either way) while staying inside i64 arithmetic.
            Err(_) => {
                let negative = v
                    .lt(0i64)
                    .unwrap_or(false);
                Tok::Int(if negative { -SATURATED_INT } else { SATURATED_INT })
            }
        };
    }
    if v.is_instance_of::<PyFloat>() {
        return match v.extract::<f64>() {
            Ok(f) => Tok::Float(f),
            Err(_) => Tok::Other,
        };
    }
    // Lists, dicts, objects: `int()` raises `TypeError` on them and they never compare
    // equal to a name. Note that Python would additionally raise `TypeError:
    // unhashable type` for a list/dict used as an *op* or *item*; those inputs are out
    // of contract (documented in parity/README.md) and are treated as no-ops here.
    Tok::Other
}

/// A large but arithmetic-safe stand-in for a Python int that overflows i64.
const SATURATED_INT: i64 = 1 << 60;

/// `list(x)` if `isinstance(x, list)`, else `None`.
fn toks_from_py(v: &Bound<'_, PyAny>) -> Option<Vec<Tok>> {
    let list = v.cast::<PyList>().ok()?;
    Some(list.iter().map(|e| tok_from_py(&e)).collect())
}

fn parse_err(kind: ActionParseError, where_: &str) -> PyErr {
    match kind {
        ActionParseError::IntConversion => PyValueError::new_err(format!(
            "{where_}: the quantity argument is not int-coercible. CPython's \
             _apply_unit_action calls int(action[2]) unguarded, so this input raises \
             out of the Kaggle interpreter and aborts the episode — there is no \
             bit-exact behaviour to reproduce. Do not emit it from a policy."
        )),
    }
}

/// One player's action dict, exactly as `interpreter()` destructures it.
///
/// * a non-dict action → `{"farmer": ["PASS"], "hands": [], "market": []}`
/// * `hands` that is not a list → `[]`
/// * `market` that is not a list → `[]`
/// * the `market` list is kept at full length (including unparseable slots as `None`),
///   because `_process_market` truncates it itself and the slot count drives how often
///   `_refresh_prices` runs.
fn player_action_from_py(v: &Bound<'_, PyAny>, player: usize) -> PyResult<PlayerAction> {
    let Ok(d) = v.cast::<PyDict>() else {
        // `action = s.action if isinstance(s.action, dict) else {}` then
        // `action.get("farmer", ["PASS"])`.
        return Ok(PlayerAction::pass());
    };

    let farmer = match d.get_item("farmer")? {
        Some(f) => match toks_from_py(&f) {
            Some(t) => parse_unit_op(&t)
                .map_err(|e| parse_err(e, &format!("player {player} farmer action")))?,
            None => UnitOp::Noop, // `not isinstance(action, list)` → bare return
        },
        None => UnitOp::Pass, // the key is absent → ["PASS"]
    };

    let mut hands = Vec::new();
    if let Some(h) = d.get_item("hands")? {
        if let Ok(list) = h.cast::<PyList>() {
            hands.reserve(list.len());
            for (i, a) in list.iter().enumerate() {
                let op = match toks_from_py(&a) {
                    Some(t) => parse_unit_op(&t)
                        .map_err(|e| parse_err(e, &format!("player {player} hands[{i}]")))?,
                    None => UnitOp::Noop,
                };
                hands.push(op);
            }
        }
        // A non-list `hands` becomes `[]` (`if not isinstance(hands_actions, list)`).
    }

    let mut market: Vec<Option<MarketOrder>> = Vec::new();
    if let Some(m) = d.get_item("market")? {
        if let Ok(list) = m.cast::<PyList>() {
            market.reserve(list.len());
            for o in list.iter() {
                // `_parse_order` returns None for a non-list order.
                let parsed = match toks_from_py(&o) {
                    Some(t) => parse_market_order(&t).unwrap_or(None),
                    None => None,
                };
                market.push(parsed);
            }
        }
    }

    Ok(PlayerAction {
        farmer,
        hands,
        market,
    })
}

// ===========================================================================
// configuration
// ===========================================================================

/// Configuration keys the Kaggle framework injects that this port ignores on purpose.
const IGNORED_CONFIG_KEYS: [&str; 5] = [
    "actTimeout",
    "agentTimeout",
    "runTimeout",
    "agentExec",
    "__raw_path__",
];

fn structure_name(s: Structure) -> &'static str {
    match s {
        Structure::Coop => "COOP",
        Structure::Pasture => "PASTURE",
    }
}

/// Parse one sparse `marketParams` patch, mirroring `_resolve_market_params`.
///
/// Python does `resolved[item].update(patch)` and lets `market_price` ignore anything
/// it does not know, so unknown patch keys are silently dropped here too. Unknown
/// *product* keys and non-dict patch values are dropped as well (`if item in resolved
/// and isinstance(patch, dict)`).
fn market_params_from_py(v: &Bound<'_, PyAny>) -> PyResult<MarketParams> {
    let Ok(d) = v.cast::<PyDict>() else {
        // `_resolve_market_params(overrides) if overrides else None` — a non-dict is
        // truthy garbage in Python and would raise on `.items()`; refuse loudly.
        return Err(PyTypeError::new_err(
            "config key \"marketParams\": expected a dict keyed by product name",
        ));
    };
    let mut overrides: Vec<(Product, ItemParamsPatch)> = Vec::new();
    for (k, patch) in d.iter() {
        let Ok(name) = k.cast::<PyString>() else {
            continue; // not a product name → `item in resolved` is False
        };
        let Some(item) = Product::from_name(name.to_str()?) else {
            continue;
        };
        let Ok(pd) = patch.cast::<PyDict>() else {
            continue; // `isinstance(patch, dict)` is False
        };
        let mut p = ItemParamsPatch::default();
        if let Some(v) = pd.get_item("base")? {
            p.base = Some(py_float_arg(&v, "marketParams.base")?);
        }
        if let Some(v) = pd.get_item("I0")? {
            p.i0 = Some(py_int_arg(&v, "marketParams.I0")?);
        }
        if let Some(v) = pd.get_item("T")? {
            p.t = Some(py_float_arg(&v, "marketParams.T")?);
        }
        if let Some(v) = pd.get_item("below_func")? {
            p.below_func = Some(Shape::from_name(v.cast::<PyString>()?.to_str()?));
        }
        if let Some(v) = pd.get_item("below_target")? {
            p.below_target = Some(py_float_arg(&v, "marketParams.below_target")?);
        }
        if let Some(v) = pd.get_item("above_func")? {
            p.above_func = Some(Shape::from_name(v.cast::<PyString>()?.to_str()?));
        }
        if let Some(v) = pd.get_item("above_target")? {
            p.above_target = Some(py_float_arg(&v, "marketParams.above_target")?);
        }
        overrides.push((item, p));
    }
    Ok(MarketParams::resolve(&overrides))
}

/// `(Config, MarketParams)` from the camelCase dict the Kaggle env takes.
///
/// Keys absent from the dict take the `kaggriculture.json` default. `seed` must be the
/// **resolved** episode seed (`env.info["seed"]`); absent or `None` means 0.
fn config_from_py(d: Option<&Bound<'_, PyDict>>) -> PyResult<(Config, MarketParams, i8)> {
    let mut cfg = Config::default();
    let mut params = MARKET_PARAMS;
    let mut phantom_seat: i8 = NO_PHANTOM_SEAT;
    let Some(d) = d else {
        return Ok((cfg, params, phantom_seat));
    };
    for (k, v) in d.iter() {
        let key_obj = k.cast::<PyString>().map_err(|_| {
            PyTypeError::new_err("configuration keys must be strings".to_string())
        })?;
        let key = key_obj.to_str()?;
        // `type: ["integer", "null"]` — a null seed means "unset".
        if v.is_none() {
            continue;
        }
        match key {
            "episodeSteps" => {
                let n = py_int_arg(&v, key)?;
                if n < 0 {
                    return Err(PyValueError::new_err("episodeSteps must be >= 0"));
                }
                cfg.episode_steps = n as u32;
            }
            "boardSize" => {
                let n = py_int_arg(&v, key)?;
                if n < 4 {
                    return Err(PyValueError::new_err(
                        "boardSize must be >= 4 (kaggriculture.json minimum)",
                    ));
                }
                cfg.board_size = n as usize;
            }
            // `int(get(configuration, "startingMoney", 3000))`, then `float(...)`.
            "startingMoney" => cfg.starting_money = py_int_arg(&v, key)? as f64,
            "maxMarketOrdersPerTurn" => {
                cfg.max_market_orders_per_turn = py_int_arg(&v, key)?.max(0) as usize
            }
            "turnsPerDay" => cfg.turns_per_day = py_int_arg(&v, key)?.max(0) as u32,
            "shedCapacity" => cfg.shed_capacity = py_int_arg(&v, key)?.max(0) as usize,
            "weedSpawnChance" => cfg.weed_spawn_chance = py_float_arg(&v, key)?,
            "townShopUnlockInterval" => {
                cfg.town_shop_unlock_interval = py_int_arg(&v, key)?.max(0) as u32
            }
            "townShopSellInterval" => {
                cfg.town_shop_sell_interval = py_int_arg(&v, key)?.max(0) as u32
            }
            "townCenterSellInterval" => {
                cfg.town_center_sell_interval = py_int_arg(&v, key)?.max(0) as u32
            }
            "farmHandCostMult" => cfg.farm_hand_cost_mult = py_int_arg(&v, key)?,
            "seed" => cfg.seed = py_int_arg(&v, key)?,
            "marketParams" => params = market_params_from_py(&v)?,
            // NOT a Kaggle key, and deliberately outside `Config` (whose field list
            // `PORT_SPEC` pins). See `PyVecEnv::set_phantom_seats`.
            "phantomSeat" => {
                let n = py_int_arg(&v, key)?;
                if !(-1..=1).contains(&n) {
                    return Err(PyValueError::new_err(
                        "phantomSeat must be -1 (none), 0 or 1",
                    ));
                }
                phantom_seat = n as i8;
            }
            other => {
                if !IGNORED_CONFIG_KEYS.contains(&other) {
                    return Err(PyValueError::new_err(format!(
                        "unknown configuration key {other:?}. Known keys: episodeSteps, \
                         boardSize, startingMoney, maxMarketOrdersPerTurn, turnsPerDay, \
                         shedCapacity, weedSpawnChance, townShopUnlockInterval, \
                         townShopSellInterval, townCenterSellInterval, farmHandCostMult, \
                         seed, marketParams, phantomSeat (plus the framework's \
                         {IGNORED_CONFIG_KEYS:?})"
                    )));
                }
            }
        }
    }
    Ok((cfg, params, phantom_seat))
}

/// "No seat is a phantom" — the value `phantomSeat` defaults to, and the only one that
/// reproduces the environment's Kaggle-exact behaviour.
const NO_PHANTOM_SEAT: i8 = -1;

/// `-1 / 0 / 1` -> the per-seat flag pair `kaggri_core::sim::step_with_phantom` takes.
#[inline]
fn phantom_mask(seat: i8) -> [bool; 2] {
    [seat == 0, seat == 1]
}

/// The reverse direction, so a caller can see what was actually parsed.
fn config_to_py<'py>(
    py: Python<'py>,
    cfg: &Config,
    params: &MarketParams,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("episodeSteps", cfg.episode_steps)?;
    d.set_item("boardSize", cfg.board_size)?;
    d.set_item("startingMoney", cfg.starting_money as i64)?;
    d.set_item("maxMarketOrdersPerTurn", cfg.max_market_orders_per_turn)?;
    d.set_item("turnsPerDay", cfg.turns_per_day)?;
    d.set_item("shedCapacity", cfg.shed_capacity)?;
    d.set_item("weedSpawnChance", cfg.weed_spawn_chance)?;
    d.set_item("townShopUnlockInterval", cfg.town_shop_unlock_interval)?;
    d.set_item("townShopSellInterval", cfg.town_shop_sell_interval)?;
    d.set_item("townCenterSellInterval", cfg.town_center_sell_interval)?;
    d.set_item("farmHandCostMult", cfg.farm_hand_cost_mult)?;
    d.set_item("seed", cfg.seed)?;
    let mp = PyDict::new(py);
    for item in PRODUCTS {
        let p: &ItemParams = params.get(item);
        let row = PyDict::new(py);
        row.set_item("base", p.base)?;
        row.set_item("I0", p.i0)?;
        row.set_item("T", p.t)?;
        row.set_item("below_func", p.below_func.as_str())?;
        row.set_item("below_target", p.below_target)?;
        row.set_item("above_func", p.above_func.as_str())?;
        row.set_item("above_target", p.above_target)?;
        mp.set_item(item.as_str(), row)?;
    }
    d.set_item("marketParams", mp)?;
    Ok(d)
}

// ===========================================================================
// state -> plain Python
// ===========================================================================

fn tile_to_py<'py>(py: Python<'py>, t: Tile) -> PyResult<Bound<'py, PyAny>> {
    Ok(match t {
        Tile::Empty => py.None().into_bound(py),
        Tile::Locked => PyString::new(py, "LOCKED").into_any(),
        Tile::Weed => {
            let d = PyDict::new(py);
            d.set_item("kind", "WEED")?;
            d.into_any()
        }
        Tile::Coop => {
            let d = PyDict::new(py);
            d.set_item("kind", "COOP")?;
            d.into_any()
        }
        Tile::Pasture => {
            let d = PyDict::new(py);
            d.set_item("kind", "PASTURE")?;
            d.into_any()
        }
        Tile::Plant(p) => {
            let d = PyDict::new(py);
            d.set_item("kind", "PLANT")?;
            d.set_item("crop", p.crop.as_str())?;
            d.set_item("planted_day", p.planted_day)?;
            d.set_item("watered_today", p.watered_today)?;
            d.set_item("consecutive_unwatered", p.consecutive_unwatered)?;
            d.set_item("yield_units", p.yield_units)?;
            d.set_item("max_lifespan_step", p.max_lifespan_step)?;
            d.set_item("fertilized_until_day", p.fertilized_until_day)?;
            d.into_any()
        }
        Tile::Animal(a) => {
            let d = PyDict::new(py);
            d.set_item("kind", structure_name(a.animal.data().structure))?;
            d.set_item("animal", a.animal.as_str())?;
            d.set_item("placed_day", a.placed_day)?;
            d.set_item("yield_units", a.yield_units)?;
            d.set_item("consecutive_unfed", a.consecutive_unfed)?;
            d.set_item("fed_today", a.fed_today)?;
            d.set_item("cared_today", a.cared_today)?;
            d.set_item("fertilizer_available", a.fertilizer_available)?;
            d.set_item("pending_care_bonus", a.pending_care_bonus)?;
            d.into_any()
        }
    })
}

fn farm_to_py<'py>(py: Python<'py>, f: &Farm) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Trap 4: money is f64 and must arrive as a Python float, never an int.
    d.set_item("money", PyFloat::new(py, f.money))?;
    let rows = PyList::empty(py);
    for y in 0..f.board_size as i64 {
        let row = PyList::empty(py);
        for x in 0..f.board_size as i64 {
            row.append(tile_to_py(py, f.tile(x, y))?)?;
        }
        rows.append(row)?;
    }
    d.set_item("tiles", rows)?;
    d.set_item("farmer", PyList::new(py, [f.farmer[0], f.farmer[1]])?)?;
    let hands = PyList::empty(py);
    for h in &f.hands {
        hands.append(PyList::new(py, [h[0], h[1]])?)?;
    }
    d.set_item("hands", hands)?;
    d.set_item(
        "unlocked_quadrants",
        PyList::new(py, f.unlocked_quadrants.iter().map(|q| q.as_str()))?,
    )?;
    d.set_item("hires_today", f.hires_today)?;
    Ok(d)
}

fn private_to_py<'py>(py: Python<'py>, p: &Private) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Fixed key order: PRODUCTS then ANIMALS, exactly as `_new_private()` builds it.
    let shed = PyDict::new(py);
    for item in SHED_ITEMS {
        shed.set_item(item.as_str(), p.shed.get(item))?;
    }
    d.set_item("shed", shed)?;
    let seeds = PyDict::new(py);
    for c in CROP_LIST {
        seeds.set_item(c.as_str(), p.seeds[c.idx()])?;
    }
    d.set_item("seeds", seeds)?;
    // Trap 2: insertion order is observable game state. Emitted both as a dict (whose
    // insertion order CPython preserves) and as an explicit list of pairs, so a caller
    // that re-sorts the dict can still see the truth.
    let invs = PyList::empty(py);
    let ordered = PyList::empty(py);
    for inv in &p.inventories {
        let di = PyDict::new(py);
        let li = PyList::empty(py);
        for (item, n) in inv.iter() {
            di.set_item(item.as_str(), n)?;
            li.append(PyTuple::new(py, [item.as_str().into_pyobject(py)?.into_any(), n.into_pyobject(py)?.into_any()])?)?;
        }
        invs.append(di)?;
        ordered.append(li)?;
    }
    d.set_item("inventories", invs)?;
    d.set_item("inventories_ordered", ordered)?;
    Ok(d)
}

/// The state-dict shape pinned by `parity/canon.py`.
fn state_dict_of<'py>(py: Python<'py>, gs: &GameState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("step", gs.step)?;
    d.set_item("day", gs.day)?;
    d.set_item("hour", gs.hour)?;
    d.set_item("done", gs.done)?;
    d.set_item(
        "rewards",
        PyList::new(
            py,
            [PyFloat::new(py, gs.rewards[0]), PyFloat::new(py, gs.rewards[1])],
        )?,
    )?;
    // `state[i].status`. The framework and the interpreter flip to DONE on exactly the
    // same step: the interpreter sets it when `step >= episodeSteps - 2` and
    // `Environment.step` when the post-increment `obs.step >= episodeSteps - 1`, and
    // `obs.step == step + 1`, so the two conditions are the same inequality. Emitting
    // it makes `parity/canon.py` compare DONE *timing* directly (trap 6) instead of
    // inferring it from the rewards. The ERROR / INVALID / TIMEOUT statuses are
    // framework agent bookkeeping with no game-state equivalent; if a trace ever drives
    // Python into one of them this field diverges, which is the correct signal — that
    // trace is outside the port's contract.
    d.set_item(
        "statuses",
        PyList::new(
            py,
            if gs.done {
                ["DONE", "DONE"]
            } else {
                ["ACTIVE", "ACTIVE"]
            },
        )?,
    )?;
    let market = PyDict::new(py);
    let inv = PyDict::new(py);
    let px = PyDict::new(py);
    for item in PRODUCTS {
        inv.set_item(item.as_str(), gs.market.inv(item))?;
        px.set_item(item.as_str(), gs.market.price(item))?;
    }
    market.set_item("inventory", inv)?;
    market.set_item("prices", px)?;
    d.set_item("market", market)?;
    let town = PyDict::new(py);
    town.set_item(
        "unlocked_shops",
        PyList::new(py, gs.town.unlocked_shops.iter().map(|s| s.as_str()))?,
    )?;
    d.set_item("town", town)?;
    let farms = PyList::empty(py);
    for f in &gs.farms {
        farms.append(farm_to_py(py, f)?)?;
    }
    d.set_item("farms", farms)?;
    let privs = PyList::empty(py);
    for p in &gs.privates {
        privs.append(private_to_py(py, p)?)?;
    }
    d.set_item("privates", privs)?;
    d.set_item("seed", gs.seed)?;
    Ok(d)
}

/// `renderer()`-ish ASCII board dump, for eyeballing a divergence.
fn render_state(gs: &GameState) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "step={} day={} hour={} done={} seed={}\n",
        gs.step, gs.day, gs.hour, gs.done, gs.seed
    ));
    for p in 0..2 {
        let f = &gs.farms[p];
        out.push_str(&format!(
            "player {p}  money={:.0}  hands={}  hires_today={}  quadrants={:?}\n",
            f.money,
            f.hands.len(),
            f.hires_today,
            f.unlocked_quadrants
                .iter()
                .map(|q| q.as_str())
                .collect::<Vec<_>>()
        ));
        for y in 0..f.board_size as i64 {
            out.push_str("  ");
            for x in 0..f.board_size as i64 {
                out.push(f.tile(x, y).render());
            }
            out.push('\n');
        }
        let shed: Vec<String> = SHED_ITEMS
            .iter()
            .filter(|i| p_shed(gs, p, **i) > 0)
            .map(|i| format!("{}:{}", i.as_str(), p_shed(gs, p, *i)))
            .collect();
        out.push_str(&format!("  shed {{{}}}\n", shed.join(", ")));
    }
    let px: Vec<String> = PRODUCTS
        .iter()
        .map(|i| format!("{}:{}@{}", i.as_str(), gs.market.inv(*i), gs.market.price(*i)))
        .collect();
    out.push_str(&format!("market {}\n", px.join(" ")));
    out
}

fn p_shed(gs: &GameState, p: usize, i: ShedItem) -> i64 {
    gs.privates[p].shed.get(i)
}

// ===========================================================================
// PyGame — single episode, parity/debug
// ===========================================================================

/// A single Kaggriculture episode, driven one step at a time.
///
/// ```python
/// import kaggri
/// g = kaggri.PyGame({"episodeSteps": 720, "seed": 12345})
/// g.step({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 5]]},
///        {"farmer": ["PASS"],           "hands": [], "market": []})
/// sd = g.state_dict()
/// ```
#[pyclass(module = "kaggri")]
pub struct PyGame {
    cfg: Config,
    params: MarketParams,
    gs: GameState,
    /// `-1` (none) / `0` / `1` — the `phantomSeat` configuration key. A phantom seat is
    /// an exogenous market-pressure stream rather than a player: see
    /// [`PyVecEnv::set_phantom_seats`] and `kaggri_core::sim::step_with_phantom`.
    phantom_seat: i8,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let (cfg, params, phantom_seat) = config_from_py(config)?;
        let gs = GameState::new_with_params(&cfg, params.clone());
        Ok(PyGame { cfg, params, gs, phantom_seat })
    }

    /// Alternative constructor, for callers that probe `PyGame.new(cfg)`.
    #[staticmethod]
    #[pyo3(signature = (config=None), name = "new")]
    fn new_static(config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        PyGame::new(config)
    }

    /// Restart the episode from scratch, optionally with a new resolved seed.
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<i64>) {
        if let Some(s) = seed {
            self.cfg.seed = s;
        }
        self.gs = GameState::new_with_params(&self.cfg, self.params.clone());
    }

    /// One `interpreter()` call. `a0` / `a1` are the action dicts verbatim.
    ///
    /// A no-op once the episode is done, mirroring the framework's
    /// `if env.done: return state`.
    fn step(&mut self, a0: &Bound<'_, PyAny>, a1: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.gs.done {
            return Ok(());
        }
        let acts = [
            player_action_from_py(a0, 0)?,
            player_action_from_py(a1, 1)?,
        ];
        sim::step_with_phantom(
            &mut self.gs,
            &acts,
            &self.cfg,
            phantom_mask(self.phantom_seat),
        );
        Ok(())
    }

    /// The configured phantom seat: `-1` (none), `0` or `1`. See the `phantomSeat`
    /// configuration key.
    #[getter]
    fn phantom_seat(&self) -> i8 {
        self.phantom_seat
    }

    /// The full state, in the shape `parity/canon.py` compares.
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_dict_of(py, &self.gs)
    }

    /// The parsed configuration, so a caller can confirm what was actually used.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        config_to_py(py, &self.cfg, &self.params)
    }

    #[getter]
    fn done(&self) -> bool {
        self.gs.done
    }

    #[getter]
    fn seed(&self) -> i64 {
        self.gs.seed
    }

    #[getter]
    fn step_no(&self) -> i64 {
        self.gs.step
    }

    #[getter]
    fn day(&self) -> i64 {
        self.gs.day
    }

    #[getter]
    fn hour(&self) -> i64 {
        self.gs.hour
    }

    #[getter]
    fn rewards(&self) -> (f64, f64) {
        (self.gs.rewards[0], self.gs.rewards[1])
    }

    #[getter]
    fn money(&self) -> (f64, f64) {
        (self.gs.farms[0].money, self.gs.farms[1].money)
    }

    fn render(&self) -> String {
        render_state(&self.gs)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyGame(seed={}, step={}, day={}, hour={}, done={})",
            self.gs.seed, self.gs.step, self.gs.day, self.gs.hour, self.gs.done
        )
    }
}

// ===========================================================================
// The RL layer  ─────────────────────────────────────────────────────────────
//
// The observation layout, the verb table, the market heads and the packer all live in
// `kaggri_core::{obs, macros}` and are specified by `rust/OBS_LAYOUT.md`. This section
// is the numpy-facing decode only. `obs::encode_with_cfg`, `obs::legal_mask_with_cfg`
// and `macros::resolve_turn` all take `&GameState`, so nothing here can change a game
// outcome.
//
// The `_with_cfg` variants are mandatory: the pinned `encode` / `legal_mask` hard-assume
// `Config::default()` (PORT_SPEC pins their signatures without a `Config`), which would
// silently mis-normalise every non-default episode — wrong `shedCapacity` divisors,
// wrong day/hour scales, wrong board size.
// ===========================================================================

/// Observation layout version, re-exported as `kaggri.OBS_VERSION`.
const OBS_VERSION_EXPOSED: u32 = OBS_VERSION;

/// Action-space version, re-exported as `kaggri.ACTION_VERSION`.
///
/// 0 was the development seam (19 primitive unit ops, no market head). 1 is the shipped
/// macro action space: per unit a target tile and a verb, plus 21 categorical market
/// heads.
const ACTION_VERSION_EXPOSED: u32 = 1;

/// Total market logits, i.e. `sum(MARKET_HEAD_SIZES)`.
const MARKET_LOGITS: usize = obs_dims().market_logits;

// The numbers `OBS_LAYOUT.md` §0 and §11 pin, and that the training code asserts at
// startup. Checked here at compile time so a layout change cannot ship without also
// moving the version this module advertises.
const _: () = assert!(OBS_VERSION_EXPOSED == 1);
const _: () = assert!(N_TOKENS == 140 && N_FEATURES == 48 && OBS_LEN == 6720);
const _: () = assert!(MAX_UNITS == 17 && N_VERBS == 38 && MASK_LEN == 646);

/// A negative value in the first legacy verb slot marks a row whose first unit block is
/// the exact low-level categorical ABI (`src/raw_actions.py`).  Valid macro policies
/// emit only `0..N_VERBS`, so this is backward-compatible and permits raw-vs-macro games
/// in one batch without a per-environment mode array.
const RAW_ROW_SENTINEL: i64 = -1;
const RAW_MARKET_SENTINEL: i64 = -1;
const _: () = assert!(N_TARGETS == 100);
const _: () = assert!(N_MARKET_HEADS == 21 && MARKET_LOGITS == 128);

/// Default envs per rayon chunk: 1, i.e. rayon's own adaptive splitting.
///
/// Deliberately *not* tuned. See [`PyVecEnv::chunk_envs`] for the measured table — no
/// single value wins at every batch size, and the default has to be the one that is never
/// pathological.
const DEFAULT_CHUNK_ENVS: usize = 1;

/// Values one player's action row carries: `units` targets, `units` verbs, then the
/// [`N_MARKET_HEADS`] market choices.
#[inline]
const fn player_action_width(units: usize) -> usize {
    2 * units + N_MARKET_HEADS
}

/// Decode one player's unit-order block into `out` (cleared and refilled).
///
/// `row` is `[target * units | verb * units | market * 21]`; only the first `2 * units`
/// values are read here.
///
/// Out-of-range on **either** head collapses that unit to [`MacroOrder::idle`], i.e. a
/// `PASS`. Clamping would be worse than idling: clamping a target silently retargets the
/// order at tile 99 and clamping a verb silently turns a garbage sample into `Drop`,
/// while idling cannot spend money, destroy an item or move a unit. An untrained policy
/// samples plenty of both.
#[inline]
fn decode_orders(row: &[i64], units: usize, out: &mut Vec<MacroOrder>) {
    out.clear();
    for u in 0..units {
        let (t, v) = (row[u], row[units + u]);
        let order = if t >= 0 && v >= 0 {
            // `from_indices` bounds-checks both heads and returns None outside them.
            MacroOrder::from_indices(t as usize, v as usize).unwrap_or_else(MacroOrder::idle)
        } else {
            MacroOrder::idle()
        };
        out.push(order);
    }
}

/// Decode one player's 21 market choices.
///
/// A negative choice becomes category 0, which is the *do nothing* category of every
/// market head (`hire = 0`, no `BUY_LAND`, quantity bucket `0`, sell fraction `0.0`), so
/// a garbage sample cannot spend money. An over-range choice is clamped to the head's
/// last category by [`MarketHead::from_choices`] — that rule lives there, on purpose, so
/// this file cannot drift from it.
#[inline]
fn decode_market(row: &[i64]) -> MarketHead {
    let mut c = [0u8; N_MARKET_HEADS];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = row[k].clamp(0, u8::MAX as i64) as u8;
    }
    MarketHead::from_choices(&c)
}

/// Decode the first `units` entries as exact low-level unit commands.
fn decode_raw_turn(
    row: &[i64],
    units: usize,
    live_hands: usize,
    out: &mut PlayerAction,
) {
    out.farmer = raw_unit_op(row.first().copied().unwrap_or(0));
    out.hands.clear();
    for h in 0..live_hands {
        out.hands.push(if h + 1 < units {
            raw_unit_op(row[h + 1])
        } else {
            UnitOp::Pass
        });
    }
}

/// Decode ten ordered market tokens. Zero terminates the queue; a nonzero quantity-zero
/// token becomes a `None` slot just as Python's `_parse_order` does.
fn decode_raw_market(row: &[i64], out: &mut Vec<Option<MarketOrder>>) {
    out.clear();
    for &token in row.iter().take(RAW_MARKET_SLOTS) {
        if token == 0 {
            break;
        }
        if token < 0 || token >= RAW_MARKET_ACTIONS as i64 {
            break;
        }
        out.push(raw_market_order(token));
    }
}

/// Human-readable verb name, for `kaggri.VERB_NAMES` and for debugging.
fn verb_name(v: Verb) -> String {
    match v {
        Verb::Idle => "IDLE".to_string(),
        Verb::Water => "WATER".to_string(),
        Verb::Harvest => "HARVEST".to_string(),
        Verb::Fertilize => "FERTILIZE".to_string(),
        Verb::Dig => "DIG".to_string(),
        Verb::BuildCoop => "BUILD_COOP".to_string(),
        Verb::BuildPasture => "BUILD_PASTURE".to_string(),
        Verb::Feed => "FEED".to_string(),
        Verb::Care => "CARE".to_string(),
        Verb::CollectFertilizer => "COLLECT_FERTILIZER".to_string(),
        Verb::Plant(c) => format!("PLANT_{}", c.as_str()),
        Verb::PlaceAnimal(a) => format!("PLACE_{}", a.as_str()),
        Verb::Pickup {
            item_slot,
            qty_slot,
        } => format!(
            "PICKUP_{}_{}",
            PICKUP_ITEMS[item_slot as usize].as_str(),
            PICKUP_QTYS[qty_slot as usize]
        ),
        Verb::Deposit(p) => format!("DEPOSIT_{}", p.as_str()),
        Verb::Drop => "DROP".to_string(),
    }
}

/// One resolved [`UnitOp`] as the token list the Kaggle environment takes.
///
/// This is the inverse of [`parse_unit_op`] over the ops [`macros::resolve_turn`] can
/// emit, and exists so [`PyVecEnv::resolve_actions`] can hand [`PyGame`] the *exact*
/// action the batched env submitted — which is what makes a `PyVecEnv`-vs-`PyGame`
/// lockstep test possible (STATUS §7.3).
fn unit_op_to_py<'py>(py: Python<'py>, op: UnitOp) -> PyResult<Bound<'py, PyList>> {
    let one = |s: &str| PyList::new(py, [s]);
    Ok(match op {
        // `Noop` is unreachable out of `resolve_turn`; an empty list is Python's own
        // no-op (`not action` falls straight through `_apply_unit_action`).
        UnitOp::Noop => PyList::empty(py),
        UnitOp::Pass => one("PASS")?,
        UnitOp::Move(d) => one(d.as_str())?,
        UnitOp::Drop => one("DROP")?,
        UnitOp::Water => one("WATER")?,
        UnitOp::Harvest => one("HARVEST")?,
        UnitOp::Fertilize => one("FERTILIZE")?,
        UnitOp::Dig => one("DIG")?,
        UnitOp::BuildCoop => one("BUILD_COOP")?,
        UnitOp::BuildPasture => one("BUILD_PASTURE")?,
        UnitOp::Feed => one("FEED")?,
        UnitOp::CollectFertilizer => one("COLLECT_FERTILIZER")?,
        UnitOp::Care => one("CARE")?,
        UnitOp::Plant(c) => PyList::new(py, ["PLANT", c.as_str()])?,
        UnitOp::Pickup { item, n } => {
            let l = PyList::new(py, ["PICKUP", item.as_str()])?;
            l.append(n)?;
            l
        }
        UnitOp::Place { item, n } => {
            let l = PyList::new(py, ["PLACE", item.as_str()])?;
            l.append(n)?;
            l
        }
    })
}

/// One packed [`MarketOrder`] as the token list the Kaggle environment takes.
fn market_order_to_py<'py>(py: Python<'py>, o: MarketOrder) -> PyResult<Bound<'py, PyList>> {
    let two = |op: &str, item: &str, n: i64| -> PyResult<Bound<'py, PyList>> {
        let l = PyList::new(py, [op, item])?;
        l.append(n)?;
        Ok(l)
    };
    match o {
        MarketOrder::Hire => PyList::new(py, ["HIRE"]),
        MarketOrder::BuyLand => PyList::new(py, ["BUY_LAND"]),
        MarketOrder::Sell { item, n } => two("SELL", item.as_str(), n),
        MarketOrder::BuyProduct { item, n } => two("BUY_PRODUCT", item.as_str(), n),
        MarketOrder::BuySeed { crop, n } => two("BUY_SEED", crop.as_str(), n),
        MarketOrder::BuyAnimal { animal, n } => two("BUY_ANIMAL", animal.as_str(), n),
    }
}

/// A whole resolved [`PlayerAction`] as the `{"farmer", "hands", "market"}` dict
/// `PyGame.step` takes.
fn player_action_to_py<'py>(py: Python<'py>, a: &PlayerAction) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("farmer", unit_op_to_py(py, a.farmer)?)?;
    let hands = PyList::empty(py);
    for h in &a.hands {
        hands.append(unit_op_to_py(py, *h)?)?;
    }
    d.set_item("hands", hands)?;
    let market = PyList::empty(py);
    for o in &a.market {
        match o {
            Some(o) => market.append(market_order_to_py(py, *o)?)?,
            None => market.append(py.None())?,
        }
    }
    d.set_item("market", market)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// numpy integer input
// ---------------------------------------------------------------------------

/// A flat, C-contiguous view of an `int64` or `int32` numpy array of any rank.
///
/// Both dtypes are accepted because torch's `argmax`/`multinomial` produce `int64` while
/// hand-written action tables and `np.int32` buffers are common; everything else is
/// refused rather than silently truncated.
enum FlatInts<'a> {
    I64(&'a [i64]),
    I32(&'a [i32]),
}

impl FlatInts<'_> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            FlatInts::I64(s) => s.len(),
            FlatInts::I32(s) => s.len(),
        }
    }

    /// Copy into an env-owned `i64` buffer of exactly the same length.
    ///
    /// Parallel above a threshold. At 65,536 envs with a 17-unit crew the action array is
    /// 58 MB and the serial `memcpy` measured **5.3 ms in isolation** — the only serial
    /// section of a vec-step, and the only part of it that does not shrink when work is
    /// added elsewhere. Parallelising it does not remove that traffic, only serialises it
    /// less: the honest end-to-end effect at 65,536 envs is ~1 ms of 43, which is inside
    /// this box's run-to-run noise, and it is nil at 4,096 envs (where the array is 3.6 MB
    /// and stays on the serial path). It is kept because below `PAR_MIN` it *is* the serial
    /// path, so it cannot be slower than what it replaced.
    ///
    /// Rayon is safe here even though the caller still holds the GIL: no worker touches
    /// Python, and holding the GIL is precisely what stops another Python thread from
    /// mutating the source while it is being read.
    fn copy_into(&self, dst: &mut [i64]) {
        // Below this the fan-out costs more than the copy (a 4,096-env array is 3.6 MB
        // and copies in ~0.2 ms serially).
        const PAR_MIN: usize = 1 << 21;
        let chunk = (dst.len() / (rayon::current_num_threads() * 2)).max(1 << 14);
        match self {
            FlatInts::I64(s) if dst.len() < PAR_MIN => dst.copy_from_slice(s),
            FlatInts::I64(s) => dst
                .par_chunks_mut(chunk)
                .zip(s.par_chunks(chunk))
                .for_each(|(d, x)| d.copy_from_slice(x)),
            FlatInts::I32(s) if dst.len() < PAR_MIN => {
                for (d, v) in dst.iter_mut().zip(s.iter()) {
                    *d = *v as i64;
                }
            }
            FlatInts::I32(s) => dst
                .par_chunks_mut(chunk)
                .zip(s.par_chunks(chunk))
                .for_each(|(d, x)| {
                    for (d, v) in d.iter_mut().zip(x.iter()) {
                        *d = *v as i64;
                    }
                }),
        }
    }
}

/// `(flat data, shape)` of a C-contiguous int64/int32 numpy array, or `None`.
///
/// The returned slice borrows the caller's array; the copy into env-owned scratch
/// happens before the GIL is released, so no rayon worker ever reads Python memory.
fn flat_ints<'a, 'py>(v: &'a Bound<'py, PyAny>) -> Option<(FlatInts<'a>, &'a [usize])> {
    if let Ok(a) = v.cast::<PyArrayDyn<i64>>() {
        // SAFETY: `as_slice` refuses a non-contiguous array; the data is only read, and
        // only while the GIL is held.
        let s = unsafe { a.as_slice() }.ok()?;
        return Some((FlatInts::I64(s), a.shape()));
    }
    if let Ok(a) = v.cast::<PyArrayDyn<i32>>() {
        let s = unsafe { a.as_slice() }.ok()?;
        return Some((FlatInts::I32(s), a.shape()));
    }
    None
}

// ===========================================================================
// PyVecEnv — batched RL interface
// ===========================================================================

/// A `*mut T` that rayon is allowed to move into worker threads.
///
/// Every use below slices strictly disjoint, index-derived ranges of an array this
/// struct owns exclusively, so no two threads ever touch the same element.
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// `&mut buf[offset .. offset + len]`.
    ///
    /// Accessed through a method rather than the tuple field on purpose: Rust 2021
    /// closures capture *individual fields*, so `ptr.0.add(i)` inside a rayon closure
    /// would capture the bare `*mut T` and bypass the `Send`/`Sync` wrapper entirely.
    /// A method call captures `&SendPtr<T>`, which is what the impls above cover.
    ///
    /// # Safety
    /// The caller must guarantee that no two concurrent calls produce overlapping
    /// ranges and that `offset + len` is within the allocation.
    #[inline]
    unsafe fn slice(&self, offset: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(offset), len)
    }
}

// ---------------------------------------------------------------------------
// The published dtype  (OBS_LAYOUT.md §11)
//
// The observation is *computed* in f32 — every divisor, clip and `ln` in `obs.rs` is an
// f32 expression, and that does not change. What crosses into numpy is a separate,
// deliberate choice, and the default is **f16**:
//
//   * A batched vec-step is ~85 % memcpy into the shared observation plane (STATUS §3.3),
//     so the element width *is* the binding cost of collection.
//   * It halves the host->device transfer and the rollout buffer at the same time.
//   * Every slot is normalised into [-4, 4] (`OBS_LAYOUT.md` §1, asserted by
//     `obs_macros.rs`), where f16 quantises to at worst 2^-9 = 0.002 absolute and
//     2^-11 = 4.9e-4 relative. `parity/f16_obs_error.py` measures the realised error per
//     slot over full episodes.
//
// `obs_dtype="float32"` restores the wide plane bit-for-bit, so an A/B is one keyword
// argument. Both planes are governed by the one knob: the mask's alphabet is {0.0, 1.0},
// which is exact in f16, so there is nothing to trade off there and no reason for the two
// input planes to disagree about their dtype.
//
// NOT covered by this knob, on purpose: `rewards` and `info["final_money"]` stay f32.
// They carry money, which reaches ~1e5 in a good episode and would quantise to steps of
// 64 in f16 (and saturate to inf past 65504). A reward plane is 2 floats per env against
// the observation's 13 440, so there is no bandwidth argument for it either.
// ---------------------------------------------------------------------------

/// IEEE-754 binary16, as the 2-byte element numpy sees as `float16`.
///
/// Conversion lives in `kaggri_core::obs` (round-to-nearest-ties-to-even, `vcvtps2ph`
/// where the CPU has it); this type exists only so `numpy` will hand back an array whose
/// dtype *is* `float16` rather than a `uint16` array the caller has to `.view()`.
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
struct F16(u16);

// SAFETY: `F16` is `#[repr(transparent)]` over `u16`, so its size (2), alignment and bit
// pattern are exactly what numpy's `float16` descriptor describes, and it holds no Python
// object (so `IS_COPY` is true and `clone_ref` is a copy).
unsafe impl Element for F16 {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::new(py, "float16").expect("numpy always provides a float16 dtype")
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }
}

/// Which dtype the observation and mask planes are published in.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum ObsDtype {
    F32,
    F16,
}

/// The shipped default. Changing this changes the policy's input dtype, so checkpoints
/// are not comparable across it — see `OBS_LAYOUT.md` §11 and `STATUS.md` §7.2.
const DEFAULT_OBS_DTYPE: ObsDtype = ObsDtype::F16;

impl ObsDtype {
    const fn name(self) -> &'static str {
        match self {
            ObsDtype::F32 => "float32",
            ObsDtype::F16 => "float16",
        }
    }

    /// Bytes per published feature.
    const fn itemsize(self) -> usize {
        match self {
            ObsDtype::F32 => 4,
            ObsDtype::F16 => 2,
        }
    }

    /// Parse the `obs_dtype` argument.
    ///
    /// Anything numpy itself resolves to a float dtype is accepted — `"float16"`, `"f2"`,
    /// `np.float16`, `np.dtype("float32")` — because the caller is a training script that
    /// already has a dtype object in hand. `None` means the shipped default. Everything
    /// else is refused loudly rather than silently downgraded: a policy trained against
    /// the wrong input dtype is a wasted run, not a warning.
    fn from_py(py: Python<'_>, v: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(v) = v else {
            return Ok(DEFAULT_OBS_DTYPE);
        };
        if v.is_none() {
            return Ok(DEFAULT_OBS_DTYPE);
        }
        let bad = || {
            PyValueError::new_err(format!(
                "obs_dtype must be \"float16\" (the default) or \"float32\", or anything \
                 numpy resolves to one of them (np.float16, np.dtype(\"f4\"), ...); got {}",
                v.repr().map(|r| r.to_string()).unwrap_or_default()
            ))
        };
        let d = PyArrayDescr::new(py, v).map_err(|_| bad())?;
        match (d.kind(), d.itemsize()) {
            (b'f', 2) => Ok(ObsDtype::F16),
            (b'f', 4) => Ok(ObsDtype::F32),
            _ => Err(bad()),
        }
    }
}

/// One published float plane — the observation or the mask — in either dtype.
///
/// Two variants rather than one `PyArrayDyn` because the element type has to be known
/// statically for numpy to report the right dtype and for the write path to be a plain
/// typed store.
enum Plane {
    F32(Py<PyArray3<f32>>),
    F16(Py<PyArray3<F16>>),
}

impl Plane {
    fn zeros(py: Python<'_>, n_envs: usize, dim: usize, dt: ObsDtype) -> Self {
        match dt {
            ObsDtype::F32 => {
                Plane::F32(PyArray3::<f32>::zeros(py, [n_envs, 2, dim], false).unbind())
            }
            ObsDtype::F16 => {
                Plane::F16(PyArray3::<F16>::zeros(py, [n_envs, 2, dim], false).unbind())
            }
        }
    }

    fn as_any<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Plane::F32(a) => a.bind(py).as_any().clone(),
            Plane::F16(a) => a.bind(py).as_any().clone(),
        }
    }

    fn ptr(&self, py: Python<'_>) -> PlanePtr {
        match self {
            Plane::F32(a) => PlanePtr::F32(SendPtr(a.bind(py).data())),
            // `F16` is `#[repr(transparent)]` over `u16`; the conversion writes `u16`.
            Plane::F16(a) => PlanePtr::F16(SendPtr(a.bind(py).data().cast::<u16>())),
        }
    }
}

/// The write end of a [`Plane`], as something rayon workers may hold.
enum PlanePtr {
    F32(SendPtr<f32>),
    F16(SendPtr<u16>),
}

impl PlanePtr {
    /// Run `fill` — `encode_with_cfg` or `legal_mask_with_cfg` — and land its `len` floats
    /// at `offset` in the plane, in the plane's dtype.
    ///
    /// On the f32 path this is exactly what the code did before: `fill` writes straight
    /// into the numpy array, no copy. On the f16 path `fill` writes into a per-worker
    /// scratch buffer that stays in L1/L2 (26.9 KB, reused for every env that worker
    /// touches) and only the converted half-width result reaches the shared array — which
    /// is the whole point: it halves the DRAM traffic that dominates a batched step, and
    /// it also keeps `encode`'s opening `out.fill(0.0)` out of the shared array entirely.
    ///
    /// # Safety
    /// As [`SendPtr::slice`]: `offset + len` must be inside the plane and no two
    /// concurrent calls may overlap. `scratch.len()` must be at least `len`.
    #[inline]
    unsafe fn publish(
        &self,
        offset: usize,
        len: usize,
        scratch: &mut [f32],
        fill: impl FnOnce(&mut [f32]),
    ) {
        match self {
            PlanePtr::F32(p) => fill(p.slice(offset, len)),
            PlanePtr::F16(p) => {
                let staged = &mut scratch[..len];
                fill(staged);
                f32_to_f16_into(staged, p.slice(offset, len));
            }
        }
    }
}

/// splitmix64, used to derive each episode's seed from the env's base seed.
///
/// Masked to 31 bits so every episode a training run touches is reproducible by the
/// Python environment via `configuration={"seed": N}`.
fn derive_seed(base: i64, episode: u64) -> i64 {
    let mut z = (base as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(episode.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z & 0x7FFF_FFFF) as i64
}

/// The episode an env runs on its `episode`'th rollout, from its base seed.
///
/// Episode 0 uses the base seed verbatim so `seeds=[N, ...]` reproduces exactly what
/// `configuration={"seed": N}` does in Python; later episodes derive a fresh 31-bit seed.
fn make_env(cfg: &Config, params: &MarketParams, base_seed: i64, episode: u64) -> GameState {
    let mut c = cfg.clone();
    c.seed = if episode == 0 {
        base_seed
    } else {
        derive_seed(base_seed, episode)
    };
    GameState::new_with_params(&c, params.clone())
}

/// `n_envs` independent episodes stepped in parallel, with the shipped RL interface
/// (`OBS_VERSION = 1`, `ACTION_VERSION = 1`).
///
/// ```python
/// v = kaggri.PyVecEnv(4096, {"episodeSteps": 720}, list(range(4096)))
/// obs, mask = v.reset()                 # (n, 2, 6720) f16,  (n, 2, 646) f16
/// a = np.zeros((v.n_envs, 2, v.player_action_dim), dtype=np.int64)   # all-idle
/// obs, mask, rew, done, info = v.step(a)
///
/// # ...and the same thing with the wide plane, for an A/B against an f16 run:
/// w = kaggri.PyVecEnv(4096, {"episodeSteps": 720}, list(range(4096)),
///                     obs_dtype="float32")
/// ```
///
/// The two input planes are `float16` by default — see [`PyVecEnv::obs_dtype`], which is
/// where the reasoning and the measurements live. `rew` and `info["final_money"]` are
/// `float32` regardless.
///
/// All five returned objects are **env-owned buffers, reused every step**. Copy what you
/// need to keep (`np.array(obs)`, or a write straight into a rollout buffer); that reuse
/// is what keeps the hot path free of Python allocation. Do not read them from another
/// thread while `step` is running — `step` releases the GIL and writes them from rayon
/// workers.
#[pyclass(module = "kaggri")]
pub struct PyVecEnv {
    cfg: Config,
    params: MarketParams,
    n_envs: usize,
    envs: Vec<GameState>,
    base_seeds: Vec<i64>,
    episode_idx: Vec<u64>,
    auto_reset: bool,
    /// [`OBS_LEN`] — the trailing axis of the observation plane.
    obs_dim: usize,
    /// [`MASK_LEN`] — the trailing axis of the mask plane.
    mask_dim: usize,
    /// [`player_action_width`] for this env's `units_per_player`.
    player_action_dim: usize,
    /// Envs per rayon chunk (`with_min_len`). See [`PyVecEnv::chunk_envs`].
    chunk_envs: usize,
    /// Dtype of the `obs` and `mask` planes. See [`PyVecEnv::obs_dtype`].
    obs_dtype: ObsDtype,
    // cached numpy buffers, allocated once
    obs: Plane,
    mask: Plane,
    rewards: Py<PyArray2<f32>>,
    dones: Py<PyArray1<bool>>,
    final_money: Py<PyArray2<f32>>,
    /// Per-seat money of the state the `obs` plane describes — published AFTER any
    /// auto-reset, exactly like `obs`, so on a done step it is the *fresh* episode's
    /// starting bank while `final_money` holds the terminal one. Potential-based
    /// shaping on cash needs an exact per-step bank (`info["final_money"]` is only
    /// written on terminal steps and the observation's money feature is f16, which
    /// quantises to ~$2 at $3,000 and ~$20 at $20,000).
    money: Py<PyArray2<f32>>,
    steps_arr: Py<PyArray1<i64>>,
    /// Optional `(n_envs, 2, TILE_MASK_LEN)` u8 plane (`tile_mask=True`), published with
    /// `obs`/`mask` on every `reset`/`step` and reachable as `info["tile_mask"]` and
    /// [`PyVecEnv::tile_mask`]. `None` keeps the shipped surface byte-identical.
    tile_mask: Option<Py<PyArray3<u8>>>,
    info: Py<PyDict>,
    // scratch, reused every step (nothing on the hot path allocates)
    act_scratch: Vec<i64>,
    units_per_player: usize,
    total_steps: u64,
    /// Per-env phantom seat: `-1` none, `0` or `1`. See [`PyVecEnv::set_phantom_seats`].
    /// Persists across auto-resets — it describes the *env*, not the episode.
    phantom_seats: Vec<i8>,
}

impl PyVecEnv {
    fn reset_env(&mut self, i: usize, fresh_episode: bool) {
        if fresh_episode {
            self.episode_idx[i] += 1;
        }
        self.envs[i] = make_env(&self.cfg, &self.params, self.base_seeds[i], self.episode_idx[i]);
    }

    /// Fill the observation and mask planes for every env, in one rayon fan-out.
    ///
    /// Used by `reset` only: `step` fuses exactly this work into its stepping closure so
    /// a step costs **one** fan-out instead of two (STATUS §6.2 — the fan-out is a fixed
    /// ~450–600 µs per call at any batch size).
    fn publish_all(&self, py: Python<'_>) {
        let obs_ptr = self.obs.ptr(py);
        let mask_ptr = self.mask.ptr(py);
        let tm_ptr = self.tile_mask.as_ref().map(|a| SendPtr(a.bind(py).data()));
        let cfg = &self.cfg;
        let envs = &self.envs;
        let min_len = self.chunk_envs;
        let scratch_len = self.scratch_len();
        py.detach(|| {
            envs.par_iter()
                .enumerate()
                .with_min_len(min_len)
                .for_each_init(
                    || vec![0f32; scratch_len],
                    |scratch, (i, gs)| {
                        for p in 0..2 {
                            // SAFETY: disjoint per (i, p); the arrays are (n_envs, 2, dim)
                            // and C-contiguous, and `self` holds the only reference to
                            // them.
                            unsafe {
                                obs_ptr.publish((i * 2 + p) * OBS_LEN, OBS_LEN, scratch, |o| {
                                    encode_with_cfg(gs, p, cfg, o)
                                });
                                mask_ptr.publish((i * 2 + p) * MASK_LEN, MASK_LEN, scratch, |m| {
                                    legal_mask_with_cfg(gs, p, cfg, m)
                                });
                                if let Some(tm) = tm_ptr.as_ref() {
                                    tile_verb_mask_with_cfg(
                                        gs,
                                        p,
                                        cfg,
                                        tm.slice((i * 2 + p) * TILE_MASK_LEN, TILE_MASK_LEN),
                                    );
                                }
                            }
                        }
                    },
                );
        });
    }

    /// Copy every env's current per-seat bank into the `money` plane.
    ///
    /// Serial on purpose: it is two f32 stores per env and is called from `reset` /
    /// `warmup_pass` only. `step` writes the same plane inside its own rayon closure.
    fn publish_money(&self, py: Python<'_>) -> PyResult<()> {
        let m = self.money.bind(py);
        // SAFETY: this crate owns the array and holds the GIL.
        let slice = unsafe { m.as_slice_mut()? };
        for (i, gs) in self.envs.iter().enumerate() {
            slice[i * 2] = gs.farms[0].money as f32;
            slice[i * 2 + 1] = gs.farms[1].money as f32;
        }
        Ok(())
    }

    /// Copy every env's current internal step counter into the `steps_arr` plane.
    fn publish_steps(&self, py: Python<'_>) -> PyResult<()> {
        let st = self.steps_arr.bind(py);
        // SAFETY: as `publish_money`.
        let slice = unsafe { st.as_slice_mut()? };
        for (i, gs) in self.envs.iter().enumerate() {
            slice[i] = gs.step;
        }
        Ok(())
    }

    /// Length of the per-worker f32 staging buffer: [`OBS_LEN`] when the planes are f16
    /// (the observation is the larger of the two), zero when they are f32 and `encode`
    /// writes straight into numpy.
    fn scratch_len(&self) -> usize {
        match self.obs_dtype {
            ObsDtype::F32 => 0,
            ObsDtype::F16 => OBS_LEN,
        }
    }
}

#[pymethods]
impl PyVecEnv {
    /// `PyVecEnv(n_envs, config=None, seeds=None, auto_reset=True, units_per_player=17)`
    ///
    /// `seeds` is one resolved episode seed per env; absent means `0..n_envs`.
    ///
    /// `units_per_player` is how many **unit macro orders** the action array carries per
    /// player. The default 17 (`MAX_UNITS` = farmer + 16 hands) is the full addressable
    /// crew and is what the training code should use; a smaller value narrows the action
    /// array and leaves the trailing units permanently idle, which is useful for cheap
    /// benchmarks and for policies that cap their crew.
    ///
    /// `chunk_envs` is the rayon `with_min_len` granularity — see
    /// [`PyVecEnv::chunk_envs`]. It changes nothing about the result and everything about
    /// the memory behaviour.
    ///
    /// `obs_dtype` is the dtype of the `obs` and `mask` planes: `"float16"` (the default)
    /// or `"float32"`. See [`PyVecEnv::obs_dtype`] — it is a deliberate, versionable
    /// choice, not a tuning knob, because it changes the policy's input dtype.
    #[new]
    #[pyo3(signature = (n_envs, config=None, seeds=None, auto_reset=true,
                        units_per_player=MAX_UNITS, chunk_envs=DEFAULT_CHUNK_ENVS,
                        obs_dtype=None, tile_mask=false))]
    fn new(
        py: Python<'_>,
        n_envs: usize,
        config: Option<&Bound<'_, PyDict>>,
        seeds: Option<&Bound<'_, PyAny>>,
        auto_reset: bool,
        units_per_player: usize,
        chunk_envs: usize,
        obs_dtype: Option<&Bound<'_, PyAny>>,
        tile_mask: bool,
    ) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(PyValueError::new_err("n_envs must be >= 1"));
        }
        if chunk_envs == 0 {
            return Err(PyValueError::new_err("chunk_envs must be >= 1"));
        }
        if units_per_player == 0 || units_per_player > MAX_UNITS {
            return Err(PyValueError::new_err(format!(
                "units_per_player must be in 1..={MAX_UNITS} (MAX_UNITS = farmer + \
                 {MAX_HANDS} hands); got {units_per_player}"
            )));
        }
        let dtype = ObsDtype::from_py(py, obs_dtype)?;
        let (cfg, params, phantom_seat) = config_from_py(config)?;
        let base_seeds: Vec<i64> = match seeds {
            None => (0..n_envs as i64).collect(),
            Some(s) => {
                let v: Vec<i64> = s.extract().map_err(|_| {
                    PyTypeError::new_err(
                        "seeds must be a sequence of ints (list or 1-D int64 numpy array)",
                    )
                })?;
                if v.len() != n_envs {
                    return Err(PyValueError::new_err(format!(
                        "seeds has {} entries, expected n_envs = {n_envs}",
                        v.len()
                    )));
                }
                v
            }
        };
        let paw = player_action_width(units_per_player);
        let envs = base_seeds
            .iter()
            .map(|s| make_env(&cfg, &params, *s, 0))
            .collect();
        let me = PyVecEnv {
            cfg,
            params,
            n_envs,
            envs,
            base_seeds,
            episode_idx: vec![0; n_envs],
            auto_reset,
            obs_dim: OBS_LEN,
            mask_dim: MASK_LEN,
            player_action_dim: paw,
            chunk_envs,
            obs_dtype: dtype,
            obs: Plane::zeros(py, n_envs, OBS_LEN, dtype),
            mask: Plane::zeros(py, n_envs, MASK_LEN, dtype),
            rewards: PyArray2::<f32>::zeros(py, [n_envs, 2], false).unbind(),
            dones: PyArray1::<bool>::zeros(py, [n_envs], false).unbind(),
            final_money: PyArray2::<f32>::zeros(py, [n_envs, 2], false).unbind(),
            money: PyArray2::<f32>::zeros(py, [n_envs, 2], false).unbind(),
            steps_arr: PyArray1::<i64>::zeros(py, [n_envs], false).unbind(),
            tile_mask: if tile_mask {
                Some(PyArray3::<u8>::zeros(py, [n_envs, 2, TILE_MASK_LEN], false).unbind())
            } else {
                None
            },
            info: PyDict::new(py).unbind(),
            act_scratch: vec![0i64; n_envs * 2 * paw],
            units_per_player,
            total_steps: 0,
            // `phantomSeat` in the config seeds every env; `set_phantom_seats` then
            // overrides per env. Absent (the default) leaves the Kaggle-exact behaviour.
            phantom_seats: vec![phantom_seat; n_envs],
        };
        let info = me.info.bind(py);
        info.set_item("final_money", me.final_money.bind(py))?;
        info.set_item("money", me.money.bind(py))?;
        info.set_item("step", me.steps_arr.bind(py))?;
        if let Some(tm) = me.tile_mask.as_ref() {
            info.set_item("tile_mask", tm.bind(py))?;
        }
        me.publish_money(py)?;
        Ok(me)
    }

    /// Restart every episode. Returns `(obs, mask)` — both env-owned, reused buffers.
    #[pyo3(signature = (seeds=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(s) = seeds {
            let v: Vec<i64> = s.extract().map_err(|_| {
                PyTypeError::new_err("seeds must be a sequence of ints")
            })?;
            if v.len() != self.n_envs {
                return Err(PyValueError::new_err(format!(
                    "seeds has {} entries, expected n_envs = {}",
                    v.len(),
                    self.n_envs
                )));
            }
            self.base_seeds = v;
            self.episode_idx = vec![0; self.n_envs];
        }
        for i in 0..self.n_envs {
            self.episode_idx[i] = 0;
            self.reset_env(i, false);
        }
        // Zero the reward/done/final buffers so a caller cannot read stale values.
        {
            let r = self.rewards.bind(py);
            let d = self.dones.bind(py);
            let fm = self.final_money.bind(py);
            let st = self.steps_arr.bind(py);
            // SAFETY: this crate owns these arrays and holds the GIL.
            unsafe {
                r.as_slice_mut()?.fill(0.0);
                d.as_slice_mut()?.fill(false);
                fm.as_slice_mut()?.fill(0.0);
                st.as_slice_mut()?.fill(0);
            }
        }
        self.publish_money(py)?;
        self.publish_all(py);
        PyTuple::new(py, [self.obs.as_any(py), self.mask.as_any(py)])
    }

    /// Advance env `i` by `offsets[i]` **all-`PASS`** turns, then republish everything.
    ///
    /// ## What it is for: desynchronising the episode boundaries
    ///
    /// `episodeSteps` is one global number and `step` advances every env in lockstep, so
    /// a freshly constructed `PyVecEnv` has **all** `n_envs` episodes starting on step 0
    /// and therefore **all** of them terminating on the same step. Against a 720-step
    /// episode and a 144-step rollout that puts a terminal in exactly one rollout in
    /// five; at 64 it is one in eleven. Worse, every rollout then samples a single phase
    /// of the game — one update sees only days 0–6, the next only days 6–12 — so the
    /// advantage statistics, the value targets and the minibatch composition are all
    /// computed on one slice of the episode at a time.
    ///
    /// Calling `warmup_pass(offsets)` once, right after `reset`, fixes both permanently.
    /// Env `i` is put at internal step `offsets[i]`, so its episode ends
    /// `episodeSteps - 1 - offsets[i]` transitions from now; the replay's initial frame
    /// is not a transition. `auto_reset` then restarts it at step 0. The offsets survive
    /// every later auto-reset because they are a property of the *phase*, not of the
    /// state — nothing in the loop can resynchronise them.
    ///
    /// Cost is one-time and negligible: less than one episode of all-`PASS` simulator
    /// transitions per env for a run that will do millions.
    ///
    /// ## The one caveat the caller must handle
    ///
    /// Each env's **first** episode after this call is a junk episode: its first
    /// `offsets[i]` turns were nobody-does-anything. Those steps are a legitimate game
    /// state but not one any policy produced, so a trainer must mark them (and the
    /// terminal that ends them) invalid rather than learn from them. The per-env
    /// `episode_indices` and [`PyVecEnv::env_steps`] are what a caller uses to detect it.
    ///
    /// `offsets` is a sequence of `n_envs` non-negative ints, each
    /// `< episodeSteps - 1`. A replay contains an initial frame, so the transition at
    /// offset `episodeSteps - 1` is already terminal (719 for the default 720-frame
    /// season). It is rejected so warmup cannot silently auto-reset and lose the requested
    /// phase. Returns `(obs, mask)`, the same env-owned buffers `reset` returns.
    fn warmup_pass<'py>(
        &mut self,
        py: Python<'py>,
        offsets: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let offs: Vec<i64> = offsets.extract().map_err(|_| {
            PyTypeError::new_err(
                "offsets must be a sequence of ints (list or 1-D int64 numpy array)",
            )
        })?;
        if offs.len() != self.n_envs {
            return Err(PyValueError::new_err(format!(
                "offsets has {} entries, expected n_envs = {}",
                offs.len(),
                self.n_envs
            )));
        }
        // A replay includes its initial frame: an `episodeSteps = N` episode reaches
        // DONE on simulator transition N - 1. Warmup must stop strictly before that
        // transition, particularly when auto-reset is enabled, or the apparent largest
        // offset becomes phase 0 of episode 1 instead.
        let limit = self.cfg.episode_steps.saturating_sub(1) as i64;
        for (i, o) in offs.iter().enumerate() {
            if *o < 0 || *o >= limit {
                return Err(PyValueError::new_err(format!(
                    "offsets[{i}] = {o} is outside 0..{limit}; a replay contains an \
                     initial frame, so offset {limit} performs the terminal transition \
                     for episodeSteps = {}",
                    self.cfg.episode_steps,
                )));
            }
        }
        let total: i64 = offs.iter().sum();
        let cfg = self.cfg.clone();
        let cfg = &cfg;
        let params = self.params.clone();
        let params = &params;
        let auto_reset = self.auto_reset;
        let base_seeds = &self.base_seeds;
        let phantom_seats = &self.phantom_seats;
        let envs = &mut self.envs;
        let episode_idx = &mut self.episode_idx;
        let min_len = self.chunk_envs;
        py.detach(|| {
            envs.par_iter_mut()
                .zip(episode_idx.par_iter_mut())
                .enumerate()
                .with_min_len(min_len)
                .for_each(|(i, (gs, ep))| {
                    let pa = [PlayerAction::pass(), PlayerAction::pass()];
                    let phantom = phantom_mask(phantom_seats[i]);
                    for _ in 0..offs[i] {
                        let was_done = gs.done;
                        sim::step_with_phantom(gs, &pa, cfg, phantom);
                        if gs.done && !was_done && auto_reset {
                            *ep += 1;
                            *gs = make_env(cfg, params, base_seeds[i], *ep);
                        }
                    }
                });
        });
        self.total_steps += total as u64;
        self.publish_money(py)?;
        self.publish_steps(py)?;
        self.publish_all(py);
        PyTuple::new(py, [self.obs.as_any(py), self.mask.as_any(py)])
    }

    /// Per-seat bank of every env, as a **fresh** `(n_envs, 2)` f32 array.
    ///
    /// The same numbers `info["money"]` carries, read off the live `GameState` instead of
    /// out of the reused buffer, so it is safe to call after `reset` / `warmup_pass`
    /// (which have no `info` to return). Allocates; not a hot path.
    fn env_money<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let mut v: Vec<f32> = Vec::with_capacity(self.n_envs * 2);
        for gs in &self.envs {
            v.push(gs.farms[0].money as f32);
            v.push(gs.farms[1].money as f32);
        }
        PyArray1::from_vec(py, v).reshape([self.n_envs, 2]).expect("shape")
    }

    /// Per-env internal step counter, as a **fresh** `int64` array of length `n_envs`.
    ///
    /// `info["step"]` is a reused buffer written by `step` *before* any auto-reset, so on
    /// a done step it reads `episodeSteps` rather than the fresh episode's 0. This reads
    /// the live `GameState`, so it is the phase an env is actually in — which is what a
    /// test of [`PyVecEnv::warmup_pass`] has to check. Allocates; not a hot path.
    fn env_steps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.envs.iter().map(|gs| gs.step).collect();
        PyArray1::from_vec(py, v)
    }

    /// Step every env once.
    ///
    /// `actions` is a C-contiguous `int64` or `int32` numpy array holding
    /// `n_envs * 2 * player_action_dim` values — shape `(n_envs, 2, player_action_dim)`
    /// or the flattened `(n_envs, 2 * player_action_dim)`, whichever the caller finds
    /// easier. Axis 1 is the player.
    ///
    /// One player's row is three concatenated blocks, in the order the policy's heads
    /// produce them (`torch.cat([targets, verbs, market], -1)`):
    ///
    /// | offset | width | head | range |
    /// |---|---|---|---|
    /// | `0` | `units_per_player` | target tile per unit, `(x, y) = (k % bs, k / bs)` | `0..100` |
    /// | `units_per_player` | `units_per_player` | verb per unit (`kaggri.VERB_NAMES`) | `0..38` |
    /// | `2 * units_per_player` | 21 | market choices (`kaggri.MARKET_HEAD_SIZES`) | per head |
    ///
    /// Unit `0` is the farmer and unit `1 + h` is hand `h`, matching observation tokens
    /// `101 + u` and mask rows `u`. An all-zero array is "every unit idle, no market
    /// orders" — i.e. an all-`PASS` turn.
    ///
    /// `bs` is `boardSize`; the target head is a fixed 100 (`OBS_LAYOUT.md` §1), so on a
    /// board smaller than 10x10 the targets past `bs * bs` are simply unreachable and
    /// resolve to `PASS`. `int32` is accepted as well as `int64` and halves the bytes the
    /// action array moves, which is measurable only above ~30,000 envs.
    ///
    /// Returns `(obs, mask, rewards, dones, info)`. **All five are reused buffers**; see
    /// the class docstring.
    #[pyo3(signature = (actions, exact_actions=None))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'_, PyAny>,
        exact_actions: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self.load_actions(actions)?;
        // Optional real action dictionaries bypass the finite policy vocabulary.
        // None entries retain the normal packed-policy path. Both paths share
        // exactly the same interpreter, terminal bookkeeping and output buffers.
        let exact = if let Some(rows) = exact_actions {
            if rows.len() != self.n_envs {
                return Err(PyValueError::new_err("exact_actions needs one pair per env"));
            }
            let mut parsed = Vec::with_capacity(self.n_envs);
            for row in rows.iter() {
                if row.len()? != 2 {
                    return Err(PyValueError::new_err("exact_actions rows need two seats"));
                }
                let mut pair: [Option<PlayerAction>; 2] = [None, None];
                for p in 0..2 {
                    let action = row.get_item(p)?;
                    if !action.is_none() { pair[p] = Some(player_action_from_py(&action, p)?); }
                }
                parsed.push(pair);
            }
            Some(parsed)
        } else { None };

        let n = self.n_envs;
        let upp = self.units_per_player;
        let paw = self.player_action_dim;
        // Cloned so the rayon closure can borrow them while `self.envs` is borrowed
        // mutably. Neither clone touches the heap: `Config` is a dozen scalars and
        // `MarketParams` is `[ItemParams; 9]` of `Copy` fields, so this is two memcpys of
        // a few hundred bytes onto the stack. Together with the pre-sized `act_scratch`,
        // the per-worker `for_each_init` state below and the reused output buffers, the
        // only thing `step` allocates is the returned `PyTuple`.
        let cfg = self.cfg.clone();
        let cfg = &cfg;
        let params = self.params.clone();
        let params = &params;
        let auto_reset = self.auto_reset;

        let obs_ptr = self.obs.ptr(py);
        let mask_ptr = self.mask.ptr(py);
        let rew_ptr = SendPtr(self.rewards.bind(py).data());
        let done_ptr = SendPtr(self.dones.bind(py).data());
        let fm_ptr = SendPtr(self.final_money.bind(py).data());
        let mn_ptr = SendPtr(self.money.bind(py).data());
        let st_ptr = SendPtr(self.steps_arr.bind(py).data());
        let tm_ptr = self.tile_mask.as_ref().map(|a| SendPtr(a.bind(py).data()));

        let scratch_len = self.scratch_len();
        let acts = &self.act_scratch;
        let base_seeds = &self.base_seeds;
        let phantom_seats = &self.phantom_seats;
        let envs = &mut self.envs;
        let episode_idx = &mut self.episode_idx;
        let min_len = self.chunk_envs;

        // ONE rayon fan-out per step (STATUS §6.2): decode, resolve, step, publish the
        // per-env scalars, auto-reset, then encode — all inside the same closure. The
        // previous shape (par_iter to step, a serial reset pass, par_iter again to
        // encode) paid the ~450-600 us fan-out twice.
        py.detach(|| {
            envs.par_iter_mut()
                .zip(episode_idx.par_iter_mut())
                .enumerate()
                .with_min_len(min_len)
                // `for_each_init` gives each rayon worker one reusable `PlayerAction`
                // pair, one reusable order buffer and (on the f16 path) one reusable
                // OBS_LEN f32 staging buffer, so the `hands` / `market` / `orders` Vecs
                // and the staging buffer are allocated once per worker rather than once
                // per env per step. A tuple rather than a struct because the closure needs
                // `&orders` and `&mut pa[p]` at the same time. `scratch_len` is 0 when the
                // planes are f32, so that path allocates exactly what it did before.
                .for_each_init(
                    || {
                        (
                            [PlayerAction::pass(), PlayerAction::pass()],
                            Vec::<MacroOrder>::with_capacity(MAX_UNITS),
                            vec![0f32; scratch_len],
                        )
                    },
                    |(pa, orders, scratch), (i, (gs, ep))| {
                        // Both players are resolved against the SAME pre-step state,
                        // exactly as the Kaggle interpreter sees them.
                        for p in 0..2 {
                            if let Some(action) = exact.as_ref().and_then(|rows| rows[i][p].as_ref()) {
                                pa[p].clone_from(action);
                                continue;
                            }
                            let row = &acts[(i * 2 + p) * paw..(i * 2 + p + 1) * paw];
                            let head = decode_market(&row[2 * upp..]);
                            if row[upp] == RAW_ROW_SENTINEL {
                                decode_raw_turn(row, upp, gs.farms[p].hands.len(), &mut pa[p]);
                                if row[2 * upp] == RAW_MARKET_SENTINEL {
                                    decode_raw_market(
                                        &row[2 * upp + 1..],
                                        &mut pa[p].market,
                                    );
                                } else {
                                    macros::pack_market(gs, p, cfg, &head, &mut pa[p].market);
                                }
                            } else {
                                decode_orders(row, upp, orders);
                                macros::resolve_turn(gs, p, cfg, orders, &head, &mut pa[p]);
                            }
                        }
                        let was_done = gs.done;
                        // `phantom_mask(-1)` is `[false, false]`, i.e. the untouched
                        // Kaggle-exact path, so the default costs one compare per env.
                        sim::step_with_phantom(gs, pa, cfg, phantom_mask(phantom_seats[i]));
                        let newly_done = gs.done && !was_done;

                        // Published BEFORE any auto-reset, so `dones`, `info["step"]`
                        // and `info["final_money"]` describe the terminal state even
                        // though `obs` below describes the fresh episode.
                        // SAFETY: index-disjoint writes into arrays this struct owns.
                        unsafe {
                            let rew = rew_ptr.slice(i * 2, 2);
                            let fm = fm_ptr.slice(i * 2, 2);
                            for p in 0..2 {
                                rew[p] = gs.rewards[p] as f32;
                                fm[p] = if newly_done {
                                    gs.farms[p].money as f32
                                } else {
                                    0.0
                                };
                            }
                            done_ptr.slice(i, 1)[0] = gs.done;
                            st_ptr.slice(i, 1)[0] = gs.step;
                        }

                        // Auto-reset in-closure. Each env's next seed is a pure function
                        // of `(base_seed, episode_idx)`, so the rollout stays
                        // bit-identical whatever order rayon happens to schedule in.
                        if newly_done && auto_reset {
                            *ep += 1;
                            *gs = make_env(cfg, params, base_seeds[i], *ep);
                        }

                        // Published AFTER the auto-reset so it describes exactly the
                        // state `obs` below describes -- on a done step that is the fresh
                        // episode's starting bank, while `final_money` (written above)
                        // still holds the terminal one. Potential-based cash shaping
                        // needs both halves and cannot reconstruct either from the other.
                        // SAFETY: index-disjoint writes into an array this struct owns.
                        unsafe {
                            let mn = mn_ptr.slice(i * 2, 2);
                            for p in 0..2 {
                                mn[p] = gs.farms[p].money as f32;
                            }
                        }

                        for p in 0..2 {
                            // SAFETY: disjoint per (i, p), as in `publish_all`.
                            unsafe {
                                obs_ptr.publish((i * 2 + p) * OBS_LEN, OBS_LEN, scratch, |o| {
                                    encode_with_cfg(gs, p, cfg, o)
                                });
                                mask_ptr.publish(
                                    (i * 2 + p) * MASK_LEN,
                                    MASK_LEN,
                                    scratch,
                                    |m| legal_mask_with_cfg(gs, p, cfg, m),
                                );
                                if let Some(tm) = tm_ptr.as_ref() {
                                    tile_verb_mask_with_cfg(
                                        gs,
                                        p,
                                        cfg,
                                        tm.slice((i * 2 + p) * TILE_MASK_LEN, TILE_MASK_LEN),
                                    );
                                }
                            }
                        }
                    },
                );
        });
        self.total_steps += n as u64;

        PyTuple::new(
            py,
            [
                self.obs.as_any(py),
                self.mask.as_any(py),
                self.rewards.bind(py).as_any().clone(),
                self.dones.bind(py).as_any().clone(),
                self.info.bind(py).as_any().clone(),
            ],
        )
    }

    /// The `PyGame`-shaped action dicts `step` would submit for env `env` **in its
    /// current state**, given one env's action row.
    ///
    /// Debug and parity path only: it runs the same `macros::resolve_turn` the hot path
    /// runs and formats the result as the `{"farmer", "hands", "market"}` dicts the
    /// Kaggle environment takes, which is what lets a test drive `PyVecEnv` and `PyGame`
    /// in lockstep from one action array and compare `state_dict()` (STATUS §7.3).
    /// Allocates freely; do not call it on the hot path.
    ///
    /// `actions` holds `2 * player_action_dim` ints, shaped `(2, player_action_dim)` or
    /// flat.
    fn resolve_actions<'py>(
        &self,
        py: Python<'py>,
        env: usize,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        if env >= self.n_envs {
            return Err(PyIndexError::new_err(format!(
                "env index {env} out of range for n_envs = {}",
                self.n_envs
            )));
        }
        let paw = self.player_action_dim;
        let want = 2 * paw;
        let Some((flat, _)) = flat_ints(actions) else {
            return Err(PyTypeError::new_err(
                "actions must be a C-contiguous numpy array of dtype int64 or int32",
            ));
        };
        if flat.len() != want {
            return Err(PyValueError::new_err(format!(
                "actions holds {} values; expected {want} = 2 players x {paw}",
                flat.len()
            )));
        }
        let mut row = vec![0i64; want];
        flat.copy_into(&mut row);

        let out = PyList::empty(py);
        let mut orders = Vec::with_capacity(MAX_UNITS);
        let mut pa = PlayerAction::pass();
        for p in 0..2 {
            let r = &row[p * paw..(p + 1) * paw];
            let head = decode_market(&r[2 * self.units_per_player..]);
            if r[self.units_per_player] == RAW_ROW_SENTINEL {
                decode_raw_turn(
                    r,
                    self.units_per_player,
                    self.envs[env].farms[p].hands.len(),
                    &mut pa,
                );
                if r[2 * self.units_per_player] == RAW_MARKET_SENTINEL {
                    decode_raw_market(&r[2 * self.units_per_player + 1..], &mut pa.market);
                } else {
                    macros::pack_market(&self.envs[env], p, &self.cfg, &head, &mut pa.market);
                }
            } else {
                decode_orders(r, self.units_per_player, &mut orders);
                macros::resolve_turn(&self.envs[env], p, &self.cfg, &orders, &head, &mut pa);
            }
            out.append(player_action_to_py(py, &pa)?)?;
        }
        Ok(out)
    }

    /// Full state of one env, in the `parity/canon.py` shape (debug path).
    /// Turn the given seat of each env into a **phantom** market-pressure stream.
    ///
    /// `seats` is one `int8` per env: `-1` none (the default and the only Kaggle-exact
    /// setting), `0` or `1`. A phantom seat is not a player. Its unit actions are
    /// ignored, and so are its `HIRE` / `BUY_LAND` / `BUY_SEED` / `BUY_ANIMAL` orders;
    /// its `SELL` fills with no shed and banks nothing, its `BUY_PRODUCT` fills with no
    /// cash and no shed space. Only `market.inventory` moves. Everything else — the
    /// per-unit lockstep against the real seat, the `$1` floor rule, `refresh_prices`,
    /// `town_consume`, `end_of_day` — is unchanged.
    ///
    /// The point is domain-randomised market pressure with exact CRN control: the real
    /// seat trades into the price curve a *recorded* opponent's realised order stream
    /// produced, and that stream is reproducible per group because it no longer depends
    /// on an opponent policy's own state.
    ///
    /// The assignment describes the **env**, not the episode: it persists across
    /// auto-resets until changed.
    fn set_phantom_seats(&mut self, seats: &Bound<'_, PyAny>) -> PyResult<()> {
        let v: Vec<i8> = seats.extract().map_err(|_| {
            PyTypeError::new_err(
                "seats must be a sequence of ints (list or 1-D int8 numpy array)",
            )
        })?;
        if v.len() != self.n_envs {
            return Err(PyValueError::new_err(format!(
                "seats has {} entries, expected n_envs = {}",
                v.len(),
                self.n_envs
            )));
        }
        for (i, seat) in v.iter().enumerate() {
            if !(-1..=1).contains(seat) {
                return Err(PyValueError::new_err(format!(
                    "seats[{i}] = {seat} is not -1 (none), 0 or 1"
                )));
            }
        }
        self.phantom_seats = v;
        Ok(())
    }

    /// The current per-env phantom seat assignment, as a fresh `int8` list.
    fn phantom_seats(&self) -> Vec<i8> {
        self.phantom_seats.clone()
    }

    fn state_dict<'py>(&self, py: Python<'py>, env: usize) -> PyResult<Bound<'py, PyDict>> {
        if env >= self.n_envs {
            return Err(PyIndexError::new_err(format!(
                "env index {env} out of range for n_envs = {}",
                self.n_envs
            )));
        }
        state_dict_of(py, &self.envs[env])
    }

    fn render(&self, env: usize) -> PyResult<String> {
        if env >= self.n_envs {
            return Err(PyIndexError::new_err("env index out of range"));
        }
        Ok(render_state(&self.envs[env]))
    }

    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        config_to_py(py, &self.cfg, &self.params)
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.n_envs
    }

    /// Values the action array carries per env: `2 * player_action_dim`.
    #[getter]
    fn action_dim(&self) -> usize {
        2 * self.player_action_dim
    }

    /// Values one player's action row carries: `2 * units_per_player + 21`.
    #[getter]
    fn player_action_dim(&self) -> usize {
        self.player_action_dim
    }

    #[getter]
    fn units_per_player(&self) -> usize {
        self.units_per_player
    }

    /// Envs per rayon chunk (`with_min_len`) — a pure **memory-behaviour** knob.
    ///
    /// It cannot change a single returned value: every chunk boundary falls between envs,
    /// the closure is indexed by env, and each env's outcome depends only on its own
    /// state, its own action row and its own seed. `chunk_envs_does_not_change_the_result`
    /// in `tests/test_pyo3_surface.py` asserts a rollout is bit-identical across the range,
    /// including a chunk larger than `n_envs`.
    ///
    /// What it changes is how many workers stream into the shared observation plane at
    /// once, and a batched step is ~85 % exactly that memcpy (STATUS 3.3). Measured on this
    /// box — 168 rayon workers, two sockets, random macro policy with `hire = 2`, one fresh
    /// process per cell, min of 3 runs inside each, **and the whole table run twice** because
    /// this regime is noisy. Both runs, steps/s:
    ///
    /// | `chunk_envs` | 512 envs | 4,096 | 16,384 | 65,536 |
    /// |---|---|---|---|---|
    /// | **1 (default)** | **749k / 718k** | 404k / 407k | 589k / 606k | 1.48M / 1.56M |
    /// | 8 | 362k / 445k | 400k / 433k | 829k / 559k | 1.55M / 1.55M |
    /// | 16 | 393k / 416k | 421k / 451k | 599k / 607k | **1.56M / 1.64M** |
    /// | 64 | 436k / 416k | 454k / 458k | 607k / 589k | 1.55M / 1.55M |
    /// | 128 | 243k / 312k | **456k / 481k** | 567k / 601k | 1.53M / 1.46M |
    ///
    /// Two of those columns say something reproducible and two do not, which is the useful
    /// part:
    ///
    /// * **At 4,096 envs, 64–128 is reproducibly +12–19 %** over the default. Worth setting.
    /// * **At 512 envs, chunking is a 2–3x regression.** 128 leaves four chunks for 168
    ///   workers, so most of them sit idle. This is the one way to hold the knob wrong.
    /// * At 16,384 and 65,536 the spread is inside the ±20 % run-to-run noise of this box;
    ///   do not read the 829k cell as real.
    ///
    /// **The hard rule: keep `n_envs / chunk_envs` at or above the rayon worker count**
    /// (168 here, `available_parallelism`). Nothing caps this for you, because a silent cap
    /// would make the knob lie about what it did.
    #[getter]
    fn chunk_envs(&self) -> usize {
        self.chunk_envs
    }

    #[setter]
    fn set_chunk_envs(&mut self, v: usize) -> PyResult<()> {
        if v == 0 {
            return Err(PyValueError::new_err("chunk_envs must be >= 1"));
        }
        self.chunk_envs = v;
        Ok(())
    }

    /// Dtype of the `obs` and `mask` planes: `"float16"` (default) or `"float32"`.
    ///
    /// **This is the policy's input dtype, so it is a run-level decision, not a knob to
    /// sweep**: checkpoints are not comparable across a change to it. It is exposed so an
    /// f32 A/B is possible, and so a training run can record what it trained against.
    ///
    /// Why f16 is the default (measured, `STATUS.md` §3.3/§7.2): a batched vec-step is
    /// ~85 % memcpy into the shared observation plane, so the element width *is* the
    /// binding cost of collection — and it halves the host→device transfer and the
    /// rollout buffer at the same time. Numerically it costs nothing that matters: every
    /// slot is normalised into `[-4, 4]` (`OBS_LAYOUT.md` §1), where f16 quantises to at
    /// worst `2^-9 = 0.002` absolute and `2^-11 = 4.9e-4` relative; the mask's `{0.0,
    /// 1.0}` alphabet, every one-hot and every boolean flag are **exact**.
    /// `parity/f16_obs_error.py` measures the realised per-slot error over full episodes.
    ///
    /// The conversion is round-to-nearest-ties-to-even, so the published plane is
    /// bit-identical to `np.float16(f32_plane)` — which is asserted, not assumed
    /// (`parity/diff_vecenv.py`, `tests/test_pyo3_surface.py`).
    ///
    /// `rewards` and `info["final_money"]` are deliberately **not** covered: they carry
    /// money, which reaches ~1e5 and would quantise to steps of 64 in f16 (and saturate
    /// past 65504), and they are 2 floats per env against the observation's 13 440.
    #[getter]
    fn obs_dtype(&self) -> &'static str {
        self.obs_dtype.name()
    }

    /// Bytes the `obs` + `mask` planes hold in total, i.e. what `step` republishes every
    /// call: `n_envs * 2 * (6720 + 646) * itemsize`.
    #[getter]
    fn published_bytes(&self) -> usize {
        self.n_envs * 2 * (OBS_LEN + MASK_LEN) * self.obs_dtype.itemsize()
    }

    /// Trailing axis of the observation plane ([`OBS_LEN`] = 6720).
    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Trailing axis of the legality-mask plane ([`MASK_LEN`] = 646).
    #[getter]
    fn mask_dim(&self) -> usize {
        self.mask_dim
    }

    /// The optional `(n_envs, 2, TILE_MASK_LEN)` u8 tile-verb plausibility plane, or
    /// `None` unless the env was built with `tile_mask=True`. Env-owned, overwritten by
    /// every `reset`/`step`/`warmup_pass`, and also reachable as `info["tile_mask"]`.
    #[getter]
    fn tile_mask<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray3<u8>>> {
        self.tile_mask.as_ref().map(|a| a.bind(py).clone())
    }

    /// Trailing axis of the tile-verb plane ([`TILE_MASK_LEN`] = 3800).
    #[getter]
    fn tile_mask_dim(&self) -> usize {
        TILE_MASK_LEN
    }

    /// The `(n_envs, 2)` f32 per-seat bank of the state `obs` currently describes — the same
    /// array as `info["money"]`, exposed so a caller can read it after `reset` / `warmup_pass`
    /// before any `step` has returned an `info` dict. Env-owned, overwritten every call.
    #[getter]
    fn money<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        self.money.bind(py).clone()
    }

    /// The `info` dict `step` returns (`final_money`, `money`, `step`, and `tile_mask` when
    /// requested), reachable before the first `step`. Its arrays are env-owned and reused.
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.info.bind(py).clone()
    }

    /// `(n_tokens, n_features)` the flat observation reshapes to: `(140, 48)`.
    #[getter]
    fn obs_shape(&self) -> (usize, usize) {
        (N_TOKENS, N_FEATURES)
    }

    /// `(max_units, n_verbs)` the flat mask reshapes to: `(17, 38)`.
    #[getter]
    fn mask_shape(&self) -> (usize, usize) {
        (MAX_UNITS, N_VERBS)
    }

    #[getter]
    fn obs_version(&self) -> u32 {
        OBS_VERSION_EXPOSED
    }

    #[getter]
    fn action_version(&self) -> u32 {
        ACTION_VERSION_EXPOSED
    }

    #[getter]
    fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// The resolved seed each env's *current* episode is running under.
    #[getter]
    fn seeds(&self) -> Vec<i64> {
        self.envs.iter().map(|g| g.seed).collect()
    }

    #[getter]
    fn episode_indices(&self) -> Vec<u64> {
        self.episode_idx.clone()
    }

    #[getter]
    fn auto_reset(&self) -> bool {
        self.auto_reset
    }

    fn __repr__(&self) -> String {
        format!(
            "PyVecEnv(n_envs={}, obs_dim={}, mask_dim={}, action_dim={}, \
             units_per_player={}, auto_reset={}, obs_dtype={})",
            self.n_envs,
            self.obs_dim,
            self.mask_dim,
            2 * self.player_action_dim,
            self.units_per_player,
            self.auto_reset,
            self.obs_dtype.name()
        )
    }
}

impl PyVecEnv {
    /// Copy the action array into `act_scratch`, laid out `(n_envs, 2,
    /// player_action_dim)`.
    ///
    /// The copy is deliberate: it happens with the GIL held, so no rayon worker ever
    /// reads Python-owned memory. `act_scratch` is allocated once in the constructor and
    /// is exactly the right length, so this is a `memcpy` and nothing more.
    fn load_actions(&mut self, actions: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.n_envs;
        let paw = self.player_action_dim;
        let want = n * 2 * paw;
        debug_assert_eq!(self.act_scratch.len(), want);

        let Some((flat, shape)) = flat_ints(actions) else {
            return Err(PyTypeError::new_err(format!(
                "actions must be a C-contiguous numpy array of dtype int64 or int32 \
                 with shape (n_envs, 2, {paw}) or (n_envs, {}); one player's row is \
                 [{} targets | {} verbs | {} market heads]",
                2 * paw,
                self.units_per_player,
                self.units_per_player,
                N_MARKET_HEADS
            )));
        };
        // A rank-1 array is accepted flat; anything with a leading axis must index envs.
        if flat.len() != want || (shape.len() > 1 && shape[0] != n) {
            return Err(PyValueError::new_err(format!(
                "actions.shape {shape:?} holds {} values; expected {want} = n_envs({n}) \
                 x 2 players x player_action_dim({paw})",
                flat.len()
            )));
        }
        flat.copy_into(&mut self.act_scratch);
        Ok(())
    }
}

// ===========================================================================
// module-level helpers (mostly so tests can poke the traps directly)
// ===========================================================================

/// `market_price(item, inventory)` — trap 1, banker's rounding included.
#[pyfunction]
#[pyo3(signature = (item, inventory, params=None))]
fn market_price(
    item: &str,
    inventory: i64,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<i64> {
    let p = Product::from_name(item)
        .ok_or_else(|| PyValueError::new_err(format!("unknown product {item:?}")))?;
    let table = match params {
        None => MARKET_PARAMS,
        Some(v) => market_params_from_py(v)?,
    };
    Ok(sim::market_price(p, inventory, &table))
}

/// Python's builtin `round(float)` — half to EVEN (trap 1).
#[pyfunction]
fn py_round(x: f64) -> i64 {
    kaggri_core::pyrng::py_round(x)
}

/// The `kaggriculture.json` defaults, as this port sees them.
#[pyfunction]
fn default_config(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    config_to_py(py, &Config::default(), &MARKET_PARAMS)
}

/// The CPython `random.Random((seed * 1_000_003) ^ day)` stream, for RNG spot checks.
#[pyfunction]
fn rng_stream(seed: i64, day: i64, n: usize) -> Vec<f64> {
    let mut r = kaggri_core::pyrng::PyRandom::for_day(seed, day);
    (0..n).map(|_| r.random()).collect()
}

/// `random.Random((seed * 1_000_003) ^ day).choice(sorted(SHOPS))` index after `skip`
/// `random()` draws.
#[pyfunction]
fn rng_choice_index(seed: i64, day: i64, skip: usize, n: usize) -> usize {
    let mut r = kaggri_core::pyrng::PyRandom::for_day(seed, day);
    for _ in 0..skip {
        r.random();
    }
    r.choice_index(n)
}

/// Every number the training code should assert at startup (`OBS_LAYOUT.md` §11).
///
/// Mirrors `kaggri_core::obs::obs_dims()` field for field, plus the action-array layout
/// this binding adds on top of it.
#[pyfunction]
fn obs_dims_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = obs_dims();
    let out = PyDict::new(py);
    out.set_item("version", d.version)?;
    out.set_item("n_tokens", d.n_tokens)?;
    out.set_item("n_features", d.n_features)?;
    out.set_item("obs_len", d.obs_len)?;
    out.set_item("n_header", d.n_header)?;
    out.set_item("off_global", d.off_global)?;
    out.set_item("n_global", d.n_global)?;
    out.set_item("off_tile", d.off_tile)?;
    out.set_item("n_tile", d.n_tile)?;
    out.set_item("off_unit", d.off_unit)?;
    out.set_item("n_unit", d.n_unit)?;
    out.set_item("off_product", d.off_product)?;
    out.set_item("n_product", d.n_product)?;
    out.set_item("off_opp", d.off_opp)?;
    out.set_item("n_opp", d.n_opp)?;
    out.set_item("off_private", d.off_private)?;
    out.set_item("n_private", d.n_private)?;
    out.set_item("off_scratch", d.off_scratch)?;
    out.set_item("n_scratch", d.n_scratch)?;
    out.set_item("n_targets", d.n_targets)?;
    out.set_item("n_verbs", d.n_verbs)?;
    out.set_item("max_units", d.max_units)?;
    out.set_item("mask_len", d.mask_len)?;
    out.set_item("n_market_heads", d.n_market_heads)?;
    out.set_item("market_logits", d.market_logits)?;
    out.set_item("action_version", ACTION_VERSION_EXPOSED)?;
    // Not a layout dimension, but part of the same startup contract: the dtype the
    // observation and mask planes are published in unless the caller overrides it.
    out.set_item("obs_dtype", DEFAULT_OBS_DTYPE.name())?;
    Ok(out)
}

/// Sanity helper: is a quadrant unlocked-at-start for `(x, y)`?
#[pyfunction]
fn quadrant_name(x: i64, y: i64, board_size: usize) -> PyResult<String> {
    if board_size == 0 {
        return Err(PyRuntimeError::new_err("board_size must be > 0"));
    }
    Ok(quadrant_of(x, y, board_size).as_str().to_string())
}

// ===========================================================================
// The submission path: a Kaggle observation dict -> GameState -> obs / mask
// ===========================================================================
//
// The training interface ([`PyVecEnv`]) encodes from a [`GameState`] it owns. The real
// competition harness never hands an agent a `GameState`: it hands it the observation
// dict `kaggriculture.json` describes (`player`, `farms`, `private`, `market`, `town`,
// `day`, `hour`, plus the framework's `step` and `remainingOverageTime`), wrapped in
// `kaggle_environments.utils.Struct` — a `dict` subclass, so everything below treats it
// as a mapping and falls back to attribute access.
//
// This section reconstructs exactly as much of a `GameState` as `obs::encode_with_cfg`,
// `obs::legal_mask_with_cfg` and `macros::resolve_turn` read, so the submitted agent sees
// the **bit-identical** input plane it was trained on. `parity/diff_obs_submission.py`
// asserts that, bit for bit, against `PyVecEnv` over full episodes.
//
// ## What is reachable from the observation, and why that is enough
//
// The observation carries **both farms' public state** (`farms` is `"shared": true` in
// `kaggriculture.json`: tiles, money, farmer/hand positions, unlocked quadrants,
// `hires_today`) but only **our own** `private` (shed, seeds, per-unit inventories).
//
// `obs::encode_with_cfg` reads `gs.privates[player]` and nothing else out of `privates`
// (verified by inspection: the only two `gs.privates[...]` expressions in `obs.rs` are
// `gs.privates[player]`, in `encode_with_cfg` and in `legal_mask_with_cfg`), and
// `macros::resolve_turn` / `macros::pack_market` likewise touch only
// `gs.privates[player]` and `gs.farms[player]`. So the opponent's `Private` is never
// read, and this reconstruction leaves it at [`Private::default`] — that is a
// *provable* no-op, not a papered-over zero.
//
// Neither `encode` nor `legal_mask` nor `resolve_turn` reads `gs.seed`, `gs.done` or
// `gs.rewards`, which is what makes the reconstruction possible at all: the framework's
// `resolve_episode_seed` deliberately scrubs the episode seed out of the configuration
// before any agent sees it (`configuration["seed"] is None` at act time), so anything
// that depended on it would be unreconstructable at submission time.

/// `obs[key]` for a mapping, falling back to `obs.key` for an attribute holder.
///
/// `Struct` satisfies both; a plain dict only the first; a namespace object only the
/// second. Returns `None` for a missing key rather than raising, so the caller decides
/// whether the field is required.
fn obs_get<'py>(o: &Bound<'py, PyAny>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(d) = o.cast::<PyDict>() {
        if let Some(v) = d.get_item(key)? {
            return Ok(if v.is_none() { None } else { Some(v) });
        }
        return Ok(None);
    }
    match o.getattr(key) {
        Ok(v) => Ok(if v.is_none() { None } else { Some(v) }),
        Err(_) => Ok(None),
    }
}

/// [`obs_get`], but a missing key is an error naming the path that was missing.
fn obs_req<'py>(o: &Bound<'py, PyAny>, key: &str, what: &str) -> PyResult<Bound<'py, PyAny>> {
    obs_get(o, key)?.ok_or_else(|| {
        PyValueError::new_err(format!(
            "observation is missing {what:?} (looked for both obs[{key:?}] and obs.{key})"
        ))
    })
}

/// `int(o[key])` with a default for an absent key.
fn obs_int(o: &Bound<'_, PyAny>, key: &str, what: &str, default: i64) -> PyResult<i64> {
    match obs_get(o, key)? {
        Some(v) => py_int_arg(&v, what),
        None => Ok(default),
    }
}

/// `float(o[key])`; the key is required.
fn obs_float(o: &Bound<'_, PyAny>, key: &str, what: &str) -> PyResult<f64> {
    let v = obs_req(o, key, what)?;
    py_float_arg(&v, what)
}

/// `bool(o[key])` with a default. Python stores these as real bools.
fn obs_bool(o: &Bound<'_, PyAny>, key: &str, default: bool) -> PyResult<bool> {
    match obs_get(o, key)? {
        Some(v) => Ok(v.is_truthy()?),
        None => Ok(default),
    }
}

/// `[x, y]` as a two-element integer sequence.
fn pos_from_py(v: &Bound<'_, PyAny>, what: &str) -> PyResult<[i64; 2]> {
    let n = v.len().map_err(|_| {
        PyTypeError::new_err(format!("{what}: expected a 2-element [x, y] sequence"))
    })?;
    if n != 2 {
        return Err(PyValueError::new_err(format!(
            "{what}: expected a 2-element [x, y] sequence, got length {n}"
        )));
    }
    Ok([
        py_int_arg(&v.get_item(0)?, what)?,
        py_int_arg(&v.get_item(1)?, what)?,
    ])
}

/// One `farm["tiles"][y][x]` value.
///
/// Python's four representations, in the order the environment's own predicates test
/// them: `None` (empty and unlocked), `"LOCKED"`, and a dict that is a weed, a plant, an
/// **occupied** structure (`"animal" in tile`, which must be tested *before* the empty
/// `COOP` / `PASTURE` cases because an occupied structure also carries that `kind`), or
/// an empty structure.
fn tile_from_py(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Tile> {
    if v.is_none() {
        return Ok(Tile::Empty);
    }
    if let Ok(ps) = v.cast::<PyString>() {
        let name = ps.to_str()?;
        if name == "LOCKED" {
            return Ok(Tile::Locked);
        }
        return Err(PyValueError::new_err(format!(
            "{what}: unknown string tile {name:?} (only \"LOCKED\" exists)"
        )));
    }
    let kind = match obs_get(v, "kind")? {
        Some(k) => k.cast::<PyString>()?.to_str()?.to_string(),
        None => String::new(),
    };
    if kind == "WEED" {
        return Ok(Tile::Weed);
    }
    if kind == "PLANT" {
        let crop_name = obs_req(v, "crop", &format!("{what}.crop"))?;
        let crop_name = crop_name.cast::<PyString>()?.to_str()?.to_string();
        let crop = Crop::from_name(&crop_name).ok_or_else(|| {
            PyValueError::new_err(format!("{what}.crop: unknown crop {crop_name:?}"))
        })?;
        return Ok(Tile::Plant(PlantTile {
            crop,
            planted_day: obs_int(v, "planted_day", what, 0)?,
            watered_today: obs_bool(v, "watered_today", false)?,
            consecutive_unwatered: obs_int(v, "consecutive_unwatered", what, 0)?,
            yield_units: obs_int(v, "yield_units", what, 0)?,
            max_lifespan_step: obs_int(v, "max_lifespan_step", what, -1)?,
            fertilized_until_day: obs_int(v, "fertilized_until_day", what, -1)?,
        }));
    }
    // An occupied structure carries `kind` COOP/PASTURE *and* an `animal` key. The
    // animal decides the structure in `Tile::Animal`, so `kind` is redundant here — but
    // a mismatch would mean the observation and `ANIMALS[a]["structure"]` disagree, which
    // cannot happen and is worth refusing rather than silently re-deriving.
    if let Some(a) = obs_get(v, "animal")? {
        let name = a.cast::<PyString>()?.to_str()?.to_string();
        let animal = Animal::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("{what}.animal: unknown animal {name:?}"))
        })?;
        let want = structure_name(animal.data().structure);
        if !kind.is_empty() && kind != want {
            return Err(PyValueError::new_err(format!(
                "{what}: animal {name} lives in a {want} but the tile kind is {kind:?}"
            )));
        }
        return Ok(Tile::Animal(AnimalTile {
            animal,
            placed_day: obs_int(v, "placed_day", what, 0)?,
            yield_units: obs_int(v, "yield_units", what, 0)?,
            consecutive_unfed: obs_int(v, "consecutive_unfed", what, 0)?,
            fed_today: obs_bool(v, "fed_today", false)?,
            cared_today: obs_bool(v, "cared_today", false)?,
            fertilizer_available: obs_bool(v, "fertilizer_available", false)?,
            pending_care_bonus: obs_int(v, "pending_care_bonus", what, 0)?,
        }));
    }
    match kind.as_str() {
        "COOP" => Ok(Tile::Coop),
        "PASTURE" => Ok(Tile::Pasture),
        other => Err(PyValueError::new_err(format!(
            "{what}: unknown tile kind {other:?}"
        ))),
    }
}

/// One `obs["farms"][i]` dict.
///
/// `board_size` comes from the configuration (that is what the interpreter itself uses);
/// a `tiles` grid of any other shape means the configuration and the observation
/// disagree, which would silently shift every tile token, so it is refused.
fn farm_from_py(v: &Bound<'_, PyAny>, board_size: usize, who: &str) -> PyResult<Farm> {
    let tiles_obj = obs_req(v, "tiles", &format!("{who}.tiles"))?;
    let rows = tiles_obj.len()?;
    if rows != board_size {
        return Err(PyValueError::new_err(format!(
            "{who}.tiles has {rows} rows but boardSize is {board_size}; pass the episode \
             configuration so the two agree"
        )));
    }
    let mut tiles = Vec::with_capacity(board_size * board_size);
    for y in 0..board_size {
        let row = tiles_obj.get_item(y)?;
        let cols = row.len()?;
        if cols != board_size {
            return Err(PyValueError::new_err(format!(
                "{who}.tiles[{y}] has {cols} entries but boardSize is {board_size}"
            )));
        }
        for x in 0..board_size {
            let cell = row.get_item(x)?;
            tiles.push(tile_from_py(&cell, &format!("{who}.tiles[{y}][{x}]"))?);
        }
    }

    let hands_obj = obs_req(v, "hands", &format!("{who}.hands"))?;
    let n_hands = hands_obj.len()?;
    let mut hands = Vec::with_capacity(n_hands);
    for h in 0..n_hands {
        let p = hands_obj.get_item(h)?;
        hands.push(pos_from_py(&p, &format!("{who}.hands[{h}]"))?);
    }

    let uq_obj = obs_req(v, "unlocked_quadrants", &format!("{who}.unlocked_quadrants"))?;
    let n_uq = uq_obj.len()?;
    let mut unlocked_quadrants = Vec::with_capacity(n_uq.max(1));
    for k in 0..n_uq {
        let name_obj = uq_obj.get_item(k)?;
        let name = name_obj.cast::<PyString>()?.to_str()?.to_string();
        let q = Quadrant::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{who}.unlocked_quadrants[{k}]: unknown quadrant {name:?}"
            ))
        })?;
        unlocked_quadrants.push(q);
    }

    let farmer_obj = obs_req(v, "farmer", &format!("{who}.farmer"))?;
    Ok(Farm {
        money: obs_float(v, "money", &format!("{who}.money"))?,
        board_size,
        tiles,
        farmer: pos_from_py(&farmer_obj, &format!("{who}.farmer"))?,
        hands,
        unlocked_quadrants,
        hires_today: obs_int(v, "hires_today", &format!("{who}.hires_today"), 0)?,
    })
}

/// `obs["private"]` — shed, seeds and the insertion-ordered per-unit inventories.
///
/// Inventory **order** is load-bearing (trap 2: `_drop_inventories_to_shed` fills the
/// shed in it and discards the remainder, and UNIT slots 19–20 publish its head), and it
/// survives the whole way here: CPython preserves dict insertion order, `structify`
/// rebuilds via `Struct(**{...})`, and `copy.deepcopy` preserves it too. Iterating the
/// mapping in its own order is therefore exactly what the environment would iterate.
fn private_from_py(v: &Bound<'_, PyAny>, n_units: usize) -> PyResult<Private> {
    let mut p = Private::new();

    if let Some(shed) = obs_get(v, "shed")? {
        let d = shed
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("private.shed: expected a dict"))?;
        for (k, n) in d.iter() {
            let name = k.cast::<PyString>()?.to_str()?.to_string();
            let item = ShedItem::from_name(&name).ok_or_else(|| {
                PyValueError::new_err(format!("private.shed: unknown item {name:?}"))
            })?;
            p.shed.set(item, py_int_arg(&n, "private.shed value")?);
        }
    }

    if let Some(seeds) = obs_get(v, "seeds")? {
        let d = seeds
            .cast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("private.seeds: expected a dict"))?;
        for (k, n) in d.iter() {
            let name = k.cast::<PyString>()?.to_str()?.to_string();
            let crop = Crop::from_name(&name).ok_or_else(|| {
                PyValueError::new_err(format!("private.seeds: unknown crop {name:?}"))
            })?;
            p.seeds[crop.idx()] = py_int_arg(&n, "private.seeds value")?;
        }
    }

    p.inventories.clear();
    if let Some(invs) = obs_get(v, "inventories")? {
        let n = invs.len()?;
        for i in 0..n {
            let row = invs.get_item(i)?;
            let d = row.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err(format!("private.inventories[{i}]: expected a dict"))
            })?;
            let mut inv = Inv::new();
            for (k, cnt) in d.iter() {
                let name = k.cast::<PyString>()?.to_str()?.to_string();
                let item = ShedItem::from_name(&name).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "private.inventories[{i}]: unknown item {name:?}"
                    ))
                })?;
                let c = py_int_arg(&cnt, "private.inventories value")?;
                // `_inv_take` deletes a key at 0, so a live inventory never holds one;
                // `add` here appends in the order the mapping yields, which is the order
                // Python built it in.
                inv.add(item, c);
            }
            p.inventories.push(inv);
        }
    }
    // `_farmer_inventory` grows the list on demand, so Python's can be longer than
    // `1 + len(hands)` but never shorter than 1. Pad rather than index out of range.
    while p.inventories.len() < n_units.max(1) {
        p.inventories.push(Inv::new());
    }
    Ok(p)
}

/// `obs["market"]` — per-product inventory and price, plus the resolved curve parameters.
///
/// `_new_market` only stores `market["params"]` when `configuration["marketParams"]`
/// overrode something, so the observation is the *authoritative* source when present and
/// the configuration-derived `fallback` is used otherwise.
fn market_from_py(v: &Bound<'_, PyAny>, fallback: &MarketParams) -> PyResult<Market> {
    let params = match obs_get(v, "params")? {
        Some(p) => {
            // The observation carries the fully resolved table, so every product is
            // present and `_resolve_market_params`' merge has already happened. Parsing
            // it through the same patch path keeps one implementation of the merge.
            market_params_from_py(&p)?
        }
        None => fallback.clone(),
    };
    let mut m = Market::new(params);
    let inv_obj = obs_req(v, "inventory", "market.inventory")?;
    let inv = inv_obj
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("market.inventory: expected a dict"))?;
    for (k, n) in inv.iter() {
        let name = k.cast::<PyString>()?.to_str()?.to_string();
        let item = Product::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("market.inventory: unknown product {name:?}"))
        })?;
        m.inventory[item.idx()] = py_int_arg(&n, "market.inventory value")?;
    }
    let px_obj = obs_req(v, "prices", "market.prices")?;
    let px = px_obj
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("market.prices: expected a dict"))?;
    for (k, n) in px.iter() {
        let name = k.cast::<PyString>()?.to_str()?.to_string();
        let item = Product::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("market.prices: unknown product {name:?}"))
        })?;
        m.prices[item.idx()] = py_int_arg(&n, "market.prices value")?;
    }
    Ok(m)
}

/// `obs["town"]` — the ordered list of unlocked shop instances (duplicates are real).
fn town_from_py(v: &Bound<'_, PyAny>) -> PyResult<Town> {
    let mut t = Town::default();
    let Some(shops) = obs_get(v, "unlocked_shops")? else {
        return Ok(t);
    };
    let n = shops.len()?;
    for i in 0..n {
        let s = shops.get_item(i)?;
        let name = s.cast::<PyString>()?.to_str()?.to_string();
        let shop = Shop::from_name(&name).ok_or_else(|| {
            PyValueError::new_err(format!("town.unlocked_shops[{i}]: unknown shop {name:?}"))
        })?;
        t.unlocked_shops.push(shop);
    }
    Ok(t)
}

/// Reconstruct `(GameState, player)` from one agent's Kaggle observation.
///
/// Everything `obs::encode_with_cfg`, `obs::legal_mask_with_cfg` and
/// `macros::resolve_turn` read is filled from the observation; the opponent's `Private`
/// is left at [`Private::default`] because none of the three ever reads it (see the
/// section header), and `seed` / `done` / `rewards` are left at their neutral values for
/// the same reason.
fn gamestate_from_obs(
    obs: &Bound<'_, PyAny>,
    cfg: &Config,
    params: &MarketParams,
) -> PyResult<(GameState, usize)> {
    let player = obs_int(obs, "player", "observation.player", 0)?;
    if !(0..2).contains(&player) {
        return Err(PyValueError::new_err(format!(
            "observation.player must be 0 or 1, got {player}"
        )));
    }
    let player = player as usize;

    let farms_obj = obs_req(obs, "farms", "observation.farms")?;
    let n_farms = farms_obj.len()?;
    if n_farms != 2 {
        return Err(PyValueError::new_err(format!(
            "observation.farms has {n_farms} entries; this port is the 2-player game"
        )));
    }
    let f0 = farms_obj.get_item(0)?;
    let f1 = farms_obj.get_item(1)?;
    let farm0 = farm_from_py(&f0, cfg.board_size, "farms[0]")?;
    let farm1 = farm_from_py(&f1, cfg.board_size, "farms[1]")?;

    let priv_obj = obs_req(obs, "private", "observation.private")?;
    let n_units = 1 + if player == 0 {
        farm0.hands.len()
    } else {
        farm1.hands.len()
    };
    let mine = private_from_py(&priv_obj, n_units)?;
    let opp_units = 1 + if player == 0 {
        farm1.hands.len()
    } else {
        farm0.hands.len()
    };
    // Hidden, and provably unread. Sized anyway so nothing downstream can index past it.
    let mut theirs = Private::new();
    while theirs.inventories.len() < opp_units {
        theirs.inventories.push(Inv::new());
    }

    let market_obj = obs_req(obs, "market", "observation.market")?;
    let town_obj = obs_req(obs, "town", "observation.town")?;

    let step = obs_int(obs, "step", "observation.step", 0)?;
    let tpd = cfg.turns_per_day_eff();
    // `interpreter` writes both, but default them off `step` the way it computes them so
    // a hand-built observation without them still lands on the right clock.
    let day = obs_int(obs, "day", "observation.day", step / tpd)?;
    let hour = obs_int(obs, "hour", "observation.hour", step % tpd)?;

    let privates = if player == 0 {
        [mine, theirs]
    } else {
        [theirs, mine]
    };
    Ok((
        GameState {
            farms: [farm0, farm1],
            privates,
            market: market_from_py(&market_obj, params)?,
            town: town_from_py(&town_obj)?,
            step,
            day,
            hour,
            seed: 0,
            done: false,
            rewards: [0.0, 0.0],
        },
        player,
    ))
}

// ---------------------------------------------------------------------------
// one published 1-D plane, in either dtype
// ---------------------------------------------------------------------------

/// The single-observation analogue of [`Plane`]: one `(dim,)` array in the published
/// dtype, owned and reused by [`PyEncoder`].
enum Vec1 {
    F32(Py<PyArray1<f32>>),
    F16(Py<PyArray1<F16>>),
}

impl Vec1 {
    fn zeros(py: Python<'_>, n: usize, dt: ObsDtype) -> Self {
        match dt {
            ObsDtype::F32 => Vec1::F32(PyArray1::<f32>::zeros(py, [n], false).unbind()),
            ObsDtype::F16 => Vec1::F16(PyArray1::<F16>::zeros(py, [n], false).unbind()),
        }
    }

    fn as_any<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self {
            Vec1::F32(a) => a.bind(py).as_any().clone(),
            Vec1::F16(a) => a.bind(py).as_any().clone(),
        }
    }

    /// Publish `src` (f32, exactly `len` long) into the array, converting if needed.
    fn write(&self, py: Python<'_>, src: &[f32]) {
        match self {
            Vec1::F32(a) => {
                let b = a.bind(py);
                // SAFETY: `b` is a freshly allocated, C-contiguous `(src.len(),)` array
                // this struct owns exclusively; no Python code can alias it here.
                let dst = unsafe { std::slice::from_raw_parts_mut(b.data(), src.len()) };
                dst.copy_from_slice(src);
            }
            Vec1::F16(a) => {
                let b = a.bind(py);
                // SAFETY: as above; `F16` is `#[repr(transparent)]` over `u16`.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(b.data().cast::<u16>(), src.len()) };
                f32_to_f16_into(src, dst);
            }
        }
    }
}

/// Width of a policy action row: `2 * units + 21`, solved for `units`.
fn units_from_action_len(len: usize) -> PyResult<usize> {
    if len < N_MARKET_HEADS + 2 || (len - N_MARKET_HEADS) % 2 != 0 {
        return Err(PyValueError::new_err(format!(
            "action row holds {len} values; expected 2 * units_per_player + \
             {N_MARKET_HEADS} for some units_per_player in 1..={MAX_UNITS}"
        )));
    }
    let u = (len - N_MARKET_HEADS) / 2;
    if u == 0 || u > MAX_UNITS {
        return Err(PyValueError::new_err(format!(
            "action row implies units_per_player = {u}, which is outside 1..={MAX_UNITS}"
        )));
    }
    Ok(u)
}

/// One action row as `Vec<i64>`, from a numpy int array or any sequence of ints.
fn action_row_from_py(v: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Some((flat, _)) = flat_ints(v) {
        let mut row = vec![0i64; flat.len()];
        flat.copy_into(&mut row);
        return Ok(row);
    }
    v.extract::<Vec<i64>>().map_err(|_| {
        PyTypeError::new_err(
            "actions must be a C-contiguous int64/int32 numpy array or a sequence of ints",
        )
    })
}

/// Resolve one action row against a reconstructed state into the Kaggle action dict.
fn resolve_row<'py>(
    py: Python<'py>,
    gs: &GameState,
    player: usize,
    cfg: &Config,
    row: &[i64],
    orders: &mut Vec<MacroOrder>,
    pa: &mut PlayerAction,
) -> PyResult<Bound<'py, PyDict>> {
    let units = units_from_action_len(row.len())?;
    let head = decode_market(&row[2 * units..]);
    if row[units] == RAW_ROW_SENTINEL {
        decode_raw_turn(row, units, gs.farms[player].hands.len(), pa);
        if row[2 * units] == RAW_MARKET_SENTINEL {
            decode_raw_market(&row[2 * units + 1..], &mut pa.market);
        } else {
            macros::pack_market(gs, player, cfg, &head, &mut pa.market);
        }
    } else {
        decode_orders(row, units, orders);
        macros::resolve_turn(gs, player, cfg, orders, &head, pa);
    }
    player_action_to_py(py, pa)
}

/// A reusable submission-side encoder: parse the configuration once, then encode and
/// decode every turn with no per-turn Python allocation beyond the action dict.
///
/// ```python
/// enc = kaggri.Encoder(config)              # once, at agent construction
/// obs_arr, mask_arr = enc.encode(obs)       # per turn; env-owned, overwritten buffers
/// action = enc.action(obs, action_row)      # the {"farmer","hands","market"} dict
/// ```
///
/// `encode` returns the **same two arrays** every call, overwritten in place, exactly as
/// [`PyVecEnv`] does. Copy what must outlive the turn.
#[pyclass(module = "kaggri", name = "Encoder")]
pub struct PyEncoder {
    cfg: Config,
    params: MarketParams,
    obs_dtype: ObsDtype,
    obs: Vec1,
    mask: Vec1,
    obs_scratch: Vec<f32>,
    mask_scratch: Vec<f32>,
    tile_mask: Py<PyArray1<u8>>,
    orders: Vec<MacroOrder>,
    pa: PlayerAction,
}

#[pymethods]
impl PyEncoder {
    /// `Encoder(config=None, obs_dtype=None)`.
    ///
    /// `config` is the episode configuration in the same camelCase form
    /// [`PyVecEnv`] takes — pass the framework's `configuration` straight through, minus
    /// any key this port does not know. `obs_dtype` defaults to the dtype `PyVecEnv`
    /// publishes, which is the dtype the policy was trained on; overriding it is only
    /// correct if the run overrode it too.
    #[new]
    #[pyo3(signature = (config=None, obs_dtype=None))]
    fn new(
        py: Python<'_>,
        config: Option<&Bound<'_, PyDict>>,
        obs_dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dtype = ObsDtype::from_py(py, obs_dtype)?;
        let (cfg, params, _) = config_from_py(config)?;
        Ok(PyEncoder {
            cfg,
            params,
            obs_dtype: dtype,
            obs: Vec1::zeros(py, OBS_LEN, dtype),
            mask: Vec1::zeros(py, MASK_LEN, dtype),
            obs_scratch: vec![0f32; OBS_LEN],
            mask_scratch: vec![0f32; MASK_LEN],
            tile_mask: PyArray1::<u8>::zeros(py, [TILE_MASK_LEN], false).unbind(),
            orders: Vec::with_capacity(MAX_UNITS),
            pa: PlayerAction::pass(),
        })
    }

    /// `(obs, mask)` for the player the observation belongs to.
    ///
    /// Bit-identical to the `[i, p]` slice `PyVecEnv` publishes for the same underlying
    /// state — `parity/diff_obs_submission.py` asserts exactly that over full episodes.
    fn encode<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        encode_with_cfg(&gs, player, &self.cfg, &mut self.obs_scratch);
        legal_mask_with_cfg(&gs, player, &self.cfg, &mut self.mask_scratch);
        self.obs.write(py, &self.obs_scratch);
        self.mask.write(py, &self.mask_scratch);
        PyTuple::new(py, [self.obs.as_any(py), self.mask.as_any(py)])
    }

    /// The `(TILE_MASK_LEN,)` u8 tile-verb plausibility plane for this observation —
    /// bit-identical to `PyVecEnv(..., tile_mask=True).tile_mask[i, p]` for the same
    /// state. Env-owned, overwritten on the next call.
    fn tile_mask<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        let arr = self.tile_mask.bind(py);
        // SAFETY: this struct owns the array and holds the GIL.
        let out = unsafe { arr.as_slice_mut()? };
        tile_verb_mask_with_cfg(&gs, player, &self.cfg, out);
        Ok(arr.clone())
    }

    /// `(obs, mask, tile_mask)` in one reconstruction instead of two.
    fn encode_full<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        encode_with_cfg(&gs, player, &self.cfg, &mut self.obs_scratch);
        legal_mask_with_cfg(&gs, player, &self.cfg, &mut self.mask_scratch);
        self.obs.write(py, &self.obs_scratch);
        self.mask.write(py, &self.mask_scratch);
        {
            let arr = self.tile_mask.bind(py);
            // SAFETY: this struct owns the array and holds the GIL.
            let out = unsafe { arr.as_slice_mut()? };
            tile_verb_mask_with_cfg(&gs, player, &self.cfg, out);
        }
        PyTuple::new(
            py,
            [self.obs.as_any(py), self.mask.as_any(py), self.tile_mask.bind(py).as_any().clone()],
        )
    }

    /// Turn a policy action row into the Kaggle action dict for this observation.
    ///
    /// `actions` is `[target * U | verb * U | market * 21]` — exactly one player's slice
    /// of what `Policy.act` returns and what [`PyVecEnv::step`] consumes. `U` is inferred
    /// from the row length. The emitted `"hands"` list is always exactly as long as the
    /// live crew, because a phantom hand slot would still count towards the interpreter's
    /// atomic-`PLANT` demand.
    fn action<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        let row = action_row_from_py(actions)?;
        resolve_row(
            py,
            &gs,
            player,
            &self.cfg,
            &row,
            &mut self.orders,
            &mut self.pa,
        )
    }

    /// `(obs, mask, action)` in one reconstruction instead of two.
    fn encode_and_action<'py>(
        &mut self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
        actions: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        encode_with_cfg(&gs, player, &self.cfg, &mut self.obs_scratch);
        legal_mask_with_cfg(&gs, player, &self.cfg, &mut self.mask_scratch);
        self.obs.write(py, &self.obs_scratch);
        self.mask.write(py, &self.mask_scratch);
        let row = action_row_from_py(actions)?;
        let act = resolve_row(
            py,
            &gs,
            player,
            &self.cfg,
            &row,
            &mut self.orders,
            &mut self.pa,
        )?;
        PyTuple::new(
            py,
            [self.obs.as_any(py), self.mask.as_any(py), act.into_any()],
        )
    }

    /// The reconstructed state in the `parity/canon.py` shape, for debugging a mismatch.
    ///
    /// `privates[1 - player]` is a placeholder: the opponent's shed, seeds and
    /// inventories are not in the observation and nothing in the encode, mask or resolve
    /// path reads them.
    fn state_dict<'py>(
        &self,
        py: Python<'py>,
        obs: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (gs, player) = gamestate_from_obs(obs, &self.cfg, &self.params)?;
        let d = state_dict_of(py, &gs)?;
        d.set_item("player", player)?;
        d.set_item("opponent_private_is_placeholder", true)?;
        Ok(d)
    }

    /// The configuration this encoder parsed, as `PyVecEnv.config()` reports it.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        config_to_py(py, &self.cfg, &self.params)
    }

    /// Dtype of the two published planes: `"float16"` or `"float32"`.
    #[getter]
    fn obs_dtype(&self) -> &'static str {
        self.obs_dtype.name()
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_LEN
    }

    #[getter]
    fn mask_dim(&self) -> usize {
        MASK_LEN
    }

    fn __repr__(&self) -> String {
        format!(
            "kaggri.Encoder(obs_dtype={:?}, obs_dim={}, mask_dim={}, boardSize={}, \
             episodeSteps={}, turnsPerDay={}, shedCapacity={})",
            self.obs_dtype.name(),
            OBS_LEN,
            MASK_LEN,
            self.cfg.board_size,
            self.cfg.episode_steps,
            self.cfg.turns_per_day,
            self.cfg.shed_capacity,
        )
    }
}

/// `(obs, mask)` for one Kaggle observation dict, freshly allocated.
///
/// The one-shot form of [`PyEncoder::encode`]. Bit-identical to the `[i, p]` slice
/// `PyVecEnv` publishes for the same underlying state. Prefer [`PyEncoder`] in an agent:
/// it parses the configuration once instead of once per turn.
#[pyfunction]
#[pyo3(signature = (obs, config=None, obs_dtype=None))]
fn encode_observation<'py>(
    py: Python<'py>,
    obs: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyDict>>,
    obs_dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    let dtype = ObsDtype::from_py(py, obs_dtype)?;
    let (cfg, params, _) = config_from_py(config)?;
    let (gs, player) = gamestate_from_obs(obs, &cfg, &params)?;
    let mut o = vec![0f32; OBS_LEN];
    let mut m = vec![0f32; MASK_LEN];
    encode_with_cfg(&gs, player, &cfg, &mut o);
    legal_mask_with_cfg(&gs, player, &cfg, &mut m);
    let oa = Vec1::zeros(py, OBS_LEN, dtype);
    let ma = Vec1::zeros(py, MASK_LEN, dtype);
    oa.write(py, &o);
    ma.write(py, &m);
    PyTuple::new(py, [oa.as_any(py), ma.as_any(py)])
}

/// The Kaggle action dict for one policy action row against one observation dict.
///
/// The one-shot form of [`PyEncoder::action`].
#[pyfunction]
#[pyo3(signature = (obs, actions, config=None))]
fn observation_action<'py>(
    py: Python<'py>,
    obs: &Bound<'_, PyAny>,
    actions: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (cfg, params, _) = config_from_py(config)?;
    let (gs, player) = gamestate_from_obs(obs, &cfg, &params)?;
    let row = action_row_from_py(actions)?;
    let mut orders = Vec::with_capacity(MAX_UNITS);
    let mut pa = PlayerAction::pass();
    resolve_row(py, &gs, player, &cfg, &row, &mut orders, &mut pa)
}

/// The `GameState` reconstructed from one observation dict, in the `canon.py` shape.
///
/// A debugging and parity aid: comparing this against `PyVecEnv.state_dict(i)` localises
/// a field-mapping error to a field, instead of to one of 6 720 observation slots.
#[pyfunction]
#[pyo3(signature = (obs, config=None))]
fn state_from_observation<'py>(
    py: Python<'py>,
    obs: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (cfg, params, _) = config_from_py(config)?;
    let (gs, player) = gamestate_from_obs(obs, &cfg, &params)?;
    let d = state_dict_of(py, &gs)?;
    d.set_item("player", player)?;
    d.set_item("opponent_private_is_placeholder", true)?;
    Ok(d)
}

#[pymodule]
fn kaggri(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_class::<PyGame>()?;
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(market_price, m)?)?;
    m.add_function(wrap_pyfunction!(py_round, m)?)?;
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    m.add_function(wrap_pyfunction!(rng_stream, m)?)?;
    m.add_function(wrap_pyfunction!(rng_choice_index, m)?)?;
    m.add_function(wrap_pyfunction!(quadrant_name, m)?)?;
    m.add_function(wrap_pyfunction!(obs_dims_dict, m)?)?;
    // --- submission path: a Kaggle observation dict -> obs / mask / action dict ---
    m.add_class::<PyEncoder>()?;
    m.add_function(wrap_pyfunction!(encode_observation, m)?)?;
    m.add_function(wrap_pyfunction!(observation_action, m)?)?;
    m.add_function(wrap_pyfunction!(state_from_observation, m)?)?;
    // --- RL interface (OBS_LAYOUT.md) ---
    m.add("OBS_VERSION", OBS_VERSION_EXPOSED)?;
    m.add("ACTION_VERSION", ACTION_VERSION_EXPOSED)?;
    m.add("OBS_DIM", OBS_LEN)?;
    m.add("MASK_DIM", MASK_LEN)?;
    m.add("TILE_MASK_DIM", TILE_MASK_LEN)?;
    // The published dtype of the obs/mask planes. `OBS_DTYPE` is what `PyVecEnv` uses
    // when `obs_dtype` is not given; `OBS_DTYPES` is what it accepts. Changing the
    // default changes the policy's input dtype — see `PyVecEnv.obs_dtype`.
    m.add("OBS_DTYPE", DEFAULT_OBS_DTYPE.name())?;
    m.add("OBS_DTYPES", vec![ObsDtype::F16.name(), ObsDtype::F32.name()])?;
    m.add("N_TOKENS", N_TOKENS)?;
    m.add("N_FEATURES", N_FEATURES)?;
    m.add("MAX_UNITS", MAX_UNITS)?;
    m.add("MAX_HANDS", MAX_HANDS)?;
    m.add("N_TARGETS", N_TARGETS)?;
    m.add("N_VERBS", N_VERBS)?;
    m.add("RAW_ACTION_VERSION", RAW_ACTION_VERSION)?;
    m.add("RAW_UNIT_ACTIONS", RAW_UNIT_ACTIONS)?;
    m.add("RAW_ROW_SENTINEL", RAW_ROW_SENTINEL)?;
    m.add("MARKET_ACTION_VERSION", MARKET_ACTION_VERSION)?;
    m.add("RAW_MARKET_ACTIONS", RAW_MARKET_ACTIONS)?;
    m.add("RAW_MARKET_SLOTS", RAW_MARKET_SLOTS)?;
    m.add("RAW_MARKET_SENTINEL", RAW_MARKET_SENTINEL)?;
    m.add("N_MARKET_HEADS", N_MARKET_HEADS)?;
    m.add("MARKET_LOGITS", MARKET_LOGITS)?;
    m.add("MARKET_HEAD_SIZES", MARKET_HEAD_SIZES.to_vec())?;
    m.add("MAX_HIRE_PER_TURN", MAX_HIRE_PER_TURN)?;
    m.add("VERB_NAMES", VERBS.map(verb_name).to_vec())?;
    m.add("QTY_BUCKETS", QTY_BUCKETS.to_vec())?;
    m.add("SELL_FRACS", SELL_FRACS.to_vec())?;
    m.add("PICKUP_ITEMS", PICKUP_ITEMS.map(|i| i.as_str()).to_vec())?;
    m.add("PICKUP_QTYS", PICKUP_QTYS.to_vec())?;
    m.add("DEPOSIT_QTY", DEPOSIT_QTY)?;
    // --- game vocabulary ---
    m.add("PRODUCTS", PRODUCTS.map(|p| p.as_str()).to_vec())?;
    m.add("SHED_ITEMS", SHED_ITEMS.map(|p| p.as_str()).to_vec())?;
    m.add("CROPS", CROP_LIST.map(|p| p.as_str()).to_vec())?;
    m.add("ANIMALS", ANIMAL_LIST.map(|p| p.as_str()).to_vec())?;
    m.add("N_PRODUCTS", N_PRODUCTS)?;
    Ok(())
}
