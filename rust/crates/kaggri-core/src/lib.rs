//! `kaggri-core` — a bit-exact Rust port of the Kaggle `kaggriculture` environment.
//!
//! Ground truth: `kaggle_environments/envs/kaggriculture/kaggriculture.py`
//! (kaggle-environments 1.32.7). Every module here is a transliteration of a part of
//! that file, and where a construct looks clumsy it is because CPython's behaviour is
//! being reproduced deliberately:
//!
//! | module      | ports |
//! |-------------|-------|
//! | [`consts`]  | `CROPS`, `ANIMALS`, `PRODUCTS`, `MARKET_PARAMS`, `SHOPS`, `LAND_*`, the geometry helpers |
//! | [`pyrng`]   | CPython's `random.Random` (MT19937) and `round()` |
//! | [`state`]   | `Config`, `GameState`, `Farm`, `Tile`, `Private`, `Market`, `Town` |
//! | [`actions`] | `_parse_order` and the `_apply_unit_action` op dispatch |
//! | [`sim`]     | `interpreter()` and every `_helper` it calls |
//!
//! The parity traps that shape these modules (banker's rounding, insertion-ordered
//! inventories, the state-dependent RNG stream, f64 money, the `LOCKED` guard sitting
//! *after* the shed ops, the atomic-`PLANT` rule, the market lockstep, the decay
//! cadence and the two daily-refresh orderings) are documented at the code that
//! implements them. Do not "simplify" any of them without re-running the parity suite.
//!
//! Two things the Python side does live *outside* [`state::Config`], because
//! `PORT_SPEC` pins `Config`'s field list:
//!
//! * `configuration["marketParams"]` — build the table with
//!   [`consts::MarketParams::resolve`] and hand it to
//!   [`state::GameState::new_with_params`] (or mutate `gs.market.params` in place).
//! * `resolve_episode_seed` — the port takes an explicit `cfg.seed`. Callers that want
//!   Python's "draw a random 31-bit int when unset" behaviour must draw it themselves
//!   and record it, exactly as `env.info["seed"]` does.

pub mod actions;
pub mod consts;
pub mod pyrng;
pub mod raw;
pub mod sim;
pub mod state;

pub mod macros;
pub mod obs;

pub use actions::{MarketOrder, PlayerAction, Tok, UnitOp};
pub use consts::{
    Animal, AnimalData, Crop, CropData, Dir, ItemParams, MarketParams, Product, Quadrant, Shape,
    ShedItem, Shop, Structure, ANIMALS, CROPS, LAND_ORDER, LAND_PRICES, MARKET_I0, MARKET_PARAMS,
    MAX_SHOP_INSTANCES, PRICE_FLOOR, PRODUCTS, SHOPS_SORTED, TOWN_CENTER_PRODUCTS,
};
pub use macros::{
    pack_market, resolve_turn, MacroOrder, MarketHead, Verb, MAX_HANDS, MAX_UNITS,
    N_MARKET_HEADS, N_TARGETS, N_VERBS,
};
pub use obs::{
    encode, encode_with_cfg, legal_mask, legal_mask_with_cfg, obs_dims, tile_verb_mask,
    tile_verb_mask_with_cfg, ObsDims, MASK_LEN, OBS_LEN, OBS_VERSION, TILE_MASK_LEN,
};
pub use sim::{market_price, step, step_with_phantom, NO_PHANTOM};
pub use state::{AnimalTile, Config, Farm, GameState, Inv, Market, PlantTile, Private, Shed, Tile, Town};
