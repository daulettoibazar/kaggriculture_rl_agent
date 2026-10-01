//! RL observation encoding and the optional legality mask.
//!
//! `OBS_LAYOUT.md` at the repository root is the normative description of this layout and
//! is what the training code is written against. This module is its implementation; the
//! slot comments below are the same numbers in the same order.
//!
//! # Shape
//!
//! A fixed `[N_TOKENS, N_FEATURES] = [140, 48]` `f32` matrix, written row-major into a
//! caller-provided flat buffer. Nothing is allocated: every intermediate is a stack array
//! and the only pass over the board is a single scan per farm.
//!
//! | tokens | group | what |
//! |---|---|---|
//! | 1 | `GLOBAL` | clock, both bankrolls, both crews, town cadence |
//! | 100 | `TILE` | the player's own farm, one token per tile, full detail |
//! | 17 | `UNIT` | farmer + up to 16 hands: position and carried inventory |
//! | 9 | `PRODUCT` | one per market product: price, inventory, demand, own supply |
//! | 8 | `OPPONENT` | 4 quadrant summaries + economy + crops + animals + supply timeline |
//! | 1 | `PRIVATE` | seeds, shed vector, own portfolio aggregates |
//! | 4 | `SCRATCH` | zero-payload registers for the network to use as workspace |
//!
//! # Why the opponent gets 8 tokens and not 100
//!
//! The opponent's board is fully visible, but it is not *actionable*: nothing the policy
//! can emit touches it. What it changes is market timing — when their produce lands and
//! how much of it — and that compresses to per-quadrant histograms plus a
//! "when does their supply arrive" timeline. Spending 100 tokens to mirror the own-farm
//! encoding would double inference cost for a signal that is already summarised losslessly
//! at the granularity decisions are made on.
//!
//! # Normalisation
//!
//! Every slot is scaled to roughly unit range and clipped, so no feature can blow up an
//! untrained network. Counts divide by a structural bound (25 tiles in a quadrant, 100
//! tiles on a board, `max_yield` of 6, `shedCapacity` of 100); money uses both a linear
//! `/20 000` clip and a `ln(1+x)/12` companion so it stays informative across the two
//! orders of magnitude an episode spans; times are relative to *now* and divide by a day.
//! Booleans are 0.0/1.0. The exact divisor for each slot is in the tables below.
//!
//! # Configuration
//!
//! `PORT_SPEC` pins `encode(gs, player, out)` with no `Config`, so the pinned entry points
//! assume [`Config::default`] (720 steps, 24 turns/day, `shedCapacity` 100, board 10).
//! Non-default episodes must call [`encode_with_cfg`] / [`legal_mask_with_cfg`], which take
//! the config explicitly. The two are the same function; the pinned one just supplies the
//! default.

use crate::consts::*;
use crate::macros::{
    dist_to_shed, resource_ok, target_plausible, Verb, MAX_UNITS, N_TARGETS, N_VERBS, VERBS,
};
use crate::sim::market_price;
use crate::state::{AnimalTile, Config, Farm, GameState, PlantTile, Tile};

/// Bump on **any** change to the layout below. Training checkpoints are only comparable
/// within one version.
pub const OBS_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Feature width of every token.
pub const N_FEATURES: usize = 48;
/// Header width: 7-way token-type one-hot plus a constant 1.0 bias.
pub const N_HEADER: usize = 8;
/// Payload width available to each token type (`N_FEATURES - N_HEADER`).
pub const N_PAYLOAD: usize = N_FEATURES - N_HEADER;

pub const N_GLOBAL_TOK: usize = 1;
pub const N_TILE_TOK: usize = N_TARGETS;
pub const N_UNIT_TOK: usize = MAX_UNITS;
pub const N_PRODUCT_TOK: usize = N_PRODUCTS;
pub const N_OPP_TOK: usize = 8;
pub const N_PRIVATE_TOK: usize = 1;
pub const N_SCRATCH_TOK: usize = 4;

pub const OFF_GLOBAL: usize = 0;
pub const OFF_TILE: usize = OFF_GLOBAL + N_GLOBAL_TOK;
pub const OFF_UNIT: usize = OFF_TILE + N_TILE_TOK;
pub const OFF_PRODUCT: usize = OFF_UNIT + N_UNIT_TOK;
pub const OFF_OPP: usize = OFF_PRODUCT + N_PRODUCT_TOK;
pub const OFF_PRIVATE: usize = OFF_OPP + N_OPP_TOK;
pub const OFF_SCRATCH: usize = OFF_PRIVATE + N_PRIVATE_TOK;

/// 1 + 100 + 17 + 9 + 8 + 1 + 4.
pub const N_TOKENS: usize = OFF_SCRATCH + N_SCRATCH_TOK;

/// Length of the buffer [`encode`] writes.
pub const OBS_LEN: usize = N_TOKENS * N_FEATURES;

/// Length of the buffer [`legal_mask`] writes: one flag per (unit, verb).
pub const MASK_LEN: usize = MAX_UNITS * N_VERBS;

/// Trailing axis of the optional per-tile verb plausibility plane
/// ([`tile_verb_mask_with_cfg`]): `N_TARGETS * N_VERBS` = 3800.
pub const TILE_MASK_LEN: usize = N_TARGETS * N_VERBS;

// Token-type ids, written as a one-hot in header slots 0..7.
const TY_GLOBAL: usize = 0;
const TY_TILE: usize = 1;
const TY_UNIT: usize = 2;
const TY_PRODUCT: usize = 3;
const TY_OPP: usize = 4;
const TY_PRIVATE: usize = 5;
const TY_SCRATCH: usize = 6;

/// Token counts, feature widths and action-head sizes, for the training code to assert
/// against at startup.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ObsDims {
    pub version: u32,
    pub n_tokens: usize,
    pub n_features: usize,
    pub obs_len: usize,
    pub n_header: usize,

    pub off_global: usize,
    pub n_global: usize,
    pub off_tile: usize,
    pub n_tile: usize,
    pub off_unit: usize,
    pub n_unit: usize,
    pub off_product: usize,
    pub n_product: usize,
    pub off_opp: usize,
    pub n_opp: usize,
    pub off_private: usize,
    pub n_private: usize,
    pub off_scratch: usize,
    pub n_scratch: usize,

    /// Per-unit target head width (tiles).
    pub n_targets: usize,
    /// Per-unit verb head width.
    pub n_verbs: usize,
    /// Units addressable in one turn (farmer + hands).
    pub max_units: usize,
    /// Length of the [`legal_mask`] buffer.
    pub mask_len: usize,
    /// Number of categorical market heads.
    pub n_market_heads: usize,
    /// Total market logits (sum of the market head widths).
    pub market_logits: usize,
}

/// Token counts and feature widths of the current layout.
pub const fn obs_dims() -> ObsDims {
    let mut market_logits = 0;
    let mut i = 0;
    while i < crate::macros::N_MARKET_HEADS {
        market_logits += crate::macros::MARKET_HEAD_SIZES[i];
        i += 1;
    }
    ObsDims {
        version: OBS_VERSION,
        n_tokens: N_TOKENS,
        n_features: N_FEATURES,
        obs_len: OBS_LEN,
        n_header: N_HEADER,
        off_global: OFF_GLOBAL,
        n_global: N_GLOBAL_TOK,
        off_tile: OFF_TILE,
        n_tile: N_TILE_TOK,
        off_unit: OFF_UNIT,
        n_unit: N_UNIT_TOK,
        off_product: OFF_PRODUCT,
        n_product: N_PRODUCT_TOK,
        off_opp: OFF_OPP,
        n_opp: N_OPP_TOK,
        off_private: OFF_PRIVATE,
        n_private: N_PRIVATE_TOK,
        off_scratch: OFF_SCRATCH,
        n_scratch: N_SCRATCH_TOK,
        n_targets: N_TARGETS,
        n_verbs: N_VERBS,
        max_units: MAX_UNITS,
        mask_len: MASK_LEN,
        n_market_heads: crate::macros::N_MARKET_HEADS,
        market_logits,
    }
}

// ---------------------------------------------------------------------------
// Publication dtype: f32 -> IEEE-754 binary16
//
// The layout above is *computed* in f32 and always will be — every divisor, clip and
// `ln` here is an f32 expression. What the batched interface *publishes* is a separate
// question, and the answer is f16 (`OBS_LAYOUT.md` §11): a batched vec-step is ~85 %
// memcpy into the shared observation plane (`STATUS.md` §3.3), so halving the element
// width halves the binding cost of collection, the host->device transfer and the rollout
// buffer. These functions are that conversion and nothing else; they are here rather
// than in `kaggri-py` because the wire format of the observation belongs to the module
// that defines the observation, and because `tests/perf_report.rs` measures them.
//
// Rounding is **round-to-nearest, ties-to-even**, i.e. exactly what `vcvtps2ph` and
// `numpy.float16(x)` do, so the published plane is bit-identical to
// `np.float16(f32_plane)` and can be checked against numpy rather than against itself.
// ---------------------------------------------------------------------------

/// One f32 as the bits of an IEEE-754 binary16, round-to-nearest-ties-to-even.
///
/// Total: overflow saturates to `±inf`, underflow to `±0` (both signed), subnormals are
/// produced where they are representable, and a NaN stays a NaN (quiet, mantissa
/// truncated) rather than becoming an infinity. `encode` never emits a non-finite value —
/// `obs_macros.rs` asserts every slot is finite and inside `[-4, 4]` — so those branches
/// are total-ness insurance, not a hot path.
#[inline]
pub const fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x007F_FFFF;

    // Inf / NaN. A NaN keeps its top mantissa bits and is forced quiet, so it cannot be
    // mistaken for an infinity downstream.
    if exp == 0xFF {
        if mant == 0 {
            return sign | 0x7C00;
        }
        return sign | 0x7C00 | 0x0200 | (mant >> 13) as u16;
    }

    let unbiased = exp - 127;
    // Larger than the largest binary16 (65504 rounds up to inf at 2^16).
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    // Below half the smallest subnormal (2^-25): rounds to a signed zero. This also
    // catches every f32 zero and f32 subnormal (`exp == 0`, `unbiased == -127`), which
    // must be handled here because the shift below would overflow for them.
    if unbiased < -25 {
        return sign;
    }
    if unbiased >= -14 {
        // Normal in binary16: 5-bit exponent, 10-bit mantissa.
        let e = (unbiased + 15) as u32;
        let m = mant >> 13;
        let rem = mant & 0x1FFF;
        let mut bits = (e << 10) | m;
        // Ties to even. A carry out of the mantissa walks into the exponent, and a carry
        // out of exponent 30 lands exactly on 0x7C00 = inf, which is the correct answer.
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            bits += 1;
        }
        return sign | bits as u16;
    }
    // Subnormal in binary16: shift the implicit 1 back in and round at the new position.
    let total_shift = (13 + (-unbiased - 14)) as u32; // 14 ..= 24
    let full = 0x0080_0000u32 | mant;
    let m = full >> total_shift;
    let half = 1u32 << (total_shift - 1);
    let rem = full & ((1u32 << total_shift) - 1);
    let mut bits = m;
    if rem > half || (rem == half && (m & 1) == 1) {
        bits += 1;
    }
    sign | bits as u16
}

/// `dst[i] = f32_to_f16_bits(src[i])`, eight at a time where the CPU can.
///
/// `vcvtps2ph` is one instruction per eight floats and rounds exactly as
/// [`f32_to_f16_bits`] does; `f16_paths_agree_bit_for_bit` in `tests/obs_macros.rs`
/// asserts that over every exponent, both zeros, both infinities, NaNs, f32 subnormals,
/// the f16 subnormal range, the rounding ties and 4 M random bit patterns.
pub fn f32_to_f16_into(src: &[f32], dst: &mut [u16]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "f32_to_f16_into: source and destination must be the same length"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") {
            // SAFETY: both features were just detected.
            unsafe { f32_to_f16_into_f16c(src, dst) };
            return;
        }
    }
    f32_to_f16_into_scalar(src, dst);
}

/// The portable path, and the reference the vectorised one is tested against.
pub fn f32_to_f16_into_scalar(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = f32_to_f16_bits(*s);
    }
}

/// `F16C` + `AVX`: `vcvtps2ph` on eight floats per iteration, scalar tail.
///
/// # Safety
/// The caller must have verified `avx` and `f16c`. `src.len() == dst.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub unsafe fn f32_to_f16_into_f16c(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let n = src.len();
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(src.as_ptr().add(i));
        // imm8 = _MM_FROUND_TO_NEAREST_INT (0): take the rounding mode from the immediate
        // rather than from MXCSR, and make it round-to-nearest-ties-to-even — which is
        // what `f32_to_f16_bits` and `numpy.float16` do. `_MM_FROUND_NO_EXC` (bit 3) is
        // deliberately absent: stdarch pins this immediate to 3 bits, and the only effect
        // of leaving it out is that an inexact conversion sets a sticky MXCSR flag nobody
        // reads. Exceptions are masked by default, so nothing traps.
        let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        _mm_storeu_si128(dst.as_mut_ptr().add(i).cast::<__m128i>(), h);
        i += 8;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = f32_to_f16_bits(*src.get_unchecked(i));
        i += 1;
    }
}

/// The inverse, for tests and for anything that wants to read a published plane back.
#[inline]
pub const fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = ((h as u32) & 0x03FF) << 13;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal binary16 -> normal f32: renormalise by shifting the mantissa up.
        let mut m = mant;
        let mut e: u32 = 127 - 14;
        while m & 0x0080_0000 == 0 {
            m <<= 1;
            e -= 1;
        }
        return f32::from_bits(sign | (e << 23) | (m & 0x007F_FFFF));
    }
    if exp == 0x1F {
        return f32::from_bits(sign | 0x7F80_0000 | mant);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | mant)
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

#[inline]
fn clip(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline]
fn b(x: bool) -> f32 {
    if x {
        1.0
    } else {
        0.0
    }
}

/// Money → two slots' worth of scale: a clipped linear one and a log companion.
///
/// Bankrolls run from $3 000 to roughly $50 000 in a strong episode, and PPO wants both
/// "how much, linearly" (for affordability) and "what order of magnitude" (for stable
/// gradients when it is large).
#[inline]
fn money_pair(m: f64) -> (f32, f32) {
    let lin = clip((m / 20_000.0) as f32, -1.0, 3.0);
    let log = clip(((1.0 + m.max(0.0)).ln() / 12.0) as f32, 0.0, 1.5);
    (lin, log)
}

/// Is this plant's yield harvestable right now? Same predicate as `HARVEST`.
#[inline]
fn plant_harvestable(p: &PlantTile, day: i64) -> bool {
    p.yield_units > 0 && day - p.planted_day >= p.crop.data().first_yield_day
}

/// Days until this plant's yield next increases: for an ongoing crop the next scheduled
/// production that is still under `max_yield`, for a one-time crop the start of its
/// watering bonus window (`0` while inside it). `9` means "never again".
fn plant_days_to_next_gain(p: &PlantTile, day: i64) -> i64 {
    let cd = p.crop.data();
    let age = day - p.planted_day;
    if cd.ongoing {
        // `_daily_refresh_plants` fires when `next_day - planted_day - first_yield_day`
        // is >= 0, divisible by `interval`, and the resulting count is <= `max_yield`.
        for d in 1..=8 {
            let dsf = day + d - p.planted_day - cd.first_yield_day;
            if dsf >= 0
                && cd.interval > 0
                && dsf % cd.interval == 0
                && dsf / cd.interval + 1 <= cd.max_yield
            {
                return d;
            }
        }
        9
    } else {
        let window_start = (cd.max_yield_day + 1) / 2;
        if age < window_start {
            window_start - age
        } else if age <= cd.max_yield_day {
            0
        } else {
            9
        }
    }
}

/// Days until this animal's next scheduled production (`0` is impossible: production is
/// evaluated for `day + 1`). `7` means "not within a week", which cannot happen for a
/// living animal but keeps the function total.
fn animal_days_to_next_prod(a: &AnimalTile, day: i64) -> i64 {
    let ad = a.animal.data();
    for d in 1..=6 {
        let dsf = day + d - a.placed_day - ad.first_yield_day;
        if dsf >= 0 && ad.interval > 0 && dsf % ad.interval == 0 {
            return d;
        }
    }
    7
}

/// `_daily_refresh_plants` turns this plant into a weed tonight.
#[inline]
fn plant_dies_tonight(p: &PlantTile) -> bool {
    !p.watered_today && p.consecutive_unwatered + 1 >= 2
}

/// `_daily_refresh_animals` lets this animal escape tonight.
#[inline]
fn animal_escapes_tonight(a: &AnimalTile) -> bool {
    !a.fed_today && a.consecutive_unfed + 1 >= 2
}

/// `_decay_plants` removes a unit from this plant on the current step.
#[inline]
fn plant_decays_now(p: &PlantTile, step: i64) -> bool {
    p.max_lifespan_step >= 0 && step >= p.max_lifespan_step && (step - p.max_lifespan_step) % 2 == 0
}

// ---------------------------------------------------------------------------
// One-pass farm scan
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Default)]
struct QuadStats {
    locked: f32,
    empty: f32,
    weed: f32,
    coop: f32,
    pasture: f32,
    plants: f32,
    animals: f32,
    harvest: f32,
    age_sum: f32,
    dying: f32,
}

/// Aggregates over one farm, computed once and read by several token groups.
#[derive(Copy, Clone, Default)]
struct FarmStats {
    quad: [QuadStats; 4],
    /// Units currently harvestable, keyed by the product they would yield.
    harvest_by_product: [f32; N_PRODUCTS],
    plants_by_crop: [f32; N_CROPS],
    harvest_by_crop: [f32; N_CROPS],
    animals_by_animal: [f32; N_ANIMALS],
    harvest_by_animal: [f32; N_ANIMALS],
    /// Plants whose yield next grows in 0, 1, 2, 3, 4+ days.
    plant_eta: [f32; 5],
    /// Animals producing in 1, 2, 3, 4+ days.
    animal_eta: [f32; 4],
    empty: f32,
    weed: f32,
    locked: f32,
    empty_coop: f32,
    empty_pasture: f32,
    plants: f32,
    animals: f32,
    harvest_total: f32,
    plant_age_sum: f32,
    dying: f32,
    escaping: f32,
    fed: f32,
    cared: f32,
    plants_at_cap: f32,
}

fn scan_farm(farm: &Farm, day: i64) -> FarmStats {
    let mut s = FarmStats::default();
    let bs = farm.board_size;
    for y in 0..bs as i64 {
        for x in 0..bs as i64 {
            let q = quadrant_of(x, y, bs) as usize;
            let qs = &mut s.quad[q];
            match farm.tile(x, y) {
                Tile::Empty => {
                    s.empty += 1.0;
                    qs.empty += 1.0;
                }
                Tile::Locked => {
                    s.locked += 1.0;
                    qs.locked += 1.0;
                }
                Tile::Weed => {
                    s.weed += 1.0;
                    qs.weed += 1.0;
                }
                Tile::Coop => {
                    s.empty_coop += 1.0;
                    qs.coop += 1.0;
                }
                Tile::Pasture => {
                    s.empty_pasture += 1.0;
                    qs.pasture += 1.0;
                }
                Tile::Plant(p) => {
                    let cd = p.crop.data();
                    s.plants += 1.0;
                    qs.plants += 1.0;
                    s.plants_by_crop[p.crop.idx()] += 1.0;
                    let age = (day - p.planted_day) as f32;
                    s.plant_age_sum += age;
                    qs.age_sum += age;
                    if plant_harvestable(&p, day) {
                        let u = p.yield_units as f32;
                        s.harvest_total += u;
                        qs.harvest += u;
                        s.harvest_by_crop[p.crop.idx()] += u;
                        s.harvest_by_product[p.crop.product().idx()] += u;
                    }
                    if p.yield_units >= cd.max_yield {
                        s.plants_at_cap += 1.0;
                    }
                    if plant_dies_tonight(&p) {
                        s.dying += 1.0;
                        qs.dying += 1.0;
                    }
                    let eta = plant_days_to_next_gain(&p, day).clamp(0, 4) as usize;
                    s.plant_eta[eta] += 1.0;
                }
                Tile::Animal(a) => {
                    let ad = a.animal.data();
                    s.animals += 1.0;
                    qs.animals += 1.0;
                    s.animals_by_animal[a.animal.idx()] += 1.0;
                    let u = a.yield_units as f32;
                    if a.yield_units > 0 {
                        s.harvest_total += u;
                        qs.harvest += u;
                        s.harvest_by_animal[a.animal.idx()] += u;
                        s.harvest_by_product[ad.product.idx()] += u;
                    }
                    if a.fed_today {
                        s.fed += 1.0;
                    }
                    if a.cared_today {
                        s.cared += 1.0;
                    }
                    if animal_escapes_tonight(&a) {
                        s.escaping += 1.0;
                    }
                    let eta = (animal_days_to_next_prod(&a, day) - 1).clamp(0, 3) as usize;
                    s.animal_eta[eta] += 1.0;
                }
            }
        }
    }
    s
}

/// Town consumption per product: `(units per shop tick, units per day)`.
///
/// `_town_consume` runs every `townShopSellInterval` steps for every unlocked shop
/// instance (single-product shops at double rate) and every `townCenterSellInterval` steps
/// for one of each non-fertilizer product.
fn town_demand(gs: &GameState, cfg: &Config) -> ([f32; N_PRODUCTS], [f32; N_PRODUCTS]) {
    let mut per_tick = [0f32; N_PRODUCTS];
    for shop in &gs.town.unlocked_shops {
        let m = shop.multiplier() as f32;
        for p in shop.products() {
            per_tick[p.idx()] += m;
        }
    }
    let shop_ticks = (cfg.turns_per_day_eff() / cfg.shop_sell_interval_eff()) as f32;
    let center_ticks = (cfg.turns_per_day_eff() / cfg.center_sell_interval_eff()) as f32;
    let mut per_day = [0f32; N_PRODUCTS];
    for p in PRODUCTS {
        per_day[p.idx()] = per_tick[p.idx()] * shop_ticks;
    }
    for p in TOWN_CENTER_PRODUCTS {
        per_day[p.idx()] += center_ticks;
    }
    (per_tick, per_day)
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// Header: token-type one-hot in slots 0..7, constant bias in slot 7.
#[inline]
fn header(tok: &mut [f32], ty: usize) {
    tok[ty] = 1.0;
    tok[7] = 1.0;
}

/// Encode the observation for `player` into `out` (length [`OBS_LEN`]).
///
/// Assumes [`Config::default`]; see [`encode_with_cfg`].
pub fn encode(gs: &GameState, player: usize, out: &mut [f32]) {
    encode_with_cfg(gs, player, &Config::default(), out);
}

/// Encode the observation for `player` under an explicit config.
///
/// Zero allocation: `out` is filled in place and every intermediate is a stack array.
///
/// Boards smaller than 10x10 leave the trailing tile tokens all-zero (their type one-hot
/// included, so they are distinguishable from real tiles); boards larger than 10x10 are
/// not representable and are truncated — [`N_TILE_TOK`] is fixed because the target head
/// is fixed.
pub fn encode_with_cfg(gs: &GameState, player: usize, cfg: &Config, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_LEN, "observation buffer must be OBS_LEN long");
    out.fill(0.0);

    let opp = 1 - player;
    let farm = &gs.farms[player];
    let ofarm = &gs.farms[opp];
    let private = &gs.privates[player];
    let bs = farm.board_size;
    let step = gs.step;
    let day = gs.day;
    let hour = gs.hour;
    let tpd = cfg.turns_per_day_eff();
    let cap = cfg.shed_capacity_i64();
    let last_step = (cfg.episode_steps as i64 - 2).max(0);
    let last_day = last_step / tpd;

    let me = scan_farm(farm, day);
    let them = scan_farm(ofarm, day);
    let (demand_tick, demand_day) = town_demand(gs, cfg);

    // Units standing on each tile, for the tile tokens' congestion slot.
    let mut occupancy = [0u8; N_TARGETS];
    let mut mark = |p: [i64; 2]| {
        let i = p[1] as usize * bs + p[0] as usize;
        if i < N_TARGETS {
            occupancy[i] = occupancy[i].saturating_add(1);
        }
    };
    mark(farm.farmer);
    for h in &farm.hands {
        mark(*h);
    }

    // =====================================================================
    // GLOBAL — 1 token, 35 payload slots
    // =====================================================================
    {
        let t = &mut out[OFF_GLOBAL * N_FEATURES..(OFF_GLOBAL + 1) * N_FEATURES];
        header(t, TY_GLOBAL);
        let p = &mut t[N_HEADER..];
        let (my_lin, my_log) = money_pair(farm.money);
        let (op_lin, op_log) = money_pair(ofarm.money);
        p[0] = clip(step as f32 / last_step.max(1) as f32, 0.0, 1.0);
        p[1] = clip(day as f32 / last_day.max(1) as f32, 0.0, 1.0);
        p[2] = hour as f32 / tpd.saturating_sub(1).max(1) as f32;
        p[3] = clip((last_step - step) as f32 / last_step.max(1) as f32, 0.0, 1.0);
        p[4] = clip((tpd - 1 - hour) as f32 / tpd.saturating_sub(1).max(1) as f32, 0.0, 1.0);
        p[5] = b(day >= last_day);
        p[6] = b(hour + 1 >= tpd);
        p[7] = my_lin;
        p[8] = my_log;
        p[9] = op_lin;
        p[10] = op_log;
        p[11] = clip(((farm.money - ofarm.money) / 20_000.0) as f32, -1.0, 1.0);
        p[12] = b(farm.money > ofarm.money);
        p[13] = clip(farm.hires_today as f32 / 16.0, 0.0, 2.0);
        p[14] = farm.hands.len() as f32 / 16.0;
        p[15] = farm.unlocked_quadrants.len() as f32 / 4.0;
        p[16] = b(farm.unlocked_quadrants.contains(&Quadrant::NE));
        p[17] = b(farm.unlocked_quadrants.contains(&Quadrant::SW));
        p[18] = b(farm.unlocked_quadrants.contains(&Quadrant::SE));
        p[19] = clip(ofarm.hires_today as f32 / 16.0, 0.0, 2.0);
        p[20] = clip(ofarm.hands.len() as f32 / 16.0, 0.0, 2.0);
        p[21] = ofarm.unlocked_quadrants.len() as f32 / 4.0;
        p[22] = b(ofarm.unlocked_quadrants.contains(&Quadrant::NE));
        p[23] = b(ofarm.unlocked_quadrants.contains(&Quadrant::SW));
        p[24] = b(ofarm.unlocked_quadrants.contains(&Quadrant::SE));
        p[25] = gs.town.unlocked_shops.len() as f32 / MAX_SHOP_INSTANCES as f32;
        // `_end_of_day` unlocks when `(day + 1) % townShopUnlockInterval == 0`.
        let iv = cfg.shop_unlock_interval_eff();
        let mut k = 0;
        while (day + 1 + k) % iv != 0 {
            k += 1;
        }
        p[26] = b(gs.town.unlocked_shops.len() < MAX_SHOP_INSTANCES) * (1.0 - k as f32 / iv as f32);
        let si = cfg.shop_sell_interval_eff();
        p[27] = 1.0 - ((si - step % si) % si) as f32 / si as f32;
        let ci = cfg.center_sell_interval_eff();
        p[28] = 1.0 - ((ci - step % ci) % ci) as f32 / ci as f32;
        let next_hire = crate::sim::hire_cost(farm.hires_today, cfg.farm_hand_cost_mult);
        p[29] = clip(next_hire as f32 / 64.0, 0.0, 2.0);
        let extra = farm.unlocked_quadrants.len() - 1;
        let land_price = LAND_PRICES.get(extra).copied().unwrap_or(0);
        p[30] = land_price as f32 / 4000.0;
        p[31] = b(land_price > 0 && farm.money >= land_price as f64);
        p[32] = (1 + farm.hands.len()) as f32 / MAX_UNITS as f32;
        p[33] = clip(private.shed.total() as f32 / cap.max(1) as f32, 0.0, 1.0);
        p[34] = clip((cap - private.shed.total()) as f32 / cap.max(1) as f32, 0.0, 1.0);
    }

    // =====================================================================
    // TILE — 100 tokens, 37 payload slots
    // =====================================================================
    for y in 0..bs.min(N_TARGETS) as i64 {
        for x in 0..bs as i64 {
            let idx = y as usize * bs + x as usize;
            if idx >= N_TILE_TOK {
                break;
            }
            let tok = OFF_TILE + idx;
            let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
            header(t, TY_TILE);
            let p = &mut t[N_HEADER..];
            let tile = farm.tile(x, y);
            // kind one-hot
            let kind = match tile {
                Tile::Empty => 0,
                Tile::Locked => 1,
                Tile::Weed => 2,
                Tile::Coop => 3,
                Tile::Pasture => 4,
                Tile::Plant(_) => 5,
                Tile::Animal(_) => 6,
            };
            p[kind] = 1.0;
            // geometry (always present)
            p[15] = x as f32 / bs.saturating_sub(1).max(1) as f32;
            p[16] = y as f32 / bs.saturating_sub(1).max(1) as f32;
            p[17] = b(is_shed_adjacent(x, y, bs));
            p[18] = dist_to_shed([x, y], bs) as f32 / 9.0;
            p[19] = occupancy[idx] as f32 / 4.0;
            match tile {
                Tile::Plant(pl) => {
                    let cd = pl.crop.data();
                    p[7 + pl.crop.idx()] = 1.0;
                    p[20] = clip((day - pl.planted_day) as f32 / 30.0, 0.0, 1.0);
                    p[21] = clip(pl.yield_units as f32 / 6.0, 0.0, 1.0);
                    p[22] = b(pl.yield_units >= cd.max_yield);
                    p[23] = b(plant_harvestable(&pl, day));
                    p[24] = clip(plant_days_to_next_gain(&pl, day) as f32 / 8.0, 0.0, 1.125);
                    p[25] = b(pl.watered_today);
                    p[26] = clip(pl.consecutive_unwatered as f32 / 2.0, 0.0, 1.0);
                    p[27] = b(plant_dies_tonight(&pl));
                    p[28] = clip((pl.fertilized_until_day - day + 1) as f32 / 3.0, 0.0, 1.0);
                    let ws = (cd.max_yield_day + 1) / 2;
                    let age = day - pl.planted_day;
                    p[29] = b(!cd.ongoing && age >= ws && age <= cd.max_yield_day);
                    p[30] = b(pl.max_lifespan_step >= 0);
                    p[31] = if pl.max_lifespan_step >= 0 {
                        clip((pl.max_lifespan_step - step) as f32 / tpd as f32, -1.0, 1.0)
                    } else {
                        0.0
                    };
                    p[32] = b(plant_decays_now(&pl, step));
                    p[36] = b(cd.ongoing);
                }
                Tile::Animal(a) => {
                    let ad = a.animal.data();
                    p[12 + a.animal.idx()] = 1.0;
                    p[20] = clip((day - a.placed_day) as f32 / 30.0, 0.0, 1.0);
                    p[21] = clip(a.yield_units as f32 / 6.0, 0.0, 1.0);
                    p[22] = b(a.yield_units >= ad.max_held);
                    p[23] = b(a.yield_units > 0);
                    p[24] = clip(animal_days_to_next_prod(&a, day) as f32 / 8.0, 0.0, 1.0);
                    p[25] = b(a.fed_today);
                    p[26] = clip(a.consecutive_unfed as f32 / 2.0, 0.0, 1.0);
                    p[27] = b(animal_escapes_tonight(&a));
                    p[33] = b(a.cared_today);
                    p[34] = b(a.fertilizer_available);
                    p[35] = clip(a.pending_care_bonus as f32 / 4.0, 0.0, 1.5);
                }
                _ => {}
            }
        }
    }

    // =====================================================================
    // UNIT — 17 tokens, 22 payload slots
    // =====================================================================
    for u in 0..N_UNIT_TOK {
        let tok = OFF_UNIT + u;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_UNIT);
        let p = &mut t[N_HEADER..];
        p[21] = u as f32 / (MAX_UNITS - 1) as f32;
        let Some(pos) = farm.unit_pos(u) else {
            continue;
        };
        p[0] = 1.0;
        p[1] = b(u == 0);
        p[2] = pos[0] as f32 / bs.saturating_sub(1).max(1) as f32;
        p[3] = pos[1] as f32 / bs.saturating_sub(1).max(1) as f32;
        p[4] = dist_to_shed(pos, bs) as f32 / 9.0;
        p[5] = b(is_shed_adjacent(pos[0], pos[1], bs));
        if let Some(inv) = private.inventories.get(u) {
            for (item, n) in inv.iter() {
                p[6 + item.idx()] = clip(n as f32 / 32.0, 0.0, 4.0);
            }
            p[18] = clip(inv.total() as f32 / cap.max(1) as f32, 0.0, 4.0);
            // Insertion order is observable: `_drop_inventories_to_shed` fills the shed in
            // this order and discards whatever no longer fits, so the head of the order is
            // what survives a full shed.
            if let Some((first, count)) = inv.iter().next() {
                p[19] = first.idx() as f32 / (N_SHED_ITEMS - 1) as f32;
                p[20] = clip(count as f32 / 32.0, 0.0, 4.0);
            }
        }
    }

    // =====================================================================
    // PRODUCT — 9 tokens, 27 payload slots
    // =====================================================================
    for item in PRODUCTS {
        let i = item.idx();
        let tok = OFF_PRODUCT + i;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_PRODUCT);
        let p = &mut t[N_HEADER..];
        let par = gs.market.params.get(item);
        let inv = gs.market.inv(item);
        let price = gs.market.price(item);
        let base = par.base;
        let held = private.shed.get(item.shed());
        p[i] = 1.0;
        p[9] = clip((inv - par.i0) as f32 / 2000.0, -1.0, 2.0);
        p[10] = clip(price as f32 / base as f32, 0.0, 3.0);
        p[11] = clip((price as f32 - base as f32) / base as f32, -1.0, 2.0);
        p[12] = clip(price as f32 / 300.0, 0.0, 2.0);
        p[13] = b(price <= PRICE_FLOOR);
        p[14] = clip(((1.0 + price as f32).ln()) / 6.0, 0.0, 1.5);
        p[15] = clip(held as f32 / cap.max(1) as f32, 0.0, 1.0);
        p[16] = clip(me.harvest_by_product[i] / 50.0, 0.0, 2.0);
        p[17] = clip(them.harvest_by_product[i] / 50.0, 0.0, 2.0);
        p[18] = clip(demand_day[i] / 24.0, 0.0, 2.0);
        p[19] = clip(demand_tick[i] / 8.0, 0.0, 2.0);
        p[20] = b(item.is_buyable());
        p[21] = clip(
            market_price(item, inv - 1, &gs.market.params) as f32 / 300.0,
            0.0,
            2.0,
        );
        // Depth: what the price becomes after dumping everything in the shed, and the
        // revenue that dump would raise. This is the single most important market signal —
        // premium products crash to the $1 floor within one quadrant's production.
        let after = market_price(item, inv + held, &gs.market.params);
        p[22] = clip(after as f32 / price.max(1) as f32, 0.0, 1.0);
        p[23] = clip(
            held as f32 * (price + after) as f32 * 0.5 / 5000.0,
            0.0,
            2.0,
        );
        p[24] = clip(base as f32 / 300.0, 0.0, 2.0);
        p[25] = Crop::from_idx(i)
            .map(|c| c.data().seed as f32 / 100.0)
            .unwrap_or(0.0);
        p[26] = ANIMAL_LIST
            .iter()
            .find(|a| a.data().product == item)
            .map(|a| a.data().cost as f32 / 500.0)
            .unwrap_or(0.0);
    }

    // =====================================================================
    // OPPONENT — 8 tokens: 4 quadrants + econ + crops + animals + timeline
    // =====================================================================
    const OSUB_QUAD: usize = 0;
    const OSUB_ECON: usize = 1;
    const OSUB_CROPS: usize = 2;
    const OSUB_ANIMALS: usize = 3;
    const OSUB_TIMELINE: usize = 4;
    let quad_tiles = ((bs / 2) * (bs / 2)).max(1) as f32;
    for q in 0..4usize {
        let tok = OFF_OPP + q;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_OPP);
        let p = &mut t[N_HEADER..];
        p[OSUB_QUAD] = 1.0;
        p[5 + q] = 1.0;
        let qs = &them.quad[q];
        p[9] = qs.locked / quad_tiles;
        p[10] = qs.empty / quad_tiles;
        p[11] = qs.weed / quad_tiles;
        p[12] = qs.coop / quad_tiles;
        p[13] = qs.pasture / quad_tiles;
        p[14] = qs.plants / quad_tiles;
        p[15] = qs.animals / quad_tiles;
        p[16] = clip(qs.harvest / 50.0, 0.0, 2.0);
        p[17] = if qs.plants > 0.0 {
            clip(qs.age_sum / qs.plants / 30.0, 0.0, 1.0)
        } else {
            0.0
        };
        p[18] = qs.dying / quad_tiles;
    }
    {
        let tok = OFF_OPP + 4;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_OPP);
        let p = &mut t[N_HEADER..];
        p[OSUB_ECON] = 1.0;
        let (lin, log) = money_pair(ofarm.money);
        p[9] = lin;
        p[10] = log;
        p[11] = clip(((ofarm.money - farm.money) / 20_000.0) as f32, -1.0, 1.0);
        p[12] = clip(ofarm.hires_today as f32 / 16.0, 0.0, 2.0);
        p[13] = clip(ofarm.hands.len() as f32 / 16.0, 0.0, 2.0);
        p[14] = ofarm.unlocked_quadrants.len() as f32 / 4.0;
        p[15] = b(ofarm.unlocked_quadrants.contains(&Quadrant::NE));
        p[16] = b(ofarm.unlocked_quadrants.contains(&Quadrant::SW));
        p[17] = b(ofarm.unlocked_quadrants.contains(&Quadrant::SE));
        p[18] = clip((1 + ofarm.hands.len()) as f32 / MAX_UNITS as f32, 0.0, 2.0);
    }
    {
        let tok = OFF_OPP + 5;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_OPP);
        let p = &mut t[N_HEADER..];
        p[OSUB_CROPS] = 1.0;
        for c in 0..N_CROPS {
            p[9 + c] = clip(them.plants_by_crop[c] / 25.0, 0.0, 4.0);
            p[14 + c] = clip(them.harvest_by_crop[c] / 50.0, 0.0, 2.0);
        }
        p[19] = clip(them.plants / 100.0, 0.0, 1.0);
        p[20] = if them.plants > 0.0 {
            clip(them.plant_age_sum / them.plants / 30.0, 0.0, 1.0)
        } else {
            0.0
        };
    }
    {
        let tok = OFF_OPP + 6;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_OPP);
        let p = &mut t[N_HEADER..];
        p[OSUB_ANIMALS] = 1.0;
        for a in 0..N_ANIMALS {
            p[9 + a] = clip(them.animals_by_animal[a] / 25.0, 0.0, 4.0);
            p[12 + a] = clip(them.harvest_by_animal[a] / 24.0, 0.0, 2.0);
        }
        p[15] = if them.animals > 0.0 {
            them.fed / them.animals
        } else {
            0.0
        };
        p[16] = if them.animals > 0.0 {
            them.cared / them.animals
        } else {
            0.0
        };
        p[17] = clip(them.animals / 100.0, 0.0, 1.0);
        p[18] = clip((them.empty_coop + them.empty_pasture) / 100.0, 0.0, 1.0);
    }
    {
        let tok = OFF_OPP + 7;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_OPP);
        let p = &mut t[N_HEADER..];
        p[OSUB_TIMELINE] = 1.0;
        for k in 0..5 {
            p[9 + k] = clip(them.plant_eta[k] / 25.0, 0.0, 4.0);
        }
        for k in 0..4 {
            p[14 + k] = clip(them.animal_eta[k] / 25.0, 0.0, 4.0);
        }
        p[18] = clip(them.harvest_total / 100.0, 0.0, 2.0);
        p[19] = clip(them.plants_at_cap / 25.0, 0.0, 4.0);
    }

    // =====================================================================
    // PRIVATE — 1 token, 38 payload slots
    // =====================================================================
    {
        let tok = OFF_PRIVATE;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_PRIVATE);
        let p = &mut t[N_HEADER..];
        for c in CROP_LIST {
            p[c.idx()] = clip(private.seeds[c.idx()] as f32 / 32.0, 0.0, 4.0);
        }
        for a in ANIMAL_LIST {
            p[5 + a.idx()] = clip(private.shed.get(a.shed()) as f32 / 8.0, 0.0, 4.0);
        }
        let total = private.shed.total();
        p[8] = clip(total as f32 / cap.max(1) as f32, 0.0, 1.0);
        p[9] = clip((cap - total) as f32 / cap.max(1) as f32, 0.0, 1.0);
        p[10] = b(total >= cap);
        for item in PRODUCTS {
            p[11 + item.idx()] = clip(
                private.shed.get(item.shed()) as f32 / cap.max(1) as f32,
                0.0,
                1.0,
            );
        }
        for c in 0..N_CROPS {
            p[20 + c] = clip(me.plants_by_crop[c] / 100.0, 0.0, 1.0);
        }
        for a in 0..N_ANIMALS {
            p[25 + a] = clip(me.animals_by_animal[a] / 100.0, 0.0, 1.0);
        }
        p[28] = me.empty / 100.0;
        p[29] = me.weed / 100.0;
        p[30] = me.empty_coop / 100.0;
        p[31] = me.empty_pasture / 100.0;
        p[32] = me.locked / 100.0;
        p[33] = clip(me.harvest_total / 100.0, 0.0, 2.0);
        p[34] = clip(me.dying / 25.0, 0.0, 4.0);
        p[35] = clip(me.escaping / 25.0, 0.0, 4.0);
        let carrying = private.inventories.iter().filter(|i| !i.is_empty()).count();
        p[36] = clip(carrying as f32 / MAX_UNITS as f32, 0.0, 2.0);
        let carried: i64 = private.inventories.iter().map(|i| i.total()).sum();
        p[37] = clip(carried as f32 / cap.max(1) as f32, 0.0, 4.0);
    }

    // =====================================================================
    // SCRATCH — 4 zero-payload register tokens
    // =====================================================================
    for k in 0..N_SCRATCH_TOK {
        let tok = OFF_SCRATCH + k;
        let t = &mut out[tok * N_FEATURES..(tok + 1) * N_FEATURES];
        header(t, TY_SCRATCH);
        t[N_HEADER] = k as f32 / (N_SCRATCH_TOK - 1) as f32;
    }
}

// ---------------------------------------------------------------------------
// legal_mask
// ---------------------------------------------------------------------------

/// Which tile-state preconditions are satisfiable *somewhere* on the board right now.
#[derive(Copy, Clone, Default)]
struct Feasible {
    unwatered_plant: bool,
    harvestable: bool,
    any_plant: bool,
    diggable: bool,
    empty: bool,
    unfed_animal: bool,
    uncared_animal: bool,
    fertilizer_ready: bool,
    empty_coop: bool,
    empty_pasture: bool,
    shed_tile: bool,
}

fn scan_feasible(farm: &Farm, day: i64) -> Feasible {
    let mut f = Feasible::default();
    let bs = farm.board_size;
    f.shed_tile = bs >= 2;
    for y in 0..bs as i64 {
        for x in 0..bs as i64 {
            match farm.tile(x, y) {
                Tile::Empty => f.empty = true,
                Tile::Locked => {}
                Tile::Weed => f.diggable = true,
                Tile::Coop => {
                    f.empty_coop = true;
                    f.diggable = true;
                }
                Tile::Pasture => {
                    f.empty_pasture = true;
                    f.diggable = true;
                }
                Tile::Plant(p) => {
                    f.any_plant = true;
                    f.diggable = true;
                    if !p.watered_today {
                        f.unwatered_plant = true;
                    }
                    if plant_harvestable(&p, day) {
                        f.harvestable = true;
                    }
                }
                Tile::Animal(a) => {
                    if a.yield_units > 0 {
                        f.harvestable = true;
                    }
                    if !a.fed_today {
                        f.unfed_animal = true;
                    }
                    if !a.cared_today {
                        f.uncared_animal = true;
                    }
                    if a.fertilizer_available {
                        f.fertilizer_ready = true;
                    }
                }
            }
        }
    }
    f
}

/// Is there any target tile whose current state admits this verb?
#[inline]
fn verb_has_target(verb: Verb, f: &Feasible) -> bool {
    match verb {
        Verb::Idle => true,
        Verb::Water => f.unwatered_plant,
        Verb::Harvest => f.harvestable,
        Verb::Fertilize => f.any_plant,
        Verb::Dig => f.diggable,
        Verb::BuildCoop | Verb::BuildPasture | Verb::Plant(_) => f.empty,
        Verb::Feed => f.unfed_animal,
        Verb::Care => f.uncared_animal,
        Verb::CollectFertilizer => f.fertilizer_ready,
        Verb::PlaceAnimal(a) => match a.data().structure {
            Structure::Coop => f.empty_coop,
            Structure::Pasture => f.empty_pasture,
        },
        Verb::Pickup { .. } | Verb::Deposit(_) | Verb::Drop => f.shed_tile,
    }
}

/// Write a `[MAX_UNITS, N_VERBS]` legality mask for `player` into `out`
/// (length [`MASK_LEN`], row-major, `1.0` = legal).
///
/// `mask[u][v] == 1.0` means "there is at least one target tile for which unit `u`'s verb
/// `v` would take effect this turn", i.e. exactly the condition under which
/// [`crate::macros::resolve_turn`] can emit that verb for that unit. Rows for units that
/// do not exist are all zero, `Idle` included, so "no unit here" is unambiguous.
///
/// The mask is **not** applied anywhere: `resolve_turn` does not consult it and the policy
/// should not hard-mask with it. It is provided as an optional extra input plane and as an
/// optional auxiliary loss target, to be A/B'd — the Orbit Wars winner found that hard
/// masking obviously-silly actions made his model measurably worse, and this environment
/// makes unmasked exploration free: every illegal action is a silent no-op that mutates
/// nothing, so the worst case is a wasted turn rather than a corrupted game.
///
/// Deliberately *not* modelled, because it would cost more than it is worth:
/// per-(unit, tile, verb) legality is 17x100x38 = 64 600 flags, ten times the size of the
/// observation itself. The tile tokens carry kind, crop, animal, yield and every
/// once-per-day flag, so tile-level legality is a linear readout the network can learn.
pub fn legal_mask(gs: &GameState, player: usize, out: &mut [f32]) {
    legal_mask_with_cfg(gs, player, &Config::default(), out);
}

/// [`legal_mask`] under an explicit config (`shedCapacity` and `boardSize` matter).
pub fn legal_mask_with_cfg(gs: &GameState, player: usize, cfg: &Config, out: &mut [f32]) {
    assert_eq!(out.len(), MASK_LEN, "mask buffer must be MASK_LEN long");
    out.fill(0.0);
    let farm = &gs.farms[player];
    let private = &gs.privates[player];
    let feasible = scan_feasible(farm, gs.day);
    let cap = cfg.shed_capacity_i64();
    let empty_inv = crate::state::Inv::new();

    let mut verb_ok = [false; N_VERBS];
    for (v, verb) in VERBS.iter().enumerate() {
        verb_ok[v] = verb_has_target(*verb, &feasible);
    }

    for u in 0..MAX_UNITS {
        if farm.unit_pos(u).is_none() {
            continue;
        }
        let inv = private.inventories.get(u).unwrap_or(&empty_inv);
        let row = &mut out[u * N_VERBS..(u + 1) * N_VERBS];
        for (v, verb) in VERBS.iter().enumerate() {
            if !verb_ok[v] {
                continue;
            }
            if resource_ok(*verb, inv, &private.shed, &private.seeds, cap) {
                row[v] = 1.0;
            }
        }
        // `Idle` is always available and is the only verb with no precondition at all.
        row[Verb::Idle.index()] = 1.0;
    }
}


/// Write the `[N_TARGETS, N_VERBS]` tile-verb plausibility plane for `player` into `out`
/// (length [`TILE_MASK_LEN`], row-major over the action row's target index
/// `t = y * boardSize + x`, `1` = plausible).
///
/// `out[t * N_VERBS + v] == 1` iff [`crate::macros::target_plausible`] holds for verb `v`
/// at tile `t` **right now** — the resolver's own step-2 tile-state gate, verbatim. It
/// ignores per-unit resources and same-turn cross-unit claims on purpose: those live in
/// [`legal_mask`] and in `resolve_turn`. AND-ing row `t` of this plane with row `u` of the
/// unit mask therefore gives exactly the `(target, verb)` pairs at which unit `u` can fire
/// once it stands on `t`; a pair outside it is a guaranteed structural `PASS`.
///
/// Consistency with [`legal_mask`]: for every unit `u` and non-`Idle` verb `v` with
/// `mask[u][v] == 1` there is at least one tile `t` with `out[t][v] == 1` (both derive
/// from the same tile predicates). `Idle` has no target and is `0` everywhere. Board
/// positions beyond `N_TARGETS` (never at the default 10x10) are not written.
///
/// Optional plane, `u8`, ~1.1 µs per call: it exists for policies that sample the verb
/// first and then the target, where masking the target head with it removes the 93% of
/// non-idle orders an untrained policy otherwise wastes on the structural gate. Whether
/// hard-masking with it *helps a trained policy* is an A/B, not a given — see the note
/// on [`legal_mask`].
pub fn tile_verb_mask_with_cfg(gs: &GameState, player: usize, _cfg: &Config, out: &mut [u8]) {
    assert_eq!(out.len(), TILE_MASK_LEN, "tile mask buffer must be TILE_MASK_LEN long");
    out.fill(0);
    let farm = &gs.farms[player];
    let bs = farm.board_size;
    let day = gs.day;
    for y in 0..bs as i64 {
        for x in 0..bs as i64 {
            let t = y as usize * bs + x as usize;
            if t >= N_TARGETS {
                continue;
            }
            let tile = farm.tile(x, y);
            let row = &mut out[t * N_VERBS..(t + 1) * N_VERBS];
            for (v, verb) in VERBS.iter().enumerate() {
                if target_plausible(*verb, tile, day, x, y, bs) {
                    row[v] = 1;
                }
            }
        }
    }
}

/// [`tile_verb_mask_with_cfg`] under the default config.
pub fn tile_verb_mask(gs: &GameState, player: usize, out: &mut [u8]) {
    tile_verb_mask_with_cfg(gs, player, &Config::default(), out);
}
