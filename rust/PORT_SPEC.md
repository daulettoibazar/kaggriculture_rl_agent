# Kaggriculture Rust engine — port notes

Ground truth: `kaggle_environments/envs/kaggriculture/kaggriculture.py` (kaggle-environments 1.32.7).
**That file is the specification.** Where these notes and the Python source disagree, the Python source
wins and this document is the bug.

The engine is a bit-exact re-implementation of that interpreter, fast enough for multi-GPU self-play
(about 9 ms per 720-step game on one core). Bit-exact is not aspirational: a silent divergence trains the
policy on a different game than the one it is submitted to.

## Module layout

```
crates/kaggri-core/src/
  lib.rs      module declarations and re-exports
  consts.rs   CROPS, ANIMALS, PRODUCTS, MARKET_PARAMS, SHOPS, LAND_ORDER/PRICES, geometry helpers
  pyrng.rs    CPython-exact MT19937 (`random.Random`) and `round()`
  state.rs    Config, GameState, Farm, Tile, Private, Market, Town
  actions.rs  UnitOp, MarketOrder, PlayerAction and the `_parse_order` mirror
  sim.rs      `step()`: the interpreter port and every helper it calls; `step_with_phantom` for phantom seats
  obs.rs      the versioned observation encoding and legality masks (OBS_LAYOUT.md)
  macros.rs   the (target tile, verb) macro action space -> low-level ops, and the market packer
  raw.rs      a stable categorical vocabulary for exact low-level unit and market commands (replay decoding)
crates/kaggri-py/src/lib.rs   PyO3 module `kaggri`: PyGame (single game, parity and debugging),
                              PyVecEnv (batched self-play, rayon + numpy), Encoder (one Kaggle observation
                              -> obs/mask/action, used by the submission bundle)
```

## Parity traps — ALL CONFIRMED EMPIRICALLY. Get these wrong and everything downstream is wrong.

### 1. `round()` is banker's rounding (half-to-even), NOT half-away-from-zero
`market_price` ends with `max(PRICE_FLOOR, int(round(price)))`. Python's `round()` rounds half to **even**.
Rust's `f64::round()` rounds half **away from zero**. These differ at reachable states — verified:

| product | inventory | raw price | Python `round` | Rust `.round()` |
|---|---|---|---|---|
| CARROT | 9595 | 66.5 | **66** | 67 |
| CARROT | 9775 | 52.5 | **52** | 53 |
| CARROT | 9955 | 38.5 | **38** | 39 |

7 exact-`.5` cases exist in inventory range 9000..11000 alone. Implement `py_round(f64) -> i64` as
round-half-to-even and unit-test it against CPython over these exact cases plus a fuzz range.

### 2. Unit inventories are INSERTION-ORDERED and the order is observable
`DROP` and `_drop_inventories_to_shed` both iterate `list(inv.items())` and fill the shed until
`shedCapacity`, **discarding the remainder**. Python dicts are insertion-ordered, so *the order a unit
harvested items in decides what survives a full shed*. Verified: shed at 96/100, unit carrying
`{MELON:5, WHEAT:50}` →
- MELON inserted first: 4 melons survive (~$1000 of melon)
- WHEAT inserted first: melons **entirely discarded**, wheat fills to 100

Use `indexmap::IndexMap` (or an ordered Vec of pairs) for unit inventories AND the shed. Never
`HashMap`/`BTreeMap`. Insertion order must match the order `_inv_add` is first called for each item.
Note `_inv_take` deletes the key when it hits 0 — a later re-add therefore goes to the BACK of the order.
Replicate that exactly.

### 3. RNG is CPython's `random.Random` and its consumption pattern is state-dependent
`_end_of_day` does `rng = random.Random((seed * 1_000_003) ^ day)` then, **with that one shared rng
object**, in this exact order:
1. player 0: `_spawn_weeds` — iterates `y` outer, `x` inner, and calls `rng.random()` **only for tiles
   that are currently `None`**. So how many draws are consumed depends on the board. Match exactly.
2. player 1: `_spawn_weeds` (same rng, continues the stream)
3. then, if `next_day % townShopUnlockInterval == 0` and under the 8-instance cap:
   `rng.choice(sorted(SHOPS))`

`sorted(SHOPS)` = `["BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"]`.
`choice` uses `_randbelow(8)` → `getrandbits(3)` with rejection sampling, NOT `random()*n`.

Implement in `pyrng.rs`:
- MT19937 with CPython's `init_by_array` seeding (`random.Random(int)` seeds from `abs(int)` split into
  32-bit little-endian words; a 0 seed still produces a 1-word array `[0]`)
- `random()` = `((a >> 5) * 67108864.0 + (b >> 6)) * (1.0 / 9007199254740992.0)` for two successive
  `genrand_uint32` calls `a`, `b`
- `getrandbits(k)` and `_randbelow(n)` with CPython's rejection loop
- Unit-test the whole thing against CPython for many seeds and long streams.

Seed origin: `resolve_episode_seed` takes `configuration["seed"]` if set, else `random.randrange(2**31)`,
stores it in `env.info["seed"]`, and **scrubs it from configuration** so agents cannot observe it.
`(seed * 1_000_003) ^ day` can exceed 2**31 — use i64/u64 and match Python's arbitrary-precision `^`
semantics on non-negative values.

### 4. Money is `f64`
`_new_farm` sets `"money": float(starting_money)`. Prices added/subtracted are ints. Keep money as f64
and match the *order* of arithmetic in `_commit_unit` / `_do_hire` / `_do_buy_land`.

### 5. Order of operations inside `_apply_unit_action` (the LOCKED guard placement)
Exact sequence: movement ops → `PASS` → **`DROP` / `PICKUP` / `PLACE` (these run BEFORE the LOCKED
guard)** → `if tile == "LOCKED": return` → `PLANT`, `WATER`, `HARVEST`, `FERTILIZE`, `DIG`, `BUILD_COOP`,
`BUILD_PASTURE`, `FEED`, `COLLECT_FERTILIZER`, `CARE`.
Shed ops work from a locked tile because three of the four shed-access tiles start locked. Movement onto
locked tiles is allowed; off-board moves are no-ops.

### 6. `interpreter()` per-turn order
For each player in index order: compute atomic-PLANT blocking, then apply farmer action, then each hand
action in order. Then `_process_market` (both players), `_town_consume(step)`, `_decay_plants(step)` for
each farm, then `_end_of_day(day)` if `(step + 1) % turnsPerDay == 0`.
`step` used by `_town_consume`/`_decay_plants` is the **pre-increment** step.
DONE fires when `step >= episodeSteps - 2`, setting `reward = money`.

### 7. Atomic PLANT validation
In `interpreter`, sum PLANT requests per crop across `[farmer, *hands]`. For any crop where
`demand > seeds[crop]`, **every** PLANT request for that crop this turn becomes `PASS` (not just the
excess). Seeds are read from the observation *before* any action applies.

### 8. `_process_market` lockstep details
- Queue truncated to `maxMarketOrdersPerTurn` per player.
- Iterate order index `i` across both players together.
- `HIRE` / `BUY_LAND` are atomic: handled once, in player order, then that slot is cleared.
- Then a per-unit loop: quote both players from the **same pre-commit inventory**, then commit both in
  player order. `SELL` quotes at current inventory; `BUY_PRODUCT` quotes at `inventory - 1`.
- A failed commit clears that player's order (aborts the rest of it).
- Break when nothing was committed. 100_000-iteration escape hatch.
- `_refresh_prices` after each order index.
- `SELL` at price 1 does **not** increment market inventory (floor stays responsive → the $1 floor is an
  infinite sink).
- `BUY_PRODUCT` / `BUY_ANIMAL` fail if `sum(shed.values()) >= shedCapacity`.
- `BUY_SEED` has no shed check (seeds live outside the shed and are uncapped).

### 9. `_decay_plants` cadence
Runs every step. Skips unless `max_lifespan_step >= 0 && step >= max_lifespan_step`, then only acts when
`(step - max_lifespan_step) % 2 == 0`. Decrements `yield_units` by 1; at `<= 0` the tile becomes `WEED`.

### 10. `_daily_refresh_animals` ordering subtlety
Within one call: feed/escape check → production (which POPS `pending_care_bonus`, using it only if
`fed_today`) → **then** the care increment for today → `fertilizer_available = True` → reset
`fed_today`/`cared_today`. Production for a day therefore cannot consume that same day's care bonus.
`fertilizer_available` is set for every surviving animal regardless of fed/cared.

### 11. `_daily_refresh_plants`
Uses `next_day = current_day + 1`. Ongoing-crop production fires when
`days_since_first = next_day - planted_day - first_yield_day` is `>= 0` and `% interval == 0`, capped at
`production_count <= max_yield`. Fertilizer doubles a production only if the plant was **also watered
that day** (`was_watered` is captured BEFORE `watered_today` is reset). When
`production_count == max_yield`, sets `max_lifespan_step = (next_day + 1) * turnsPerDay`.
Weed conversion at `consecutive_unwatered >= 2` happens **before** the production logic (a plant that
dies that night produces nothing).

## Tests and build

* `crates/kaggri-core/tests/sim_parity.rs`: full-episode lockstep parity. `tests/gen_sim_fixtures.py` drives
  the official interpreter over recorded action traces (ten scenarios in `tests/data/sim_trace_*.txt`,
  covering malformed orders, PLANT contention, full sheds, escaped animals, decayed plants, all quadrants,
  prices at the floor) and writes a canonical digest of the entire state after every step; the test replays
  the actions through `sim::step` and asserts the digest matches byte for byte, reporting the first
  divergent step and a field diff on failure.
* `tests/market_price_vectors.rs`: the price function against vectors from `tests/gen_market_vectors.py`.
* `tests/obs_macros.rs`: the observation and action contract (OBS_LAYOUT.md) and the mask invariants.
* `crates/kaggri-py/tests/`: the PyO3 surface (`test_pyo3_surface.py`) and the `Encoder` used by the bundle
  (`test_encode_observation.py`); run with `pytest` after installing the wheel.
* Unit tests inside the crate cover the RNG, rounding and inventory-order traps above.

```
cd rust && cargo test --release -p kaggri-core
pip install maturin && maturin build --release -m crates/kaggri-py/Cargo.toml   # wheel in target/wheels/
```
