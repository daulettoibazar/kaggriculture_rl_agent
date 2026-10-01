# Kaggriculture observation & action layout — `OBS_VERSION = 1`

**This document is the contract the training code is written against.** It is normative:
`crates/kaggri-core/src/obs.rs` and `crates/kaggri-core/src/macros.rs` implement exactly
what is written here, and `tests/obs_macros.rs` pins the numbers. Any change to a slot, a
divisor, a token count or a head width is a **breaking change** and must bump
`OBS_VERSION`. Checkpoints are only comparable within one version.

Environment semantics referenced throughout come from
`kaggle_environments/envs/kaggriculture/kaggriculture.py` (v1.32.7) via `rust/PORT_SPEC.md`.

---

## 0. Headline numbers

| quantity | value |
|---|---|
| Tokens | **140** |
| Features per token | **48** |
| Observation floats | **6 720** (row-major `[140, 48]`; computed in `f32`, **published as `f16`** — §11) |
| Legality mask floats | **646** (`[17, 38]`, published in the same dtype) |
| Per-unit action logits | **138** (100 target + 38 verb) |
| Market action logits | **128** (21 categorical heads) |
| Total policy logits | **2 474** (17 x 138 + 128) |
| Joint action cardinality | `3800^U x 1.214e16`, `U` = live units; **~8.7e76** at a full 17-unit crew |

---

## 1. Buffer shape and conventions

```
encode(gs, player, out)                out.len() == 6720   // OBS_LEN
encode_with_cfg(gs, player, cfg, out)   // same, for non-default Config
legal_mask(gs, player, out)             out.len() == 646    // MASK_LEN
legal_mask_with_cfg(gs, player, cfg, out)
obs_dims() -> ObsDims                   // every constant below, for a startup assert
```

`encode` and `legal_mask` compute in `f32` and always will: every divisor, clip and `ln`
in `obs.rs` is an `f32` expression, and every number in §2–§10 is an `f32` value. What the
batched interface *publishes* is a separate decision, taken in §11: **`f16` by default.**
The two are independent — the layout is unchanged, the published plane is
`np.float16(f32_plane)` exactly.

* Row-major: feature `f` of token `t` lives at `out[t * 48 + f]`.
* `out` is **fully overwritten** every call (zeroed first), so buffers are reusable and
  nothing carries over between steps. Zero heap allocation on the hot path.
* Everything is **player-relative**: `player` is always "me", `1 - player` is "the
  opponent". Self-play needs no side-swapping anywhere else.
* `encode` / `legal_mask` assume `Config::default()` because `PORT_SPEC` pins their
  signatures without a `Config`. Non-default episodes must use the `_with_cfg` variants.
* Boards smaller than 10x10 leave the trailing tile tokens entirely zero — including the
  type one-hot, so "no such tile" is distinguishable from "empty tile". Boards larger than
  10x10 are **not representable** (the target head is a fixed 100).

### Header — features 0..8, identical on every token

| slot | meaning |
|---|---|
| 0..7 | token-type one-hot: `0` GLOBAL, `1` TILE, `2` UNIT, `3` PRODUCT, `4` OPPONENT, `5` PRIVATE, `6` SCRATCH |
| 7 | constant `1.0` bias |

Features `8..48` are the 40-slot **payload**, reused per token type. Slots past each
type's documented range are `0.0`. The widest payload in use is PRIVATE's 38 slots
(payload `0..38` = features `8..46`), so **features 46 and 47 are reserved** on every token
type — spare capacity for a future feature that will not move any existing slot.
`encode_is_deterministic_and_uses_no_padding_slots` asserts they are never written, in both
the opening state and a state where every documented slot is non-zero.

### Token blocks

| tokens | block | offset | count |
|---|---|---|---|
| `[0]` | GLOBAL | 0 | 1 |
| `[1..101]` | TILE (own farm, `t = 1 + y*10 + x`) | 1 | 100 |
| `[101..118]` | UNIT (`101` = farmer, `101+h` = hand `h-1`) | 101 | 17 |
| `[118..127]` | PRODUCT (`PRODUCTS` order) | 118 | 9 |
| `[127..135]` | OPPONENT (4 quadrants, econ, crops, animals, timeline) | 127 | 8 |
| `[135]` | PRIVATE | 135 | 1 |
| `[136..140]` | SCRATCH | 136 | 4 |

### Normalisation rules

Every slot lands in roughly `[-1, 1]` and is hard-clipped; the fuzz test asserts every
float in every reachable state is finite and within `[-4, 4]`.

* **Counts** divide by a structural bound: 25 (a quadrant), 100 (a board or
  `shedCapacity`), 16 (`MAX_HANDS`), 6 (the largest `max_yield` / `max_held`), 32 (a
  quantity bucket ceiling).
* **Money** gets two slots: linear `money / 20 000` clipped to `[-1, 3]`, and
  `ln(1 + max(0, money)) / 12` clipped to `[0, 1.5]`. An episode spans $3 000 → ~$50 000,
  so the linear slot answers "can I afford this" and the log slot keeps a gradient at
  scale.
* **Times** are always **relative to now** and divide by one day (`turnsPerDay`) or by the
  season (30 days). Nothing is an absolute step number the model would have to subtract.
* **Booleans** are exactly `0.0` / `1.0`.
* **Imminence** features (`1 - turns_until / period`) are `1.0` on the turn the event
  fires and fall linearly to just above `0.0` the turn after — monotone in urgency, which
  a ReLU can read directly.

---

## 2. GLOBAL token — `[0]`, payload 0..35

| slot | feature | normalisation |
|---|---|---|
| 0 | `step` | `/ 718` clip `[0,1]` |
| 1 | `day` | `/ 29` clip `[0,1]` |
| 2 | `hour` | `/ 23` |
| 3 | steps remaining | `(718 - step) / 718` clip `[0,1]` |
| 4 | turns left today | `(23 - hour) / 23` clip `[0,1]` |
| 5 | is final day (`day >= 29`) | bool |
| 6 | is final turn of the day (`hour + 1 >= 24`) | bool |
| 7 | own money, linear | `/ 20 000` clip `[-1,3]` |
| 8 | own money, log | `ln(1+m)/12` clip `[0,1.5]` |
| 9 | opponent money, linear | as slot 7 |
| 10 | opponent money, log | as slot 8 |
| 11 | money lead | `(mine - theirs) / 20 000` clip `[-1,1]` |
| 12 | I am ahead | bool |
| 13 | own `hires_today` | `/ 16` clip `[0,2]` |
| 14 | own live hands | `/ 16` |
| 15 | own unlocked quadrant count | `/ 4` |
| 16, 17, 18 | own NE / SW / SE unlocked | bool each |
| 19 | opponent `hires_today` | `/ 16` clip `[0,2]` |
| 20 | opponent live hands | `/ 16` clip `[0,2]` |
| 21 | opponent unlocked quadrant count | `/ 4` |
| 22, 23, 24 | opponent NE / SW / SE unlocked | bool each |
| 25 | town shop instances | `/ 8` (`MAX_SHOP_INSTANCES`) |
| 26 | next shop unlock imminence | `0` if capped at 8, else `1 - days_until / 3`; `1.0` when tonight's `_end_of_day` unlocks one |
| 27 | shop consumption tick imminence | `1 - turns_until / townShopSellInterval`; `1.0` on a tick turn |
| 28 | town-centre tick imminence | `1 - turns_until / townCenterSellInterval`; `1.0` on a tick turn |
| 29 | cost of my next `HIRE` (`mult * fib(hires_today)`) | `/ 64` clip `[0,2]` |
| 30 | price of my next quadrant (`0` if none left) | `/ 4000` |
| 31 | I can afford that quadrant now | bool |
| 32 | own unit count (`1 + hands`) | `/ 17` |
| 33 | own shed fill | `total / shedCapacity` clip `[0,1]` |
| 34 | own shed room | `(cap - total) / cap` clip `[0,1]` |

Slots 26–28 exist because every town event is a *market price* event: shop and centre ticks
drain inventory (raising prices), and an unlock permanently raises demand.

---

## 3. TILE tokens — `[1..101]`, payload 0..37

One token per tile of **the player's own farm**, at `t = 1 + y*10 + x`. This is the block
the target head indexes into: target `k` in the action space is tile token `1 + k`, i.e.
`(x, y) = (k % 10, k / 10)`.

### Always written

| slot | feature | normalisation |
|---|---|---|
| 0..7 | kind one-hot: `0` empty(`None`), `1` `LOCKED`, `2` `WEED`, `3` empty `COOP`, `4` empty `PASTURE`, `5` `PLANT`, `6` occupied structure | one-hot |
| 15 | `x` | `/ 9` |
| 16 | `y` | `/ 9` |
| 17 | is a shed-access tile | bool |
| 18 | moves to the nearest shed-access tile | `/ 9` |
| 19 | own units standing here | `/ 4` |

Slots 7–11 are the crop one-hot and 12–14 the animal one-hot; a tile is never both, and
both regions are zero for the other five kinds.

### If `kind == PLANT` (slot 5 set)

| slot | feature | normalisation |
|---|---|---|
| 7..12 | crop one-hot in `CROPS` order (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON) | one-hot |
| 20 | age in days (`day - planted_day`) | `/ 30` clip `[0,1]` |
| 21 | `yield_units` | `/ 6` clip `[0,1]` |
| 22 | at yield cap (`yield_units >= max_yield`) | bool |
| 23 | harvestable **now** (`yield_units > 0` and `day - planted_day >= first_yield_day`) | bool |
| 24 | days until the yield next grows | `/ 8`; ongoing crops: next scheduled production still under `max_yield`; one-time crops: days to the start of the watering window (`0` inside it); `9/8` = never again |
| 25 | `watered_today` | bool |
| 26 | `consecutive_unwatered` | `/ 2` clip `[0,1]` |
| 27 | **turns into a weed tonight** (`!watered_today && consecutive_unwatered >= 1`) | bool |
| 28 | fertilizer days left (`fertilized_until_day - day + 1`) | `/ 3` clip `[0,1]` |
| 29 | inside the watering bonus window (one-time crops: `(max_yield_day+1)/2 <= age <= max_yield_day`) | bool |
| 30 | has a lifespan (`max_lifespan_step >= 0`) | bool |
| 31 | lifespan relative to now (`(max_lifespan_step - step) / 24`) | clip `[-1,1]`; negative means already decaying |
| 32 | `_decay_plants` removes a unit **this step** | bool |
| 36 | ongoing crop (TOMATO / STRAWBERRY) | bool |

Slot 27 is the single most action-relevant feature on the board: it is the difference
between a plant and a weed, and it is one `WATER` away.

### If the tile is an occupied structure (slot 6 set)

| slot | feature | normalisation |
|---|---|---|
| 12..15 | animal one-hot in `ANIMALS` order (GOOSE, COW, SHEEP) | one-hot |
| 20 | age in days (`day - placed_day`) | `/ 30` clip `[0,1]` |
| 21 | `yield_units` | `/ 6` clip `[0,1]` |
| 22 | at hold cap (`yield_units >= max_held`) | bool |
| 23 | harvestable now (`yield_units > 0`) | bool |
| 24 | days to the next scheduled production | `/ 8` |
| 25 | `fed_today` | bool |
| 26 | `consecutive_unfed` | `/ 2` clip `[0,1]` |
| 27 | **escapes tonight** (`!fed_today && consecutive_unfed >= 1`) | bool |
| 33 | `cared_today` | bool |
| 34 | `fertilizer_available` | bool |
| 35 | `pending_care_bonus` | `/ 4` clip `[0,1.5]` |

Slots 21/22/23/24/25/26/27 are deliberately shared with the plant encoding — they mean the
same thing (stock on the tile, cap, ready, timer, daily-need met, misses, dies tonight) and
sharing keeps the payload inside 40 slots. The kind one-hot disambiguates.

---

## 4. UNIT tokens — `[101..118]`, payload 0..22

Token `101` is the main farmer; `101 + h` is hand `h - 1`. Rows for units that do not exist
carry only slot 21.

| slot | feature | normalisation |
|---|---|---|
| 0 | unit exists | bool (`0.0` for every absent hand) |
| 1 | is the main farmer | bool |
| 2 | `x` | `/ 9` |
| 3 | `y` | `/ 9` |
| 4 | moves to the nearest shed-access tile | `/ 9` |
| 5 | standing on a shed-access tile | bool |
| 6..18 | carried count per shed item, in `SHED_ITEMS` order (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER, GOOSE, COW, SHEEP) | `/ 32` clip `[0,4]` |
| 18 | total carried | `/ shedCapacity` clip `[0,4]` |
| 19 | **first item in insertion order**, as an id | `item_index / 11` |
| 20 | count of that first item | `/ 32` clip `[0,4]` |
| 21 | unit index | `/ 16` (written even for absent units) |

Slots 19–20 exist because unit inventories are insertion-ordered *and the order is
observable*: `_drop_inventories_to_shed` fills the shed in that order and deletes whatever
no longer fits. A unit carrying `{MELON:5, WHEAT:50}` into a shed with 4 slots free keeps 4
melons; the same unit with the keys in the other order loses all the melon. The head of the
order is therefore what survives, and the policy needs to see it.

Unit inventories are **uncapped** (only the shed obeys `shedCapacity`), which is why slot
18 clips at `4x` capacity rather than at `1x`.

---

## 5. PRODUCT tokens — `[118..127]`, payload 0..27

One token per market product, in `PRODUCTS` order, at `118 + product_index`.

| slot | feature | normalisation |
|---|---|---|
| 0..9 | product one-hot (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER) | one-hot |
| 9 | market inventory vs anchor (`inventory - I0`) | `/ 2000` clip `[-1,2]` |
| 10 | **price relative to base** | `price / base` clip `[0,3]` |
| 11 | price move from base | `(price - base) / base` clip `[-1,2]` |
| 12 | absolute price | `/ 300` clip `[0,2]` |
| 13 | price is at the `$1` floor | bool |
| 14 | log price | `ln(1+price)/6` clip `[0,1.5]` |
| 15 | own shed holding of this product | `/ shedCapacity` clip `[0,1]` |
| 16 | own harvestable units of it on the farm | `/ 50` clip `[0,2]` |
| 17 | opponent's harvestable units of it | `/ 50` clip `[0,2]` |
| 18 | town demand per day (shops x ticks/day + centre) | `/ 24` clip `[0,2]` |
| 19 | town demand per shop tick | `/ 8` clip `[0,2]` |
| 20 | `BUY_PRODUCT`-able (WHEAT, FERTILIZER only) | bool |
| 21 | buy price, quoted as the env does at `inventory - 1` | `/ 300` clip `[0,2]` |
| 22 | **market depth**: price after dumping the whole shed holding, over the current price | `market_price(inv + held) / price` clip `[0,1]` |
| 23 | estimated revenue from dumping the whole holding | `held * (price + price_after)/2 / 5000` clip `[0,2]` |
| 24 | `base` | `/ 300` clip `[0,2]` |
| 25 | seed cost, for the five crops (`0` otherwise) | `/ 100` |
| 26 | cost of the animal that produces it (`0` otherwise) | `/ 500` |

Slots 22–23 are the load-bearing market features and are worth their two `market_price`
calls per product per encode (18 calls, a few hundred nanoseconds total). The premium
products (STRAWBERRY, MELON, MILK, WOOL) use `above_target > 1`, so a single quadrant's
production drives them from base to the `$1` floor: the difference between "sell 60 melons"
and "sell 10 melons six times" is thousands of dollars, and it is not readable from `price`
alone. Slot 10 rather than raw price is the primary signal because `base` spans 25→250.

Town demand (18–19) is exposed **per product**, not as shop names. Which shop unlocked is
irrelevant; that it consumes 6 wheat/day is not. Two bakeries and one bakery differ only in
this number, and it is monotone over the season.

---

## 6. OPPONENT tokens — `[127..135]`, payload 0..21

The opponent's board is fully visible but not actionable. What matters is *when and how much
supply they will dump*, so it is summarised rather than mirrored. All eight tokens share
the same header type (`4`) and use a subtype one-hot.

| slot | feature |
|---|---|
| 0..5 | subtype one-hot: `0` QUADRANT, `1` ECON, `2` CROPS, `3` ANIMALS, `4` TIMELINE |
| 5..9 | quadrant one-hot (NW, NE, SW, SE) — QUADRANT tokens only, zero elsewhere |

### QUADRANT — tokens `[127..131]`, one per quadrant in NW, NE, SW, SE order

All fractions divide by the 25 tiles in a quadrant.

| slot | feature | normalisation |
|---|---|---|
| 9 | locked tiles | `/ 25` |
| 10 | empty tiles | `/ 25` |
| 11 | weeds | `/ 25` |
| 12 | empty coops | `/ 25` |
| 13 | empty pastures | `/ 25` |
| 14 | plants | `/ 25` |
| 15 | occupied structures | `/ 25` |
| 16 | harvestable units in the quadrant | `/ 50` clip `[0,2]` |
| 17 | mean plant age (`0` if no plants) | `/ 30` clip `[0,1]` |
| 18 | plants dying tonight | `/ 25` |

### ECON — token `[131]`

| slot | feature | normalisation |
|---|---|---|
| 9 | opponent money, linear | `/ 20 000` clip `[-1,3]` |
| 10 | opponent money, log | `ln(1+m)/12` clip `[0,1.5]` |
| 11 | their lead over me | `(theirs - mine) / 20 000` clip `[-1,1]` |
| 12 | their `hires_today` | `/ 16` clip `[0,2]` |
| 13 | their live hands | `/ 16` clip `[0,2]` |
| 14 | their unlocked quadrant count | `/ 4` |
| 15, 16, 17 | their NE / SW / SE unlocked | bool each |
| 18 | their unit count | `/ 17` clip `[0,2]` |

### CROPS — token `[132]`

| slot | feature | normalisation |
|---|---|---|
| 9..14 | their plant count per crop, `CROPS` order | `/ 25` clip `[0,4]` |
| 14..19 | their harvestable units per crop | `/ 50` clip `[0,2]` |
| 19 | their total plants | `/ 100` clip `[0,1]` |
| 20 | their mean plant age | `/ 30` clip `[0,1]` |

### ANIMALS — token `[133]`

| slot | feature | normalisation |
|---|---|---|
| 9..12 | their animal count per species, `ANIMALS` order | `/ 25` clip `[0,4]` |
| 12..15 | their harvestable product per species | `/ 24` clip `[0,2]` |
| 15 | fraction of their animals fed today | `[0,1]` |
| 16 | fraction cared for today | `[0,1]` |
| 17 | their total animals | `/ 100` clip `[0,1]` |
| 18 | their total empty structures | `/ 100` clip `[0,1]` |

### TIMELINE — token `[134]`

The supply-arrival histogram: how much of their production lands, and when.

| slot | feature | normalisation |
|---|---|---|
| 9..14 | plants whose yield next grows in 0, 1, 2, 3, 4+ days | `/ 25` clip `[0,4]` |
| 14..18 | animals producing in 1, 2, 3, 4+ days | `/ 25` clip `[0,4]` |
| 18 | their total harvestable-now units | `/ 100` clip `[0,2]` |
| 19 | their plants at yield cap | `/ 25` clip `[0,4]` |

---

## 7. PRIVATE token — `[135]`, payload 0..38

Everything in `private` that is not already on a unit token, plus own-farm aggregates.

| slot | feature | normalisation |
|---|---|---|
| 0..5 | seeds per crop, `CROPS` order | `/ 32` clip `[0,4]` |
| 5..8 | unplaced animals in the shed, `ANIMALS` order | `/ 8` clip `[0,4]` |
| 8 | shed fill | `total / shedCapacity` clip `[0,1]` |
| 9 | shed room | `(cap - total) / cap` clip `[0,1]` |
| 10 | shed is full (blocks `BUY_PRODUCT` and `BUY_ANIMAL`) | bool |
| 11..20 | shed count per product, `PRODUCTS` order | `/ shedCapacity` clip `[0,1]` |
| 20..25 | own plant count per crop | `/ 100` clip `[0,1]` |
| 25..28 | own animal count per species | `/ 100` clip `[0,1]` |
| 28 | own empty tiles | `/ 100` |
| 29 | own weeds | `/ 100` |
| 30 | own empty coops | `/ 100` |
| 31 | own empty pastures | `/ 100` |
| 32 | own locked tiles | `/ 100` |
| 33 | own harvestable-now units | `/ 100` clip `[0,2]` |
| 34 | own plants dying tonight | `/ 25` clip `[0,4]` |
| 35 | own animals escaping tonight | `/ 25` clip `[0,4]` |
| 36 | own units carrying anything | `/ 17` clip `[0,2]` |
| 37 | own total carried items | `/ shedCapacity` clip `[0,4]` |

Seeds live outside the shed and are uncapped, hence the separate slots and the `/32` scale.
Note slots 5..8: **unplaced animals count against `shedCapacity`**, which is a documented
failure chain — a full shed silently refuses `BUY_PRODUCT WHEAT`, animals go unfed, and a
$300–500 asset walks away.

---

## 8. SCRATCH tokens — `[136..140]`, payload 0..1

| slot | feature |
|---|---|
| 0 | token index `/ 3` |

Zero-information registers. They cost 4 tokens (2.9 % of the sequence) and give the trunk
somewhere to accumulate global computation without overwriting a real token's
representation — the standard "register token" trick. The market head reads the **last**
token (`[139]`); the value head reads the GLOBAL token (`[0]`).

---

## 9. Action space

Two independent halves per player per turn: one macro order per unit, and one market head.

### 9.1 Unit macros — `(target tile, verb)`

* **Target head**: 100 logits, tile `k` = `(k % 10, k / 10)`, aligning 1:1 with tile tokens
  `1 + k`. Emit one target distribution per unit token.
* **Verb head**: 38 logits per unit token.

The policy never emits raw movement. `resolve_turn` converts each `(target, verb)` into the
single `UnitOp` for this turn: a step along the shortest path while the unit is not standing
on the target, the verb once it is. Orders are **re-issued every turn**, so a macro is a
soft intention, not a commitment — the wrapper carries no state between turns.

#### Verb table

| index | verb | env op emitted |
|---|---|---|
| 0 | `Idle` | `PASS` |
| 1 | `Water` | `WATER` |
| 2 | `Harvest` | `HARVEST` |
| 3 | `Fertilize` | `FERTILIZE` |
| 4 | `Dig` | `DIG` |
| 5 | `BuildCoop` | `BUILD_COOP` |
| 6 | `BuildPasture` | `BUILD_PASTURE` |
| 7 | `Feed` | `FEED` |
| 8 | `Care` | `CARE` |
| 9 | `CollectFertilizer` | `COLLECT_FERTILIZER` |
| 10..15 | `Plant(crop)`, `CROPS` order | `PLANT <crop>` |
| 15..18 | `PlaceAnimal(animal)`, `ANIMALS` order | `PLACE <animal>` (onto a matching empty structure) |
| 18..28 | `Pickup(item, qty)`, `index = 18 + 2*item + qty` | `PICKUP <item> <n>` |
| 28..37 | `Deposit(product)`, `PRODUCTS` order | `PLACE <product> 1000000` |
| 37 | `Drop` | `DROP` |

`Pickup` items are `[WHEAT, FERTILIZER, GOOSE, COW, SHEEP]` (index 0..5) and quantities are
`[1, 8]` (index 0..2).

#### Omissions, and why

* **`NORTH/SOUTH/EAST/WEST`** — replaced by the target head. This is the whole design.
* **`PASS`** — is `Idle`.
* **`PICKUP` of the other 7 shed items** — produce is sold *out of the shed*; moving it into
  an inventory can only expose it to the end-of-day overflow. Only WHEAT (feeding),
  FERTILIZER (fertilizing) and the three animals (placing) have a downstream use.
* **A `Pickup` quantity larger than 8** — a hand has at most 23 actions in a day and must
  walk between tiles, so more than 8 single-use consumables per trip is unreachable. The
  env clamps `n` to what the shed holds, so `8` against a shed holding 3 takes 3.
* **`Deposit` of an animal** — `PLACE <animal>` hits the animal-placement branch when the
  unit stands on a matching empty structure, and a shed-access tile *can* hold a coop, so
  such a verb would silently mean "place" on exactly the tiles a deposit targets. Surplus
  animals return to the shed for free in the end-of-day drop. Use `PlaceAnimal` to place.
* **A `Deposit` quantity head** — `PLACE` clamps to `min(n, held)` then to the shed room and
  leaves the remainder in the inventory, so an oversized `n` *is* the optimal deposit.
* **`Drop` is kept but dominated.** `DROP` `del`etes every key it visits whether or not it
  fit; `PLACE` clamps losslessly. `Drop`'s only edge is emptying a mixed inventory in one
  action, so the resolver only emits it when the shed has room for the whole inventory
  (see 9.3).

#### Cardinality

Per unit: 100 x 38 = **3 800** joint, **138** logits factorised. Across a full 17-unit
crew: 2 346 logits, `3800^17 ~= 7.2e60` joint.

### 9.2 Market head — 21 categorical heads, 128 logits

In `MarketHead::from_choices` order:

| head | index | categories | meaning |
|---|---|---|---|
| `hire` | 0 | 11 | hands to hire this turn, `0..=10` |
| `buy_land` | 1 | 2 | buy the next quadrant |
| `seed_qty[crop]` | 2..7 | 7 each | bucket into `QTY_BUCKETS`, `CROPS` order |
| `animal_qty[animal]` | 7..10 | 7 each | bucket into `QTY_BUCKETS`, `ANIMALS` order |
| `buy_product[i]` | 10..12 | 7 each | bucket into `QTY_BUCKETS` for `[WHEAT, FERTILIZER]` |
| `sell_frac[product]` | 12..21 | 5 each | bucket into `SELL_FRACS`, `PRODUCTS` order |

```
QTY_BUCKETS = [0, 1, 2, 4, 8, 16, 32]
SELL_FRACS  = [0.0, 0.25, 0.5, 0.75, 1.0]      // fraction of the shed holding
```

Quantities are buckets rather than one-per-unit orders because **a single order index drains
its whole quantity**: the inner lockstep loop repeats until `remaining` hits 0 or a commit
fails, so quantity is free in slot terms. `32` covers a whole 25-tile quadrant of seeds in
one order. Sells are *fractions of the holding* because holdings span three orders of
magnitude over an episode and "sell half" is the scale-free thing to say; a non-zero
fraction of a non-zero holding always rounds **up** to at least 1 unit.

Out-of-range choices clamp to the last category rather than panicking, so a malformed
sample can never kill an env worker.

Market joint cardinality: `11 * 2 * 7^10 * 5^9 = 1.2138e16`.

### 9.3 Packer — head to market queue

`pack_market` turns a `MarketHead` into an ordered queue of at most
`maxMarketOrdersPerTurn` (10) orders. Slot **inclusion** and queue **position** are ranked
separately, because they are ranked by different things.

**Inclusion priority** — what earns one of the 10 slots. Ranked by what dropping the
request costs *permanently*, since anything dropped is re-issuable next turn at no cost:

| rank | order | cost of a one-turn deferral |
|---|---|---|
| 1 | `HIRE` | a hand hired at hour `h` gets `23 - h` actions and every hand is wiped at midnight, so a deferred hire is permanently one action-day shorter |
| 2 | `BUY_PRODUCT` | wheat feeds animals; a missed buy on the second unfed day loses a $300–500 animal outright |
| 3 | `BUY_LAND` | ~25 tile-days, and there are only ever three of these per episode |
| 4 | `SELL` | ~0: the goods sit in the shed and town consumption *raises* the price while they wait |
| 5 | `BUY_SEED` | one turn of planting delay |
| 6 | `BUY_ANIMAL` | one turn of production delay on the most expensive purchase in the game |

The consequence is deliberate: asking for 10 hands at hour 0 leaves no slots for anything
else that turn, which is exactly the measured advice ("plan hour 0 as a pure hiring turn and
start selling at hour 1, or hire 8 and keep 2 slots"). The policy controls `hire` and can
express both.

**Execution order** — where in the queue. `_process_market` walks order index `i` across
both players, quotes both against the *same* pre-commit inventory, commits in player order,
then refreshes prices. Two consequences:

* an earlier index sells into a fresher market — measured at **+$644 (12.6 %)** on a
  100-carrot sale when the opponent sold at index 3 instead of index 0;
* atomic orders at index `i` see the money left by every commit at index `< i`, so a `SELL`
  at index 0 **funds** a `BUY_LAND` or `HIRE` later in the same turn.

So the queue is written:

```
SELL (descending units x current price)  ->  BUY_PRODUCT (WHEAT, FERTILIZER)
  ->  BUY_SEED (crop order)  ->  BUY_ANIMAL (animal order)  ->  BUY_LAND  ->  HIRE x n
```

Sells first (freshest quote, and they fund what follows), largest sale first. Buys after the
sells because a sell raises market inventory and therefore *lowers* the buy price. `HIRE`
last because its value does not depend on its index at all — a hand hired at index 0 and one
hired at index 9 both start acting next turn — so it is the one order that can afford to be
funded by everything before it.

**Requests dropped as provably dead** (never a wasted slot):

* `SELL` of a product the shed does not hold;
* `BUY_PRODUCT` / `BUY_ANIMAL` when the shed will still be at capacity after this turn's
  sells (`_commit_unit` refuses both at `sum(shed) >= shedCapacity`; `BUY_SEED` has no shed
  check and is never dropped for this);
* `BUY_LAND` when all three quadrants are unlocked;
* `HIRE` beyond `MAX_HANDS = 16`, so every unit keeps an addressable macro order and the
  unit-token block stays a fixed 17;
* when the queue contains **no sells** — and therefore money can only fall during the turn,
  making current money a hard upper bound — any purchase the balance cannot cover, and the
  unaffordable tail of a hire run priced at `mult * fib(hires_today + j)`. With sells queued
  the packer makes *no* affordability judgement, because money at execution time is
  strictly higher than it is now.

**Known conservatism**: sell volumes are quoted against the shed as of the *start* of the
turn, so items deposited by this turn's unit actions are sold next turn. This matches the
recommended pipeline anyway (sell on turn `t`, deposit on turn `t+1`, because `PLACE`
resolves in phase 1 and sees the shed *before* the same turn's `SELL` empties it).

### 9.4 What the resolver refuses, and what it does not

For each unit, in order:

1. unit absent / verb `Idle` / target off-board → `PASS`
2. the target tile's current state does not admit the verb → `PASS` (**target became
   invalid**: plant turned to weed, animal escaped, tile already built on)
3. not standing on the target → one move along the shortest path
4. the verb would consume something the unit does not have → `PASS`
5. another unit already claimed this (tile, verb class) this turn → `PASS`
6. otherwise → the verb

Steps 2, 4 and 5 make every emitted op *effective*: never a guaranteed no-op, never a
destruction the policy did not ask for. Note the asymmetry in step 3 — **walking is not
gated on resources**, only on tile state. A unit with no seeds may legitimately be sent
towards the tile it will plant, because `BUY_SEED` settles in the market phase, *after* this
turn's unit actions.

Three cross-unit protections, all inside one turn:

* **Atomic `PLANT`.** The interpreter voids *every* `PLANT` for a crop whose summed demand
  across `[farmer, *hands]` exceeds `seeds[crop]` — a 12-action wipe with a big crew. The
  resolver hands out a per-crop seed budget in unit order, so the rule can never fire.
  `out.hands` is also always exactly `hands.len()` long, because phantom hand slots would
  still count towards that demand.
* **Once-per-tile verb classes.** `WATER`, `FEED`, `CARE`, `COLLECT_FERTILIZER`, `HARVEST`,
  `FERTILIZE` and `PLANT` (as one class) are claimed per target tile; a second unit issuing
  the same class on the same tile gets `PASS`. A duplicate is a wasted action for all of
  them and a *destroyed item* for `FERTILIZE`. Deliberately **not** grouped: `DIG` then
  `BUILD_COOP` on one tile is a legitimate two-unit combo, and `FERTILIZE` then `WATER` on
  one plant doubles the day's bonus.
* **Shared shed room.** `Deposit` and `Drop` reserve the room they will use, so two units
  cannot both pass a "there is room for my whole inventory" test and have the second one's
  overflow deleted.

`DIG` on a plant that still holds yield is **not** refused — clearing a decaying plant to
reuse the tile is legitimate play, and the tile token exposes everything needed to judge it.

---

## 10. Legality mask — `legal_mask`, `[17, 38]`

`mask[u * 38 + v] == 1.0` means "there is at least one target tile for which unit `u`'s verb
`v` would take effect this turn" — exactly the condition under which `resolve_turn` can emit
that verb for that unit. Rows for absent units are all zero, `Idle` included, so "no unit
here" is unambiguous.

**The mask is not applied anywhere.** `resolve_turn` does not consult it and the policy
should not hard-mask with it. It is supplied as an optional extra input plane and as an
optional auxiliary loss target, to be A/B'd: the Orbit Wars winner found that hard-masking
obviously-silly actions made his model measurably worse, and this environment makes unmasked
exploration free — every illegal action is a silent no-op that mutates nothing, so the worst
case is a wasted turn, never a corrupted game.

Per-verb condition (tile-state existence AND the unit's own resources):

| verb | legal when |
|---|---|
| `Idle` | always (for a live unit) |
| `Water` | some plant is not watered today |
| `Harvest` | some tile has `yield_units > 0`, and if it is a plant, `day - planted_day >= first_yield_day` |
| `Fertilize` | some tile is a plant **and** the unit holds FERTILIZER |
| `Dig` | some tile is a weed, an empty coop/pasture, or a plant |
| `BuildCoop` / `BuildPasture` | some tile is empty |
| `Feed` | some animal is unfed **and** the unit holds WHEAT |
| `Care` | some animal is not cared for today |
| `CollectFertilizer` | some animal has `fertilizer_available` |
| `Plant(crop)` | some tile is empty **and** `seeds[crop] >= 1` |
| `PlaceAnimal(a)` | some matching empty structure exists **and** the unit holds `a` |
| `Pickup(item, _)` | `shed[item] >= 1` |
| `Deposit(product)` | the unit holds it **and** the shed has room |
| `Drop` | the unit carries something **and** the shed has room for *all* of it |

The mask is deliberately the **more permissive** of the two: it is computed per unit and
ignores cross-unit contention (the shared seed budget, the tile claims, the shared shed
room), so `resolve_turn` may still decline a verb the mask calls legal. The converse never
happens, and `check_mask_agrees_with_resolver` asserts it over full random episodes.

Per-`(unit, tile, verb)` legality is deliberately **not** provided: at 17 x 100 x 38 =
64 600 flags it is ten times the size of the observation. Tile-level legality is a linear
readout of the tile tokens, which carry kind, crop, animal, yield and every once-per-day
flag.

---

### 10.1 Optional tile-verb plausibility plane — `tile_verb_mask`, `[100, 38]` u8

An optional plane for the self-play trainer. `out[t * 38 + v] == 1` iff
`macros::target_plausible(verb v, tile t)` holds right now — the resolver's own step-2
tile-state gate (§9.4), verbatim, over the action row's target index `t = y * boardSize + x`.
It ignores per-unit resources and same-turn cross-unit claims (those live in §10 and in
`resolve_turn`), so AND-ing row `t` with unit row `u` of §10 for the sampled verb gives exactly
the `(target, verb)` pairs at which unit `u` can fire once it stands on `t`. `Idle` has no
target and is `0` everywhere. Invariant, pinned by `tests/obs_macros.rs`
(`tile_mask_agrees_with_unit_mask_and_resolver`) and `test_pyo3_surface.py`
(`vecenv_tile_mask_plane`): every non-`Idle` verb the unit mask calls legal has at least one
plausible tile, and a unit standing on a plausible tile that orders that `(tile, verb)` is
never a structural `PASS`.

Surface: `PyVecEnv(..., tile_mask=True)` allocates an `(n_envs, 2, 3800)` `uint8` plane,
republished with `obs`/`mask` on every `reset`/`step`/`warmup_pass`, reachable as
`env.tile_mask` and `info["tile_mask"]` (`None` / absent unless requested, so the shipped
surface is unchanged); `Encoder.tile_mask(obs)` and `Encoder.encode_full(obs)` give the same
plane for one Kaggle observation; `kaggri.TILE_MASK_DIM == 3800`. Cost ~1 µs per seat-step.


## 11. The batched interface — `kaggri.PyVecEnv`

Sections 1–10 specify the layout; this section specifies how it crosses into numpy, which
is what the training code actually touches. `kaggri.ACTION_VERSION == 1` covers the
packing below; a change to it bumps `ACTION_VERSION`, and a change to a slot, divisor,
token count or head width bumps `OBS_VERSION` as before.

```python
v = kaggri.PyVecEnv(n_envs, config=None, seeds=None,
                    auto_reset=True, units_per_player=17, chunk_envs=1,
                    obs_dtype=None)          # None -> kaggri.OBS_DTYPE == "float16"
obs, mask                     = v.reset()
obs, mask, rew, done, info    = v.step(actions)
```

### Outputs

| object | shape | dtype | notes |
|---|---|---|---|
| `obs` | `(n_envs, 2, 6720)` | **`float16`** | `obs_dtype`; reshape the last axis to `(140, 48)` |
| `mask` | `(n_envs, 2, 646)` | **`float16`** | `obs_dtype`; reshape to `(17, 38)`; strictly `0.0`/`1.0` |
| `rew` | `(n_envs, 2)` | `float32` | `0.0` until DONE, then final money |
| `done` | `(n_envs,)` | `bool` | the terminal step, before any auto-reset |
| `info["final_money"]` | `(n_envs, 2)` | `float32` | non-zero only on the DONE step |
| `info["step"]` | `(n_envs,)` | `int64` | the terminal step number |

Axis 1 is always the player, and every plane is player-relative: `[i, p]` is what player
`p` of env `i` sees, with itself as "me". Self-play needs no side-swapping.

### The published dtype — `f16`, deliberately

**The two input planes are published as `float16`. This is the shipped default and it is a
run-level decision, not a tuning knob: it is the policy's input dtype, so checkpoints are
not comparable across a change to it.** `kaggri.OBS_DTYPE` is the default,
`kaggri.OBS_DTYPES` is what is accepted, `kaggri.obs_dims_dict()["obs_dtype"]` carries it
in the startup contract, and `PyVecEnv(..., obs_dtype="float32")` restores the wide plane
for an A/B. Anything numpy resolves to a 2- or 4-byte float is accepted (`"float16"`,
`"f2"`, `np.float16`, `np.dtype("float32")`, …); anything else raises.

`OBS_VERSION` stays **1**. Nothing in §1–§10 moved: no slot, divisor, token count or head
width changed, and the published plane is *exactly* `np.float16` of the `f32` plane, value
for value. The dtype is part of the *interface*, and it is recorded here and in
`obs_dims_dict()` so a run can pin what it trained against.

The published plane is exactly `np.float16` of the `f32` plane, value for value: worst absolute
error 9.7e-4, worst relative error 4.8e-4 (`2^-11`), and the legality mask is lossless since
`{0.0, 1.0}` are exact in `f16`. Every one-hot, flag and small-integer count is exact; only the
log-money, money-over-20000 and price-ratio slots round, and their companion slots carry the
precision. One slot to watch: **`(inventory - I0) / 2000`** (PRODUCT 9) resolves one unit of
inventory as `5e-4`, so above `inventory - I0 ≈ 2 000` two inventories one unit apart share an
`f16`; if a policy ever dumps thousands of units into one product, rescale this slot rather than
widening the whole plane.

`rewards` and `info["final_money"]` are deliberately **not** covered by `obs_dtype`: they
carry money, which reaches ~1e5 in a good episode, quantises to steps of 64 in `f16` and
saturates to `inf` past 65 504. They are also 2 floats per env against the observation's
13 440, so there is no bandwidth argument for them.

Pinned by `f16_paths_agree_bit_for_bit` and `f16_publication_of_a_real_episode_is_faithful` in
`crates/kaggri-core/tests/obs_macros.rs` and `vecenv_published_dtype` in
`crates/kaggri-py/tests/test_pyo3_surface.py`.

**Every returned array is an env-owned buffer that the next `step` overwrites in place.**
Copy what must survive the step (`np.array(obs)`, or a write straight into the rollout
buffer). Nothing may touch them from another thread while `step` runs — `step` releases the
GIL and writes them from rayon workers.

With `auto_reset=True`, a finished env is restarted *inside* the same step, so `obs`/`mask`
describe the **fresh** episode while `done`, `rew`, `info["step"]` and
`info["final_money"]` describe the **terminal** one. Seeds are reproducible: episode 0 uses
`seeds[i]` verbatim (identical to Python's `configuration={"seed": N}`), and later episodes
use splitmix64 masked to 31 bits, so any episode a run touches can be replayed in the real
environment.

### Actions

A C-contiguous `int64` or `int32` array holding `n_envs * 2 * player_action_dim` values —
shape `(n_envs, 2, player_action_dim)` or the flat `(n_envs, 2 * player_action_dim)`.
With the default `units_per_player = 17`, `player_action_dim = 2*17 + 21 = 55` and
`action_dim = 110`.

One player's row is three concatenated blocks, in the order the heads come out of the
network (`torch.cat([targets, verbs, market], -1)`):

| offset | width | head | range |
|---|---|---|---|
| `0` | `units_per_player` | target tile per unit (§9.1) | `0..100` |
| `units_per_player` | `units_per_player` | verb per unit (§9.1, `kaggri.VERB_NAMES`) | `0..38` |
| `2 * units_per_player` | `21` | market choices (§9.2, `kaggri.MARKET_HEAD_SIZES`) | per head |

Unit `0` is the farmer and unit `1 + h` is hand `h`, matching observation tokens `101 + u`
and mask rows `u`. **An all-zero array is a legal all-`PASS` turn** — every verb slot is
`Idle` and every market head is its do-nothing category.

The decode is total, and the two out-of-range directions differ on purpose:

* **an out-of-range target *or* verb idles that unit** (a `PASS`). Clamping would silently
  retarget the order at tile 99, or turn a garbage verb into `Drop`; idling cannot spend
  money, destroy an item or move a unit.
* **a negative market choice becomes category 0** — do nothing on every one of the 21 heads.
* **an over-range market choice is clamped to the head's last category** by
  `MarketHead::from_choices` (§9.2). Note the consequence: a row of large positive integers
  is *not* a no-op, it is the maximal market request.

So a malformed sample from an untrained policy can neither panic a worker nor spend money
the policy did not ask to spend.

### Startup assertions

```python
d = kaggri.obs_dims_dict()
assert (kaggri.OBS_VERSION, kaggri.ACTION_VERSION) == (1, 1)
assert d["version"] == 1
assert d["obs_dtype"] == kaggri.OBS_DTYPE == "float16"   # the published input dtype
assert list(kaggri.OBS_DTYPES) == ["float16", "float32"]
assert (d["n_tokens"], d["n_features"], d["obs_len"]) == (140, 48, 6720)
assert (d["max_units"], d["n_targets"], d["n_verbs"]) == (17, 100, 38)
assert d["mask_len"] == 646
assert (d["n_market_heads"], d["market_logits"]) == (21, 128)
assert (d["off_tile"], d["off_unit"], d["off_product"]) == (1, 101, 118)
assert (d["off_opp"], d["off_private"], d["off_scratch"]) == (127, 135, 136)
assert sum(kaggri.MARKET_HEAD_SIZES) == kaggri.MARKET_LOGITS == 128
```

The same constants in Rust:

```rust
let d = obs_dims();
assert_eq!(d.version, 1);
assert_eq!((d.n_tokens, d.n_features, d.obs_len), (140, 48, 6720));
assert_eq!((d.max_units, d.n_targets, d.n_verbs), (17, 100, 38));
assert_eq!(d.mask_len, 646);
assert_eq!((d.n_market_heads, d.market_logits), (21, 128));
assert_eq!((d.off_tile, d.off_unit, d.off_product), (1, 101, 118));
assert_eq!((d.off_opp, d.off_private, d.off_scratch), (127, 135, 136));
```

`crates/kaggri-py/src/lib.rs` asserts the first five of these **at compile time**, so the
layout and the version it advertises cannot drift apart.

### Sizing

A batched step is dominated by the store into the observation plane, so `n_envs` should be at least
1,024 (below that the fixed rayon fan-out dominates). `chunk_envs` is the rayon chunk granularity and
cannot change a returned value; keep `n_envs / chunk_envs` at or above the worker count.

---

Tests that pin this document: `crates/kaggri-core/tests/obs_macros.rs` (layout numbers, mask
invariants, the `f16` publication), `crates/kaggri-py/tests/test_pyo3_surface.py` (shapes, dtypes,
buffer reuse, decode rules, `PyVecEnv`-vs-`PyGame` lockstep) and
`crates/kaggri-py/tests/test_encode_observation.py` (the single-observation `Encoder` used by the
submission bundle). Game semantics are pinned separately by `tests/sim_parity.rs`.
