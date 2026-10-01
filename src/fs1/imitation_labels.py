"""Convert a public program's raw Kaggle turn into the policy's macro action row, for imitation.

The programs act in the raw grammar (`["NORTH"]` one tile per turn, `["WATER"]` on the tile the
unit stands on, market orders with unit counts); the policy acts in the macro grammar (per unit a
target tile and a verb the resolver walks to and performs; market heads with quantity buckets and
sell fractions). The mapping (spec of 2026-09-24, cited in research/fs1/PLAN.md §23):

* a unit op performed in place -> (target = the unit's tile y*10+x, verb id); the raw op and the
  macro verb coincide on that tile;
* a raw move (NORTH/SOUTH/EAST/WEST) is a step along the resolver's own shortest path, so every
  move step since the unit's previous op is labelled with the (tile, verb) of the NEXT op it performs
  ("look-ahead"); a walk never followed by an op has no macro equivalent and stays unlabelled;
* `PASS` -> IDLE (no target); `PLACE animal` on a structure -> PlaceAnimal; `PLACE product` ->
  Deposit(product) (the macro deposits everything that fits); `PICKUP item n` -> the 1 or 8 bucket;
  `DROP` -> Drop (illegal labels are dropped by the loss);
* market: HIRE count -> head 0; BUY_LAND -> head 1; BUY_SEED / BUY_ANIMAL / BUY_PRODUCT quantities
  -> the nearest QTY bucket [0,1,2,4,8,16,32]; SELL n -> the sell-fraction bucket whose unit count
  against the start-of-turn shed holding is closest to n (c(f) = h if f == 1 else clamp(ceil(h f), 1, h)).
  All 21 heads are labelled every teacher step (a head without an order is the no-op bucket 0).
"""
from __future__ import annotations

import math

import numpy as np

from . import constants as C

VERB_OF = {"WATER": 1, "HARVEST": 2, "FERTILIZE": 3, "DIG": 4, "BUILD_COOP": 5, "BUILD_PASTURE": 6,
           "FEED": 7, "CARE": 8, "COLLECT_FERTILIZER": 9, "DROP": 37}
MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
CROP_IDX = {name: i for i, name in enumerate(C.PRODUCTS[:5])}
PRODUCT_IDX = {name: i for i, name in enumerate(C.PRODUCTS)}
ANIMAL_IDX = {name: i for i, name in enumerate(C.ANIMALS)}
PICKUP_SLOT = {"WHEAT": 0, "FERTILIZER": 1, "GOOSE": 2, "COW": 3, "SHEEP": 4}
QTY_BUCKETS = (0, 1, 2, 4, 8, 16, 32)
SELL_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
H_HIRE, H_LAND, H_SEED0, H_ANIMAL0, H_BUY_WHEAT, H_BUY_FERT, H_SELL0 = 0, 1, 2, 7, 10, 11, 12
MK = 2 * C.MAX_UNITS


def unit_verb(op) -> int | None:
    """Raw unit op list -> macro verb id, None for a move / unknown / unmappable op."""
    if not isinstance(op, (list, tuple)) or not op or not isinstance(op[0], str):
        return C.VERB_IDLE
    name = op[0].upper()
    if name in MOVES:
        return None
    if name == "PASS":
        return C.VERB_IDLE
    if name in VERB_OF:
        return VERB_OF[name]
    if name == "PLANT" and len(op) >= 2 and str(op[1]).upper() in CROP_IDX:
        return 10 + CROP_IDX[str(op[1]).upper()]
    if name == "PICKUP" and len(op) >= 2 and str(op[1]).upper() in PICKUP_SLOT:
        n = int(op[2]) if len(op) >= 3 else 1
        return 18 + 2 * PICKUP_SLOT[str(op[1]).upper()] + (1 if n >= 8 else 0)
    if name == "PLACE" and len(op) >= 2:
        item = str(op[1]).upper()
        if item in ANIMAL_IDX:
            return 15 + ANIMAL_IDX[item]
        if item in PRODUCT_IDX:
            return 28 + PRODUCT_IDX[item]
    return None


def is_move(op) -> bool:
    return isinstance(op, (list, tuple)) and bool(op) and isinstance(op[0], str) and op[0].upper() in MOVES


def qty_bucket(n: float) -> int:
    return int(np.argmin([abs(b - n) for b in QTY_BUCKETS]))


def sell_bucket(n: float, have: float) -> int:
    if n <= 0 or have <= 0:
        return 0
    best, best_d = 0, float("inf")
    for i, f in enumerate(SELL_FRACS):
        if f == 0:
            continue
        c = have if f >= 1.0 else min(max(math.ceil(have * f), 1), have)
        d = abs(c - n)
        if d < best_d:
            best, best_d = i, d
    return best


def market_bins(orders, shed: dict) -> np.ndarray:
    """Raw market order list + start-of-turn shed holdings -> the 21 head bins."""
    out = np.zeros(C.N_MARKET_HEADS, dtype=np.int64)
    hires = 0
    seed_q = np.zeros(5)
    animal_q = np.zeros(3)
    buy_q = np.zeros(2)
    sell_q = np.zeros(9)
    for o in orders or []:
        if not isinstance(o, (list, tuple)) or not o or not isinstance(o[0], str):
            continue
        kind = o[0].upper()
        try:
            n = float(o[2]) if len(o) >= 3 else 1.0
        except (TypeError, ValueError):
            continue
        if kind == "HIRE":
            hires += 1
        elif kind == "BUY_LAND":
            out[H_LAND] = 1
        elif kind == "BUY_SEED" and len(o) >= 2 and str(o[1]).upper() in CROP_IDX and n > 0:
            seed_q[CROP_IDX[str(o[1]).upper()]] += n
        elif kind == "BUY_ANIMAL" and len(o) >= 2 and str(o[1]).upper() in ANIMAL_IDX and n > 0:
            animal_q[ANIMAL_IDX[str(o[1]).upper()]] += n
        elif kind == "BUY_PRODUCT" and len(o) >= 2 and str(o[1]).upper() in ("WHEAT", "FERTILIZER") and n > 0:
            buy_q[0 if str(o[1]).upper() == "WHEAT" else 1] += n
        elif kind == "SELL" and len(o) >= 2 and str(o[1]).upper() in PRODUCT_IDX and n > 0:
            sell_q[PRODUCT_IDX[str(o[1]).upper()]] += n
    out[H_HIRE] = min(hires, C.MARKET_HEAD_SIZES[0] - 1)
    for i in range(5):
        out[H_SEED0 + i] = qty_bucket(seed_q[i]) if seed_q[i] > 0 else 0
    for i in range(3):
        out[H_ANIMAL0 + i] = qty_bucket(animal_q[i]) if animal_q[i] > 0 else 0
    out[H_BUY_WHEAT] = qty_bucket(buy_q[0]) if buy_q[0] > 0 else 0
    out[H_BUY_FERT] = qty_bucket(buy_q[1]) if buy_q[1] > 0 else 0
    for i, name in enumerate(C.PRODUCTS):
        out[H_SELL0 + i] = sell_bucket(sell_q[i], float(shed.get(name, 0))) if sell_q[i] > 0 else 0
    return out


class LabelBuffer:
    """Look-ahead labelling of one seat's teacher steps within a rollout.

    `observe(t, state, turn)` records the raw turn taken at rollout step `t` from `state` (the
    state BEFORE the turn); unit ops performed in place label that step at once, moves are held
    until the unit's next in-place op, which then labels the whole walk with (that tile, verb).
    Hands are re-hired every day, so pending walks of hands are dropped at a day boundary.
    `drain()` returns the rows touched since the last drain as (t, row55, unit_label17, market21)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.rows: dict[int, np.ndarray] = {}
        self.unit_label: dict[int, np.ndarray] = {}
        self.pending: dict[int, list[int]] = {}         # unit index -> rollout steps of its unfinished walk
        self.touched: set[int] = set()
        self.last_day = None

    def _row(self, t: int):
        if t not in self.rows:
            self.rows[t] = np.zeros(C.PLAYER_ACTION_DIM, dtype=np.int64)
            self.unit_label[t] = np.zeros(C.MAX_UNITS, dtype=bool)
        self.touched.add(t)
        return self.rows[t], self.unit_label[t]

    def observe(self, t: int, state: dict, seat: int, turn: dict):
        farm = state["farms"][seat]
        day = state.get("day")
        if self.last_day is not None and day != self.last_day:
            for u in list(self.pending):
                if u != 0:
                    self.pending.pop(u)                  # yesterday's hands are gone
        self.last_day = day
        row, ulab = self._row(t)
        row[MK:] = market_bins((turn or {}).get("market", []), state["privates"][seat].get("shed", {}))
        ops = [(turn or {}).get("farmer", ["PASS"])] + list((turn or {}).get("hands", []) or [])
        positions = [farm["farmer"]] + list(farm["hands"])
        for u, op in enumerate(ops[:C.MAX_UNITS]):
            if u >= len(positions):
                break
            if is_move(op):
                self.pending.setdefault(u, []).append(t)
                continue
            v = unit_verb(op)
            if v is None:
                self.pending.pop(u, None)
                continue
            x, y = positions[u]
            tile = int(y) * 10 + int(x)
            for tp in self.pending.pop(u, []):
                if tp in self.rows:
                    r, ul = self._row(tp)
                    r[u], r[C.MAX_UNITS + u], ul[u] = (tile if v != C.VERB_IDLE else 0), v, v != C.VERB_IDLE
            row[u] = tile if v != C.VERB_IDLE else 0
            row[C.MAX_UNITS + u] = v
            ulab[u] = True

    def drain(self):
        out = []
        for t in sorted(self.touched):
            out.append((t, self.rows[t].copy(), self.unit_label[t].copy(), np.ones(C.N_MARKET_HEADS, dtype=bool)))
        self.touched.clear()
        return out
