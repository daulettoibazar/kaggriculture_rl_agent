#!/usr/bin/env python3
"""
Tests for the PyO3 surface itself (crates/kaggri-py/src/lib.rs).

Bit-exactness against the Python environment is `parity/diff_test.py`'s job. This
file covers the things that live only in the binding layer and that the parity
harness cannot see:

  * config parsing (camelCase keys, defaults, marketParams merge, typo rejection)
  * malformed-action tolerance (the `isinstance(..., list)` / `isinstance(..., dict)`
    ladder) and the one input CPython genuinely raises on
  * `state_dict()` key ORDER and Python types (trap 2, trap 4)
  * PyGame vs PyVecEnv agreement -- two independent drive paths over one core sim
  * PyVecEnv numpy plumbing: shapes, dtypes, buffer reuse, determinism, auto-reset
  * the published dtype: f16 by default, f32 on request, and the f16 plane being
    exactly `np.float16` of the f32 one

Run:  python3 crates/kaggri-py/tests/test_pyo3_surface.py
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

import kaggri

FAILURES: list[str] = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILURES.append(f"{name}: {detail}")


def case(fn):
    print(f"\n--- {fn.__name__} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(f"{fn.__name__}: raised\n{traceback.format_exc()}")
        print(traceback.format_exc())
    return fn


PASS = {"farmer": ["PASS"], "hands": [], "market": []}

#: The dtype `PyVecEnv` publishes the obs/mask planes in unless asked otherwise.
OBS_DT = np.dtype(kaggri.OBS_DTYPE)


def obs_tol(arr, want):
    """Tolerance for comparing a published slot against a Python-computed reference.

    The layout is computed in f32 inside Rust and published in `arr.dtype`, while a
    reference computed here is float64. So a legitimate difference is one f32 rounding
    plus the published dtype's quantum -- about 1e-3 at f16 and 1e-7 at f32. Anything
    larger is a real disagreement, and every check that uses this is separating values
    that differ by 0.1 or more.
    """
    return 1e-6 + float(np.spacing(np.asarray(want, dtype=arr.dtype)))


# ==========================================================================
@case
def config_parsing():
    d = kaggri.default_config()
    check("defaults match kaggriculture.json",
          (d["episodeSteps"], d["boardSize"], d["startingMoney"],
           d["maxMarketOrdersPerTurn"], d["turnsPerDay"], d["shedCapacity"],
           d["weedSpawnChance"], d["townShopUnlockInterval"],
           d["townShopSellInterval"], d["townCenterSellInterval"],
           d["farmHandCostMult"]) ==
          (720, 10, 3000, 10, 24, 100, 0.005, 3, 4, 24, 1),
          repr(d))
    check("MARKET_PARAMS round-trips",
          d["marketParams"]["CARROT"] == {"base": 35.0, "I0": 10000, "T": 450.0,
                                          "below_func": "hinge", "below_target": 1.0,
                                          "above_func": "sqrt", "above_target": 0.7},
          repr(d["marketParams"]["CARROT"]))

    g = kaggri.PyGame({"boardSize": 6, "turnsPerDay": 8, "seed": 99})
    c = g.config()
    check("overrides applied", (c["boardSize"], c["turnsPerDay"], c["seed"]) == (6, 8, 99), repr(c))
    check("board is boardSize x boardSize",
          len(g.state_dict()["farms"][0]["tiles"]) == 6, "")

    # A null seed means "unset" -> 0 (this port does not draw a random seed).
    check("null seed -> 0", kaggri.PyGame({"seed": None}).seed == 0, "")
    # The framework's own keys must be tolerated...
    kaggri.PyGame({"actTimeout": 1, "runTimeout": 9600, "seed": 1})
    check("framework keys tolerated", True)
    # ...but a typo must not be silently ignored.
    try:
        kaggri.PyGame({"boardsize": 6})
        check("typo'd config key rejected", False, "no exception")
    except ValueError as e:
        check("typo'd config key rejected", "boardsize" in str(e), str(e))

    # sparse marketParams merge
    g = kaggri.PyGame({"marketParams": {"WHEAT": {"base": 99},
                                        "NOT_A_PRODUCT": {"base": 1},
                                        "CARROT": "not a dict"}})
    c = g.config()
    check("marketParams sparse merge",
          c["marketParams"]["WHEAT"]["base"] == 99.0
          and c["marketParams"]["WHEAT"]["T"] == 400.0
          and c["marketParams"]["CARROT"]["base"] == 35.0,
          repr(c["marketParams"]["WHEAT"]))
    check("market seeded from the overridden base",
          g.state_dict()["market"]["prices"]["WHEAT"] == 99, "")


# ==========================================================================
@case
def state_dict_shape():
    g = kaggri.PyGame({"seed": 7})
    sd = g.state_dict()
    check("shed key order",
          list(sd["privates"][0]["shed"]) ==
          ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
           "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"], "")
    check("seeds key order",
          list(sd["privates"][0]["seeds"]) ==
          ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"], "")
    prods = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
             "WOOL", "FERTILIZER"]
    check("market key order",
          list(sd["market"]["inventory"]) == prods
          and list(sd["market"]["prices"]) == prods, "")
    check("money is float (trap 4)", isinstance(sd["farms"][0]["money"], float),
          type(sd["farms"][0]["money"]).__name__)
    check("rewards are floats and zero pre-DONE",
          sd["rewards"] == [0.0, 0.0] and all(isinstance(r, float) for r in sd["rewards"]), "")
    check("inventories non-empty (farmer at [0])",
          len(sd["privates"][0]["inventories"]) == 1, "")
    check("initial tiles: 25 None + 75 LOCKED",
          sum(1 for row in sd["farms"][0]["tiles"] for t in row if t is None) == 25
          and sum(1 for row in sd["farms"][0]["tiles"] for t in row if t == "LOCKED") == 75, "")

    # Drive a plant + an animal into existence and check the field SETS exactly.
    g = kaggri.PyGame({"seed": 7, "startingMoney": 5000})
    a = {"farmer": ["PASS"], "hands": [],
         "market": [["BUY_SEED", "WHEAT", 1], ["BUY_ANIMAL", "GOOSE", 1]]}
    g.step(a, PASS)
    g.step({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, PASS)
    g.step({"farmer": ["BUILD_COOP"], "hands": [], "market": []}, PASS)  # occupied -> no-op
    sd = g.state_dict()
    plant = next(t for row in sd["farms"][0]["tiles"] for t in row
                 if isinstance(t, dict) and t.get("kind") == "PLANT")
    check("plant tile field set",
          set(plant) == {"kind", "crop", "planted_day", "watered_today",
                         "consecutive_unwatered", "yield_units",
                         "max_lifespan_step", "fertilized_until_day"}, repr(plant))
    check("watered_today is a bool, not 0/1",
          isinstance(plant["watered_today"], bool), repr(plant["watered_today"]))
    check("shed holds the bought goose", sd["privates"][0]["shed"]["GOOSE"] == 1,
          repr(sd["privates"][0]["shed"]))

    # Place the goose on a coop. The farmer starts on (4,4), which is shed-adjacent
    # but now carries a wheat plant, so PICKUP happens there and the coop is built one
    # tile north (unlocked, empty).
    for op in (["PICKUP", "GOOSE", 1], ["NORTH"], ["BUILD_COOP"], ["PLACE", "GOOSE", 1]):
        g.step({"farmer": op, "hands": [], "market": []}, PASS)
    sd = g.state_dict()
    animals = [t for row in sd["farms"][0]["tiles"] for t in row
               if isinstance(t, dict) and "animal" in t]
    if animals:
        check("animal tile field set",
              set(animals[0]) == {"kind", "animal", "placed_day", "yield_units",
                                  "consecutive_unfed", "fed_today", "cared_today",
                                  "fertilizer_available", "pending_care_bonus"},
              repr(animals[0]))
        check("animal kind matches its structure", animals[0]["kind"] == "COOP",
              repr(animals[0]["kind"]))
    else:
        check("animal placed", False, "no animal tile appeared; ops: " + g.render()[:200])


# ==========================================================================
@case
def inventory_insertion_order_is_visible():
    """Trap 2: the order a unit picked items up in must survive into Python."""
    g = kaggri.PyGame({"seed": 3, "startingMoney": 100000})
    g.step({"farmer": ["PASS"], "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 5], ["BUY_PRODUCT", "FERTILIZER", 5]]},
           PASS)
    # farmer starts at (4,4), which is shed-adjacent
    g.step({"farmer": ["PICKUP", "FERTILIZER", 2], "hands": [], "market": []}, PASS)
    g.step({"farmer": ["PICKUP", "WHEAT", 3], "hands": [], "market": []}, PASS)
    inv = g.state_dict()["privates"][0]["inventories"][0]
    ordered = g.state_dict()["privates"][0]["inventories_ordered"][0]
    check("dict order is FERTILIZER then WHEAT",
          list(inv) == ["FERTILIZER", "WHEAT"], repr(inv))
    check("inventories_ordered agrees with the dict",
          [tuple(x) for x in ordered] == list(inv.items()), repr(ordered))

    # _inv_take deleting a key at 0 must move a re-add to the BACK
    g.step({"farmer": ["FERTILIZE"], "hands": [], "market": []}, PASS)  # no plant -> no-op
    g.step({"farmer": ["PLACE", "FERTILIZER", 2], "hands": [], "market": []}, PASS)
    inv = g.state_dict()["privates"][0]["inventories"][0]
    check("dropping a key leaves only WHEAT", list(inv) == ["WHEAT"], repr(inv))
    g.step({"farmer": ["PICKUP", "FERTILIZER", 1], "hands": [], "market": []}, PASS)
    inv = g.state_dict()["privates"][0]["inventories"][0]
    check("re-added key goes to the back",
          list(inv) == ["WHEAT", "FERTILIZER"], repr(inv))


# ==========================================================================
@case
def malformed_actions():
    g = kaggri.PyGame({"seed": 11})
    before = g.state_dict()
    junk = [
        None, 7, 3.5, True, "PASS", {}, [],
        {"farmer": "PASS"},                        # non-list farmer
        {"farmer": None},
        {"farmer": ["FLY"], "hands": "nope", "market": "nope"},
        {"farmer": [""], "hands": [None, 42, {}, []], "market": [None, 42, {}, []]},
        {"farmer": [42], "hands": [["PLANT"], ["PLANT", "wheat"], ["PICKUP", "GOLD", 5]],
         "market": [["NOPE"], ["SELL"], ["SELL", "WHEAT"], ["SELL", "WHEAT", "abc"],
                    ["SELL", "WHEAT", 0], ["SELL", "WHEAT", -3], ["SELL", "GOOSE", 1],
                    ["BUY_PRODUCT", "MELON", 1], ["SELL", "WHEAT", float("nan")]]},
    ]
    for j in junk:
        g.step(j, j)
    after = g.state_dict()
    # The market DOES move under all-PASS: `_town_consume` runs every step and the town
    # center buys one of every non-fertilizer product whenever
    # `step % townCenterSellInterval == 0` (so on step 0). The reference is therefore an
    # all-PASS episode of the same length, not the initial state.
    ref = kaggri.PyGame({"seed": 11})
    for _ in junk:
        ref.step(PASS, PASS)
    ref_sd = ref.state_dict()
    check("malformed actions == all-PASS (market)", after["market"] == ref_sd["market"],
          f'{after["market"]["inventory"]} vs {ref_sd["market"]["inventory"]}')
    check("malformed actions left town untouched", after["town"] == before["town"], "")
    check("malformed actions left money untouched",
          [f["money"] for f in after["farms"]] == [f["money"] for f in before["farms"]],
          "")
    check("malformed actions left the boards untouched",
          [f["tiles"] for f in after["farms"]] == [f["tiles"] for f in before["farms"]], "")
    check("the clock still advanced", (after["step"], after["hour"]) == (len(junk), len(junk)),
          f"{after['step']} vs {len(junk)}")

    # The one input CPython genuinely raises on must raise here too, not no-op.
    for bad in ("abc", "3.5", None, [1]):
        try:
            kaggri.PyGame({"seed": 1}).step(
                {"farmer": ["PICKUP", "WHEAT", bad], "hands": [], "market": []}, PASS)
            check(f"PICKUP qty {bad!r} raises", False, "no exception")
        except ValueError:
            check(f"PICKUP qty {bad!r} raises", True)

    # step() is a no-op once done
    g = kaggri.PyGame({"episodeSteps": 4, "seed": 1})
    for _ in range(10):
        g.step(PASS, PASS)
    check("done fires at episodeSteps - 2 and freezes the state",
          g.done and g.step_no == 3, f"done={g.done} step={g.step_no}")
    check("rewards are the final money once done", g.rewards == g.money,
          f"{g.rewards} vs {g.money}")


# ==========================================================================
@case
def trap1_banker_rounding():
    """market_price must round half to EVEN, and it is reachable at CARROT 10450."""
    cases = {8875: 752, 9325: 158, 9595: 66, 9685: 60, 9775: 52,
             9865: 46, 9955: 38, 10450: 10}
    bad = {inv: kaggri.market_price("CARROT", inv) for inv, want in cases.items()
           if kaggri.market_price("CARROT", inv) != want}
    check("all 8 exact-.5 CARROT states round half-to-even", not bad, repr(bad))
    check("py_round is half-to-even",
          [kaggri.py_round(x) for x in (0.5, 1.5, 2.5, -0.5, -1.5, 157.5, 66.5)]
          == [0, 2, 2, 0, -2, 158, 66], "")
    check("price floor applies", kaggri.market_price("CARROT", 10**6) == 1, "")


# ==========================================================================
# The RL interface (OBS_VERSION 1 / ACTION_VERSION 1, rust/OBS_LAYOUT.md)
# ==========================================================================
V = kaggri.VERB_NAMES


def random_actions(rng, v, n, hire=0):
    """A uniform-random macro action array: targets, verbs, then the market heads.

    `hire` forces the hire head so hands actually get hired and the multi-unit path
    (atomic-PLANT contention, per-tile verb claims, shared shed room) is exercised.
    """
    upp = v.units_per_player
    a = np.empty((n, 2, v.player_action_dim), dtype=np.int64)
    a[:, :, :upp] = rng.integers(0, kaggri.N_TARGETS, size=(n, 2, upp))
    a[:, :, upp:2 * upp] = rng.integers(0, kaggri.N_VERBS, size=(n, 2, upp))
    for k, size in enumerate(kaggri.MARKET_HEAD_SIZES):
        a[:, :, 2 * upp + k] = rng.integers(0, size, size=(n, 2))
    if hire:
        a[:, :, 2 * upp] = hire
    return a


def verb_index_of_op(op):
    """The verb index `resolve_turn` must have used to emit this Kaggle op.

    `None` for `PASS` and for a path step, which every verb can produce and which
    therefore says nothing about legality.
    """
    if not op:
        return None
    head = op[0]
    if head in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"):
        return None
    if head == "PLANT":
        return V.index("PLANT_" + op[1])
    if head == "PICKUP":
        return V.index(f"PICKUP_{op[1]}_{int(op[2])}")
    if head == "PLACE":
        # PLACE of an animal is Verb::PlaceAnimal; PLACE of a product is Verb::Deposit.
        return V.index(("PLACE_" if op[1] in kaggri.ANIMALS else "DEPOSIT_") + op[1])
    return V.index(head)


@case
def vecenv_matches_pygame():
    """STATUS 7.3: two independent drive paths over one core sim must agree exactly.

    `PyVecEnv` decodes a macro action array and calls `macros::resolve_turn`; `PyGame`
    parses the action dicts the Kaggle environment takes. `resolve_actions` hands the
    dicts the vec env actually submitted straight to `PyGame`, so any divergence is in
    the batched drive path (decode, auto-reset, seed derivation) and nowhere else.
    """
    cfg = {"episodeSteps": 240, "startingMoney": 9000}
    seeds = [4, 17, 2024, 0]
    n = len(seeds)
    v = kaggri.PyVecEnv(n, cfg, seeds, auto_reset=False)
    v.reset()
    games = [kaggri.PyGame(dict(cfg, seed=s)) for s in seeds]

    rng = np.random.default_rng(12345)
    max_hands = 0
    for t in range(239):
        acts = random_actions(rng, v, n, hire=3 if t < 40 else 0)
        # The dicts the vec env is about to submit, resolved against the CURRENT state.
        dicts = [v.resolve_actions(i, acts[i]) for i in range(n)]
        for i in range(n):
            max_hands = max(max_hands, max(len(d["hands"]) for d in dicts[i]))
        v.step(acts)
        for i, g in enumerate(games):
            g.step(dicts[i][0], dicts[i][1])
    mism = []
    for i, g in enumerate(games):
        a, b = g.state_dict(), v.state_dict(i)
        for k in ("step", "day", "hour", "done", "rewards", "statuses", "market",
                  "town", "farms", "privates", "seed"):
            if a[k] != b[k]:
                mism.append(f"env {i} field {k}")
    check("PyVecEnv and PyGame agree over 239 steps x 4 seeds x 17 units", not mism,
          repr(mism))
    planted = sum(1 for i in range(n) for row in v.state_dict(i)["farms"][0]["tiles"]
                  for t in row if isinstance(t, dict))
    check("the random macro stream actually mutated the boards", planted > 0,
          f"{planted} dict tiles")
    check("hands were actually hired and driven (multi-unit path is live)",
          max_hands >= 5, f"max hands driven = {max_hands}")

    # Same again ACROSS episode boundaries, which is the part the run above cannot reach:
    # the auto-reset now happens inside the stepping closure, and the next episode's seed
    # comes from `derive_seed`. PyGame has no auto-reset, so it is rebuilt from the seed
    # PyVecEnv reports for the new episode -- if the derivation or the reset placement were
    # wrong, the two would diverge immediately after the boundary.
    cfg = {"episodeSteps": 48, "startingMoney": 9000}
    v = kaggri.PyVecEnv(n, cfg, seeds, auto_reset=True)
    v.reset()
    games = [kaggri.PyGame(dict(cfg, seed=s)) for s in seeds]
    rng = np.random.default_rng(777)
    boundaries, mism = 0, []
    for t in range(140):
        acts = random_actions(rng, v, n, hire=2)
        dicts = [v.resolve_actions(i, acts[i]) for i in range(n)]
        v.step(acts)
        for i, g in enumerate(games):
            g.step(dicts[i][0], dicts[i][1])
            if g.done:
                boundaries += 1
                games[i] = kaggri.PyGame(dict(cfg, seed=v.seeds[i]))
        for i, g in enumerate(games):
            a, b = g.state_dict(), v.state_dict(i)
            for k in ("step", "day", "hour", "done", "market", "town", "farms",
                      "privates", "seed"):
                if a[k] != b[k]:
                    mism.append(f"t={t} env {i} field {k}")
        if mism:
            break
    check("PyVecEnv and PyGame stay in lockstep across auto-reset boundaries",
          not mism, repr(mism[:3]))
    check("boundaries were actually crossed", boundaries == n * 2,
          f"{boundaries} resets over 140 steps of a 48-step episode (expected {n * 2})")


# ==========================================================================
@case
def vecenv_numpy_plumbing():
    v = kaggri.PyVecEnv(16, {"episodeSteps": 48}, list(range(16)))
    check("obs_dim / mask_dim are the shipped layout",
          (v.obs_dim, v.mask_dim, v.obs_shape, v.mask_shape)
          == (6720, 646, (140, 48), (17, 38)),
          f"{v.obs_dim} {v.mask_dim} {v.obs_shape} {v.mask_shape}")
    check("obs_version / action_version", (v.obs_version, v.action_version) == (1, 1),
          f"{v.obs_version} {v.action_version}")
    check("action_dim = 2 * (2 * 17 + 21)",
          (v.units_per_player, v.player_action_dim, v.action_dim) == (17, 55, 110),
          f"{v.units_per_player} {v.player_action_dim} {v.action_dim}")

    obs, mask = v.reset()
    check("obs shape/dtype", obs.shape == (16, 2, 6720) and obs.dtype == OBS_DT,
          f"{obs.shape} {obs.dtype}")
    check("mask shape/dtype", mask.shape == (16, 2, 646) and mask.dtype == OBS_DT,
          f"{mask.shape} {mask.dtype}")
    check("the shipped default published dtype is float16",
          (kaggri.OBS_DTYPE, v.obs_dtype, OBS_DT) == ("float16", "float16", np.float16),
          f"{kaggri.OBS_DTYPE} {v.obs_dtype}")

    a = np.zeros((16, 2, 55), dtype=np.int64)
    o, m, r, d, info = v.step(a)
    check("step returns (obs, mask, rewards, dones, info)",
          o.shape == (16, 2, 6720) and m.shape == (16, 2, 646), f"{o.shape} {m.shape}")
    check("reward shape/dtype", r.shape == (16, 2) and r.dtype == np.float32, f"{r.shape}")
    check("done shape/dtype", d.shape == (16,) and d.dtype == np.bool_, f"{d.shape} {d.dtype}")
    check("info carries live money, terminal money, and step",
          set(info) == {"final_money", "money", "step"}, repr(set(info)))
    o2, m2, r2, d2, info2 = v.step(a)
    check("buffers are reused (no per-step allocation)",
          o2 is o and m2 is m and r2 is r and d2 is d and info2 is info, "")
    check("reset returns the same buffers too",
          all(x is y for x, y in zip(v.reset(), (o, m))), "")

    # An all-zero action array is an all-PASS turn: every verb slot is Idle and every
    # market head is its do-nothing category.
    z = kaggri.PyVecEnv(2, {"episodeSteps": 48}, [5, 6], auto_reset=False)
    z.reset()
    z.step(np.zeros((2, 110), dtype=np.int64))
    g = kaggri.PyGame({"episodeSteps": 48, "seed": 5})
    g.step(PASS, PASS)
    check("all-zero actions == all-PASS", z.state_dict(0)["farms"] == g.state_dict()["farms"],
          "")

    for shape, dt in [((16, 110), np.int64), ((16, 2, 55), np.int64),
                      ((16, 110), np.int32), ((16, 2, 55), np.int32),
                      ((16 * 110,), np.int64)]:
        v.step(np.zeros(shape, dtype=dt))
    check("every documented action shape/dtype accepted", True)
    for bad in (np.zeros((16, 5), dtype=np.int64), np.zeros((15, 110), dtype=np.int64),
                np.zeros((16, 110), dtype=np.float32), np.zeros((16, 2, 56), dtype=np.int64)):
        try:
            v.step(bad)
            check(f"bad action array {bad.shape}/{bad.dtype} rejected", False, "no exception")
        except (ValueError, TypeError):
            check(f"bad action array {bad.shape}/{bad.dtype} rejected", True)

    # A malformed sample from an untrained policy must never panic a rayon worker. The
    # two out-of-range directions are deliberately NOT the same, and both are documented:
    #
    #   negative  -> unit orders idle, market heads take category 0 (do nothing), so the
    #                turn is exactly all-PASS and no money can move;
    #   too large -> unit orders still idle (an out-of-range target or verb is never
    #                clamped into a different action), but MarketHead::from_choices
    #                clamps each market head to its LAST category, i.e. the maximal
    #                market request. That is a real turn, not a no-op.
    def rollout(seeds, rows, steps=6, cfg=None):
        e = kaggri.PyVecEnv(len(seeds), cfg or {"episodeSteps": 96}, seeds, auto_reset=False)
        e.reset()
        for _ in range(steps):
            e.step(rows)
        return [e.state_dict(i) for i in range(len(seeds))]

    seeds = [1, 2, 3, 4]
    zeros = np.zeros((4, 2, 55), dtype=np.int64)
    base = rollout(seeds, zeros)
    for filler in (-1, -(10 ** 12), np.iinfo(np.int64).min):
        got = rollout(seeds, np.full((4, 2, 55), filler, dtype=np.int64))
        check(f"negative action fill ({filler}) is exactly all-PASS", got == base, "")

    # Saturated == explicitly asking for the last category of every head.
    sat = np.zeros((4, 2, 55), dtype=np.int64)
    sat[:, :, :34] = 10 ** 12                          # targets + verbs: out of range
    for k, size in enumerate(kaggri.MARKET_HEAD_SIZES):
        sat[:, :, 34 + k] = size - 1
    for filler in (10 ** 12, 2 ** 62, np.iinfo(np.int64).max):
        got = rollout(seeds, np.full((4, 2, 55), filler, dtype=np.int64))
        check(f"over-range action fill ({filler}) clamps to the last category of "
              f"every market head", got == rollout(seeds, sat), "")

    # units_per_player < MAX_UNITS narrows the action array
    w = kaggri.PyVecEnv(4, {"episodeSteps": 48}, [1, 2, 3, 4], units_per_player=5)
    check("player_action_dim = 2 * units_per_player + 21",
          (w.player_action_dim, w.action_dim) == (31, 62),
          f"{w.player_action_dim} {w.action_dim}")
    w.reset()
    w.step(np.zeros((4, 62), dtype=np.int64))
    w.step(np.zeros((4, 2, 31), dtype=np.int64))
    check("narrowed action arrays accepted", True)
    for bad_upp in (0, 18):
        try:
            kaggri.PyVecEnv(2, None, [1, 2], units_per_player=bad_upp)
            check(f"units_per_player={bad_upp} rejected", False, "no exception")
        except ValueError:
            check(f"units_per_player={bad_upp} rejected", True)


# ==========================================================================
@case
def vecenv_warmup_stops_before_terminal_transition():
    """A 720-frame replay has 719 transitions; warmup may not consume the last one."""
    v = kaggri.PyVecEnv(2, {"episodeSteps": 720}, [17, 18], auto_reset=True)
    v.reset()

    # Validation is batch-atomic. In particular, 719 must not step the env to DONE and
    # then disappear behind auto-reset as episode 1, step 0.
    try:
        v.warmup_pass(np.array([1, 719], dtype=np.int64))
        check("terminal warmup offset 719 rejected", False, "no exception")
    except ValueError as exc:
        check("terminal warmup offset 719 rejected",
              "719" in str(exc) and "terminal transition" in str(exc), str(exc))
    check("rejected warmup is atomic and cannot auto-reset",
          np.array_equal(v.env_steps(), [0, 0])
          and list(v.episode_indices) == [0, 0]
          and v.total_steps == 0,
          f"steps={v.env_steps()} episodes={list(v.episode_indices)} total={v.total_steps}")

    # 718 is the largest non-terminal offset. One ordinary step after it must expose the
    # terminal state through done/info before resetting only that env.
    v.warmup_pass([0, 718])
    check("largest valid warmup offset remains in episode 0",
          np.array_equal(v.env_steps(), [0, 718])
          and list(v.episode_indices) == [0, 0]
          and v.total_steps == 718,
          f"steps={v.env_steps()} episodes={list(v.episode_indices)} total={v.total_steps}")
    actions = np.zeros((2, 2, v.player_action_dim), dtype=np.int64)
    _, _, _, done, info = v.step(actions)
    check("step after offset 718 captures terminal before auto-reset",
          np.array_equal(done, [False, True])
          and np.array_equal(info["step"], [1, 719])
          and list(v.episode_indices) == [0, 1]
          and np.array_equal(v.env_steps(), [1, 0])
          and bool(np.asarray(info["final_money"])[1].all()),
          f"done={done} info_step={info['step']} episodes={list(v.episode_indices)} "
          f"live_steps={v.env_steps()} final_money={info['final_money']}")


# ==========================================================================
@case
def vecenv_published_dtype():
    """The published dtype (`OBS_LAYOUT.md` 11): f16 by default, f32 on request.

    The property that makes f16 safe to ship is not "the error is small" -- it is that the
    f16 plane is *exactly* `np.float16` of the f32 plane, so the only difference between
    the two configurations is a round-to-nearest-ties-to-even quantisation with nothing
    else riding along: no lost slot, no shifted index, no reordered player. That is what
    this case asserts, over a real rollout, together with the game outcome being
    untouched.
    """
    check("module advertises the default and the alternatives",
          (kaggri.OBS_DTYPE, list(kaggri.OBS_DTYPES)) == ("float16", ["float16", "float32"]),
          f"{kaggri.OBS_DTYPE} {kaggri.OBS_DTYPES}")
    check("obs_dims_dict carries the same default",
          kaggri.obs_dims_dict()["obs_dtype"] == kaggri.OBS_DTYPE,
          repr(kaggri.obs_dims_dict().get("obs_dtype")))

    # Every spelling a training script might reasonably hold.
    for spec, want in [(None, "float16"), ("float16", "float16"), ("f2", "float16"),
                       (np.float16, "float16"), (np.dtype("float16"), "float16"),
                       ("float32", "float32"), ("f4", "float32"), (np.float32, "float32"),
                       (np.dtype("float32"), "float32")]:
        e = kaggri.PyVecEnv(2, None, [1, 2], True, 17, 1, spec)
        o, m = e.reset()
        ok = e.obs_dtype == want and o.dtype == np.dtype(want) and m.dtype == np.dtype(want)
        check(f"obs_dtype={spec!r} -> {want}", ok, f"{e.obs_dtype} {o.dtype} {m.dtype}")
    for bad in ("float64", "f8", np.int16, "uint16", np.float64, "bfloat16", "nonsense", 17):
        try:
            kaggri.PyVecEnv(2, None, [1, 2], True, 17, 1, bad)
            check(f"obs_dtype={bad!r} rejected", False, "no exception")
        except (ValueError, TypeError):
            check(f"obs_dtype={bad!r} rejected", True)

    n, U, steps = 6, 17, 96
    cfg = {"episodeSteps": 48}          # short, so auto-reset boundaries are crossed
    seeds = [101, 102, 103, 104, 105, 106]
    half = kaggri.PyVecEnv(n, cfg, seeds, True, U, 1, "float16")
    wide = kaggri.PyVecEnv(n, cfg, seeds, True, U, 1, "float32")
    check("published_bytes halves with the dtype",
          (wide.published_bytes, half.published_bytes)
          == (n * 2 * (6720 + 646) * 4, n * 2 * (6720 + 646) * 2),
          f"{wide.published_bytes} {half.published_bytes}")
    oh, mh = half.reset()
    ow, mw = wide.reset()
    exact = np.array_equal(oh, ow.astype(np.float16)) and np.array_equal(mh, mw.astype(np.float16))
    rng = np.random.default_rng(2027)
    worst_abs, worst_rel = 0.0, 0.0
    mask_lossless = True
    outcome_same = True
    for _ in range(steps):
        a = random_actions(rng, half, n, hire=2)
        oh, mh, rh, dh, ih = half.step(a)
        ow, mw, rw, dw, iw = wide.step(a)
        exact &= bool(np.array_equal(oh, ow.astype(np.float16)))
        exact &= bool(np.array_equal(mh, mw.astype(np.float16)))
        # The mask alphabet is {0.0, 1.0}; both are exact in f16, so no rounding at all.
        mask_lossless &= bool(np.array_equal(mh.astype(np.float32), mw))
        outcome_same &= bool(np.array_equal(rh, rw) and np.array_equal(dh, dw)
                             and np.array_equal(ih["final_money"], iw["final_money"])
                             and np.array_equal(ih["step"], iw["step"]))
        err = np.abs(oh.astype(np.float32) - ow)
        worst_abs = max(worst_abs, float(err.max()))
        nz = ow != 0.0
        if nz.any():
            worst_rel = max(worst_rel, float((err[nz] / np.abs(ow[nz])).max()))
    print(f"        [{n} envs x 2 players x {steps} steps published in both dtypes; "
          f"worst abs error {worst_abs:.3e}, worst relative {worst_rel:.3e}]")
    check("the f16 plane is exactly np.float16 of the f32 plane, every value", exact, "")
    check("the legality mask is published losslessly in f16", mask_lossless, "")
    check("dtype cannot change a game outcome (rewards/dones/final_money/step identical)",
          outcome_same, "")
    # Half an f16 ULP at the top of the documented [-4, 4] range is 2^-9 = 1.953e-3, and
    # the relative bound is 2^-11 = 4.883e-4. Both are properties of the format, so
    # exceeding either means a slot left its documented range.
    check("worst absolute error is under half an ULP at |x| = 4",
          worst_abs <= 1.954e-3, f"{worst_abs:.3e}")
    check("worst relative error is under 2^-11", worst_rel <= 4.884e-4, f"{worst_rel:.3e}")
    check("the error check is not vacuous (some slot really did round)",
          worst_abs > 0.0, f"{worst_abs}")
    check("both planes are C-contiguous, so no .view()/copy is needed downstream",
          oh.flags["C_CONTIGUOUS"] and mh.flags["C_CONTIGUOUS"], "")
    check("auto_reset boundaries were crossed in both dtypes",
          list(half.episode_indices) == list(wide.episode_indices)
          and min(half.episode_indices) >= 1, repr(list(half.episode_indices)))


# ==========================================================================
@case
def vecenv_obs_and_mask_are_well_formed():
    """The >= 1024-env smoke test: value ranges, mask alphabet, mask vs the resolver."""
    n = 1024
    cfg = {"episodeSteps": 96}
    v = kaggri.PyVecEnv(n, cfg, list(range(n)), auto_reset=True)
    obs, mask = v.reset()
    rng = np.random.default_rng(99)

    lo, hi = np.inf, -np.inf
    bad_mask_values = 0
    absent_rows_nonzero = 0
    live_rows_without_idle = 0
    mask_violations = []
    live_unit_mismatch = 0
    checked_units = 0
    checked_verbs = 0
    n_probe = 64               # envs deep-checked against resolve_actions per step

    for t in range(120):
        if not (np.isfinite(obs).all() and np.isfinite(mask).all()):
            check("every observation and mask float is finite", False, f"step {t}")
            return
        lo = min(lo, float(obs.min()))
        hi = max(hi, float(obs.max()))
        bad_mask_values += int(((mask != 0.0) & (mask != 1.0)).sum())

        # Mask rows: zero for absent units (Idle included), Idle set for live ones.
        rows = mask.reshape(n, 2, kaggri.MAX_UNITS, kaggri.N_VERBS)
        live = rows[:, :, :, 0] == 1.0
        absent_rows_nonzero += int(rows[~live].sum())
        live_rows_without_idle += int((rows.sum(axis=3) > 0).sum() - live.sum())
        for i in range(0, n, 97):                       # a cheap sample of envs
            sd = v.state_dict(i)
            for p in range(2):
                want = 1 + len(sd["farms"][p]["hands"])
                if int(live[i, p].sum()) != want:
                    live_unit_mismatch += 1

        acts = random_actions(rng, v, n, hire=2 if t < 30 else 0)
        # In the probe envs, point every unit at the tile it is already standing on. A
        # uniformly random target is 99 % of the time somewhere else, so the resolver
        # emits a path step and the mask check sees almost no verbs; standing on the
        # target is what makes the verb fire and the check non-vacuous.
        for i in range(n_probe):
            sd = v.state_dict(i)
            for p in range(2):
                f = sd["farms"][p]
                for u, pos in enumerate([f["farmer"]] + list(f["hands"])):
                    acts[i, p, u] = pos[1] * 10 + pos[0]
        prev = mask.copy()
        resolved = [v.resolve_actions(i, acts[i]) for i in range(n_probe)]
        obs, mask, rew, done, info = v.step(acts)

        # OBS_LAYOUT 10: the mask is the more permissive side -- the resolver may
        # decline a verb the mask calls legal, but must never emit one it calls illegal.
        for i in range(n_probe):
            for p in range(2):
                ops = [resolved[i][p]["farmer"]] + list(resolved[i][p]["hands"])
                for u, op in enumerate(ops):
                    checked_units += 1
                    vi = verb_index_of_op(op)
                    if vi is None:
                        continue
                    checked_verbs += 1
                    if prev[i, p, u * kaggri.N_VERBS + vi] != 1.0:
                        mask_violations.append((t, i, p, u, V[vi]))

    print(f"        [{n} envs x 2 players x 120 steps = {n * 2 * 120:,} encodes; "
          f"obs range [{lo:.4f}, {hi:.4f}]; {checked_verbs:,} emitted verbs checked "
          f"against the mask out of {checked_units:,} unit-ops]")
    check("every observation and mask float is finite", True)
    check("observation stays inside the documented [-4, 4]", -4.0 <= lo and hi <= 4.0,
          f"min {lo} max {hi}")
    check("mask is strictly 0.0 / 1.0", bad_mask_values == 0, f"{bad_mask_values} other values")
    check("absent units' mask rows are all zero", absent_rows_nonzero == 0,
          f"{absent_rows_nonzero} flags set")
    check("no live unit is missing Idle", live_rows_without_idle == 0,
          f"{live_rows_without_idle} rows")
    check("live mask rows == 1 + len(hands) from state_dict", live_unit_mismatch == 0,
          f"{live_unit_mismatch} mismatches")
    check("resolve_turn never emits a verb the mask calls illegal",
          not mask_violations,
          f"{len(mask_violations)} violations, first {mask_violations[:3]}")
    check("the mask check actually saw verbs (not vacuous)", checked_verbs > 4000,
          f"{checked_verbs} verbs over {checked_units} unit-ops")
    check("auto_reset ran episodes past the boundary",
          all(i >= 1 for i in v.episode_indices) and v.total_steps == n * 120,
          f"{v.episode_indices[:4]} total_steps={v.total_steps}")


# ==========================================================================
@case
def obs_uses_the_episode_config_not_the_default():
    """The pinned `obs::encode` / `legal_mask` hard-assume `Config::default()`.

    `PyVecEnv` must call the `_with_cfg` variants, or every non-default episode is
    normalised against the wrong divisors -- silently, with no test failing and no
    exception, just a policy fed garbage scales. This pins three independent slots of the
    GLOBAL token against the config-aware formula AND against the default-config value they
    would have had, so the two cannot be confused.
    """
    tpd, steps, cap = 6, 48, 5
    cfg = {"episodeSteps": steps, "turnsPerDay": tpd, "shedCapacity": cap,
           "startingMoney": 20000}
    v = kaggri.PyVecEnv(2, cfg, [21, 22], auto_reset=False)
    v.reset()
    # Buy wheat so the shed is not empty, which is what makes the shedCapacity divisor
    # observable at all.
    buy = np.zeros((2, 2, v.player_action_dim), dtype=np.int64)
    buy[:, :, 2 * v.units_per_player + 10] = len(kaggri.QTY_BUCKETS) - 1   # BUY_PRODUCT WHEAT
    obs, mask, *_ = v.step(buy)
    for _ in range(2):
        obs, mask, *_ = v.step(np.zeros_like(buy))

    sd = v.state_dict(0)
    step_no, hour = sd["step"], sd["hour"]
    shed_total = sum(sd["privates"][0]["shed"].values())
    g = obs[0, 0, 8:]                       # GLOBAL token payload (header is 8 wide)

    def clip(x, lo, hi):
        return max(lo, min(hi, x))

    want = {
        0:  clip(step_no / max(steps - 2, 1), 0.0, 1.0),        # / 46, not / 718
        2:  hour / max(tpd - 1, 1),                             # / 5,  not / 23
        4:  clip((tpd - 1 - hour) / max(tpd - 1, 1), 0.0, 1.0),
        33: clip(shed_total / cap, 0.0, 1.0),                   # / 5,  not / 100
    }
    default = {0: clip(step_no / 718, 0.0, 1.0), 2: hour / 23,
               4: clip((23 - hour) / 23, 0.0, 1.0),
               33: clip(shed_total / 100, 0.0, 1.0)}
    bad = {k: (float(g[k]), w) for k, w in want.items()
           if abs(float(g[k]) - w) > obs_tol(obs, w)}
    check("GLOBAL slots use the episode's own turnsPerDay / episodeSteps / shedCapacity",
          not bad and shed_total > 0,
          f"shed_total={shed_total} mismatches={bad}")
    # The two candidate values must be further apart than the publication tolerance, or
    # the check above could pass against either. They differ by 0.1 or more here.
    check("...and are NOT the values Config::default() would have produced",
          all(abs(want[k] - default[k]) > max(1e-2, obs_tol(obs, want[k])) for k in want),
          repr(default))

    # `legal_mask_with_cfg` reads shedCapacity too, and `Deposit` is where it shows:
    # `resource_ok` wants the unit to hold the product AND `shed.total() < shedCapacity`.
    # Drive the farmer into exactly that state -- carrying wheat, shed at capacity -- and
    # the verb must be illegal at shedCapacity=2 and legal at the default 100. Run both, so
    # neither half of the check can pass vacuously.
    def deposit_wheat_legal(shed_cap):
        e = kaggri.PyVecEnv(1, {"episodeSteps": 48, "shedCapacity": shed_cap,
                                "startingMoney": 20000}, [7], auto_reset=False)
        e.reset()
        upp = e.units_per_player
        mkt = 2 * upp
        def row(buy_bucket=0, verb=None):
            a = np.zeros((1, 2, e.player_action_dim), dtype=np.int64)
            a[:, :, mkt + 10] = buy_bucket                     # BUY_PRODUCT WHEAT
            if verb is not None:
                a[0, 0, 0] = 44                                # the farmer's own tile (4,4)
                a[0, 0, upp] = verb                            # ...which is shed-adjacent
            return a
        e.step(row(buy_bucket=2))                              # QTY_BUCKETS[2] == 2 wheat
        e.step(row(verb=V.index("PICKUP_WHEAT_1")))            # shed 2 -> 1, inv 0 -> 1
        _, mask, *_ = e.step(row(buy_bucket=1))                # shed 1 -> 2 == capacity
        sd = e.state_dict(0)
        inv = sd["privates"][0]["inventories"][0]
        row0 = mask[0, 0].reshape(kaggri.MAX_UNITS, kaggri.N_VERBS)[0]
        return (sum(sd["privates"][0]["shed"].values()), inv.get("WHEAT", 0),
                float(row0[V.index("DEPOSIT_WHEAT")]))

    # The extreme configs `parity/config_parity.py` covers for the simulator, now driven
    # through the RL layer as well: a board smaller than the fixed 100-tile target head, a
    # board larger than it (truncated, per OBS_LAYOUT.md 1), and the degenerate clocks.
    rng = np.random.default_rng(31)
    for extra in ({"boardSize": 4, "turnsPerDay": 3}, {"boardSize": 13},
                  {"turnsPerDay": 1}, {"shedCapacity": 1}, {"maxMarketOrdersPerTurn": 1},
                  {"episodeSteps": 3}, {"farmHandCostMult": 0, "startingMoney": 50000}):
        e = kaggri.PyVecEnv(8, dict(extra, episodeSteps=extra.get("episodeSteps", 60)),
                            list(range(8)), auto_reset=True)
        o, m = e.reset()
        finite = np.isfinite(o).all() and np.isfinite(m).all()
        inrange = float(o.min()) >= -4.0 and float(o.max()) <= 4.0
        binary = bool(((m == 0.0) | (m == 1.0)).all())
        for _ in range(40):
            o, m, *_ = e.step(random_actions(rng, e, 8, hire=2))
            finite &= bool(np.isfinite(o).all() and np.isfinite(m).all())
            inrange &= float(o.min()) >= -4.0 and float(o.max()) <= 4.0
            binary &= bool(((m == 0.0) | (m == 1.0)).all())
        # boardSize 4 leaves tile tokens 16.. entirely zero, type one-hot included.
        tail_zero = True
        if extra.get("boardSize") == 4:
            tok = o[:, :, :].reshape(8, 2, kaggri.N_TOKENS, kaggri.N_FEATURES)
            tail_zero = bool((tok[:, :, 1 + 16:1 + 100, :] == 0.0).all())
        check(f"non-default config {extra} stays finite / in range / binary-masked",
              finite and inrange and binary and tail_zero,
              f"finite={finite} inrange={inrange} binary={binary} tail_zero={tail_zero}")

    tight = deposit_wheat_legal(2)
    loose = deposit_wheat_legal(100)
    check("the setup really has the farmer carrying wheat with a full shed",
          tight == (2, 1, 0.0), f"(shed, carried, DEPOSIT_WHEAT legal) = {tight}")
    check("the same state with the default shedCapacity leaves Deposit legal "
          "(so the mask really reads the episode's shedCapacity)",
          loose == (2, 1, 1.0), f"(shed, carried, DEPOSIT_WHEAT legal) = {loose}")


# ==========================================================================
@case
def chunk_envs_does_not_change_the_result():
    """`chunk_envs` is a memory-behaviour knob and must be observationally inert.

    Every rayon chunk boundary falls between envs and each env's outcome depends only on
    its own state, its own action row and its own seed, so a rollout has to be
    bit-identical across the whole range -- including values that do not divide n_envs and
    values larger than it (one chunk, i.e. effectively serial).
    """
    n, cfg = 64, {"episodeSteps": 96, "startingMoney": 9000}
    seeds = list(range(100, 100 + n))
    rng = np.random.default_rng(4242)
    acts = [random_actions(rng, kaggri.PyVecEnv(1, cfg, [0]), n, hire=2 if t < 20 else 0)
            for t in range(60)]

    def roll(chunk):
        v = kaggri.PyVecEnv(n, cfg, seeds, auto_reset=True, chunk_envs=chunk)
        obs, mask = v.reset()
        trail = [(obs.copy(), mask.copy())]
        for a in acts:
            o, m, r, d, info = v.step(a)
            trail.append((o.copy(), m.copy(), r.copy(), d.copy(),
                          np.asarray(info["final_money"]).copy()))
        return trail, [v.state_dict(i) for i in range(n)]

    base_trail, base_sd = roll(1)
    for chunk in (3, 7, 16, 64, 4096):
        trail, sd = roll(chunk)
        same = (len(trail) == len(base_trail)
                and all(all(np.array_equal(x, y) for x, y in zip(a, b))
                        for a, b in zip(trail, base_trail)))
        check(f"chunk_envs={chunk} gives a bit-identical rollout", same and sd == base_sd, "")


# ==========================================================================
@case
def vecenv_determinism_and_autoreset():
    cfg = {"episodeSteps": 48}
    seeds = [11, 12, 13, 14]
    def roll(auto):
        v = kaggri.PyVecEnv(4, cfg, seeds, auto_reset=auto)
        v.reset()
        a = np.zeros((4, 2, v.player_action_dim), dtype=np.int64)
        trail = []
        for _ in range(120):
            o, m, r, d, info = v.step(a)
            trail.append((o.copy(), r.copy(), d.copy(), np.asarray(info["final_money"]).copy()))
        return v, trail

    v1, t1 = roll(True)
    v2, t2 = roll(True)
    same = all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
               and np.array_equal(x[2], y[2]) and np.array_equal(x[3], y[3])
               for x, y in zip(t1, t2))
    check("same seeds -> bit-identical rollout", same, "")
    check("auto_reset started new episodes",
          list(v1.episode_indices) == [2, 2, 2, 2], repr(v1.episode_indices))
    check("auto_reset seeds stay in the Python-reproducible 31-bit range",
          all(0 <= s < 2**31 for s in v1.seeds), repr(v1.seeds))
    dones = [int(x[2].sum()) for x in t1]
    check("done fired exactly twice per env in 120 steps of a 48-step episode",
          sum(dones) == 8, f"{sum(dones)}")
    fm = [x[3] for x in t1 if x[3].any()]
    check("final_money is reported on the DONE step", len(fm) == 2 and (fm[0] > 0).all(),
          repr(fm))

    v3, t3 = roll(False)
    check("auto_reset=False leaves the envs done",
          bool(t3[-1][2].all()) and list(v3.episode_indices) == [0, 0, 0, 0],
          repr(v3.episode_indices))

    # different seeds must actually produce different episodes
    w = kaggri.PyVecEnv(2, {"episodeSteps": 720, "weedSpawnChance": 0.5}, [1, 2],
                        auto_reset=False)
    w.reset()
    a = np.zeros((2, 2, w.player_action_dim), dtype=np.int64)
    for _ in range(30):
        w.step(a)
    check("distinct seeds diverge (weed RNG)",
          w.state_dict(0)["farms"][0]["tiles"] != w.state_dict(1)["farms"][0]["tiles"], "")


# ==========================================================================
@case
def vecenv_tile_mask_plane():
    """`tile_mask=True`: the optional (n, 2, 3800) u8 plane, its consistency with the unit
    mask and its agreement with `resolve_actions`. (`Encoder.tile_mask` parity with the
    vec-env plane lives in the repo's pytest suite, which has the Kaggle-obs builder.)"""
    n = 256
    cfg = {"episodeSteps": 96}
    v0 = kaggri.PyVecEnv(4, cfg, list(range(4)))
    check("tile_mask is None unless requested", v0.tile_mask is None and
          "tile_mask" not in v0.step(np.zeros((4, 2, v0.player_action_dim), np.int64))[4])
    v = kaggri.PyVecEnv(n, cfg, list(range(n)), auto_reset=True, tile_mask=True)
    obs, mask = v.reset()
    tm = v.tile_mask
    check("tile_mask plane shape/dtype", tm.shape == (n, 2, kaggri.TILE_MASK_DIM) and
          tm.dtype == np.uint8 and kaggri.TILE_MASK_DIM == 3800, f"{tm.shape} {tm.dtype}")
    rng = np.random.default_rng(7)
    bad_alpha = idle_set = no_tile = 0
    resolver_pass = []
    probed = 0
    for t in range(96):
        tmv = tm.reshape(n, 2, kaggri.N_TARGETS, kaggri.N_VERBS)
        rows = mask.reshape(n, 2, kaggri.MAX_UNITS, kaggri.N_VERBS)
        bad_alpha += int(((tm != 0) & (tm != 1)).sum())
        idle_set += int(tmv[:, :, :, 0].sum())
        # unit mask 1 (non-Idle) => some plausible tile for that verb
        any_tile = tmv.max(axis=2)                                   # (n, 2, V)
        legal = rows[:, :, :, 1:] == 1.0                             # (n, 2, U, V-1)
        no_tile += int((legal & ~(any_tile[:, :, None, 1:] == 1)).sum())
        acts = np.zeros((n, 2, v.player_action_dim), np.int64)
        # probe: one unit per seat orders (own tile, a verb legal on both masks)
        for i in range(48):
            sd = v.state_dict(i)
            for p in range(2):
                f = sd["farms"][p]
                units = [f["farmer"]] + list(f["hands"])
                u = int(rng.integers(len(units)))
                here = units[u][1] * 10 + units[u][0]
                ok = np.flatnonzero((rows[i, p, u] == 1.0) & (tmv[i, p, here] == 1))
                if ok.size == 0:
                    continue
                vi = int(rng.choice(ok))
                acts[i, p, u] = here
                acts[i, p, kaggri.MAX_UNITS + u] = vi
                d = v.resolve_actions(i, acts[i])[p]
                op = d["farmer"] if u == 0 else d["hands"][u - 1]
                probed += 1
                if op[0] in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"):
                    resolver_pass.append((t, i, p, u, V[vi], op))
        # random step for everyone else
        ra = random_actions(rng, v, n, hire=1 if t < 20 else 0)
        keep = acts.any(axis=2)                                      # probed seats keep theirs
        ra[keep] = acts[keep]
        obs, mask, rew, done, info = v.step(ra)
        check_same = info["tile_mask"] is not None
        if t == 0:
            check("info['tile_mask'] is the same plane as .tile_mask",
                  check_same and np.shares_memory(info["tile_mask"], v.tile_mask))
    check("tile plane is strictly 0/1", bad_alpha == 0, f"{bad_alpha}")
    check("Idle column is all zero", idle_set == 0, f"{idle_set}")
    check("every legal non-Idle verb has a plausible tile", no_tile == 0, f"{no_tile}")
    check("standing on a plausible tile, the resolver fires the verb (never PASS/move)",
          not resolver_pass, f"{len(resolver_pass)} first {resolver_pass[:3]}")
    check("the resolver probe was not vacuous", probed > 3000, f"{probed}")


# ==========================================================================
@case
def rng_helpers_match_cpython():
    import random
    bad = []
    for seed in (0, 1, 12345, 2**31 - 1, -7):
        for day in (0, 1, 29):
            r = random.Random((seed * 1_000_003) ^ day)
            mine = kaggri.rng_stream(seed, day, 50)
            theirs = [r.random() for _ in range(50)]
            if mine != theirs:
                bad.append((seed, day))
            if kaggri.rng_choice_index(seed, day, 50, 8) != r.choice(range(8)):
                bad.append((seed, day, "choice"))
    check("PyRandom.for_day matches CPython over 15 (seed, day) pairs", not bad, repr(bad))


if __name__ == "__main__":
    print(f"kaggri {kaggri.__version__} from {kaggri.__file__}")
    print(f"OBS_VERSION={kaggri.OBS_VERSION} ACTION_VERSION={kaggri.ACTION_VERSION} "
          f"OBS_DIM={kaggri.OBS_DIM} MASK_DIM={kaggri.MASK_DIM} TILE_MASK_DIM={kaggri.TILE_MASK_DIM} "
          f"OBS_DTYPE={kaggri.OBS_DTYPE}")
    print()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  " + f)
        sys.exit(1)
    print("\nall PyO3 surface checks passed")
