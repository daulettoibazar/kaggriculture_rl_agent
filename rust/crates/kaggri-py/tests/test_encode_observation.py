#!/usr/bin/env python3
"""
Unit tests for the submission surface added to `kaggri` (crates/kaggri-py/src/lib.rs):

    kaggri.Encoder(config=None, obs_dtype=None)
        .encode(obs) -> (obs, mask)      reused, env-owned buffers
        .action(obs, row) -> action dict
        .encode_and_action(obs, row) -> (obs, mask, action)
        .state_dict(obs) -> the canon.py state shape
        .config()
    kaggri.encode_observation(obs, config=None, obs_dtype=None)
    kaggri.observation_action(obs, actions, config=None)
    kaggri.state_from_observation(obs, config=None)

Bit-exactness against `PyVecEnv` over full episodes is
`parity/diff_obs_submission.py`; this file covers the binding-layer behaviour that
harness cannot see: accepted input shapes, reused-buffer semantics, dtype
selection, action-row width inference, and the error message for every field the
reconstruction requires.

Run:  python3 rust/crates/kaggri-py/tests/test_encode_observation.py
"""
from __future__ import annotations

import copy
import sys
import traceback

import numpy as np

import kaggri
from kaggle_environments import make
from kaggle_environments.utils import structify

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


KNOWN = {
    "episodeSteps", "boardSize", "startingMoney", "maxMarketOrdersPerTurn", "turnsPerDay",
    "shedCapacity", "weedSpawnChance", "townShopUnlockInterval", "townShopSellInterval",
    "townCenterSellInterval", "farmHandCostMult", "seed", "marketParams",
}
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
U = kaggri.MAX_UNITS
H = kaggri.N_MARKET_HEADS


def episode(seed=7, steps=720, extra=None, advance=0):
    """(obs for player 0, game config, env) after `advance` all-PASS steps."""
    cfg = {"episodeSteps": steps, "seed": seed, **(extra or {})}
    env = make("kaggriculture", configuration=cfg, debug=True)
    env.reset(2)
    for _ in range(advance):
        env.step([PASS, PASS])
    obs = structify(env._Environment__get_shared_state(0)).observation
    game_cfg = {k: v for k, v in dict(env.configuration).items() if k in KNOWN}
    return obs, game_cfg, env


def bits(a):
    a = np.asarray(a)
    return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])


# ===========================================================================
@case
def module_surface():
    for name in ("Encoder", "encode_observation", "observation_action",
                 "state_from_observation"):
        check(f"kaggri.{name} exists", hasattr(kaggri, name))


@case
def shapes_and_default_dtype():
    obs, cfg, _ = episode()
    o, m = kaggri.encode_observation(obs, cfg)
    check("obs shape", o.shape == (kaggri.OBS_DIM,), str(o.shape))
    check("mask shape", m.shape == (kaggri.MASK_DIM,), str(m.shape))
    check("default dtype is the published one", str(o.dtype) == kaggri.OBS_DTYPE,
          f"{o.dtype} != {kaggri.OBS_DTYPE}")
    check("mask has the same dtype", o.dtype == m.dtype)
    check("both arrays are C-contiguous", o.flags["C_CONTIGUOUS"] and m.flags["C_CONTIGUOUS"])


@case
def f16_plane_is_exactly_np_float16_of_f32():
    obs, cfg, _ = episode(advance=60)
    o32, m32 = kaggri.encode_observation(obs, cfg, obs_dtype="float32")
    o16, m16 = kaggri.encode_observation(obs, cfg, obs_dtype="float16")
    check("obs f16 == np.float16(f32)",
          np.array_equal(bits(o16), bits(np.float16(np.asarray(o32)))))
    check("mask f16 == np.float16(f32)",
          np.array_equal(bits(m16), bits(np.float16(np.asarray(m32)))))
    check("float32 request is honoured", str(o32.dtype) == "float32")
    check("np.float16 object accepted as obs_dtype",
          str(kaggri.encode_observation(obs, cfg, obs_dtype=np.float16)[0].dtype) == "float16")
    try:
        kaggri.encode_observation(obs, cfg, obs_dtype="int8")
        check("int8 obs_dtype refused", False, "no exception")
    except ValueError:
        check("int8 obs_dtype refused", True)


@case
def encoder_matches_the_one_shot_function():
    obs, cfg, _ = episode(advance=100)
    enc = kaggri.Encoder(cfg)
    eo, em = enc.encode(obs)
    fo, fm = kaggri.encode_observation(obs, cfg)
    check("Encoder.encode == encode_observation (obs)", np.array_equal(bits(eo), bits(fo)))
    check("Encoder.encode == encode_observation (mask)", np.array_equal(bits(em), bits(fm)))


@case
def encoder_reuses_its_buffers():
    obs, cfg, env = episode(advance=30)
    enc = kaggri.Encoder(cfg)
    o1, m1 = enc.encode(obs)
    o2, m2 = enc.encode(obs)
    check("same array object returned every call", o1 is o2 and m1 is m2)
    snapshot = np.array(o1)
    for _ in range(24):
        env.step([PASS, PASS])
    obs2 = structify(env._Environment__get_shared_state(0)).observation
    o3, _ = enc.encode(obs2)
    check("the buffer is overwritten in place, not reallocated", o3 is o1)
    check("a caller who did not copy sees the new step",
          not np.array_equal(bits(np.asarray(o1)), bits(snapshot)),
          "the two steps encoded identically, so this proves nothing -- pick another step")


@case
def accepts_struct_plain_dict_and_object():
    obs, cfg, _ = episode(advance=50)
    ref = bits(kaggri.encode_observation(obs, cfg)[0])

    plain = copy.deepcopy(dict(obs))          # plain dicts and lists all the way down
    check("plain dict observation", np.array_equal(bits(kaggri.encode_observation(plain, cfg)[0]), ref))

    class Holder:                              # attribute access only
        pass
    h = Holder()
    for k, v in plain.items():
        setattr(h, k, v)
    check("attribute-only observation",
          np.array_equal(bits(kaggri.encode_observation(h, cfg)[0]), ref))


@case
def player_field_selects_the_point_of_view():
    """`player` comes from the observation, and it is what makes the plane
    player-relative.  Two all-PASS farms are symmetric, so the test has to make them
    asymmetric first -- otherwise identical planes would prove nothing."""
    obs, cfg, env = episode(advance=40)
    o1 = structify(env._Environment__get_shared_state(1)).observation
    check("observation.player is 0 and 1", int(obs["player"]) == 0 and int(o1["player"]) == 1)

    asym = copy.deepcopy(dict(obs))
    asym["farms"][0]["money"] = 5000.0
    asym["farms"][1]["money"] = 1000.0
    a = np.asarray(kaggri.encode_observation(asym, cfg, obs_dtype="float32")[0])
    flipped = copy.deepcopy(asym)
    flipped["player"] = 1
    b = np.asarray(kaggri.encode_observation(flipped, cfg, obs_dtype="float32")[0])
    check("flipping observation.player changes the plane",
          not np.array_equal(bits(a), bits(b)))
    # GLOBAL payload slots 7 and 9 are own / opponent money, linear (/20 000).
    hdr = kaggri.obs_dims_dict()["n_header"]
    own_a, opp_a = float(a[hdr + 7]), float(a[hdr + 9])
    own_b, opp_b = float(b[hdr + 7]), float(b[hdr + 9])
    check("player 0 sees its own $5 000 and the opponent's $1 000",
          abs(own_a - 0.25) < 1e-6 and abs(opp_a - 0.05) < 1e-6, f"{own_a}, {opp_a}")
    check("player 1 sees the mirror image",
          abs(own_b - 0.05) < 1e-6 and abs(opp_b - 0.25) < 1e-6, f"{own_b}, {opp_b}")


@case
def market_params_come_out_of_the_observation():
    """`market["params"]` exists only under a `marketParams` override, and is the
    authoritative source when present."""
    override = {"MELON": {"I0": 0, "T": 4, "above_func": "linear", "above_target": 40.0}}
    obs, cfg, _ = episode(advance=5, extra={"marketParams": override})
    check("the observation carries market.params", "params" in dict(obs["market"]))
    enc = kaggri.Encoder(cfg)
    got = dict(dict(enc.config()["marketParams"])["MELON"])
    check("override reached the encoder", got["I0"] == 0 and got["T"] == 4.0
          and got["above_target"] == 40.0, str(got))
    # And with the override stripped from the config, the observation still supplies it.
    bare = {k: v for k, v in cfg.items() if k != "marketParams"}
    a = kaggri.encode_observation(obs, cfg)[0]
    b = kaggri.encode_observation(obs, bare)[0]
    check("market.params wins over an absent config override",
          np.array_equal(bits(a), bits(b)))


@case
def action_row_widths():
    obs, cfg, _ = episode(advance=10)
    enc = kaggri.Encoder(cfg)
    for units in (1, 4, 17):
        row = np.zeros(2 * units + H, dtype=np.int64)
        a = enc.action(obs, row)
        check(f"units={units}: all-idle row -> a PASS turn",
              a["farmer"] == ["PASS"] and a["market"] == [], str(a))
    check("int32 rows accepted",
          enc.action(obs, np.zeros(2 * U + H, dtype=np.int32))["farmer"] == ["PASS"])
    check("python list accepted",
          enc.action(obs, [0] * (2 * U + H))["farmer"] == ["PASS"])
    for bad in (2 * U + H - 1, H, 2 * 18 + H):
        try:
            enc.action(obs, np.zeros(bad, dtype=np.int64))
            check(f"row width {bad} refused", False, "no exception")
        except ValueError:
            check(f"row width {bad} refused", True)


@case
def action_matches_pyvecenv_resolve_actions():
    obs, cfg, _ = episode(advance=0)
    enc = kaggri.Encoder(cfg)
    v = kaggri.PyVecEnv(1, {k: x for k, x in cfg.items() if k != "seed"}, [7])
    v.reset()
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(50):
        rows = np.stack([
            np.concatenate([rng.integers(0, 100, U), rng.integers(0, 38, U),
                            rng.integers(0, 7, H)])
            for _ in range(2)
        ]).astype(np.int64)
        ref = v.resolve_actions(0, rows)
        if enc.action(obs, rows[0]) != ref[0]:
            bad += 1
    check("50 random rows resolve identically to PyVecEnv.resolve_actions", bad == 0,
          f"{bad} mismatches")


@case
def hands_length_tracks_the_crew():
    """`out.hands` must be exactly `len(farm["hands"])`: a phantom slot still counts
    towards the interpreter's atomic-PLANT demand."""
    cfg = {"episodeSteps": 240, "seed": 3}
    env = make("kaggriculture", configuration=cfg, debug=True)
    env.reset(2)
    gc = {k: v for k, v in dict(env.configuration).items() if k in KNOWN}
    enc = kaggri.Encoder(gc)
    hire = {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 5}
    seen = set()
    bad = 0
    for i in range(30):
        obs = structify(env._Environment__get_shared_state(0)).observation
        n = len(obs["farms"][0]["hands"])
        seen.add(n)
        row = np.zeros(2 * U + H, dtype=np.int64)
        if len(enc.action(obs, row)["hands"]) != n:
            bad += 1
        env.step([hire if i < 6 else PASS, PASS])
    check("hands length == crew size at every step", bad == 0, f"{bad} bad")
    check("the crew actually grew during the test", len(seen) > 1 and max(seen) >= 5,
          f"crew sizes seen: {sorted(seen)}")


@case
def encode_and_action_agrees_with_the_two_calls():
    obs, cfg, _ = episode(advance=70)
    enc = kaggri.Encoder(cfg)
    row = np.arange(2 * U + H, dtype=np.int64) % 7
    o1, m1 = enc.encode(obs)
    o1, m1 = np.array(o1), np.array(m1)
    a1 = enc.action(obs, row)
    o2, m2, a2 = enc.encode_and_action(obs, row)
    check("encode_and_action obs matches", np.array_equal(bits(o1), bits(np.asarray(o2))))
    check("encode_and_action mask matches", np.array_equal(bits(m1), bits(np.asarray(m2))))
    check("encode_and_action action matches", a1 == a2, f"{a1} != {a2}")


@case
def state_from_observation_shape():
    obs, cfg, _ = episode(advance=80)
    d = kaggri.state_from_observation(obs, cfg)
    for k in ("step", "day", "hour", "market", "town", "farms", "privates", "seed",
              "player", "opponent_private_is_placeholder"):
        check(f"state_dict has {k!r}", k in d)
    check("player is echoed", int(d["player"]) == 0)
    check("the opponent private is flagged as a placeholder",
          d["opponent_private_is_placeholder"] is True)
    check("seed is NOT reconstructed (it is scrubbed from the observation)",
          int(d["seed"]) == 0, str(d["seed"]))
    check("money is a float", isinstance(d["farms"][0]["money"], float))
    check("Encoder.state_dict agrees",
          kaggri.Encoder(cfg).state_dict(obs)["farms"] == d["farms"])


@case
def required_fields_are_named_in_the_error():
    obs, cfg, _ = episode(advance=3)
    base = copy.deepcopy(dict(obs))
    for key in ("farms", "private", "market", "town"):
        broken = copy.deepcopy(base)
        del broken[key]
        try:
            kaggri.encode_observation(broken, cfg)
            check(f"missing {key!r} refused", False, "no exception")
        except ValueError as e:
            check(f"missing {key!r} refused, and named", key in str(e), str(e))
    # A board that disagrees with boardSize would silently shift every tile token.
    broken = copy.deepcopy(base)
    broken["farms"][0]["tiles"] = broken["farms"][0]["tiles"][:9]
    try:
        kaggri.encode_observation(broken, cfg)
        check("wrong-sized tile grid refused", False, "no exception")
    except ValueError as e:
        check("wrong-sized tile grid refused", "boardSize" in str(e), str(e))
    # Unknown enum values, too.
    for path, value, word in (
        (("farms", 0, "tiles", 0, 0), {"kind": "NOPE"}, "kind"),
        (("farms", 0, "unlocked_quadrants"), ["NOPE"], "quadrant"),
        (("town", "unlocked_shops"), ["NOPE"], "shop"),
    ):
        broken = copy.deepcopy(base)
        tgt = broken
        for p in path[:-1]:
            tgt = tgt[p]
        tgt[path[-1]] = value
        try:
            kaggri.encode_observation(broken, cfg)
            check(f"unknown {word} refused", False, "no exception")
        except ValueError as e:
            check(f"unknown {word} refused", word in str(e).lower(), str(e))
    # player must be 0 or 1.
    broken = copy.deepcopy(base)
    broken["player"] = 3
    try:
        kaggri.encode_observation(broken, cfg)
        check("player out of range refused", False, "no exception")
    except ValueError as e:
        check("player out of range refused", "player" in str(e), str(e))


@case
def unknown_config_key_is_refused():
    obs, cfg, _ = episode()
    try:
        kaggri.Encoder({**cfg, "someNewKnob": 1})
        check("unknown config key refused", False, "no exception")
    except ValueError as e:
        check("unknown config key refused", "someNewKnob" in str(e), str(e))
    for k in ("actTimeout", "runTimeout", "agentTimeout", "__raw_path__"):
        try:
            kaggri.Encoder({**cfg, k: 1})
            check(f"framework key {k!r} tolerated", True)
        except Exception as e:
            check(f"framework key {k!r} tolerated", False, repr(e))
    check("a None seed is treated as unset",
          int(kaggri.Encoder({**cfg, "seed": None}).config()["seed"]) == 0)


@case
def inventory_insertion_order_survives():
    """Trap 2: unit inventory order is observable state, and it must round-trip."""
    order = [("MELON", 5), ("WHEAT", 50)]
    obs, cfg, _ = episode(advance=2)
    for perm in (order, list(reversed(order))):
        broken = copy.deepcopy(dict(obs))
        broken["private"]["inventories"][0] = dict(perm)
        d = kaggri.state_from_observation(broken, cfg)
        got = [tuple(x) for x in d["privates"][0]["inventories_ordered"][0]]
        check(f"order {[k for k, _ in perm]} preserved", got == [tuple(x) for x in perm],
              f"got {got}")
    # And the order changes the encoding (UNIT slots 19-20 publish the head).
    a = copy.deepcopy(dict(obs))
    a["private"]["inventories"][0] = dict(order)
    b = copy.deepcopy(dict(obs))
    b["private"]["inventories"][0] = dict(reversed(order))
    check("the two orders encode differently",
          not np.array_equal(bits(kaggri.encode_observation(a, cfg)[0]),
                             bits(kaggri.encode_observation(b, cfg)[0])))


@case
def non_default_board_size():
    """A smaller board must leave the trailing tile tokens ENTIRELY zero -- type one-hot
    included, so "no such tile" is distinguishable from "empty tile"."""
    obs, cfg, _ = episode(advance=10, extra={"boardSize": 8, "turnsPerDay": 12,
                                             "shedCapacity": 30})
    check("the observation really is 8x8", len(obs["farms"][0]["tiles"]) == 8)
    o = np.asarray(kaggri.encode_observation(obs, cfg, obs_dtype="float32")[0])
    n_f = kaggri.N_FEATURES
    off_tile = kaggri.obs_dims_dict()["off_tile"]
    zero_rows = sum(
        1 for k in range(100)
        if not o[(off_tile + k) * n_f:(off_tile + k + 1) * n_f].any()
    )
    # 8x8 fills tiles[y*8 + x] for y in 0..7, i.e. token indices 0..63; 100 - 64 = 36.
    check("an 8x8 board leaves 36 tile tokens entirely zero", zero_rows == 36,
          f"{zero_rows} zero tile tokens")
    check("a 10x10 board leaves none",
          sum(1 for k in range(100)
              if not np.asarray(kaggri.encode_observation(
                  episode(advance=10)[0], episode(advance=10)[1], obs_dtype="float32")[0]
              )[(off_tile + k) * n_f:(off_tile + k + 1) * n_f].any()) == 0)
    # And a config that disagrees with the observation is refused rather than shifted.
    ten = {k: v for k, v in cfg.items()}
    ten["boardSize"] = 10
    try:
        kaggri.encode_observation(obs, ten)
        check("boardSize disagreeing with the grid refused", False, "no exception")
    except ValueError as e:
        check("boardSize disagreeing with the grid refused", "boardSize" in str(e), str(e))


def main() -> int:
    print(f"kaggri {kaggri.__version__}  OBS_VERSION={kaggri.OBS_VERSION} "
          f"ACTION_VERSION={kaggri.ACTION_VERSION} OBS_DTYPE={kaggri.OBS_DTYPE}")
    print("\n" + "=" * 72)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for f in FAILURES:
            print("  - " + f.split("\n")[0])
        return 1
    print("all encode_observation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
