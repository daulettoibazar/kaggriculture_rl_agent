"""fs1 Kaggle agent: the exact simulator's encoder + the numpy transformer, no torch.

Bundle members (dist/fs1_<tag>/): main.py (this file), fs1_numpy_policy.py, fs1_policy.npz,
kaggri/ (the abi3 extension package, Python >= 3.11, glibc >= 2.34), manifest.json.

Per turn: rebuild the state from the observation with `kaggri.Encoder` (bit-identical to the
training simulator), append the price-delta history the trainer used, run the numpy policy
greedily, and convert the 55-int action row back to the {"farmer","hands","market"} dict with
the same resolver the trainer stepped. Any exception falls back to PASS and is logged to stderr.

An optional member `fs1_bundle.json` (rl_last) sets {"temperature": T, "relax_masks": bool}: T > 0
samples the policy at that temperature (fresh OS-seeded RNG per process), and relax_masks applies
the F1 unit-mask relaxation the policy was trained with. Absent = greedy, unrelaxed (fs1).
"""
import json
import os
import sys

# One BLAS / rayon thread, set before numpy and kaggri load: Kaggle gives the agent ~1.6 vCPU, and
# an unpinned BLAS spawns a thread per host core (a local -j 1 arena run hit 2,950% CPU).
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"
import time
import traceback

MARKER = "fs1_policy.npz"
KNOWN_CONFIG_KEYS = (
    "episodeSteps", "boardSize", "startingMoney", "maxMarketOrdersPerTurn", "turnsPerDay",
    "shedCapacity", "weedSpawnChance", "townShopUnlockInterval", "townShopSellInterval",
    "townCenterSellInterval", "farmHandCostMult", "seed", "marketParams",
)
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def _find_bundle():
    own = globals().get("__file__")
    cands = [os.path.dirname(os.path.abspath(own))] if own else []
    cands.append(os.getcwd())          # Kaggle's loader sets no __file__; its cwd is the bundle during the load
    cands += [d for d in reversed(sys.path) if d]
    cands.append("/kaggle_simulations/agent")
    for d in cands:
        try:
            if os.path.isfile(os.path.join(d, MARKER)):
                return os.path.abspath(d)
        except Exception:
            continue
    return None


BUNDLE = _find_bundle() or "/kaggle_simulations/agent"
if BUNDLE not in sys.path:
    sys.path.insert(0, BUNDLE)

import numpy as np  # noqa: E402


def _pin_blas_threads():
    """The env vars above only bind a BLAS loaded after this file; Kaggle's harness imports numpy
    first. Cap the already-loaded OpenBLAS at 1 thread too (threadpoolctl, else its C symbol)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
        return "threadpoolctl"
    except Exception:
        pass
    try:
        import ctypes
        with open("/proc/self/maps") as f:
            libs = {ln.split()[-1] for ln in f if "openblas" in ln.lower() and "/" in ln}
        for lib in sorted(libs):
            h = ctypes.CDLL(lib)
            for sym in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_", "openblas_set_num_threads"):
                fn = getattr(h, sym, None)
                if fn is not None:
                    fn(1)
                    return sym
    except Exception:
        pass
    return "none"


BLAS_PIN = _pin_blas_threads()

_T0 = time.perf_counter()
import kaggri  # noqa: E402
from fs1_numpy_policy import MAX_UNITS, N_TARGETS, N_VERBS, NumpyFs1Policy, PriceHistory, relax_unit_mask  # noqa: E402

IMPORT_MS = (time.perf_counter() - _T0) * 1e3

try:
    with open(os.path.join(BUNDLE, "fs1_bundle.json")) as _f:
        BUNDLE_CFG = json.load(_f)
except FileNotFoundError:
    BUNDLE_CFG = {}
TEMPERATURE = float(BUNDLE_CFG.get("temperature", 0.0))
RELAX_MASKS = bool(BUNDLE_CFG.get("relax_masks", False))
RNG = np.random.default_rng()

_STATE = {"enc": None, "policy": None, "hist": None, "last_step": -1, "turns": 0, "fallbacks": 0,
          "time_ms": []}


def _config_for_kaggri(config):
    d = dict(config) if config is not None else {}
    return {k: d[k] for k in KNOWN_CONFIG_KEYS if k in d and d[k] is not None}


def _ensure_loaded(config):
    if _STATE["policy"] is None:
        _STATE["policy"] = NumpyFs1Policy.load(os.path.join(BUNDLE, MARKER))
    if _STATE["enc"] is None:
        # float32 planes: the trainer computed the price history from the f32 observation
        _STATE["enc"] = kaggri.Encoder(_config_for_kaggri(config), obs_dtype="float32")
    if _STATE["hist"] is None:
        _STATE["hist"] = PriceHistory(_STATE["policy"].hist_k)


def act_row(obs, config):
    """The 55-int action row for this observation (exposed for parity tests)."""
    _ensure_loaded(config)
    o = dict(obs)
    step = int(o.get("step", 0) or 0)
    if step == 0 or step < _STATE["last_step"]:
        _STATE["hist"] = PriceHistory(_STATE["policy"].hist_k)      # new episode
    _STATE["last_step"] = step
    enc = _STATE["enc"]
    obs_arr, mask, tmask = enc.encode_full(o)
    aug = _STATE["hist"].augment(np.asarray(obs_arr))
    um = np.asarray(mask).reshape(1, MAX_UNITS, N_VERBS) > 0.5
    tm = np.asarray(tmask).reshape(1, N_TARGETS, N_VERBS) > 0
    if RELAX_MASKS:
        um = relax_unit_mask(um, tm)
    return _STATE["policy"].act(aug[None], um, tm, temperature=TEMPERATURE, rng=RNG)[0]


def agent(obs, config):
    t0 = time.perf_counter()
    try:
        row = act_row(obs, config)
        action = _STATE["enc"].action(dict(obs), row)
    except Exception:
        _STATE["fallbacks"] += 1
        traceback.print_exc(file=sys.stderr)
        action = PASS
    _STATE["turns"] += 1
    _STATE["time_ms"].append((time.perf_counter() - t0) * 1e3)
    return action
