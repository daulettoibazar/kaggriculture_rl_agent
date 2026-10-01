"""Opponent registry for the fs1 self-play trainer and the evaluators.

``struct`` records which observation flavour the program expects: the public
notebooks and the wrapped bundles read ``obs.farms[0].money`` (attribute access), so
they need the ``kaggle_environments`` ``Struct`` that the live harness hands an agent.
Getting this wrong is not silent -- the program raises inside ``KernelPolicy.act`` and the
call is counted as a fallback -- but it also costs a whole game, so the flag is declared
here once.

The three role sets are disjoint by construction and validated at import:

* ``FIELD``   -- opponents the trainer is allowed to draw during collection.
* ``HELDOUT`` -- never trained against; selection reads only these.
* ``FINAL``   -- never touched until the line is finished, so the last number is
  measured against programs that took no part in either fitting or selection.

Public notebooks are pulled into ``data/public_kernels/<author>_<slug>/`` (agent in
``extracted/main.py``). Names whose file is absent are reported by ``missing_paths()``.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ``mirror`` is not a file: it is the base itself with the residual disabled
# (all-pass).  It is the only opponent that makes the twin-vs-twin $0 draw mass
# visible during training, so it is a first-class member of the field.
MIRROR = "mirror"

OPPONENTS: dict[str, dict] = {
    # --- public kernels -----------------------------------------------------
    "router955": {"path": "data/public_kernels/thomastschinkel_kaggriculture-95-5-win-rate-via-replay-routing/extracted/main.py",
                "struct": True, "kind": "kernel"},
    "hysteresis": {"path": "data/public_kernels/boatlee_v29-r1-adaptive-market-hysteresis/extracted/main.py",
                 "struct": True, "kind": "kernel"},
    "minimax58": {"path": "data/public_kernels/kaitofukami_238-238-known-streams-v58-minimax-closed-loop/extracted/main.py",
                "struct": True, "kind": "kernel"},
    "top10_market": {"path": "data/public_kernels/indarkarhana_rank-top10-read-the-market-choose-the-farm/extracted/main.py",
                   "struct": True, "kind": "kernel"},
    "economic111": {"path": "data/public_kernels/denizeryilmaz_v111-8c4s-economic-core-premium-lead/extracted/main.py",
                  "struct": True, "kind": "kernel"},
    "reactive_aurax": {"path": "data/public_kernels/aurax7_kaggriculture-reactive-router/extracted/main.py",
                     "struct": True, "kind": "kernel"},
    "rv5": {"path": "data/public_kernels/aurax7_kaggriculture-shop-router-reactive-v5/extracted/main.py",
          "struct": True, "kind": "kernel"},
    "rv4": {"path": "data/public_kernels/aurax7_kaggriculture-shop-router-reactive-v4/extracted/main.py",
          "struct": True, "kind": "kernel"},
    "rv2": {"path": "data/public_kernels/aurax7_kaggriculture-shop-router-reactive-v2/extracted/main.py",
          "struct": True, "kind": "kernel"},
    "mengine_v4": {"path": "data/public_kernels/guruprasaathas111_kaggriculture-master-engine-v4/extracted/main.py",
                 "struct": True, "kind": "kernel"},
    "yummers_v52": {"path": "data/public_kernels/romantamrazov_kaggriculture-yummers/extracted/main.py",
                  "struct": True, "kind": "kernel"},
    "abo_v50": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v50-early-yarn-commit/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "abo_v49": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v49-funded-sale-timing-and-worker/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "abo_v48": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v48-clear-the-queue/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "farm2945": {"path": "data/public_kernels/thomastschinkel_the-2945-farm-96-vs-the-top-10-public-bots/extracted/main.py",
               "struct": True, "kind": "kernel"},
    "shiiin9": {"path": "data/public_kernels/shiiin9_beat-v48-100-0-your-herd-is-decided-on-day-6/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "goodpjw": {"path": "data/public_kernels/goodpjw2008_kaggriculture-melon-threshold-squeeze-2749/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "haideptry_2965": {"path": "data/public_kernels/haideptry_the-2965-master-hybrid-engine/extracted/main.py",
                     "struct": True, "kind": "kernel"},
    "dmitrii_herd": {"path": "data/public_kernels/dmitriigluzdov_kaggriculture-herd-safe-sale-window-lb-2700/extracted/main.py",
                   "struct": True, "kind": "kernel"},
    "prvsiyan_frontier": {"path": "data/public_kernels/prvsiyan_kaggriculture-frontier-the-soil-remembers-rain/extracted/main.py",
                        "struct": True, "kind": "kernel"},
    "abo_v57": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v57-funding-order-invariant/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "abo_v56": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v56-smarter-seeds-and-fertilizer/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "abo_v55": {"path": "data/public_kernels/ahmedberatozer_kaggriculture-v55-one-turn-market-race-edge/extracted/main.py",
              "struct": True, "kind": "kernel"},
    "cha22": {"path": "data/public_kernels/abhinav0370_cha22-agent/extracted/main.py",
            "struct": True, "kind": "kernel"},
    "harvest_ledger": {"path": "data/public_kernels/haodou092_kaggriculture-harvest-ledger/extracted/main.py",
                     "struct": True, "kind": "kernel"},
    "shepherds": {"path": "data/public_kernels/haideptry_the-shepherds-ledger-herd-safe-sovereign/extracted/main.py",
                "struct": True, "kind": "kernel"},
    "mengine_v53": {"path": "data/public_kernels/guruprasaathas111_kaggriculture-master-engine-v53e01d74d8f/extracted/main.py",
                  "struct": True, "kind": "kernel"},
    "evgendvorkin_v31": {"path": "data/public_kernels/evgendvorkin_kaggriculture-version-31-26-09-bronze-going-up/extracted/main.py",
                       "struct": True, "kind": "kernel"},
    "flexonafft": {"path": "data/public_kernels/flexonafft_kaggriculture-multi-route-farming-agent/extracted/main.py",
                 "struct": True, "kind": "kernel"},
    "mengine_v3": {"path": "data/public_kernels/guruprasaathas111_kaggriculture-master-engine-v3/extracted/main.py",
                 "struct": True, "kind": "kernel"},
    "evgendvorkin": {"path": "data/public_kernels/evgendvorkin_kaggriculture/extracted/main.py",
                   "struct": True, "kind": "kernel"},
    "arsgorynich_hs3": {"path": "data/public_kernels/arsgorynich_herd-safe-v3-experimental-risk-aware-feed/extracted/main.py",
                      "struct": True, "kind": "kernel"},
    # --- bundles: wrapper plus the vendored notebook policy (policy.py is not shipped in the public repo) ---
    "cha22_bundle": {"path": "dist/tape_v20_cha22/main.py", "struct": True, "kind": "bundle"},
    "cha22x_bundle": {"path": "dist/tape_v21_cha22x/main.py", "struct": True, "kind": "bundle"},
}

DEFAULT_BASE = "cha22x_bundle"

# --- the field: residual/self-play opponents over the cha22x family --------------------------
# CONTESTED = the cells the cha22 family does not already win outright: selection reads the
# worst of these.  A delta against a program the base already beats outright is bounded above
# by zero and can only dilute a real edge.  Every set is overridable per run from the config
# (`--field-opponents`, `--heldout-opponents`, `--final-opponents`), and `field_weights()` only
# returns its prior when the field it is handed IS this one.
FIELD_V21 = (MIRROR, "cha22x_bundle", "cha22_bundle", "prvsiyan_frontier", "harvest_ledger", "shepherds", "dmitrii_herd", "abo_v57", "abo_v56", "goodpjw", "haideptry_2965", "mengine_v4")
FIELD_WEIGHTS_V21 = (0.1042, 0.125, 0.125, 0.1042, 0.1042, 0.1042, 0.0833, 0.0625, 0.0417, 0.0625, 0.0417, 0.0415)
HELDOUT_V21 = ("farm2945", "shiiin9", "abo_v50")
FINAL_V21 = ("hysteresis", "minimax58")
CONTESTED_V21 = (MIRROR, "cha22x_bundle", "cha22_bundle", "prvsiyan_frontier", "harvest_ledger", "shepherds", "dmitrii_herd", "abo_v57")

FIELD = FIELD_V21
FIELD_WEIGHTS = FIELD_WEIGHTS_V21
HELDOUT = HELDOUT_V21
FINAL = FINAL_V21
CONTESTED = CONTESTED_V21


def path_of(name: str) -> Path:
    """Absolute path of a registered program (``mirror`` has none)."""
    if name == MIRROR:
        raise KeyError("`mirror` is the base itself, not a separate program")
    if name not in OPPONENTS:
        raise KeyError(f"Unknown opponent {name!r}")
    return ROOT / OPPONENTS[name]["path"]


def struct_of(name: str, base: str = DEFAULT_BASE) -> bool:
    """Whether this program must be handed a ``Struct`` observation."""
    return bool(OPPONENTS[base if name == MIRROR else name]["struct"])


def field_weights(field=FIELD, weights=FIELD_WEIGHTS):
    """Default draw over the field, normalised; ``None`` for any field we have no prior for."""
    if tuple(field) != tuple(FIELD):
        return None
    total = float(sum(weights))
    return tuple(w / total for w in weights)


def validate_sets(field=FIELD, heldout=HELDOUT, final=FINAL) -> None:
    """Every name registered, every role disjoint from every other role."""
    roles = {"field": tuple(field), "heldout": tuple(heldout), "final": tuple(final)}
    for role, names in roles.items():
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate opponent inside the {role} set")
        for name in names:
            if name != MIRROR and name not in OPPONENTS:
                raise KeyError(f"{role} set names unregistered opponent {name!r}")
    labels = list(roles)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = set(roles[a]) & set(roles[b])
            if shared:
                raise ValueError(f"{a} and {b} share {sorted(shared)}")


def missing_paths() -> list[str]:
    """Registered names whose file is absent (checked, never assumed)."""
    return sorted(name for name in OPPONENTS if not (ROOT / OPPONENTS[name]["path"]).is_file())


validate_sets()
if len(FIELD_WEIGHTS) != len(FIELD):
    raise ValueError("One default field weight per field opponent is required")
if abs(sum(FIELD_WEIGHTS) - 1.0) > 1e-9:
    raise ValueError("Default field weights must sum to one")
if not set(CONTESTED) <= set(FIELD):
    raise ValueError("Every contested opponent must be in the field")
