"""Knob overlay for a wrapped public base, loaded by `submission/wrap_main.py` as `overlay.py`.

Why: two copies of the same program split the contested town demand evenly (cha22 against itself is
92% exact draws, margin $0).  A twin that differs from the base only in a few of its own tunable
constants breaks that symmetry; the aurax7 reservation horizon 4 -> 12 (v16) is the precedent.

It does two things, both configured by `overlay.json` beside this file:

  set         `configure(policy)` assigns module globals of the vendored `policy.py` once, before its
              first turn.  Only names that already exist are assigned, and only when the new value has
              the same type as the old one, so a typo is a no-op rather than a new global.  The policy
              reads these names at call time, so the file on disk stays verbatim.

  term_step   from this step on, the market rows become one SELL of every shed good priced >= $1, by
              price descending (the base's own terminal layer, T62A, does exactly this from step 712;
              an earlier step starts the same liquidation sooner).  Absent or null: untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

_CFG = {}
try:
    _p = Path(__file__).with_name("overlay.json")
    if _p.is_file():
        _CFG = json.loads(_p.read_text())
except Exception:
    _CFG = {}

APPLIED = {}


def _coerce(old, new):
    if isinstance(old, tuple) and isinstance(new, list):
        return tuple(tuple(x) if isinstance(x, list) else x for x in new)
    if isinstance(old, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def configure(policy):
    for name, value in (_CFG.get("set") or {}).items():
        if not hasattr(policy, name):
            continue
        old = getattr(policy, name)
        new = _coerce(old, value)
        if type(new) is not type(old):
            continue
        setattr(policy, name, new)
        APPLIED[name] = new


def apply(obs, config, turn, seat):
    t = _CFG.get("term_step")
    if t is None:
        return turn
    step = int(obs.get("step", 0))
    if step < int(t) or step >= 718:
        return turn
    shed = (obs.get("private") or {}).get("shed") or {}
    prices = (obs.get("market") or {}).get("prices") or {}
    items = [it for it, q in shed.items() if int(q) > 0 and int(prices.get(it, 0)) >= 1]
    if not items:
        return turn
    items.sort(key=lambda it: -int(prices.get(it, 0)))
    out = dict(turn)
    out["market"] = [["SELL", it, int(shed[it])] for it in items[:10]]
    return out
