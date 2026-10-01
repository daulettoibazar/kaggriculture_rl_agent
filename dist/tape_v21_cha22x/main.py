"""v11/v12 live agent: a public policy, run unchanged, with deterministic overrides on top.

The base policy is `policy.py`, the agent code of the Kaggle notebook
`thomastschinkel/kaggriculture-93-8-win-rate-public-state-router` (id_no 133420968), vendored
verbatim.  Kaggle public notebooks carry the Apache 2.0 default and the notebook states no other
licence; the manifest records the id, the fetch time and the sha of the file.

Why a wrapper rather than an extracted tape: the policy is a portfolio of five 719-step tapes
plus five decision trees, and it RE-CHOOSES its tape at steps 0, 144, 288, 432 and 576 from 100
observable features.  Wrapping runs the real chooser on the real observation, so there is no
route-inference step to get wrong and nothing to keep in sync.  Its module state (`_SESSIONS`,
keyed by seat) is never reset: the module is imported once and its entry point is called once per
turn, exactly as Kaggle runs it.

The entry point is resolved by Kaggle's own rule -- the LAST callable in `policy.py`'s namespace,
whatever it is named -- not by the name `agent`; see `resolve_entry` for why the difference is
neither cosmetic nor rare.

Our layer adds only two deterministic things:

* the opponent-class identifier (the same `Matcher` the tape line shipped), whose references are
  rebuilt with THIS policy in our seat; and
* market-row overrides read from `overrides.json`, applied to the turn the policy emitted.  An
  override is keyed by the class label (or ``ALL``) and the step, and it rewrites SELL/BUY orders
  the policy actually emitted, so it is well defined whichever tape the policy chose.  The kinds
  are ``defer`` (re-emit the order at a later step), ``drop`` (remove the slot), ``blank``
  (cancel the order but keep its slot, so the slot race order of the turn is unchanged) and
  ``qty`` (rewrite the quantity).

Everything else the policy returns is passed through untouched.  On any internal error the
wrapper falls back to the policy's own turn, and if that fails too, to an all-PASS turn.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

REFS = "identify_refs.npz"
OVERRIDES = "overrides.json"
OVERLAY = "overlay.py"
POLICY = "policy.py"
_QTY_CAP_DEFAULT = 10


def _log(msg):
    try:
        sys.stderr.write(f"[wrap] {msg}\n")
    except Exception:
        pass


def find_bundle_dir(raw_path=None):
    """The directory holding ``policy.py``.

    MUST be resolved at IMPORT time, which is what ``BUNDLE_DIR`` below does. Kaggle's loader
    compiles the agent from source with ``get_last_callable`` and sets no ``__file__`` and no
    ``sys.path`` entry; the only moment the bundle is locatable is during the load, when the
    loader's cwd is the bundle. Resolving lazily on the first turn found nothing, and the
    wrapper then played PASS for a whole game while reporting no exception -- 8 draws against
    PASS at the 3000-dollar starting cash. The shipped tape agent resolves at import for the
    same reason.
    """
    seen = []
    for cand in (raw_path, globals().get("__file__")):
        if cand:
            q = Path(cand)
            seen.append((q.parent if q.suffix else q))
    seen.append(Path.cwd())
    seen += [Path(x) for x in reversed(sys.path) if x]
    seen += [Path("/kaggle_simulations/agent"), Path("/kaggle/working")]
    for d in seen:
        try:
            if (d / POLICY).is_file():
                return d.resolve()
        except OSError:
            continue
    return None


BUNDLE_DIR = find_bundle_dir()


def load_policy(d):
    """Compile and exec the vendored base policy from ``d``."""
    spec = importlib.util.spec_from_file_location("wrap_policy", Path(d) / POLICY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_entry(mod):
    """The callable Kaggle would actually run in ``policy.py`` -- NOT necessarily ``mod.agent``.

    ``kaggle_environments.agent.get_last_callable`` exec's the file into a fresh dict and returns
    ``[v for v in env.values() if callable(v)][-1]``: the LAST callable in the module namespace, by
    dict insertion order, whatever it is named.  A module's ``__dict__`` orders identically, so the
    same comprehension over ``vars(mod)`` reproduces the rule exactly.

    This matters because re-binding an existing name does NOT move it to the end of a dict.  The
    2026-09-2x public notebooks are composites that bind ``agent`` dozens of times and then publish
    their final layer under another name (``kaggle_agent``, ``kaggle_submission_agent``,
    ``submission_v67``, ``opening_liquidity_agent``), leaving the bare name ``agent`` pointing at an
    earlier, weaker layer; others deliberately end with ``agent = globals().pop('agent')``, whose
    pop-then-assign is precisely what moves ``agent`` last.  Calling ``mod.agent`` therefore shipped
    a DIFFERENT agent from the one the notebook runs on Kaggle, with no exception and no fallback to
    show for it -- measured at 0.962 -> 0.302 exact score on one 2026-09-24 base.

    ``agent`` remains the fallback for a policy whose namespace holds no callable at all.
    """
    try:
        callables = [v for v in vars(mod).values() if callable(v)]
        if callables:
            return callables[-1]
    except BaseException:
        _log("entry-point resolution raised; falling back to the name 'agent'")
        traceback.print_exc(file=sys.stderr)
    return getattr(mod, "agent", None)


def entry_argcount(fn) -> int:
    """Kaggle truncates ``[observation, configuration]`` to the entry point's positional arity."""
    code = getattr(fn, "__code__", None)
    n = getattr(code, "co_argcount", 2) if code is not None else 2
    return 2 if n >= 2 else max(1, n)


def _preload():
    """Exec `policy.py` at IMPORT, not on the first turn.

    Kaggle compiles and runs this module during agent initialisation, which carries a far larger
    allowance than the ~1 s per-turn budget; the first TURN does not. The base policies are now
    single self-contained files in the hundreds of kilobytes (the v15 base is 285 KB), and
    exec'ing one costs up to ~0.9 s cold -- which the lazy path spent inside turn 1, on a 1.6
    vCPU Kaggle worker, against a 1 s limit. Moving it here costs nothing and removes the whole
    class of first-turn timeout.

    Failure here is not fatal: `Runner.setup` still loads the policy lazily, so a bundle whose
    directory is only locatable later behaves exactly as before.
    """
    if BUNDLE_DIR is None:
        return None
    try:
        return load_policy(BUNDLE_DIR)
    except BaseException:
        _log("preload failed; the policy will be loaded on the first turn")
        traceback.print_exc(file=sys.stderr)
        return None


PRELOADED = _preload()


class Runner:
    def __init__(self):
        self.ready = False
        self.policy = None
        self.entry = None          # the callable Kaggle would run; see resolve_entry()
        self.entry_argc = 2
        self.overlay = None        # optional per-bundle turn post-processor; see setup()
        self.matcher = None
        self.overrides = {}
        self.dir = None
        self.pending = {}          # step -> [order, ...] injected later by a deferral
        self.seat = None
        self.label = None
        self.err = 0

    def setup(self, obs):
        """Load the base policy, then the OPTIONAL layers.

        ``ready`` is set as soon as the policy is in hand. Anything after it is optional and its
        failure must never leave ``ready`` False: an earlier version loaded the identifier module
        unconditionally, so a bundle shipped without it re-ran this whole method on every turn,
        re-imported the policy 719 times, reset its per-seat route latch each time, and turned a
        0.5 mirror into 0.02 while burning 45 ms a turn and 29 s of the overage bank.
        """
        d = BUNDLE_DIR or find_bundle_dir()
        if d is None:
            raise FileNotFoundError(f"{POLICY} not found; the bundle was not locatable at import")
        self.dir = d
        self.policy = PRELOADED if PRELOADED is not None else load_policy(d)
        self.entry = resolve_entry(self.policy)
        if self.entry is None:
            raise AttributeError(f"{POLICY} exposes no callable entry point")
        self.entry_argc = entry_argcount(self.entry)
        self.seat = int(obs.get("player", 0))
        self.ready = True                      # everything below is optional

        refs = d / REFS
        mpath = d / "matcher.py"
        if refs.exists() and mpath.exists():
            try:
                sp = importlib.util.spec_from_file_location("wrap_match", mpath)
                mm = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(mm)
                self.matcher = mm.Matcher(str(refs), self.seat)
            except BaseException:
                _log("identifier unavailable; running the base policy with ALL overrides only")
                traceback.print_exc(file=sys.stderr)
                self.matcher = None
        elif refs.exists() != mpath.exists():
            _log(f"identifier half-installed (refs {refs.exists()}, matcher {mpath.exists()}); "
                 f"running without it")

        o = d / OVERRIDES
        if o.exists():
            try:
                raw = json.loads(o.read_text())
                self.overrides = {cls: {int(k): v for k, v in tbl.items()}
                                  for cls, tbl in raw.get("by_class", {}).items()}
            except BaseException:
                _log("overrides.json unreadable; running the base policy")
                self.overrides = {}

        # Optional per-bundle turn post-processor.  Absent in every bundle shipped so far, and a
        # bundle without it behaves exactly as before.  Same contract as the identifier above: it
        # is optional, so a failure to load must leave the base policy running, not break the turn.
        ov = d / OVERLAY
        if ov.is_file():
            try:
                sp = importlib.util.spec_from_file_location("wrap_overlay", ov)
                om = importlib.util.module_from_spec(sp)
                sp.loader.exec_module(om)
                fn = getattr(om, "apply", None)
                # An optional `configure(policy)` runs once, before the policy's first turn, and may
                # set the policy module's tunable globals; `policy.py` itself stays verbatim.
                conf = getattr(om, "configure", None)
                if callable(conf):
                    conf(self.policy)
                self.overlay = fn if callable(fn) else None
                if self.overlay is None:
                    _log(f"{OVERLAY} defines no callable `apply`; ignoring it")
            except BaseException:
                _log("overlay failed to load; running the base policy without it")
                traceback.print_exc(file=sys.stderr)
                self.overlay = None

    # ------------------------------------------------------------------ overrides
    @staticmethod
    def _match(order, spec):
        if not order or str(order[0]) != str(spec.get("op")):
            return False
        if spec.get("product") is not None and str(order[1]) != str(spec["product"]):
            return False
        q = spec.get("qty")
        if q is not None and int(order[2]) != int(q):
            return False
        return True

    def _apply(self, turn, step, cls, cap):
        rules = []
        for key in ("ALL", cls):
            if key and key in self.overrides:
                rules += self.overrides[key].get(step, [])
        market = [list(o) for o in (turn.get("market") or [])]
        injected = self.pending.pop(step, [])
        if not rules and not injected:
            if injected:
                market = market + [list(o) for o in injected]
            turn["market"] = market[:cap]
            return turn
        out = []
        for order in market:
            hit = None
            for r in rules:
                if self._match(order, r):
                    hit = r
                    break
            if hit is None:
                out.append(order)
                continue
            kind = str(hit.get("k"))
            if kind == "drop":
                continue
            if kind == "blank":
                # Market orders race by SLOT INDEX, so removing one promotes every later order
                # of the same turn into an earlier slot and changes who wins each race. The
                # base policy's own `market_order` writes ['SELL','WHEAT',0] for a slot it
                # cannot fill, and the engine reads a zero-quantity sale as an empty slot, so
                # this cancels the order while leaving every other slot exactly where the
                # policy put it. `drop` remains available for the cases where the promotion is
                # itself the wanted effect.
                out.append(["SELL", "WHEAT", 0])
                continue
            if kind == "defer":
                to = int(hit.get("to", step))
                if to > step:
                    self.pending.setdefault(to, []).append(list(order))
                    continue
                out.append(order)
            elif kind == "qty":
                o2 = list(order)
                o2[2] = int(hit["new"])
                out.append(o2)
            else:
                out.append(order)
        out += [list(o) for o in injected]
        turn["market"] = out[:cap]
        return turn

    # ------------------------------------------------------------------ per turn
    def call_base(self, obs, config=None):
        """One call into the vendored policy, exactly as Kaggle would make it."""
        return self.entry(*(obs, config)[: self.entry_argc])

    def turn(self, obs, config=None):
        if not self.ready:
            try:
                self.setup(obs)
            except BaseException:
                # A base policy that failed to load must be loud: an inert all-PASS turn scores
                # like a working agent's loss and is invisible in a win-rate panel.
                _log("FATAL: the base policy did not load")
                traceback.print_exc(file=sys.stderr)
                raise
        base = self.call_base(obs, config)               # the policy, unchanged
        if not isinstance(base, dict):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        step = int(obs.get("step", 0))
        cap = _QTY_CAP_DEFAULT
        try:
            if config:
                cap = int((config.get("maxMarketOrdersPerTurn") if hasattr(config, "get")
                           else getattr(config, "maxMarketOrdersPerTurn", cap)) or cap)
        except BaseException:
            pass
        if self.matcher is not None:
            try:
                shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
                self.matcher.observe(step, obs, self.seat, shops)
                if step in (144, 216):
                    self.label = self.matcher.label
            except BaseException:
                self.err += 1
                if self.err == 1:
                    _log("identifier raised; continuing on the base policy")
                    traceback.print_exc(file=sys.stderr)
        turn = base
        if self.overrides:
            try:
                turn = self._apply(base, step, self.label, cap)
            except BaseException:
                self.err += 1
                if self.err <= 2:
                    _log("override layer raised; returning the base policy's turn")
                    traceback.print_exc(file=sys.stderr)
                turn = base
        if self.overlay is not None:
            try:
                out = self.overlay(obs, config, turn, self.seat)
                if isinstance(out, dict):
                    turn = out
            except BaseException:
                self.err += 1
                if self.err <= 2:
                    _log("overlay raised; returning the turn it was given")
                    traceback.print_exc(file=sys.stderr)
        return turn


RUNNER = Runner()


def agent(obs, config=None):
    try:
        return RUNNER.turn(obs, config)
    except BaseException:
        try:
            sys.stderr.write("[wrap] FATAL:\n")
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        if RUNNER.entry is not None:
            # the policy loaded, so something in OUR layer failed: fall back to it
            return RUNNER.call_base(obs, config)
        # the policy never loaded. An all-PASS turn would score like an ordinary loss and hide
        # a broken bundle behind a plausible number, so raise and let the harness record it.
        raise
