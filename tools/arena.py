#!/usr/bin/env python3
"""
arena.py -- reusable local benchmark harness for the Kaggle "kaggriculture" env.

Design notes (all verified against kaggle-environments 1.32.7):

* One `Environment` object is used for EXACTLY ONE episode. `env.info["seed"]`
  survives `reset()`, so re-running the same env replays the same episode, and
  `kaggle_environments.evaluate(num_episodes=N)` returns N *identical* episodes.
  Every episode here gets a fresh `make(...)` with an explicit seed.
* `debug=False` is mandatory for benchmarking: with `debug=True` an exception in
  the agent propagates out of `env.run()` instead of being converted into an
  `ERROR` status.
* The kaggriculture interpreter overwrites every agent's status with `DONE` and
  its reward with its final money on the last step, so a mid-episode
  ERROR/TIMEOUT is INVISIBLE in `env.steps[-1]`. This harness therefore scans
  every step's status and reports `err_steps` / `timeout_steps` explicitly.
  Treat any non-zero count as a failed submission.
* Both `farms[]` and the market are shared state, so results are only comparable
  across runs that use the same seed AND the same pair of agents. Weed rolls and
  town-shop draws come from one RNG stream shared by both players, so changing
  the opponent changes YOUR town-shop schedule even at a fixed seed.
* Sides are swapped: market orders are resolved in player order and player 0's
  weed rolls are drawn first, so p0 and p1 are not symmetric. `--sides both`
  (default) plays every seed from both seats.

* Local runs are SEQUENTIAL inside one process (only UrlAgent is
  `is_parallelizable`), so an episode's wall clock is the sum of BOTH agents'
  compute. On Kaggle each agent gets its own process, so it isn't.
  Consequence for self-play: two *callable* agents share module globals, two
  *file-path* agents do not (each gets its own exec'd globals dict). Prefer the
  main.py path form for self-play. (Both verified.)
* Measured throughput on a 192-core box: 1.48 s/episode serial, ~23 episodes/s
  at -j 64.

Agent specs (strings, so they survive process boundaries):
    "pass" | "random" | "starter"      built-in agents
    "/path/to/main.py"                 file agent (loaded with the same
                                       get_last_callable() rule Kaggle uses:
                                       the LAST module-level callable wins)
    "pkg.module:func"                  dotted import path
    "reg:name"                         a callable registered via register_agent()

CLI:
    python3 tools/arena.py --agent starter --opponents pass,starter,random -n 24
    python3 tools/arena.py --agent dist/tape_v21_cha22x/main.py --opponents pass -n 4 --sides both -j 1
    python3 tools/arena.py --agent a/main.py --opponents b/main.py -n 24   # head to head
    python3 tools/arena.py --selftest

Reproducibility: every run prints its seed list, its `episodeSteps`/config, the worker
count and any `KAGGRI_AGENT_*` environment override, and `--json` records all of them
under the `_run` key.  Seeds default to `range(--seed0, --seed0 + -n)`, so a run is
reproduced by repeating the printed line.

Timing: per-turn numbers measured with several workers are CONTENDED -- each worker
loads its own torch on turn 1 and they collide.  On Kaggle every agent owns its own
1.6-vCPU container.  Use `-j 1` for budget-grade timing and a larger `-j` for strength.

API:
    from arena import run_match, register_agent, MatchResult
    res = run_match("starter", "pass", n_seeds=24)
    print(res.win_rate, res.mean_money, res.p95_step_ms)
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# defaults taken from kaggriculture.json / schemas.json
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {"episodeSteps": 720}
ACT_TIMEOUT = 1.0          # kaggriculture.json  configuration.actTimeout
OVERAGE_BANK = 60.0        # kaggriculture.json  observation.remainingOverageTime
BUILTINS = ("pass", "random", "starter")

#: This box is cgroup-capped at 168 CPUs even though `os.cpu_count()` reports 192.
#: Oversubscribing measured a 4.4x slowdown (src/README.md 6.5), and every worker here
#: runs a torch agent that will happily grab threads of its own, so the pool is sized
#: against the real quota and `KAGGRI_CPU_QUOTA` narrows it further.
CPU_CAP = 168


def cpu_quota() -> int:
    """Usable CPUs: `KAGGRI_CPU_QUOTA` if set, else the cgroup cap."""
    env = os.environ.get("KAGGRI_CPU_QUOTA")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except Exception:
        affinity = os.cpu_count() or 1
    return max(1, min(CPU_CAP, affinity))

_REGISTRY: dict[str, Callable] = {}


def register_agent(name: str, fn: Callable) -> str:
    """Register an in-process callable and return its spec string ("reg:name").

    Works with the parallel runner only under the "fork" start method (the
    default on Linux) because children inherit the registry. Register BEFORE
    calling run_match().
    """
    _REGISTRY[name] = fn
    return f"reg:{name}"


def resolve_agent(spec: Any) -> Any:
    """Turn a spec into something env.run() accepts."""
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"bad agent spec: {spec!r}")
    if spec in BUILTINS:
        return spec
    if spec.startswith("reg:"):
        key = spec[4:]
        if key not in _REGISTRY:
            raise KeyError(f"agent {key!r} not in registry (register_agent before forking)")
        return _REGISTRY[key]
    if ":" in spec and not os.path.exists(spec):
        mod, _, fn = spec.partition(":")
        return getattr(importlib.import_module(mod), fn)
    # file path -> hand the path to kaggle_environments so the real
    # get_last_callable() loading rules apply.
    return os.path.abspath(spec)


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    seed: int
    my_seat: int
    my_money: float
    opp_money: float
    my_status_final: str
    opp_status_final: str
    err_steps: int
    timeout_steps: int
    invalid_steps: int
    n_acts: int
    durations: list[float] = field(default_factory=list)   # my agent, seconds
    overage_left: float = OVERAGE_BANK
    wall: float = 0.0
    exception: str | None = None

    @property
    def overage_used(self) -> float:
        """Seconds of the 60 s bank this episode consumed. >60 == TIMEOUT."""
        return max(0.0, OVERAGE_BANK - self.overage_left)

    @property
    def turn1_ms(self) -> float:
        """First turn in isolation: it carries the model load and the torch import."""
        return 1000.0 * self.durations[0] if self.durations else float("nan")

    @property
    def steady_max_ms(self) -> float:
        """Worst turn EXCLUDING turn 1, which is the number the 1 s limit bites on."""
        return 1000.0 * max(self.durations[1:]) if len(self.durations) > 1 else float("nan")

    @property
    def result(self) -> int:
        if self.my_money > self.opp_money:
            return 1
        if self.my_money < self.opp_money:
            return -1
        return 0


def play_episode(
    me: Any,
    opp: Any,
    seed: int,
    my_seat: int = 0,
    config: dict[str, Any] | None = None,
) -> EpisodeResult:
    """Run exactly one seeded episode and extract per-side money + timings."""
    from kaggle_environments import make

    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    cfg["seed"] = int(seed)

    a_me, a_opp = resolve_agent(me), resolve_agent(opp)
    agents = [a_me, a_opp] if my_seat == 0 else [a_opp, a_me]

    t0 = time.perf_counter()
    env = make("kaggriculture", configuration=cfg, debug=False)
    try:
        env.run(agents)
    except Exception as e:                                    # runTimeout etc.
        return EpisodeResult(seed, my_seat, float("nan"), float("nan"),
                             "EXC", "EXC", 0, 0, 0, 0, [], OVERAGE_BANK,
                             time.perf_counter() - t0, f"{type(e).__name__}: {e}")
    wall = time.perf_counter() - t0

    opp_seat = 1 - my_seat
    statuses = [[s.status for s in st] for st in env.steps]
    err = sum(1 for r in statuses if r[my_seat] == "ERROR")
    tmo = sum(1 for r in statuses if r[my_seat] == "TIMEOUT")
    inv = sum(1 for r in statuses if r[my_seat] == "INVALID")

    durs = [lg[my_seat]["duration"] for lg in env.logs
            if len(lg) > my_seat and isinstance(lg[my_seat], dict) and "duration" in lg[my_seat]]

    last = env.steps[-1]
    farms = last[0].observation["farms"]
    return EpisodeResult(
        seed=seed,
        my_seat=my_seat,
        my_money=float(farms[my_seat]["money"]),
        opp_money=float(farms[opp_seat]["money"]),
        my_status_final=last[my_seat].status,
        opp_status_final=last[opp_seat].status,
        err_steps=err, timeout_steps=tmo, invalid_steps=inv,
        n_acts=len(durs), durations=durs,
        overage_left=float(last[my_seat].observation["remainingOverageTime"]),
        wall=wall,
    )


def pool_context(start_method: str | None, specs: Iterable[Any] = ()):
    """A multiprocessing context that will not deadlock on this workload.

    `fork` is the fast default, but a fork AFTER `torch` has been imported in the parent
    deadlocks: the child inherits torch's OpenMP/allocator locks in a locked state and
    hangs the first time it touches an init kernel.  Observed exactly here --
    `submission/tests/test_submission.py` imports torch, then a forked worker hung
    forever inside `torch.nn.init.trunc_normal_` while building the agent's model.  So
    when torch is already loaded, spawn.

    Registry agents (`reg:name`) are in-process callables and only survive `fork`, so
    they pin the method and this warns rather than silently losing them.
    """
    if start_method in (None, "auto"):
        if any(isinstance(sp, str) and sp.startswith("reg:") for sp in specs):
            start_method = "fork"
            if "torch" in sys.modules:
                print("arena: WARNING registry agents force fork, but torch is already "
                      "imported in this process -- a worker may deadlock. Prefer a "
                      "main.py path agent, or workers=1.", file=sys.stderr)
        elif "torch" in sys.modules:
            start_method = "spawn"
        else:
            start_method = "fork"
    try:
        return mp.get_context(start_method)
    except ValueError:
        return mp.get_context()


def _worker(task: tuple) -> EpisodeResult:
    me, opp, seed, seat, cfg = task
    try:
        return play_episode(me, opp, seed, seat, cfg)
    except Exception as e:                                    # never lose a worker
        return EpisodeResult(seed, seat, float("nan"), float("nan"), "EXC", "EXC",
                             0, 0, 0, 0, [], OVERAGE_BANK, 0.0, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# a match = many seeds x both seats
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    me: str
    opp: str
    n: int
    wins: int
    losses: int
    draws: int
    my_money: list[float]
    opp_money: list[float]
    per_seat: dict[int, dict[str, float]]
    err_steps: int
    timeout_steps: int
    invalid_steps: int
    exceptions: list[str]
    durations_ms: list[float]
    wall: float
    seeds: list[int] = field(default_factory=list)
    workers: int = 1
    config: dict[str, Any] = field(default_factory=dict)
    env_overrides: dict[str, str] = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list, repr=False)

    # -- aggregates -----------------------------------------------------
    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")

    @property
    def score(self) -> float:
        """win + 0.5*draw, the quantity TrueSkill actually reacts to."""
        return (self.wins + 0.5 * self.draws) / self.n if self.n else float("nan")

    @property
    def mean_money(self) -> float:
        return statistics.fmean(self.my_money) if self.my_money else float("nan")

    @property
    def median_money(self) -> float:
        return statistics.median(self.my_money) if self.my_money else float("nan")

    @property
    def mean_opp_money(self) -> float:
        return statistics.fmean(self.opp_money) if self.opp_money else float("nan")

    @property
    def median_opp_money(self) -> float:
        return statistics.median(self.opp_money) if self.opp_money else float("nan")

    @property
    def mean_margin(self) -> float:
        return statistics.fmean(a - b for a, b in zip(self.my_money, self.opp_money))

    def _q(self, p: float) -> float:
        if not self.durations_ms:
            return float("nan")
        d = sorted(self.durations_ms)
        i = min(len(d) - 1, max(0, int(round(p * (len(d) - 1)))))
        return d[i]

    @property
    def mean_step_ms(self) -> float:
        return statistics.fmean(self.durations_ms) if self.durations_ms else float("nan")

    @property
    def p50_step_ms(self) -> float:
        return self._q(0.50)

    @property
    def p95_step_ms(self) -> float:
        return self._q(0.95)

    @property
    def p99_step_ms(self) -> float:
        return self._q(0.99)

    @property
    def p999_step_ms(self) -> float:
        return self._q(0.999)

    @property
    def max_step_ms(self) -> float:
        return max(self.durations_ms) if self.durations_ms else float("nan")

    @property
    def steps_over_actTimeout(self) -> int:
        return sum(1 for d in self.durations_ms if d > ACT_TIMEOUT * 1000.0)

    @property
    def worst_overage_used(self) -> float:
        """Largest per-episode overage consumption (s). >60 == TIMEOUT."""
        if not self.episodes:
            return 0.0
        return max(e.overage_used for e in self.episodes)

    @property
    def mean_overage_used(self) -> float:
        ok = [e for e in self.episodes if e.exception is None]
        return statistics.fmean(e.overage_used for e in ok) if ok else 0.0

    @property
    def total_overage_used(self) -> float:
        """Summed across episodes -- a fleet-level view, not a per-episode limit."""
        return sum(e.overage_used for e in self.episodes if e.exception is None)

    @property
    def episodes_over_bank(self) -> int:
        """Episodes that would have been disqualified (bank exhausted)."""
        return sum(1 for e in self.episodes
                   if e.exception is None and e.overage_used >= OVERAGE_BANK)

    @property
    def mean_turn1_ms(self) -> float:
        vals = [e.turn1_ms for e in self.episodes if e.durations]
        return statistics.fmean(vals) if vals else float("nan")

    @property
    def max_turn1_ms(self) -> float:
        vals = [e.turn1_ms for e in self.episodes if e.durations]
        return max(vals) if vals else float("nan")

    @property
    def steady_max_ms(self) -> float:
        """Worst turn excluding every episode's first turn."""
        vals = [e.steady_max_ms for e in self.episodes if len(e.durations) > 1]
        return max(vals) if vals else float("nan")

    @property
    def worst_turn_episode(self) -> tuple[int, int, float]:
        """(seed, seat, ms) of the single worst steady-state turn in the match."""
        best = (-1, -1, float("-inf"))
        for e in self.episodes:
            if len(e.durations) > 1:
                v = e.steady_max_ms
                if v > best[2]:
                    best = (e.seed, e.my_seat, v)
        return best

    @property
    def healthy(self) -> bool:
        return (self.err_steps == 0 and self.timeout_steps == 0
                and self.invalid_steps == 0 and not self.exceptions
                and self.episodes_over_bank == 0)

    def summary(self) -> str:
        flag = "" if self.healthy else "   *** UNHEALTHY ***"
        lines = [
            f"{self.me}  vs  {self.opp}   n={self.n} episodes ({self.wall:.1f}s wall){flag}",
            f"  W/L/D          {self.wins}/{self.losses}/{self.draws}"
            f"   win_rate={self.win_rate:.3f}  score={self.score:.3f}",
            f"  my money       mean={self.mean_money:9.1f}  median={self.median_money:9.1f}"
            f"  min={min(self.my_money):9.1f}  max={max(self.my_money):9.1f}",
            f"  opp money      mean={self.mean_opp_money:9.1f}  median={self.median_opp_money:9.1f}",
            f"  mean margin    {self.mean_margin:+.1f}",
        ]
        for seat in sorted(self.per_seat):
            s = self.per_seat[seat]
            lines.append(f"  as p{seat}          n={int(s['n'])} win_rate={s['win_rate']:.3f} "
                         f"mean={s['mean_money']:.1f} median={s['median_money']:.1f}")
        lines.append(
            f"  agent time/step  mean={self.mean_step_ms:.3f}ms p50={self.p50_step_ms:.3f}ms "
            f"p95={self.p95_step_ms:.3f}ms p99={self.p99_step_ms:.3f}ms "
            f"p99.9={self.p999_step_ms:.3f}ms max={self.max_step_ms:.3f}ms")
        ws, wseat, wms = self.worst_turn_episode
        if self.workers > 1:
            lines.append(
                f"  TIMING IS CONTENDED: measured with {self.workers} concurrent episodes "
                f"on one box (each loads its own torch). On Kaggle every agent owns its "
                f"own 1.6-vCPU container, so these numbers are an upper bound, not the "
                f"budget. Re-measure with -j 1 for budget-grade per-turn timing.")
        lines.append(
            f"  turn 1 (load)    mean={self.mean_turn1_ms:.1f}ms max={self.max_turn1_ms:.1f}ms"
            f"   steady worst={self.steady_max_ms:.1f}ms (seed {ws} as p{wseat})")
        lines.append(
            f"  budget           actTimeout={ACT_TIMEOUT * 1000:.0f}ms  steps_over={self.steps_over_actTimeout}"
            f"  overage used: mean={self.mean_overage_used:.3f}s worst={self.worst_overage_used:.3f}s"
            f" total={self.total_overage_used:.1f}s / {OVERAGE_BANK:.0f}s per episode"
            f"  episodes_over_bank={self.episodes_over_bank}")
        if self.seeds:
            shown = self.seeds if len(self.seeds) <= 12 else self.seeds[:12]
            lines.append(f"  seeds            {shown}"
                         f"{'' if len(self.seeds) <= 12 else f' ... ({len(self.seeds)} total)'}"
                         f"   config={self.config}"
                         f"{f'   env={self.env_overrides}' if self.env_overrides else ''}")
        lines.append(
            f"  statuses         err_steps={self.err_steps} timeout_steps={self.timeout_steps} "
            f"invalid_steps={self.invalid_steps} exceptions={len(self.exceptions)}")
        for e in self.exceptions[:3]:
            lines.append(f"      ! {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items()
             if k not in ("episodes", "durations_ms", "my_money", "opp_money")}
        d.update(
            win_rate=self.win_rate, score=self.score,
            mean_money=self.mean_money, median_money=self.median_money,
            mean_opp_money=self.mean_opp_money, median_opp_money=self.median_opp_money,
            mean_margin=self.mean_margin,
            mean_step_ms=self.mean_step_ms, p50_step_ms=self.p50_step_ms,
            p95_step_ms=self.p95_step_ms, p99_step_ms=self.p99_step_ms,
            p999_step_ms=self.p999_step_ms,
            max_step_ms=self.max_step_ms,
            steps_over_actTimeout=self.steps_over_actTimeout,
            worst_overage_used=self.worst_overage_used,
            mean_overage_used=self.mean_overage_used,
            total_overage_used=self.total_overage_used,
            episodes_over_bank=self.episodes_over_bank,
            mean_turn1_ms=self.mean_turn1_ms, max_turn1_ms=self.max_turn1_ms,
            steady_max_ms=self.steady_max_ms,
            worst_turn=list(self.worst_turn_episode),
            healthy=self.healthy,
        )
        return d


def run_match(
    me: Any,
    opp: Any,
    n_seeds: int = 24,
    seeds: Sequence[int] | None = None,
    sides: str = "both",
    workers: int | None = None,
    config: dict[str, Any] | None = None,
    progress: bool = False,
    start_method: str | None = None,
) -> MatchResult:
    """Play `me` against `opp` over seeds; returns a MatchResult.

    sides: "both" (default, each seed played from both seats), "p0", "p1".
    workers: process count; 1 or 0 forces in-process execution.
    start_method: "fork" / "spawn" / "forkserver", or None to pick one that is safe for
        the agents involved (see `pool_context`).
    """
    if seeds is None:
        seeds = list(range(n_seeds))
    seats = {"both": (0, 1), "p0": (0,), "p1": (1,)}[sides]
    me_s = me if isinstance(me, str) else register_agent(getattr(me, "__name__", "me"), me)
    opp_s = opp if isinstance(opp, str) else register_agent(getattr(opp, "__name__", "opp"), opp)

    tasks = [(me_s, opp_s, int(s), seat, config) for s in seeds for seat in seats]
    if workers is None:
        workers = min(len(tasks), cpu_quota())

    t0 = time.perf_counter()
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        ctx = pool_context(start_method, (me_s, opp_s))
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks, chunksize=1)
    wall = time.perf_counter() - t0

    per_seat: dict[int, dict[str, float]] = {}
    for seat in seats:
        rs = [r for r in results if r.my_seat == seat and r.exception is None]
        if not rs:
            continue
        per_seat[seat] = {
            "n": float(len(rs)),
            "win_rate": sum(1 for r in rs if r.result > 0) / len(rs),
            "mean_money": statistics.fmean(r.my_money for r in rs),
            "median_money": statistics.median([r.my_money for r in rs]),
        }

    ok = [r for r in results if r.exception is None]
    return MatchResult(
        me=str(me_s), opp=str(opp_s), n=len(ok),
        wins=sum(1 for r in ok if r.result > 0),
        losses=sum(1 for r in ok if r.result < 0),
        draws=sum(1 for r in ok if r.result == 0),
        my_money=[r.my_money for r in ok],
        opp_money=[r.opp_money for r in ok],
        per_seat=per_seat,
        err_steps=sum(r.err_steps for r in results),
        timeout_steps=sum(r.timeout_steps for r in results),
        invalid_steps=sum(r.invalid_steps for r in results),
        exceptions=[r.exception for r in results if r.exception],
        durations_ms=[d * 1000.0 for r in ok for d in r.durations],
        wall=wall,
        seeds=[int(x) for x in seeds],
        workers=workers,
        config=dict(config or {}),
        env_overrides={k: v for k, v in sorted(os.environ.items())
                       if k.startswith("KAGGRI_AGENT")},
        episodes=results,
    )


def run_gauntlet(
    me: Any,
    opponents: Iterable[Any] = BUILTINS,
    **kw: Any,
) -> dict[str, MatchResult]:
    return {str(o): run_match(me, o, **kw) for o in opponents}


# ---------------------------------------------------------------------------
# self-test: reproduces the published baselines
# ---------------------------------------------------------------------------
def selftest(n_seeds: int = 12, workers: int | None = None) -> bool:
    print("arena.py self-test (kaggle-environments "
          f"{importlib.import_module('kaggle_environments').__version__})")
    ok = True
    # Reference values measured on kaggle-environments 1.32.7, 200 seeds x 2 seats:
    #   pass    vs pass    -> exactly 3000.0 every episode, always a draw
    #   starter vs pass    -> 3380..3823 (median 3520), starter always wins
    #   random  vs pass    -> median 0.0, mean < 100 (random bankrupts itself);
    #                         it is NOT always exactly 0 -- it can end with a few
    #                         hundred if it happens to harvest and sell.
    #   starter vs starter -> always a draw (both sides deterministic + symmetric)
    checks = [
        ("pass", "pass", lambda m: m.mean_money == 3000.0 and m.draws == m.n),
        ("starter", "pass", lambda m: 3300 <= m.median_money <= 3700 and m.wins == m.n),
        ("random", "pass", lambda m: m.median_money == 0.0 and m.mean_money < 300
         and m.losses == m.n),
        ("starter", "starter", lambda m: m.draws == m.n),
    ]
    for me, opp, pred in checks:
        r = run_match(me, opp, n_seeds=n_seeds, workers=workers)
        good = pred(r) and r.healthy
        ok = ok and good
        print(f"[{'ok ' if good else 'FAIL'}] {r.me:8s} vs {r.opp:8s} "
              f"money mean={r.mean_money:8.1f} median={r.median_money:8.1f} "
              f"min={min(r.my_money):8.1f} max={max(r.my_money):8.1f} "
              f"W/L/D={r.wins}/{r.losses}/{r.draws} "
              f"mean_step={r.mean_step_ms:.4f}ms max_step={r.max_step_ms:.3f}ms")
    # determinism check
    a = run_match("starter", "pass", seeds=[7], sides="p0", workers=1)
    b = run_match("starter", "pass", seeds=[7], sides="p0", workers=1)
    det = a.my_money == b.my_money
    ok = ok and det
    print(f"[{'ok ' if det else 'FAIL'}] determinism: seed 7 replays to {a.my_money} twice")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", "-a", default="starter",
                   help="agent spec under test (builtin name, path/to/main.py, or mod:func)")
    p.add_argument("--opponents", "-o", default="pass,starter,random",
                   help="comma-separated opponent specs")
    p.add_argument("-n", "--n-seeds", type=int, default=24)
    p.add_argument("--seeds", default=None, help="explicit comma-separated seed list")
    p.add_argument("--sides", choices=("both", "p0", "p1"), default="both")
    p.add_argument("-j", "--workers", type=int, default=None)
    p.add_argument("--steps", type=int, default=720, help="episodeSteps")
    p.add_argument("--config", default=None, help="extra configuration as JSON")
    p.add_argument("--json", default=None, help="write results to this JSON file")
    p.add_argument("--tier", default=None,
                   help="KAGGRI_AGENT_TIER for submission/main.py agents "
                        "(neural|neural_small|macro|plain|pass). Measurement only: it "
                        "pins the tier and DISABLES the per-turn time guard. Applies to "
                        "every main.py in the match, both seats.")
    p.add_argument("--threads", type=int, default=None,
                   help="KAGGRI_AGENT_THREADS: torch threads inside the agent. Use 1 to "
                        "reproduce the 1.6-vCPU submission sandbox.")
    p.add_argument("--seed0", type=int, default=0, help="first seed when --seeds is absent")
    p.add_argument("--start-method", default=None, choices=("fork", "spawn", "forkserver"),
                   help="multiprocessing start method; default picks a safe one "
                        "(spawn when torch is already imported -- fork deadlocks)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    # Set before any fork so every worker inherits it.
    if args.tier:
        os.environ["KAGGRI_AGENT_TIER"] = args.tier
    if args.threads:
        os.environ["KAGGRI_AGENT_THREADS"] = str(args.threads)

    if args.selftest:
        return 0 if selftest(n_seeds=max(4, args.n_seeds), workers=args.workers) else 1

    cfg = {"episodeSteps": args.steps}
    if args.config:
        cfg.update(json.loads(args.config))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(range(args.seed0, args.seed0 + args.n_seeds)))

    out = {"_run": {"seeds": seeds, "sides": args.sides, "config": cfg,
                    "workers": args.workers if args.workers else cpu_quota(),
                    "agent": args.agent,
                    "env": {k: v for k, v in sorted(os.environ.items())
                            if k.startswith("KAGGRI_AGENT")}}}
    rc = 0
    print(f"arena: agent={args.agent}  seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  "
          f"sides={args.sides}  config={cfg}  workers="
          f"{args.workers if args.workers else cpu_quota()}"
          + (f"  env={out['_run']['env']}" if out["_run"]["env"] else ""))
    for opp in [o.strip() for o in args.opponents.split(",") if o.strip()]:
        r = run_match(args.agent, opp, seeds=seeds, sides=args.sides,
                      workers=args.workers, config=cfg,
                      start_method=args.start_method)
        print(r.summary())
        print()
        out[opp] = r.to_dict()
        if not r.healthy:
            rc = 2
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", args.json)
    return rc


if __name__ == "__main__":
    sys.exit(main())
