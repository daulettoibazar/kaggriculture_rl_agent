"""Closed-loop public-kernel opponents for the exact simulator.

A recorded opponent replays a fixed token tape open-loop and cannot answer the learner.  This
module runs public Kaggle notebooks **as agents**, in the loop, inside the exact Rust simulator,
so the opponent reacts to what the learner actually does.

How it works: ``kaggri.PyGame.state_dict()`` carries exactly the fields a Kaggle observation
carries.  ``observation_of`` rebuilds the harness observation from it, and the opponent seat is
driven by the program's own ``agent(obs, config)``.

Isolation is a guard against accidents, not a security sandbox: a Python agent loaded in-process
can always defeat it.  It blocks the imports that would be slow or networked (torch, tensorflow,
requests, sockets) and the process-spawning entry points, records every block, and gives each
agent its own namespace with its own directory on ``sys.path`` so a kernel that ships a data
file next to itself still finds it.  Every agent call is wrapped in a wall-clock timeout and an
exception guard that falls back to an all-PASS turn and **counts** the fallback; a policy whose
``fallbacks`` is non-zero is reported and must not be treated as a faithful opponent.
"""
from __future__ import annotations

import os

for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib
import io
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PASS_TURN = {"farmer": ["PASS"], "hands": [], "market": []}
STEPS_PER_GAME = 720
DEFAULT_CFG = {"episodeSteps": 720}


# --------------------------------------------------------------------------- isolation guards

BLOCKED_ROOTS = frozenset({
    "torch", "torchvision", "tensorflow", "keras", "jax", "flax", "transformers",
    "requests", "urllib3", "httpx", "aiohttp", "socket", "socketserver", "ftplib",
    "telnetlib", "smtplib", "paramiko", "boto3", "botocore", "kagglehub", "kaggle",
})


class _BlockImports:
    """A ``sys.meta_path`` finder that refuses a blocklisted top-level package."""

    def __init__(self, log: list[str]):
        self.log = log

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED_ROOTS:
            self.log.append(fullname)
            raise ImportError(f"kernel_opponents: import of {fullname!r} is blocked")
        return None


@contextlib.contextmanager
def _quiet():
    """Swallow agent stdout/stderr the way ``kaggle_environments.Agent.act`` does."""
    with io.StringIO() as out, io.StringIO() as err, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


def _blocked_spawn(*a, **k):
    raise PermissionError("kernel_opponents: the agent tried to spawn a process")


class _Guards:
    """Blocks heavy/networked imports and process spawning for the duration of the block."""

    _PATCH = (
        (subprocess, "Popen"), (subprocess, "run"), (subprocess, "call"),
        (subprocess, "check_call"), (subprocess, "check_output"),
        (os, "system"), (os, "popen"), (os, "fork"), (os, "execv"), (os, "execvp"),
        (os, "spawnv"), (os, "spawnvp"),
    )

    def __init__(self, log: list[str]):
        self.finder = _BlockImports(log)
        self.saved: list[tuple[Any, str, Any]] = []

    def __enter__(self):
        sys.meta_path.insert(0, self.finder)
        for obj, name in self._PATCH:
            if hasattr(obj, name):
                self.saved.append((obj, name, getattr(obj, name)))
                setattr(obj, name, _blocked_spawn)
        return self

    def __exit__(self, *exc):
        for obj, name, old in self.saved:
            setattr(obj, name, old)
        self.saved.clear()
        try:
            sys.meta_path.remove(self.finder)
        except ValueError:
            pass
        return False


class _Deadline:
    """Wall-clock timeout for one agent call. Only arms in a process's main thread."""

    __slots__ = ("seconds", "armed", "fired")

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.armed = False
        self.fired = False

    @staticmethod
    def available() -> bool:
        return threading.current_thread() is threading.main_thread()

    def __enter__(self):
        if self.seconds and self.available():
            signal.signal(signal.SIGALRM, self._fire)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            self.armed = True
        return self

    def _fire(self, signum, frame):
        self.fired = True
        raise TimeoutError("kernel_opponents: agent call exceeded its deadline")

    def __exit__(self, *exc):
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        return False


# --------------------------------------------------------------------------- loading an agent

def load_agent(path: str | Path, blocks: list[str] | None = None) -> Callable:
    """Load a single-file agent under the guards, by the rule Kaggle itself uses.

    ``kaggle_environments.agent.get_last_callable`` picks the LAST module-level callable, which
    is what ``tools/arena.py`` and the live harness both do, so a kernel that defines helpers
    after its ``agent`` is selected identically here.  The file's own directory goes on
    ``sys.path`` and stays there, because kernels load data files lazily on their first turn.
    """
    from kaggle_environments.agent import get_last_callable

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    here = str(path.parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    cwd = os.getcwd()
    log = blocks if blocks is not None else []
    try:
        os.chdir(here)
        with _quiet(), _Guards(log):
            fn = get_last_callable(path.read_text(encoding="utf-8"), path=str(path))
    finally:
        os.chdir(cwd)
    if not callable(fn):
        raise TypeError(f"{path}: no module-level callable found")
    return fn


# --------------------------------------------------------------------------- observations

def observation_of(state: dict, player: int, struct: bool = True,
                   overage: float = 60.0) -> Any:
    """The Kaggle observation for ``player``, rebuilt from ``PyGame.state_dict()``.

    ``state_dict`` carries one field the harness does not, ``privates[p].inventories_ordered``
    (a debugging aid); everything else is identical, which :func:`check_observation`
    asserts step by step against a real ``kaggle_environments`` episode.  ``struct=True``
    returns the ``Struct`` the harness hands an agent, so ``obs.farms[0].money`` works as well
    as ``obs["farms"][0]["money"]``; it costs about 49 us, so our own tape side asks for the
    plain dict, which it reads only by key.
    """
    private = dict(state["privates"][player])
    private.pop("inventories_ordered", None)
    obs = {
        "player": player,
        "farms": state["farms"],
        "private": private,
        "market": state["market"],
        "town": state["town"],
        "day": state["day"],
        "hour": state["hour"],
        "step": state["step"],
        "remainingOverageTime": overage,
    }
    if not struct:
        return obs
    from kaggle_environments.utils import structify
    return structify(obs)


_FULL_CFG: dict[str, Any] = {}


def full_configuration(cfg: dict | None = None) -> Any:
    """The ``Struct`` configuration the harness hands an agent, cached per requested override.

    ``kaggle_environments`` passes the WHOLE resolved configuration (episodeSteps, boardSize,
    marketParams, the timeouts, ``__raw_path__``), not the two keys we care about, and several
    kernels read ``config.episodeSteps`` or ``config.turnsPerDay``. Built once per process by
    asking the real environment, so nothing here can drift from it.
    """
    from kaggle_environments import make
    from kaggle_environments.utils import structify

    key = json.dumps(cfg or DEFAULT_CFG, sort_keys=True)
    if key not in _FULL_CFG:
        env = make("kaggriculture", configuration=dict(cfg or DEFAULT_CFG), debug=False)
        _FULL_CFG[key] = structify(dict(env.configuration))
    return _FULL_CFG[key]



def _sane_turn(action: Any) -> dict | None:
    """A turn the simulator will accept, or None so the caller can count a fallback."""
    if not isinstance(action, dict):
        return None
    farmer = action.get("farmer")
    if not isinstance(farmer, (list, tuple)) or not farmer:
        return None
    hands = action.get("hands") or []
    market = action.get("market") or []
    if not isinstance(hands, (list, tuple)) or not isinstance(market, (list, tuple)):
        return None
    return {"farmer": list(farmer), "hands": [list(h) for h in hands],
            "market": [list(m) for m in market]}


# --------------------------------------------------------------------------- the policy

@dataclass
class PolicyStats:
    calls: int = 0
    exceptions: int = 0
    timeouts: int = 0
    malformed: int = 0
    blocked_imports: list[str] = field(default_factory=list)
    total_s: float = 0.0
    max_s: float = 0.0
    first_error: str = ""

    @property
    def fallbacks(self) -> int:
        return self.exceptions + self.timeouts + self.malformed

    def as_dict(self) -> dict:
        return {"calls": self.calls, "exceptions": self.exceptions, "timeouts": self.timeouts,
                "malformed": self.malformed, "fallbacks": self.fallbacks,
                "blocked_imports": sorted(set(self.blocked_imports)),
                "mean_ms": round(1e3 * self.total_s / self.calls, 4) if self.calls else 0.0,
                "max_ms": round(1e3 * self.max_s, 3), "first_error": self.first_error[:400]}


class KernelPolicy:
    """One loaded agent, called once per turn for one seat, with a deadline and a PASS fallback.

    The agent object is stateful (kernels keep route locks and day counters in module globals),
    so a policy belongs to exactly one game: call :meth:`reset` -- which reloads the file -- or
    build a new policy per episode.  Fallbacks are counted, never silent.
    """

    def __init__(self, path: str | Path, timeout: float | None = 0.5,
                 config: dict | None = None, label: str = ""):
        self.path = Path(path)
        self.timeout = timeout
        self.config = full_configuration(config)
        self.label = label or self.path.parent.parent.name
        self.stats = PolicyStats()
        self._blocks: list[str] = []
        self.agent = load_agent(self.path, self._blocks)
        self._argc = self._argcount(self.agent)
        self.stats.blocked_imports = list(self._blocks)

    @staticmethod
    def _argcount(fn) -> int:
        """Kaggle truncates ``[observation, configuration]`` to the agent's positional arity."""
        code = getattr(fn, "__code__", None)
        return min(2, getattr(code, "co_argcount", 2)) if code is not None else 2

    def reset(self) -> "KernelPolicy":
        """Reload the file so module-level state from the previous episode cannot leak."""
        self._blocks = []
        self.agent = load_agent(self.path, self._blocks)
        self._argc = self._argcount(self.agent)
        return self

    def act(self, obs: Any) -> dict:
        t0 = time.perf_counter()
        out = None
        failed = False
        deadline = _Deadline(self.timeout)
        try:
            args = (obs, self.config)[: self._argc]
            with _Guards(self._blocks), deadline:
                out = self.agent(*args)
            # An opponent's broad exception handler can swallow SIGALRM's
            # TimeoutError. That is still a deadline failure, not a valid move.
            if deadline.fired:
                failed = True
                out = None
                self.stats.timeouts += 1
                if not self.stats.first_error:
                    self.stats.first_error = "TimeoutError caught internally by agent"
        except TimeoutError:
            failed = True
            self.stats.timeouts += 1
            if not self.stats.first_error:
                self.stats.first_error = "TimeoutError"
        except BaseException:
            failed = True
            self.stats.exceptions += 1
            if not self.stats.first_error:
                self.stats.first_error = traceback.format_exc()
        dt = time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.total_s += dt
        self.stats.max_s = max(self.stats.max_s, dt)
        turn = _sane_turn(out)
        if turn is None:
            # Returning None is also malformed; do not silently pass without
            # recording a fallback. Exceptions/timeouts are already counted.
            if not failed:
                self.stats.malformed += 1
            return dict(PASS_TURN)
        return turn


# --------------------------------------------------------------------------- one game



def cluster_se(values: Sequence[float], seeds: Sequence[int]) -> float:
    """Standard error of the mean with observations clustered by environment seed."""
    v = np.asarray(values, dtype=np.float64)
    s = np.asarray(seeds)
    if v.size == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    v, s = v[order], s[order]
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    sums = np.add.reduceat(v, edges[:-1])
    sizes = np.diff(edges)
    m = v.mean()
    k = len(sizes)
    if k < 2:
        return float("nan")
    resid = sums - sizes * m
    return float(np.sqrt(resid @ resid * k / (k - 1)) / v.size)

