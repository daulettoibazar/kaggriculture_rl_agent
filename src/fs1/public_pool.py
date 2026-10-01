"""Parallel drop-in for `PublicOpponents`: public programs on seat 1, in a process pool.

`PublicOpponents` calls every program in the trainer's main thread (~1.7 ms per program turn), so
a 50% program share of 2048 envs costs ~1.7 s per step. Here each spawned worker keeps a SHADOW
`kaggri.PyGame` for each of its public envs and the program that plays seat 1 in it:

* the shadow starts from the env's base seed and replays the same all-PASS warmup the vec env did
  (`warmup_pass` offsets = the env's step count at the first call), so both hold the same state;
* per step the parent sends only the learner's seat-0 Kaggle dict for each public env
  (`PyVecEnv.resolve_actions`, the resolver's own turn for the packed row, ~1-5 us) and receives
  the program's turn for the NEXT step, which the worker computes while the parent runs the env
  step and the next GPU forward;
* an episode end restarts the shadow on the vec env's derived seed (`derive_seed`, the same
  splitmix64 as `rust/crates/kaggri-py/src/lib.rs`), with a fresh program instance;
* every step the worker returns its shadow money, and the parent asserts it equals the vec env's
  money for those envs, so any drift is a hard error, not a silent bias.

The interface (`exact_actions`, `on_done`, `totals`, `summary`) matches `PublicOpponents` except
that `exact_actions` also takes the packed action rows about to be stepped. As there, a staggered
env's first (junk) episode is driven but not scored, and fallbacks are counted per program.

Workers inherit PYTHONHASHSEED from the parent environment (pinned to 0 here unless set): some
programs iterate string sets, so outcomes depend on it.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import torch

from . import constants as C

M64 = (1 << 64) - 1
PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def derive_seed(base: int, episode: int) -> int:
    """`derive_seed` of the vec env: episode 0 is the base seed, later ones splitmix64 (31 bits)."""
    if episode == 0:
        return int(base)
    z = ((int(base) & M64) * 0x9E3779B97F4A7C15 + (int(episode) & M64) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    z ^= z >> 31
    return z & 0x7FFFFFFF


class _Shard:
    """Worker side: shadow games + seat-1 programs for a set of public envs."""

    def __init__(self, envs, bases, offsets, names, episode_steps, timeout, seed):
        import kaggri
        from src.evo import kernel_opponents as K
        from src.rl3 import opponents as O
        self.kaggri, self.K, self.O = kaggri, K, O
        self.envs = list(envs)
        self.bases = [int(b) for b in bases]
        self.names = list(names)
        self.cfg = {"episodeSteps": int(episode_steps)}
        self.timeout = float(timeout)
        self.rng = np.random.default_rng(seed)
        self.episode = [0] * len(self.envs)
        self.games = [None] * len(self.envs)
        self.pols = [None] * len(self.envs)
        self.assigned = [0] * len(self.envs)
        self.finished = []                              # (env, program index, fallbacks) of ended episodes
        for j, off in enumerate(offsets):
            self._new_episode(j, warm=int(off))

    def _new_episode(self, j, warm=0):
        g = self.kaggri.PyGame({**self.cfg, "seed": derive_seed(self.bases[j], self.episode[j])})
        g.reset()
        for _ in range(warm):
            g.step(dict(PASS), dict(PASS))
        self.games[j] = g
        k = int(self.rng.integers(len(self.names)))
        self.assigned[j] = k
        self.pols[j] = self.K.KernelPolicy(self.O.path_of(self.names[k]), timeout=self.timeout)

    def turns(self):
        out = []
        for j, g in enumerate(self.games):
            st = g.state_dict()
            nm = self.names[self.assigned[j]]
            p = self.pols[j]
            steady = p.timeout
            if p.stats.calls == 0:
                p.timeout = max(steady, 30.0)           # cold import / JIT on the first call
            try:
                out.append(p.act(self.K.observation_of(st, 1, struct=self.O.struct_of(nm))))
            finally:
                p.timeout = steady
        return out

    def advance(self, seat0, turns):
        money = np.zeros((len(self.games), 2), np.float64)
        for j, g in enumerate(self.games):
            g.step(seat0[j], turns[j])
            money[j] = g.money
            if g.done:
                self.finished.append((self.envs[j], self.assigned[j], int(self.pols[j].stats.fallbacks)))
                self.episode[j] += 1
                self._new_episode(j)
        return money


def _worker(conn, cwd):
    # `scancel --signal=USR1` reaches every process of the step: a worker must not die on the
    # trainer's checkpoint signal (an early run's USR1 killed the workers and the ranks crashed on EOF)
    import signal
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    os.makedirs(cwd, exist_ok=True)
    os.chdir(cwd)
    shard, pending = None, None
    try:
        while True:
            op, arg = conn.recv()
            if op == "close":
                return
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                if op == "init":
                    shard = _Shard(**arg)
                    pending = shard.turns()
                    ans = {"turns": pending}
                elif op == "step":
                    t0 = time.perf_counter()
                    money = shard.advance(arg, pending)
                    finished, shard.finished = shard.finished, []
                    pending = shard.turns()
                    ans = {"turns": pending, "money": money, "finished": finished, "s": time.perf_counter() - t0}
                else:
                    raise ValueError(op)
            conn.send((True, ans))
    except (EOFError, ConnectionResetError, BrokenPipeError):
        pass                                          # the parent went away: exit quietly
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except OSError:
            pass
    finally:
        conn.close()


class ParallelPublicOpponents:
    def __init__(self, names, n_public: int, first_env: int, timeout: float = 1.0, seed: int = 0,
                 ema: float = 0.05, workers: int = 8, cwd_root: str = ".", episode_steps: int = C.EPISODE_STEPS,
                 deadline_s: float = 600.0):
        self.names = [nm for nm in names if nm]
        assert self.names and n_public > 0 and workers >= 1
        self.first, self.n = int(first_env), int(n_public)
        self.timeout, self.ema, self.seed = float(timeout), float(ema), int(seed)
        self.episode_steps = int(episode_steps)
        self.deadline = float(deadline_s)
        self.games = np.zeros(len(self.names))
        self.wins = np.zeros(len(self.names))
        self.fallbacks = np.zeros(len(self.names))
        self.win_ema = np.full(len(self.names), 0.5)
        self.program_s = 0.0
        os.environ.setdefault("PYTHONHASHSEED", "0")
        ctx = mp.get_context("spawn")
        k = min(int(workers), self.n)
        self.groups = [g for g in np.array_split(np.arange(self.n), k)]
        self.conns, self.procs = [], []
        for w in range(k):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, os.path.join(cwd_root, f"pub{w:02d}")), daemon=True)
            p.start()
            b.close()
            self.conns.append(a)
            self.procs.append(p)
        import atexit
        atexit.register(self.close)
        self.started = False
        self.turns = None                     # [n] program turn for the current step
        self.warm = np.zeros(self.n, bool)    # env's current episode began mid-game (staggered start)
        self.pending_done = {}                # env -> (program index, fallbacks) awaiting its final money

    def _recv(self, c):
        if not c.poll(self.deadline):
            raise TimeoutError("public-program worker deadline")
        ok, res = c.recv()
        if not ok:
            raise RuntimeError(f"public-program worker failed:\n{res}")
        return res

    def _start(self, vec_env):
        steps = np.asarray(vec_env.env_steps())
        bases = list(vec_env.seeds)
        eps = list(vec_env.episode_indices)
        envs = np.arange(self.first, self.first + self.n)
        assert all(int(eps[e]) == 0 for e in envs), "start the pool before the first auto-reset"
        self.warm = steps[envs] > 0
        for w, (c, g) in enumerate(zip(self.conns, self.groups)):
            c.send(("init", dict(envs=[int(e) for e in envs[g]], bases=[int(bases[e]) for e in envs[g]],
                                 offsets=[int(steps[e]) for e in envs[g]], names=self.names,
                                 episode_steps=self.episode_steps, timeout=self.timeout,
                                 seed=self.seed * 1009 + w)))
        self.turns = [None] * self.n
        for c, g in zip(self.conns, self.groups):
            for i, t in zip(g, self._recv(c)["turns"]):
                self.turns[i] = t
        self.started = True

    def exact_actions(self, vec_env, n_total: int, actions=None) -> list:
        """`actions` [n_total, 2, 55] (numpy or tensor): the packed rows about to be stepped. Sends
        every public env's seat-0 turn to its shadow; returns the `[seat0, seat1]` rows for step()."""
        assert actions is not None, "ParallelPublicOpponents needs the packed actions"
        if getattr(self, "_sent", False):
            assert getattr(self, "_outcome", None) is not None, "on_done must follow every step"
            self.flush()
        if not self.started:
            self._start(vec_env)
        a = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        a = a.reshape(n_total, 2, -1).astype(np.int64)
        rows = [[None, None] for _ in range(n_total)]
        seat0 = [None] * self.n
        for i in range(self.n):
            e = self.first + i
            seat0[i] = vec_env.resolve_actions(e, a[e])[0]
            rows[e] = [seat0[i], self.turns[i]]
        for c, g in zip(self.conns, self.groups):
            c.send(("step", [seat0[i] for i in g]))
        self._sent = True
        return rows

    def on_done(self, done: np.ndarray, final_money: np.ndarray, vec_env=None):
        """After EVERY `env.step` (not only when something finished): records this step's outcome.
        The workers' replies (next turns, shadow money, finished games) are collected lazily at the
        next `exact_actions` (or `flush`), so the programs run while the trainer does its GPU work."""
        money_now = None if vec_env is None else np.asarray(vec_env.env_money()).copy()
        self._outcome = (np.asarray(done).astype(bool).copy(), np.asarray(final_money, np.float64).copy(), money_now)

    def flush(self):
        """Collect the replies for the last stepped turn (idempotent)."""
        if getattr(self, "_outcome", None) is None:
            return
        done, final_money, money_now = self._outcome
        self._outcome = None
        t0 = time.perf_counter()
        for c, g in zip(self.conns, self.groups):
            res = self._recv(c)
            for i, t in zip(g, res["turns"]):
                self.turns[i] = t
            for j, i in enumerate(g):
                e = self.first + i
                want = final_money[e] if bool(done[e]) else (None if money_now is None else money_now[e])
                if want is not None and not np.allclose(res["money"][j], want, rtol=0, atol=1e-3):
                    raise AssertionError(f"shadow drift on env {e}: shadow {res['money'][j].tolist()} "
                                         f"vs vec env {np.asarray(want).tolist()}")
            for (e, k, fb) in res["finished"]:
                i = e - self.first
                if self.warm[i]:
                    self.warm[i] = False
                    continue
                score = 0.5 * (np.sign(float(final_money[e, 0]) - float(final_money[e, 1])) + 1.0)
                self.games[k] += 1
                self.wins[k] += score
                self.fallbacks[k] += fb
                self.win_ema[k] = (1 - self.ema) * self.win_ema[k] + self.ema * score
            self.program_s = max(self.program_s, res["s"])
        self.wait_s = time.perf_counter() - t0
        self._sent = False

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.flush()
        except Exception:
            pass
        for c, p in zip(self.conns, self.procs):
            try:
                c.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
            c.close()

    # --------------------------------------------------------------------- stats
    def totals(self, device) -> torch.Tensor:
        """Tallies up to the last COLLECTED step (the newest step's endings land at the next call)."""
        return torch.tensor(np.stack([self.games, self.wins, self.fallbacks]), device=device, dtype=torch.float64)

    def summary(self, totals: torch.Tensor | None = None) -> dict:
        t = (self.totals("cpu") if totals is None else totals.cpu()).numpy()
        games, wins, fb = t[0], t[1], t[2]
        return {
            "win_vs_public": {nm: (float(wins[k] / games[k]) if games[k] > 0 else float("nan"))
                              for k, nm in enumerate(self.names)},
            "win_vs_public_ema": {nm: float(self.win_ema[k]) for k, nm in enumerate(self.names)},
            "public_games": {nm: float(games[k]) for k, nm in enumerate(self.names)},
            "public_fallbacks": {nm: float(fb[k]) for k, nm in enumerate(self.names)},
        }
