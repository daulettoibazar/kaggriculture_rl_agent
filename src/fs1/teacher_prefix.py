"""Program-taught reverse curriculum: public programs play the opening of some episodes.

Iteration 1 showed that milestone bonuses alone make the agent TRY land, animals and fertilizer but
not keep them; a hand-written full-recipe teacher (2026-09-23) reached only $8-13K per game because
of unit-routing waste, below the agent's own $50K. The public programs already play the whole
recipe (3 quadrants, 9-11 hands, 14-17 animals, fertilizer, ~$180K unopposed), so on a block of
self-play envs BOTH seats follow a randomly drawn program for a random prefix of at most
`max_prefix` steps, then the network takes over the farm the program built. The prefix shrinks to
zero over training (same schedule shape as the scripted opener) so the agent first learns to RUN a
full farm and later to build one. Prefix transitions do not train.

Mechanics as in `public_opponents.py`: Kaggle-format observations from the exact simulator's
state, turns through `PyVecEnv.step(actions, exact_actions)`; one program instance per (env, seat),
reset per episode; deadline + counted fallbacks. Training-time only: the deployed agent never calls
a program.
"""
from __future__ import annotations

import time

import numpy as np

import torch

from src.evo import kernel_opponents as K
from src.fs1.imitation_labels import LabelBuffer
from src.rl3 import opponents as O


class TeacherPrefix:
    def __init__(self, names, env_indices, timeout: float = 1.0, seed: int = 0, budget_s: float = 8.0):
        self.names = [nm for nm in names if nm]
        self.envs = np.asarray(sorted(int(e) for e in env_indices), dtype=np.int64)
        assert self.names and len(self.envs) > 0
        self.timeout = float(timeout)
        self.rng = np.random.default_rng(seed)
        self.paths = {nm: O.path_of(nm) for nm in self.names}
        self.struct = {nm: O.struct_of(nm) for nm in self.names}
        self.prefix = np.zeros(len(self.envs), dtype=np.int64)        # steps the program plays this episode
        self.program = np.full(len(self.envs), -1, dtype=np.int64)
        self.policies = [[None, None] for _ in self.envs]
        self.episodes = 0
        self.fallbacks = 0.0
        self.budget_s = float(budget_s)       # wall-clock budget per `fill`: a stall guard, not a per-turn deadline
        self.budget_hits = 0
        self._rot = 0                         # rotating start so a budget hit never starves the same envs
        self.labels = [[LabelBuffer(), LabelBuffer()] for _ in self.envs]   # imitation labels per (env, seat)

    # ------------------------------------------------------------------ episodes
    def _start(self, i: int, max_prefix: int):
        k = int(self.rng.integers(0, max_prefix + 1)) if max_prefix > 0 else 0
        self.prefix[i] = k
        if k <= 0:
            return
        p = int(self.rng.integers(len(self.names)))
        path = self.paths[self.names[p]]
        for seat in range(2):
            pol = self.policies[i][seat]
            if pol is None or self.program[i] != p:
                self.policies[i][seat] = K.KernelPolicy(path, timeout=self.timeout)
            else:
                pol.reset()
                pol.stats = K.PolicyStats()   # reset() keeps the counters; fallbacks are summed per episode
        self.program[i] = p

    def start_all(self, vec_env, max_prefix: int):
        """At construction: envs already mid-game (staggered starts) get no prefix this episode."""
        steps = np.asarray(vec_env.env_steps())
        for i, e in enumerate(self.envs):
            if steps[e] > 0:
                self.prefix[i] = 0
            else:
                self._start(i, max_prefix)

    def prefix_len(self, n_total: int) -> np.ndarray:
        """Per-env prefix length [n_total] (0 outside the teacher block)."""
        out = np.zeros(n_total, dtype=np.int64)
        out[self.envs] = self.prefix
        return out

    def new_rollout(self):
        """Labels are indexed by rollout step; unfinished walks from the previous rollout are dropped."""
        for pair in self.labels:
            pair[0].reset()
            pair[1].reset()

    def fill(self, rows: list, vec_env, env_steps: np.ndarray, t: int = 0) -> int:
        """Write both seats' program turns into `rows` for teacher envs still inside their prefix and
        record them for imitation labels at rollout step `t`. Returns the number of envs driven."""
        driven = 0
        t0 = time.time()
        n_env = len(self.envs)
        self._rot = (self._rot + 1) % max(1, n_env)
        for k in range(n_env):
            i = (self._rot + k) % n_env
            e = self.envs[i]
            if env_steps[e] >= self.prefix[i]:
                continue
            if time.time() - t0 > self.budget_s:      # leave the rest to the network for this one step
                self.budget_hits += 1
                break
            nm = self.names[self.program[i]]
            state = vec_env.state_dict(int(e))
            turns = [self.policies[i][s].act(K.observation_of(state, s, struct=self.struct[nm])) for s in range(2)]
            rows[e] = turns
            for s in range(2):
                self.labels[i][s].observe(t, state, s, turns[s])
            driven += 1
        return driven

    def drain_labels(self):
        """-> list of (t, env, seat, row55 int64 tensor, unit_label bool[17], market_label bool[21])."""
        out = []
        for i, e in enumerate(self.envs):
            for s in range(2):
                for (t, row, ulab, mlab) in self.labels[i][s].drain():
                    out.append((t, int(e), s, torch.from_numpy(row), torch.from_numpy(ulab), torch.from_numpy(mlab)))
        return out

    def on_done(self, done: np.ndarray, max_prefix: int):
        for i, e in enumerate(self.envs):
            if bool(done[e]):
                if self.prefix[i] > 0:
                    self.episodes += 1
                    self.fallbacks += sum(p.stats.fallbacks for p in self.policies[i] if p is not None)
                self._start(i, max_prefix)

    def summary(self) -> dict:
        return {"teacher_episodes": self.episodes, "teacher_fallbacks": self.fallbacks,
                "teacher_budget_hits": self.budget_hits, "teacher_prefix_mean": float(self.prefix.mean())}
