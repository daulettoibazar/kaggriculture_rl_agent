"""Public Kaggle programs as frozen seat-1 opponents inside fs1 self-play training.

Self-play alone learns to beat itself; the leaderboard is a Bradley-Terry rating against a field
that is ~90% copies of a handful of public notebooks (2026-09-23 measurement: the update-1674
checkpoint won 0 of 96 games against three of them while earning a third of their money). A block
of league envs therefore plays seat 1 with a real public program, driven exactly as the panel
tool does (`tools/fs1/panel_eval.py`): the Kaggle-format observation is rebuilt from the exact
simulator's state and the program's turn dictionary bypasses the finite action vocabulary through
`PyVecEnv.step(actions, exact_actions)`. Seat 0 (the learner) trains on these games like on any
league game; seat 1 never trains.

Cost: a public program costs ~3-4 ms per turn in Python against ~10 us for the simulator, so the
block is small (tens of envs per rank) and every program call runs in the main thread under a
wall-clock deadline with an all-PASS fallback that is COUNTED (a program with fallbacks is not a
faithful opponent). Programs write files into the working directory, so the trainer chdirs into
<run-dir>/agent_cwd/rank<r> before creating this pool. One program instance per env, re-created
at every episode start (the programs keep hidden state across turns).

Per-program win rates (learner = seat 0) are the in-training panel: cumulative totals are reduced
across ranks by the trainer; the EMA is rank-local.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evo import kernel_opponents as K
from src.rl3 import opponents as O


class PublicOpponents:
    def __init__(self, names, n_public: int, first_env: int, timeout: float = 1.0, seed: int = 0,
                 ema: float = 0.05):
        self.names = [nm for nm in names if nm]
        assert self.names and n_public > 0, (names, n_public)
        self.first, self.n = int(first_env), int(n_public)
        self.timeout, self.ema = float(timeout), float(ema)
        self.rng = np.random.default_rng(seed)
        self.paths = {nm: O.path_of(nm) for nm in self.names}
        self.struct = {nm: O.struct_of(nm) for nm in self.names}
        self.games = np.zeros(len(self.names), dtype=np.float64)
        self.wins = np.zeros(len(self.names), dtype=np.float64)
        self.fallbacks = np.zeros(len(self.names), dtype=np.float64)
        self.win_ema = np.full(len(self.names), 0.5)
        self.policies: list = [None] * self.n
        self.assigned = np.zeros(self.n, dtype=np.int64)
        self._warm = None                 # envs whose first episode began mid-game (staggered starts)
        for i in range(self.n):
            self._assign(i)

    # ------------------------------------------------------------------ episodes
    def _assign(self, i: int):
        k = int(self.rng.integers(len(self.names)))
        self.assigned[i] = k
        self.policies[i] = K.KernelPolicy(self.paths[self.names[k]], timeout=self.timeout)   # a fresh load per episode

    def exact_actions(self, vec_env, n_total: int, actions=None) -> list:
        """One `[seat0, seat1]` row per env for `PyVecEnv.step`: `None` keeps the packed action,
        a dict is the program's turn. Only seat 1 of the public block is ever filled."""
        if self._warm is None:
            # staggered envs hand the program a game already in progress: it plays that first,
            # partial episode from a cold start, so that episode is driven but never scored
            steps = np.asarray(vec_env.env_steps())
            self._warm = steps[self.first:self.first + self.n] > 0
        rows = [[None, None] for _ in range(n_total)]
        for i in range(self.n):
            e = self.first + i
            nm = self.names[self.assigned[i]]
            state = vec_env.state_dict(e)
            obs = K.observation_of(state, 1, struct=self.struct[nm])
            rows[e] = [None, self.policies[i].act(obs)]
        return rows

    def on_done(self, done: np.ndarray, final_money: np.ndarray):
        """After `env.step`: record finished games (learner seat 0 vs the program) and start the
        next episode's program instance."""
        for i in range(self.n):
            e = self.first + i
            if not bool(done[e]):
                continue
            if self._warm is not None and self._warm[i]:
                self._warm[i] = False
                self._assign(i)
                continue
            k = self.assigned[i]
            score = 0.5 * (np.sign(float(final_money[e, 0]) - float(final_money[e, 1])) + 1.0)
            self.games[k] += 1
            self.wins[k] += score
            self.fallbacks[k] += self.policies[i].stats.fallbacks
            self.win_ema[k] = (1 - self.ema) * self.win_ema[k] + self.ema * score
            self._assign(i)

    # --------------------------------------------------------------------- stats
    def totals(self, device) -> torch.Tensor:
        """[3, n_names] cumulative games / wins / fallbacks, for an all-reduce."""
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
