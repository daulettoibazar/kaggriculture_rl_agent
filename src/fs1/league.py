"""Frozen opponents for fs1 self-play: the last-best snapshot and a PFSP pool.

Every rank holds an identical League (all decisions are made on rank 0 and broadcast as
state dicts / names), so the frozen forward passes run locally on each rank's league envs.

Sampling for a league env whose episode just ended: with probability `last_best_share` the
last-best snapshot, otherwise a pool member drawn with weight (1 - winrate)^pfsp_power where
`winrate` is the latest policy's exponentially averaged score against it (ties 0.5). Fresh
members start at winrate 0.5.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


class Opponent:
    def __init__(self, name: str, model: torch.nn.Module):
        self.name = name
        self.model = model
        self.winrate = 0.5       # latest-vs-this, EMA
        self.games = 0

    def record(self, score: float, ema: float = 0.02):
        self.winrate = (1 - ema) * self.winrate + ema * float(score)
        self.games += 1

    def summary(self):
        return {"name": self.name, "winrate": round(self.winrate, 4), "games": self.games}


class League:
    def __init__(
        self,
        make_model: Callable[[], torch.nn.Module],
        pool_max: int = 16,
        last_best_share: float = 0.5,
        pfsp_power: float = 2.0,
    ):
        self.make_model = make_model
        self.pool_max = int(pool_max)
        self.last_best_share = float(last_best_share)
        self.pfsp_power = float(pfsp_power)
        self.last_best: Optional[Opponent] = None
        self.pool: list[Opponent] = []

    # ------------------------------------------------------------- membership
    def _frozen(self, name: str, state_dict) -> Opponent:
        m = self.make_model()
        m.load_state_dict(state_dict)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return Opponent(name, m)

    def set_last_best(self, name: str, state_dict):
        self.last_best = self._frozen(name, state_dict)

    def add_to_pool(self, name: str, state_dict):
        self.pool.append(self._frozen(name, state_dict))
        if len(self.pool) > self.pool_max:
            # FIFO, so every rank's pool stays identical without a broadcast
            self.pool.pop(0)

    def has_opponents(self) -> bool:
        return self.last_best is not None or bool(self.pool)

    def opponents(self) -> list[Opponent]:
        out = [] if self.last_best is None else [self.last_best]
        return out + self.pool

    # --------------------------------------------------------------- sampling
    def sample(self, rng: np.random.Generator) -> Opponent:
        assert self.has_opponents()
        if self.last_best is not None and (not self.pool or rng.random() < self.last_best_share):
            return self.last_best
        w = np.array([(1.0 - o.winrate) ** self.pfsp_power + 1e-3 for o in self.pool])
        return self.pool[int(rng.choice(len(self.pool), p=w / w.sum()))]

    def summary(self):
        return {
            "last_best": None if self.last_best is None else self.last_best.summary(),
            "pool": [o.summary() for o in self.pool],
        }
