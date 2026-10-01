"""Scripted 4th-quadrant intervention for fs1 self-play (2026-09-30, the user's last experiment).

The trained policy never buys the 4th quadrant on its own (quadrant4 = 0.00 in every window of the final
run), so a bonus never fires and it cannot discover whether the extra 25 tiles pay. This forces the
purchase in a random share of episodes: at the start of an episode a target day in [day_lo, day_hi] is
drawn with probability `frac`; from that day on, the first step at which seat 0 owns exactly 3 quadrants
and holds >= $4,000 gets its BUY_LAND market head overridden. The forced step is marked invalid, so PPO
never trains on the forced action; everything after it is on-policy, which is how the policy learns to
farm the land and, through the critic, whether to buy it itself. Give-up after `patience` days.
"""
from __future__ import annotations

import numpy as np

from . import constants as C

LAND_HEAD = 2 * C.MAX_UNITS + 1        # packed action column of the buy_land market head (size 2: 0 = no, 1 = buy)
Q4_PRICE = C.LAND_PRICES[2]            # $4,000 for the 4th quadrant
STEPS_PER_DAY = 24


class Q4Intervention:
    def __init__(self, n_envs: int, frac: float, day_lo: int = 8, day_hi: int = 14, patience: int = 4, seed: int = 0):
        self.n, self.frac = int(n_envs), float(frac)
        self.day_lo, self.day_hi, self.patience = int(day_lo), int(day_hi), int(patience)
        self.rng = np.random.default_rng(seed)
        self.target = np.full(self.n, -1, dtype=np.int64)      # target day per env, -1 = none this episode
        self.forced = 0

    def select(self, steps: np.ndarray, quadrants: np.ndarray, money: np.ndarray) -> np.ndarray:
        """Env indices whose seat-0 BUY_LAND head must be forced this step (call once per env step, before it)."""
        steps = np.asarray(steps); day = steps // STEPS_PER_DAY
        fresh = steps == 0
        if fresh.any():
            draw = self.rng.random(int(fresh.sum())) < self.frac
            days = self.rng.integers(self.day_lo, self.day_hi + 1, size=int(fresh.sum()))
            self.target[fresh] = np.where(draw, days, -1)
        expired = (self.target >= 0) & (day > self.target + self.patience)   # give up first: never fire late
        self.target[expired] = -1
        armed = (self.target >= 0) & (day >= self.target)
        fire = armed & (np.asarray(quadrants) == 3) & (np.asarray(money) >= Q4_PRICE)
        self.target[fire] = -1
        idx = np.nonzero(fire)[0]
        self.forced += int(len(idx))
        return idx
