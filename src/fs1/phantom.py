"""Phantom-seat market pressure from recorded top-team order streams (rl_last review, 2026-09-28).

`kaggri.PyVecEnv.set_phantom_seats` turns seat 1 of chosen envs into a market-pressure process: its
unit / HIRE / BUY_LAND / BUY_SEED / BUY_ANIMAL actions are ignored, its SELL / BUY_PRODUCT orders move
`market.inventory` only, and it banks nothing. Feeding it a recorded top-ranked seat's realised order
stream (tools/fs1/replay_market_streams.py) exposes the training seat (seat 0) to top-of-field market
pressure that no public program produces, with nothing to exploit but the price curve itself.

Reward: a phantom has no farm, so its potential is constant and the shaped reward of the real seat would be
its ABSOLUTE money (5-10x the paired margins; the review's first smoke inflated ret_scale 0.66 -> 0.96 in 6
updates). `virtual_opponent` therefore replaces the phantom seat's Phi with the recorded seat's own money
curve, so a phantom game is a margin race against the recorded top-20 team's score:
r = (dPhi_self - d money_rec) / SCALE, terminal = our money - their recorded final.

A stream is a list of 720 per-step order lists plus the recorded money per step; the env's current step
indexes both, so a staggered env simply starts mid-stream. Every auto-reset draws a fresh stream.
"""
from __future__ import annotations

import json

import numpy as np
import torch

PASS_TURN = {"farmer": ["PASS"], "hands": [], "market": []}


class PhantomStreams:
    def __init__(self, path: str, envs, seed: int = 0, episode_steps: int = 720):
        d = json.load(open(path))
        ok = [s for s in d["streams"] if len(s["market"]) >= episode_steps and len(s.get("money", [])) >= episode_steps]
        assert ok, f"no full-length streams with money curves in {path} (rebuild with tools/fs1/replay_market_streams.py)"
        self.streams = [s["market"] for s in ok]
        self.money = [np.asarray(s["money"], dtype=np.float64) for s in ok]
        self.finals = [float(s["final"]) for s in ok]
        self.meta = {"n_streams": len(ok), "min_rank": d.get("min_rank"), "since": d.get("since"),
                     "final_median": float(np.median(self.finals))}
        self.envs = [int(e) for e in envs]
        self.rng = np.random.default_rng(seed)
        self.cur = {e: int(self.rng.integers(len(self.streams))) for e in self.envs}
        self._idx = None

    def seat_array(self, n_envs: int) -> np.ndarray:
        """int8 [n_envs]: -1 everywhere, 1 on the phantom envs (their seat 1 is the phantom)."""
        s = np.full(n_envs, -1, dtype=np.int8)
        s[self.envs] = 1
        return s

    def turns(self, steps: np.ndarray) -> dict:
        """{env: seat-1 turn dict} for the current step of every phantom env."""
        out = {}
        for e in self.envs:
            t = int(steps[e])
            mk = self.streams[self.cur[e]]
            orders = mk[t] if 0 <= t < len(mk) else []
            out[e] = {"farmer": ["PASS"], "hands": [], "market": [list(o) for o in orders]}
        return out

    def fill(self, rows: list, steps: np.ndarray):
        """Write the phantom turns into an `exact_actions` row list ([seat0, seat1] per env)."""
        for e, turn in self.turns(steps).items():
            rows[e] = [rows[e][0] if rows[e] is not None else None, turn]
        return rows

    def virtual_opponent(self, phi_prev, phi_next, final_money, done: np.ndarray, steps_prev: np.ndarray):
        """Return copies of phi_prev / phi_next / final_money [n, 2] with the phantom seat (1) of every phantom
        env replaced by the recorded seat's money at the step before / after this transition (the recorded final
        where the episode ended). Call BEFORE `on_done` (which draws new streams)."""
        prev, nxt, fin = [], [], []
        for e in self.envs:
            m = self.money[self.cur[e]]
            t = int(steps_prev[e])
            prev.append(m[min(max(t, 0), len(m) - 1)])
            if done[e]:
                nxt.append(self.finals[self.cur[e]])
                fin.append(self.finals[self.cur[e]])
            else:
                nxt.append(m[min(t + 1, len(m) - 1)])
                fin.append(float(final_money[e, 1]))
        dev = phi_prev.device
        if self._idx is None:
            self._idx = torch.tensor(self.envs, device=dev, dtype=torch.long)
        phi_prev, phi_next, final_money = phi_prev.clone(), phi_next.clone(), final_money.clone()
        phi_prev[self._idx, 1] = torch.tensor(prev, device=dev, dtype=phi_prev.dtype)
        phi_next[self._idx, 1] = torch.tensor(nxt, device=dev, dtype=phi_next.dtype)
        final_money[self._idx, 1] = torch.tensor(fin, device=dev, dtype=final_money.dtype)
        return phi_prev, phi_next, final_money

    def on_done(self, done: np.ndarray):
        for e in self.envs:
            if done[e]:
                self.cur[e] = int(self.rng.integers(len(self.streams)))
