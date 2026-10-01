"""PyVecEnv wrapper for fs1: phase stagger, price history, torch tensors, validity flags.

One instance per DDP rank. The simulator publishes f32 observations; the wrapper computes the
potential from them, appends `hist_k` price-delta columns to every token (non-zero only on the
PRODUCT tokens), and hands the policy f16-quantised inputs (storage). The bundle feeds the
Encoder's f32 planes to the numpy policy; the difference is one f16 ulp and flipped 0 of 32
greedy actions in the review's check (`research/fs1/review_opus2_2026-09-21.md` LOW 7).

Phase stagger: `warmup_pass(offsets)` puts env i at step `offsets[i]` by all-PASS turns, so
every rollout contains terminals and every game phase. The first episode of a staggered env
is junk (nobody acted for `offsets[i]` turns) and is marked invalid until its terminal.

`kaggri` is imported lazily inside the constructor, after the caller has pinned the thread
environment (RAYON_NUM_THREADS etc.), and never in a parent that later forks.
"""
from __future__ import annotations

import numpy as np
import torch

from . import constants as C
from .masks import relax_unit_mask
from .reward import Potential


class Fs1VecEnv:
    def __init__(
        self,
        n_envs: int,
        seeds,
        device,
        hist_k: int = 4,
        episode_steps: int = C.EPISODE_STEPS,
        stagger: bool = True,
        stagger_seed: int = 0,
        chunk_envs: int = 1,
        relax_masks: bool = False,
    ):
        import kaggri  # noqa: WPS433  (lazy on purpose, see module docstring)

        assert kaggri.OBS_DIM == C.OBS_DIM and kaggri.MASK_DIM == C.MASK_DIM
        assert kaggri.TILE_MASK_DIM == C.TILE_MASK_DIM
        self.k = kaggri
        self.n = int(n_envs)
        self.relax_masks = bool(relax_masks)      # rl_last F1 fix (`src/fs1/masks.py`)
        self.device = torch.device(device)
        self.hist_k = int(hist_k)
        self.episode_steps = int(episode_steps)
        self.config = {"episodeSteps": self.episode_steps}
        self.env = kaggri.PyVecEnv(
            self.n, self.config, [int(s) for s in seeds], auto_reset=True,
            chunk_envs=chunk_envs, obs_dtype="float32", tile_mask=True,
        )
        self.potential = Potential(self.device)

        obs, mask = self.env.reset()
        rng = np.random.default_rng(stagger_seed)
        if stagger:
            offsets = rng.integers(0, self.episode_steps - 1, size=self.n)
            obs, mask = self.env.warmup_pass([int(o) for o in offsets])
        else:
            offsets = np.zeros(self.n, dtype=np.int64)
        self.offsets = offsets
        # transitions of a junk first episode (including its terminal) must not train
        self.invalid = torch.from_numpy(offsets > 0).to(self.device)

        self.hist = torch.zeros(self.n, 2, len(C.PRODUCTS), self.hist_k, device=self.device)
        self.prev_rel = None
        self.phi = None
        self.money = None
        self._ingest(obs, mask, self.env.tile_mask, np.asarray(self.env.money), reset_hist=None)

    # ------------------------------------------------------------------ state
    def _ingest(self, obs, mask, tile_mask, money, reset_hist):
        """Copy the env-owned buffers to the device and derive the per-step tensors."""
        o = torch.from_numpy(np.asarray(obs)).to(self.device, copy=True)   # never alias the env buffer
        o = o.view(self.n, 2, C.N_TOKENS, C.N_FEATURES)
        self.money = torch.from_numpy(np.asarray(money)).to(self.device).clone()
        self.phi = self.potential(o, self.money)                                  # [n, 2]

        rel = o[:, :, C.TOK_PRODUCT0:C.TOK_PRODUCT1, C.HDR + C.P_PRICE_REL]      # [n, 2, 9]
        if self.prev_rel is None:
            delta = torch.zeros_like(rel)
        else:
            delta = rel - self.prev_rel
        if reset_hist is not None:                                                # done envs: new episode
            delta = torch.where(reset_hist[:, None, None], torch.zeros_like(delta), delta)
            self.hist = torch.where(reset_hist[:, None, None, None], torch.zeros_like(self.hist), self.hist)
        if self.hist_k > 0:
            self.hist = torch.roll(self.hist, shifts=1, dims=-1)
            self.hist[..., 0] = delta
        self.prev_rel = rel.clone()

        aug = torch.zeros(self.n, 2, C.N_TOKENS, self.hist_k, device=self.device)
        if self.hist_k > 0:
            aug[:, :, C.TOK_PRODUCT0:C.TOK_PRODUCT1, :] = self.hist
        self.obs_aug = torch.cat([o, aug], dim=-1).to(torch.float16)             # [n, 2, 140, 48+k]
        self.unit_mask = torch.from_numpy(np.asarray(mask)).to(self.device).view(
            self.n, 2, C.MAX_UNITS, C.N_VERBS) > 0.5
        self.tile_mask = torch.from_numpy(np.asarray(tile_mask)).to(self.device).view(
            self.n, 2, C.N_TARGETS, C.N_VERBS) > 0
        if self.relax_masks:
            self.unit_mask = relax_unit_mask(self.unit_mask, self.tile_mask)

    # ------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor, exact_actions=None):
        """actions: [n, 2, 55] int64 on any device; `exact_actions` an optional per-env list of
        `[seat0, seat1]` Kaggle-format turn dicts (None keeps the packed action for that seat).

        Returns (done [n] bool, final_money [n, 2] f32, valid [n] bool) for the transition
        just taken; `obs_aug`, `unit_mask`, `tile_mask`, `phi`, `money` now describe the next
        state (a fresh episode where `done`).
        """
        a = actions.detach().to("cpu", torch.int64).contiguous().numpy()
        obs, mask, rew, done, info = self.env.step(a, exact_actions) if exact_actions is not None else self.env.step(a)
        done_t = torch.from_numpy(np.asarray(done)).to(self.device)
        final_money = torch.from_numpy(np.asarray(info["final_money"])).to(self.device).clone()
        valid = ~self.invalid
        self.invalid = self.invalid & ~done_t
        self._ingest(obs, mask, info["tile_mask"], info["money"], reset_hist=done_t)
        return done_t, final_money, valid

    def flat(self):
        """Seat-major views for the policy: [2n, ...] with seat pairs adjacent (opp = i ^ 1)."""
        n2 = 2 * self.n
        return (
            self.obs_aug.view(n2, C.N_TOKENS, -1),
            self.unit_mask.view(n2, C.MAX_UNITS, C.N_VERBS),
            self.tile_mask.view(n2, C.N_TARGETS, C.N_VERBS),
        )

    @staticmethod
    def pair_index(n_envs: int, device) -> torch.Tensor:
        idx = torch.arange(2 * n_envs, device=device)
        return idx ^ 1

    def env_steps(self) -> np.ndarray:
        return np.asarray(self.env.env_steps())
