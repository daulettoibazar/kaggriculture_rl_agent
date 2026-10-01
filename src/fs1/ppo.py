"""Rollout storage, GAE and the clipped PPO update for fs1.

Storage lives on the GPU in the dtypes the policy consumes (f16 observations, bool masks,
int16 actions). Rewards carry the dense potential term plus the terminal win term; a second
return stream carries the win term alone as the target of the auxiliary value head.

Advantages are normalised over all valid samples of the rollout across ranks (one all-reduce
of sums) rather than per minibatch. Returns are scaled by a running RMS so the value loss
stays O(1) as the money scale grows through training.
"""
from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import constants as C


@dataclass
class PPOConfig:
    horizon: int = 64
    epochs: int = 1
    minibatch: int = 2048
    grad_accum: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    vf_win_coef: float = 0.25
    ent_coef: float = 1e-3
    kl_lb_coef: float = 0.005
    max_grad_norm: float = 1.0
    ret_scale_ema: float = 0.01
    bc_coef: float = 0.0             # imitation of teacher-labelled rows (per-component NLL), 0 = off
    bc_minibatch: int = 512          # labelled rows scored alongside every PPO minibatch
    mb_over_valid: bool = False      # minibatch count from the valid rows (max over ranks), not T*N
    debug_checks: bool = False       # host-side range check of the minibatch indices (one sync per epoch)

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Rollout:
    """Fixed-size storage for T steps x N seat-samples."""

    def __init__(self, T: int, N: int, in_dim: int, d_model: int, device, with_lb: bool, split: bool = False):
        self.T, self.N = T, N
        # rl_last two-stream PPO: the split log-prob (step | strategy) and the day clock
        self.logp2 = torch.zeros(T, N, 2, device=device) if split else None
        self.hour0 = torch.zeros(T, N, dtype=torch.bool, device=device) if split else None
        self.opp_tok = torch.zeros(T, N, d_model, dtype=torch.float16, device=device)
        self.obs = torch.zeros(T, N, C.N_TOKENS, in_dim, dtype=torch.float16, device=device)
        self.unit_mask = torch.zeros(T, N, C.MAX_UNITS, C.N_VERBS, dtype=torch.bool, device=device)
        self.tile_mask = torch.zeros(T, N, C.N_TARGETS, C.N_VERBS, dtype=torch.bool, device=device)
        self.actions = torch.zeros(T, N, C.PLAYER_ACTION_DIM, dtype=torch.int16, device=device)
        self.logp = torch.zeros(T, N, device=device)
        self.value = torch.zeros(T, N, 2, device=device)
        self.reward = torch.zeros(T, N, device=device)
        self.reward_win = torch.zeros(T, N, device=device)
        self.done = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.valid = torch.zeros(T, N, dtype=torch.bool, device=device)
        self.lb_logp = torch.zeros(T, N, device=device) if with_lb else None
        # imitation labels (teacher-played steps): the teacher's macro row and which components are labelled
        self.bc_actions = torch.zeros(T, N, C.PLAYER_ACTION_DIM, dtype=torch.int16, device=device)
        self.bc_unit = torch.zeros(T, N, C.MAX_UNITS, dtype=torch.bool, device=device)
        self.bc_market = torch.zeros(T, N, C.N_MARKET_HEADS, dtype=torch.bool, device=device)

    def add(self, t, obs, unit_mask, tile_mask, actions, logp, value, opp_tok, lb_logp=None):
        self.obs[t].copy_(obs)
        self.opp_tok[t].copy_(opp_tok)
        self.unit_mask[t].copy_(unit_mask)
        self.tile_mask[t].copy_(tile_mask)
        self.actions[t].copy_(actions.to(torch.int16))
        if logp.dim() == 2:                           # split log-prob [N, 2]
            self.logp2[t].copy_(logp)
            logp = logp.sum(-1)
        self.logp[t].copy_(logp)
        self.value[t].copy_(value)
        if lb_logp is not None and self.lb_logp is not None:
            self.lb_logp[t].copy_(lb_logp)

    def clear_bc(self):
        self.bc_unit.zero_()
        self.bc_market.zero_()

    def add_bc(self, t, row, actions, unit_label, market_label):
        """Label rollout row `row` at step `t` with a teacher macro row (int [55]) and its masks."""
        self.bc_actions[t, row] = actions.to(self.bc_actions.dtype)
        self.bc_unit[t, row] = unit_label
        self.bc_market[t, row] = market_label

    def add_outcome(self, t, reward, reward_win, done, valid):
        self.reward[t].copy_(reward)
        self.reward_win[t].copy_(reward_win)
        self.done[t].copy_(done)
        self.valid[t].copy_(valid)


class RunningScale:
    """EMA of the second moment of returns; `scale` is its square root (>= 1e-2)."""

    def __init__(self, ema: float, device):
        self.ema = ema
        self.m2 = torch.tensor(1.0, device=device)
        self.n = 0

    def update(self, x: torch.Tensor):
        """Always a collective (every rank must call it every update, even with no samples):
        the sum and count are reduced together, so an empty rank contributes nothing."""
        x = x.float()
        s = torch.stack([(x ** 2).sum(), torch.tensor(float(x.numel()), device=x.device)])
        if dist.is_initialized():
            dist.all_reduce(s, op=dist.ReduceOp.SUM)
        if float(s[1]) <= 0:
            return
        m2 = s[0] / s[1]
        if self.n == 0:
            self.m2 = m2
        else:
            self.m2 = (1 - self.ema) * self.m2 + self.ema * m2
        self.n += 1

    @property
    def scale(self) -> torch.Tensor:
        return self.m2.sqrt().clamp_min(1e-2)

    def state_dict(self):
        return {"m2": float(self.m2), "n": self.n}

    def load_state_dict(self, d):
        self.m2 = torch.tensor(float(d["m2"]), device=self.m2.device)
        self.n = int(d["n"])


def compute_gae(reward, value, done, last_value, gamma: float, lam: float):
    """reward/value/done: [T, N]; last_value: [N]. Terminal steps do not bootstrap.

    Returns (advantage [T, N], return [T, N]) with return = advantage + value.
    """
    T = reward.shape[0]
    adv = torch.zeros_like(reward)
    gae = torch.zeros_like(last_value)
    next_v = last_value
    for t in reversed(range(T)):
        nonterm = (~done[t]).float()
        delta = reward[t] + gamma * next_v * nonterm - value[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
        next_v = value[t]
    return adv, adv + value


def _global_mean_std(x: torch.Tensor, mask: torch.Tensor):
    s = torch.stack([(x * mask).sum(), ((x ** 2) * mask).sum(), mask.sum()])
    if dist.is_initialized():
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
    n = s[2].clamp_min(1.0)
    mean = s[0] / n
    var = (s[1] / n - mean ** 2).clamp_min(0.0)
    return mean, var.sqrt() + 1e-8


def ppo_update(model, optimizer, ro: Rollout, adv, ret, ret_win, cfg: PPOConfig, ret_scale,
               ent_coef: float, kl_active: bool = False, autocast_dtype=torch.bfloat16, market_lock=None,
               adv_day=None, norm_mask=None):
    """One PPO pass over the rollout. `model` may be a DDP wrapper (its forward = evaluate).

    Every rank runs exactly `epochs * ceil(T*N / minibatch)` minibatches — a constant, so the
    DDP all-reduce count is identical on every rank whatever its number of valid samples. Each
    epoch is a permutation of the rank's valid samples, wrapped around to fill; a rank with no
    valid samples runs the same passes with the loss weighted by zero. The optimizer steps
    every `grad_accum` minibatches (`no_sync` in between). The KL-to-last-best term is applied
    only when `kl_active` (a last-best exists and `lb_logp` was filled).

    With `adv_day` (rl_last) the model returns the split log-prob and the rollout holds `logp2`: the
    step part trains on `adv`, the strategy part (12 spend heads) on the day-level advantage, each
    with its own ratio, clip and normalisation (`residual_ppo.compute_gae_day`)."""
    T, N = ro.T, ro.N
    valid = ro.valid.view(-1)
    # `norm_mask` (rl_last phantom envs): rows that define the advantage statistics; valid rows outside it still
    # train but are normalised with these statistics and clamped, so an out-of-distribution block cannot inflate
    # the scale of every other row's advantage
    stat = valid if norm_mask is None else (valid & norm_mask.view(1, N).expand(T, N).reshape(-1))
    odd = valid & ~stat
    mean, std = _global_mean_std(adv.view(-1), stat.float())
    adv_n = ((adv.view(-1) - mean) / std)
    adv_n = torch.where(odd, adv_n.clamp(-3.0, 3.0), adv_n)
    two = adv_day is not None
    if two:
        assert ro.logp2 is not None, "adv_day needs a split rollout"
        m_d, s_d = _global_mean_std(adv_day.view(-1), stat.float())
        adv_d_n = (adv_day.view(-1) - m_d) / s_d
        adv_d_n = torch.where(odd, adv_d_n.clamp(-3.0, 3.0), adv_d_n)
        old2 = ro.logp2.view(-1, 2)
    ret_s = (ret.view(-1) / ret_scale).detach()
    ret_win_f = ret_win.view(-1).detach()

    obs = ro.obs.view(T * N, C.N_TOKENS, -1)
    um = ro.unit_mask.view(T * N, C.MAX_UNITS, C.N_VERBS)
    tm = ro.tile_mask.view(T * N, C.N_TARGETS, C.N_VERBS)
    acts = ro.actions.view(T * N, C.PLAYER_ACTION_DIM)
    old_lp = ro.logp.view(-1)
    lb_lp = None if ro.lb_logp is None else ro.lb_logp.view(-1)
    opp_tok = ro.opp_tok.view(T * N, -1)
    use_kl = bool(kl_active) and lb_lp is not None and cfg.kl_lb_coef > 0

    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    n_valid = int(idx_valid.numel())
    device = obs.device
    if n_valid == 0:
        pool, loss_w = torch.arange(T * N, device=device), 0.0
    else:
        pool, loss_w = idx_valid, 1.0
    # imitation rows: every minibatch also scores `bc_minibatch` teacher-labelled rows (constant count
    # per rank for DDP; a rank without labels scores dummy rows with weight 0)
    use_bc = cfg.bc_coef > 0
    bc_rows = torch.nonzero(ro.bc_unit.view(T * N, -1).any(-1) | ro.bc_market.view(T * N, -1).any(-1),
                            as_tuple=False).squeeze(-1) if use_bc else None
    n_bc = int(bc_rows.numel()) if use_bc else 0
    if use_bc:
        bc_pool, bc_w = (bc_rows, 1.0) if n_bc > 0 else (torch.arange(T * N, device=device), 0.0)
        bc_acts = ro.bc_actions.view(T * N, C.PLAYER_ACTION_DIM)
        bc_unit = ro.bc_unit.view(T * N, C.MAX_UNITS)
        bc_market = ro.bc_market.view(T * N, C.N_MARKET_HEADS)
    n_mb = max(1, math.ceil(T * N / cfg.minibatch))
    if cfg.mb_over_valid:
        # the same count on every rank (DDP): the largest rank's valid rows, each seen about once per epoch
        nv = torch.tensor(float(n_valid), device=device)
        if dist.is_initialized():
            dist.all_reduce(nv, op=dist.ReduceOp.MAX)
        n_mb = max(1, math.ceil(float(nv) / cfg.minibatch))
    stats = {k: 0.0 for k in ("loss", "pg", "vf", "vf_win", "ent", "kl_old", "kl_lb", "clipfrac", "bc_nll",
                              "bc_nll_unit", "bc_nll_market", "steps")}
    is_ddp = hasattr(model, "no_sync")
    accum = max(1, int(cfg.grad_accum))
    optimizer.zero_grad(set_to_none=True)
    for _ in range(cfg.epochs):
        perm = pool[torch.randperm(pool.numel(), device=device)]
        order = perm[torch.arange(n_mb * cfg.minibatch, device=device) % pool.numel()]
        if cfg.debug_checks:
            lo, hi = int(order.min()), int(order.max())
            if lo < 0 or hi >= T * N:
                raise RuntimeError(f"PPO minibatch index out of range: {lo}..{hi} vs T*N = {T * N} (pool {pool.numel()})")
        if use_bc:
            bperm = bc_pool[torch.randperm(bc_pool.numel(), device=device)]
            border = bperm[torch.arange(n_mb * cfg.bc_minibatch, device=device) % bc_pool.numel()]
        for j in range(n_mb):
            mb = order[j * cfg.minibatch:(j + 1) * cfg.minibatch]
            last_in_group = (j % accum == accum - 1) or (j == n_mb - 1)
            ctx = model.no_sync() if (is_ddp and not last_in_group) else _nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=obs.is_cuda):
                    if use_bc:
                        bb = border[j * cfg.bc_minibatch:(j + 1) * cfg.bc_minibatch]
                        bc_in = (obs[bb], um[bb], tm[bb], bc_acts[bb].long(), bc_unit[bb], bc_market[bb])
                        lp, ent, val, bc_nll_sum, bc_n = model(obs[mb], um[mb], tm[mb], acts[mb].long(), opp_tok[mb],
                                                               market_lock, bc_in)
                    else:
                        lp, ent, val = model(obs[mb], um[mb], tm[mb], acts[mb].long(), opp_tok[mb], market_lock)
                lp, ent, val = lp.float(), ent.float(), val.float()
                if two:
                    lp2, lp = lp, lp.sum(-1)
                bc_nll = torch.zeros((), device=device)
                bc_parts = (0.0, 0.0)
                if use_bc:
                    tot = bc_nll_sum.float().sum(0)                                          # [2] unit | market
                    cnt = bc_n.float().sum(0)
                    bc_nll = tot.sum() / cnt.sum().clamp_min(1.0)                            # per labelled component
                    bc_parts = (float(tot[0] / cnt[0].clamp_min(1.0)), float(tot[1] / cnt[1].clamp_min(1.0)))

                if two:
                    ratio = torch.exp(lp2[:, 0] - old2[mb, 0])
                    a = adv_n[mb]
                    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                    r_d = torch.exp(lp2[:, 1] - old2[mb, 1])
                    a_d = adv_d_n[mb]
                    pg = pg - torch.min(r_d * a_d, torch.clamp(r_d, 1 - cfg.clip, 1 + cfg.clip) * a_d).mean()
                else:
                    ratio = torch.exp(lp - old_lp[mb])
                    a = adv_n[mb]
                    pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vf = F.mse_loss(val[:, 0], ret_s[mb])
                vf_win = F.mse_loss(val[:, 1], ret_win_f[mb])
                loss = pg + cfg.vf_coef * vf + cfg.vf_win_coef * vf_win - ent_coef * ent.mean()
                kl_lb = torch.zeros((), device=device)
                if use_kl:
                    w = torch.exp(lb_lp[mb] - old_lp[mb]).clamp(max=5.0).detach()
                    kl_lb = (w * (lb_lp[mb] - lp)).mean()
                    loss = loss + cfg.kl_lb_coef * kl_lb
                total = loss * loss_w                       # PPO terms weigh 0 on a rank without valid samples
                if use_bc:
                    total = total + cfg.bc_coef * bc_w * bc_nll   # imitation has its own weight
                (total / accum).backward()
            if last_in_group:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                stats["loss"] += float(loss)
                stats["pg"] += float(pg)
                stats["vf"] += float(vf)
                stats["vf_win"] += float(vf_win)
                stats["ent"] += float(ent.mean())
                stats["kl_old"] += float((old_lp[mb] - lp).mean())
                stats["kl_lb"] += float(kl_lb)
                stats["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                stats["bc_nll"] += float(bc_nll)
                stats["bc_nll_unit"] += bc_parts[0]
                stats["bc_nll_market"] += bc_parts[1]
                stats["steps"] += 1
    k = max(1, int(stats.pop("steps")))
    out = {key: v / k for key, v in stats.items()}
    out["n_valid"] = n_valid
    out["n_bc"] = n_bc
    out["n_minibatches"] = n_mb * cfg.epochs
    out["adv_mean"], out["adv_std"] = float(mean), float(std)
    if two:
        out["adv_day_std"] = float(s_d)
    return out
