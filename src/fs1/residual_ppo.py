"""rl_last Option B: PPO for the follow-or-override residual over cha22x, with two advantage streams.

The policy's log-prob of a turn splits into a STEP part (unit gates + their override branches, the
9 sell heads) and a STRATEGY part (the 12 spend heads: hire, land, seeds, animals, buys), see
`Fs1Policy.evaluate_residual`. Each part gets its own clipped surrogate with its own advantage:

* step stream: the usual per-step GAE (gamma 1, lambda ~0.99, `ppo.compute_gae`);
* strategy stream: `compute_gae_day`, which bootstraps only at DAY boundaries (hour 0), so a
  hire / land / animal choice is credited with the whole day's reward and, through lambda_day,
  with the following days' (research/rl_last/DESIGN.md "Algorithm"). Every row's own value is its
  baseline, so a row mid-day uses the rest of its day as the n-step return.

The ratio and clip are per row, with the day advantage shared by the rows of a day (a valid policy
gradient: the day's log-probs sum, each weighted by the same advantage) rather than on the
day-summed ratio: minibatches are random rows, not whole days.

The FOLLOW anchor is a closed-form penalty on the override probabilities, mean over live gates
of p_override (a KL to an all-FOLLOW reference is dominated by the same term), with its own
coefficient schedule owned by the trainer.
"""
from __future__ import annotations

import math
from contextlib import nullcontext as _nullcontext

import torch
import torch.nn.functional as F

from . import constants as C
from .ppo import Rollout, _global_mean_std

N_GATES = C.MAX_UNITS + C.N_MARKET_HEADS    # 38


class ResidualRollout(Rollout):
    """`Rollout` plus the gate choices, the split log-prob and the day clock."""

    def __init__(self, T: int, N: int, in_dim: int, d_model: int, device, with_lb: bool = False):
        super().__init__(T, N, in_dim, d_model, device, with_lb)
        self.gate = torch.zeros(T, N, N_GATES, dtype=torch.bool, device=device)
        self.logp2 = torch.zeros(T, N, 2, device=device)
        self.hour0 = torch.zeros(T, N, dtype=torch.bool, device=device)     # row is a day's first turn
        self.strategy_open = torch.ones(T, N, dtype=torch.bool, device=device)

    def add_residual(self, t, obs, unit_mask, tile_mask, actions, gate, logp2, value, opp_tok, hour0,
                     strategy_open=None):
        self.add(t, obs, unit_mask, tile_mask, actions, logp2.sum(-1), value, opp_tok)
        self.gate[t].copy_(gate)
        self.logp2[t].copy_(logp2)
        self.hour0[t].copy_(hour0)
        if strategy_open is not None:
            self.strategy_open[t].copy_(strategy_open)


def compute_gae_day(reward, value, done, hour0, last_value, lam_day: float):
    """Day-level GAE (gamma = 1). reward/value/done/hour0: [T, N]; last_value: [N].

    For every row t: A_t = (sum of rewards from t to the next day boundary) + V(boundary) - V_t
    + lam_day * A(boundary), where the boundary is the next row with hour0 (or the rollout end,
    bootstrapped with `last_value`, A = 0); an episode end stops the sum and bootstraps 0.
    lam_day = 1 gives the Monte-Carlo return minus V_t; 0 gives a one-day TD error.
    Returns (advantage [T, N], return [T, N])."""
    T = reward.shape[0]
    adv = torch.zeros_like(reward)
    acc = torch.zeros_like(last_value)          # rewards from t to the next boundary
    v_nb = last_value.clone()                   # value at the next boundary
    a_nb = torch.zeros_like(last_value)         # advantage at the next boundary
    for t in reversed(range(T)):
        d = done[t]
        acc = torch.where(d, reward[t], reward[t] + acc)
        v_nb = torch.where(d, torch.zeros_like(v_nb), v_nb)
        a_nb = torch.where(d, torch.zeros_like(a_nb), a_nb)
        a_t = acc + v_nb - value[t] + lam_day * a_nb
        adv[t] = a_t
        b = hour0[t]
        v_nb = torch.where(b, value[t], v_nb)
        a_nb = torch.where(b, a_t, a_nb)
        acc = torch.where(b, torch.zeros_like(acc), acc)
    return adv, adv + value


def ppo_update_residual(model, optimizer, ro: ResidualRollout, adv_step, adv_day, ret, cfg, ret_scale,
                        ent_coef: float, follow_coef: float, autocast_dtype=torch.bfloat16, policy_frozen=False):
    """One PPO pass. `model` is a residual `Fs1Policy` or its DDP wrapper (forward routes to
    `evaluate_residual`). `policy_frozen` = critic warm-up: only the value loss trains (the policy terms are
    computed for logging but weigh 0). Minibatch count is constant per rank as in `ppo_update`."""
    T, N = ro.T, ro.N
    valid = ro.valid.view(-1)
    vf = valid.float()
    m_s, s_s = _global_mean_std(adv_step.view(-1), vf)
    # the strategy stream only has signal on rows whose strategy gates were open
    open_ = ro.strategy_open.view(-1) & valid
    m_d, s_d = _global_mean_std(adv_day.view(-1), open_.float())
    a_step = (adv_step.view(-1) - m_s) / s_s
    a_day = (adv_day.view(-1) - m_d) / s_d
    ret_s = (ret.view(-1) / ret_scale).detach()

    obs = ro.obs.view(T * N, C.N_TOKENS, -1)
    um = ro.unit_mask.view(T * N, C.MAX_UNITS, C.N_VERBS)
    tm = ro.tile_mask.view(T * N, C.N_TARGETS, C.N_VERBS)
    acts = ro.actions.view(T * N, C.PLAYER_ACTION_DIM)
    gate = ro.gate.view(T * N, N_GATES)
    sopen = ro.strategy_open.view(-1)
    old = ro.logp2.view(T * N, 2)
    opp_tok = ro.opp_tok.view(T * N, -1)
    device = obs.device
    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    n_valid = int(idx_valid.numel())
    pool, loss_w = (idx_valid, 1.0) if n_valid else (torch.arange(T * N, device=device), 0.0)
    n_mb = max(1, math.ceil(T * N / cfg.minibatch))
    stats = {k: 0.0 for k in ("loss", "pg_step", "pg_day", "vf", "ent", "p_override", "follow_pen",
                              "kl_step", "kl_day", "clip_step", "clip_day", "steps")}
    is_ddp = hasattr(model, "no_sync")
    accum = max(1, int(cfg.grad_accum))
    optimizer.zero_grad(set_to_none=True)
    for _ in range(cfg.epochs):
        perm = pool[torch.randperm(pool.numel(), device=device)]
        order = perm[torch.arange(n_mb * cfg.minibatch, device=device) % pool.numel()]
        for j in range(n_mb):
            mb = order[j * cfg.minibatch:(j + 1) * cfg.minibatch]
            last_in_group = (j % accum == accum - 1) or (j == n_mb - 1)
            ctx = model.no_sync() if (is_ddp and not last_in_group) else _nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=obs.is_cuda):
                    lp, ent, val, p_ov = model(obs[mb], um[mb], tm[mb], acts[mb].long(), gate[mb], opp_tok[mb],
                                               None, sopen[mb])      # forward -> evaluate_residual
                lp, ent, val, p_ov = lp.float(), ent.float(), val.float(), p_ov.float()
                r_s = torch.exp(lp[:, 0] - old[mb, 0])
                r_d = torch.exp(lp[:, 1] - old[mb, 1])
                a_s, a_d = a_step[mb], a_day[mb]
                w_d = sopen[mb].float()
                pg_s = -torch.min(r_s * a_s, torch.clamp(r_s, 1 - cfg.clip, 1 + cfg.clip) * a_s).mean()
                pg_d = -(torch.min(r_d * a_d, torch.clamp(r_d, 1 - cfg.clip, 1 + cfg.clip) * a_d) * w_d).sum() \
                    / w_d.sum().clamp_min(1.0)
                v_loss = F.mse_loss(val[:, 0], ret_s[mb])
                live = torch.cat([um[mb].any(-1), torch.ones_like(p_ov[:, C.MAX_UNITS:], dtype=torch.bool)], -1)
                follow_pen = (p_ov * live).sum(-1).mean()
                pol = pg_s + pg_d - ent_coef * ent.mean() + follow_coef * follow_pen
                loss = cfg.vf_coef * v_loss + (0.0 if policy_frozen else 1.0) * pol
                ((loss * loss_w) / accum).backward()
            if last_in_group:
                if policy_frozen:           # critic warm-up: drop any policy-side gradient leak
                    for nm, p in (model.module if is_ddp else model).named_parameters():
                        if not nm.startswith("value_head") and p.grad is not None:
                            p.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                stats["loss"] += float(loss)
                stats["pg_step"] += float(pg_s)
                stats["pg_day"] += float(pg_d)
                stats["vf"] += float(v_loss)
                stats["ent"] += float(ent.mean())
                stats["p_override"] += float((p_ov * live).sum(-1).mean())
                stats["follow_pen"] += float(follow_pen)
                stats["kl_step"] += float((old[mb, 0] - lp[:, 0]).mean())
                stats["kl_day"] += float((old[mb, 1] - lp[:, 1]).mean())
                stats["clip_step"] += float(((r_s - 1).abs() > cfg.clip).float().mean())
                stats["clip_day"] += float((((r_d - 1).abs() > cfg.clip).float() * w_d).sum() / w_d.sum().clamp_min(1.0))
                stats["steps"] += 1
    k = max(1, int(stats.pop("steps")))
    out = {key: v / k for key, v in stats.items()}
    out.update(n_valid=n_valid, n_minibatches=n_mb * cfg.epochs, adv_step_std=float(s_s), adv_day_std=float(s_d))
    return out
