"""Imitation of teacher rows: the per-component NLL is finite, ignores unlabelled and illegal
components, and a few gradient steps on labelled rows raise the labels' likelihood."""
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.model import Fs1Config, Fs1Policy
from src.fs1.ppo import PPOConfig, Rollout, compute_gae, ppo_update
from test_fs1_model import real_batch

pytestmark = pytest.mark.kaggri


def _tiny():
    torch.manual_seed(0)
    return Fs1Policy(Fs1Config(d_model=32, n_layers=1, n_heads=2, mlp_ratio=2, d_target=16, hist_k=4,
                               market_noop_bias=0.0, market_sell_noop_bias=0.0))


def test_imitation_nll_masks_and_counts():
    m = _tiny().eval()
    aug, um, tm, _ = real_batch(n_envs=4, steps=10)
    acts, *_ = m.act(aug, um, tm, None)                       # legal rows by construction
    B = aug.shape[0]
    unit_label = torch.zeros(B, C.MAX_UNITS, dtype=torch.bool)
    unit_label[:, 0] = True                                   # the farmer only
    market_label = torch.zeros(B, C.N_MARKET_HEADS, dtype=torch.bool)
    market_label[:, 12] = True                                # one sell head
    nll, n = m.imitation_nll(aug, um, tm, acts, unit_label, market_label)
    assert nll.shape == (B, 2) and torch.isfinite(nll).all() and (nll >= 0).all()
    assert bool((n[:, 0] >= 1).all()) and bool((n[:, 0] <= 2).all()) and bool((n[:, 1] == 1).all())   # verb (+ target) | one head
    # an illegal label (verb not legal for the unit) is dropped, not propagated as inf
    bad = acts.clone()
    bad[:, C.MAX_UNITS] = 37                                  # DROP for the farmer, illegal in most states
    nll2, n2 = m.imitation_nll(aug, um, tm, bad, unit_label, market_label)
    assert torch.isfinite(nll2).all()
    # nothing labelled -> zero
    z, nz = m.imitation_nll(aug, um, tm, acts, torch.zeros_like(unit_label), torch.zeros_like(market_label))
    assert float(z.abs().sum()) == 0.0 and float(nz.sum()) == 0.0


def test_ppo_update_with_imitation_raises_label_likelihood():
    m = _tiny()
    aug, um, tm, _ = real_batch(n_envs=4, steps=10)
    B = aug.shape[0]
    T, N = 2, B
    ro = Rollout(T, N, m.cfg.in_dim, m.cfg.d_model, "cpu", with_lb=False)
    with torch.no_grad():
        acts, lp, ent, val, vtok = m.act(aug, um, tm, None)
    teacher = acts.clone()
    teacher[:, C.MAX_UNITS] = 1                               # farmer WATER (legal when a plant needs water)
    teacher[:, 2 * C.MAX_UNITS + 12] = 4                      # sell all wheat
    unit_label = torch.zeros(B, C.MAX_UNITS, dtype=torch.bool); unit_label[:, 0] = True
    market_label = torch.zeros(B, C.N_MARKET_HEADS, dtype=torch.bool); market_label[:, 12] = True
    for t in range(T):
        ro.add(t, aug, um, tm, acts, lp, val, vtok, None)
        ro.add_outcome(t, torch.zeros(B), torch.zeros(B), torch.zeros(B, dtype=torch.bool), torch.ones(B, dtype=torch.bool))
        for row in range(B):
            ro.add_bc(t, row, teacher[row], unit_label[row], market_label[row])
    adv, ret = compute_gae(ro.reward, ro.value[..., 0], ro.done, val[:, 0], 1.0, 0.95)
    _, ret_win = compute_gae(ro.reward_win, ro.value[..., 1], ro.done, val[:, 1], 1.0, 1.0)
    cfg = PPOConfig(minibatch=8, grad_accum=1, epochs=1, bc_coef=1.0, bc_minibatch=8)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    with torch.no_grad():
        before, nb = m.imitation_nll(aug, um, tm, teacher, unit_label, market_label)
    for _ in range(12):
        stats = ppo_update(m, opt, ro, adv, ret, ret_win, cfg, torch.tensor(1.0), 0.0, autocast_dtype=torch.float32)
    assert stats["n_bc"] == T * N and stats["bc_nll"] > 0 and stats["bc_nll_unit"] > 0 and stats["bc_nll_market"] > 0
    with torch.no_grad():
        after, na = m.imitation_nll(aug, um, tm, teacher, unit_label, market_label)
    assert float(after.sum() / na.sum()) < float(before.sum() / nb.sum())
    assert stats["n_minibatches"] == 2
