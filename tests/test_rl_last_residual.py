"""rl_last model side: style embedding, F1 mask relaxation, Option B residual heads and PPO."""
import copy

import numpy as np
import torch

from src.fs1 import constants as C
from src.fs1.masks import relax_unit_mask
from src.fs1.model import Fs1Config, Fs1Policy
from src.fs1.ppo import PPOConfig
from src.fs1.residual_ppo import ResidualRollout, compute_gae_day, ppo_update_residual

SMALL = dict(d_model=32, n_layers=1, n_heads=4, d_target=16)


def _inputs(B, extra=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(B, C.N_TOKENS, C.N_FEATURES + 4 + extra, generator=g)
    um = torch.rand(B, C.MAX_UNITS, C.N_VERBS, generator=g) < 0.5
    um[:, 12:] = False                                  # absent units
    tm = torch.rand(B, C.N_TARGETS, C.N_VERBS, generator=g) < 0.3
    return obs, um, tm


def test_relax_mask_live_units_only():
    um = np.zeros((17, 38), bool)
    um[0, 0] = True
    tm = np.zeros((100, 38), bool)
    tm[4, 10] = True
    r = relax_unit_mask(um, tm)
    assert r[0, 10] and not r[1, 10] and not r[0, 3] and not um[0, 10]


def test_fold_style_matches_style_kwarg():
    torch.manual_seed(0)
    m = Fs1Policy(Fs1Config(n_styles=4, **SMALL)).eval()
    with torch.no_grad():
        m.style_emb.weight.normal_()
    obs, um, tm = _inputs(3)
    st = torch.full((3,), 2)
    a, *_ = m.act(obs, um, tm, greedy=True, style=st)
    f = m.fold_style(2)
    assert f.cfg.n_styles == 0 and f.style_emb is None
    b, *_ = f.act(obs, um, tm, greedy=True)
    assert torch.equal(a, b)
    lp1, _, v1 = m.evaluate(obs, um, tm, a, style=st)
    lp2, _, v2 = f.evaluate(obs, um, tm, a)
    assert torch.allclose(lp1, lp2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-5)


def test_residual_init_is_follow_and_base_equivalent():
    torch.manual_seed(1)
    base = Fs1Policy(Fs1Config(**SMALL)).eval()
    res = Fs1Policy(Fs1Config(n_extra_in=6, residual=True, **SMALL)).load_base(base.state_dict()).eval()
    obs, um, tm = _inputs(4, extra=6)
    a, gate, lp, val, _ = res.act_residual(obs, um, tm, greedy=True)
    assert not gate.any()                                   # greedy at init = all FOLLOW
    assert float(lp.sum(-1).abs().max()) < 38 * 4e-4        # log P(all follow) ~ 0
    # appended columns are invisible at init: the override branch equals the base policy
    b, _, _, vb, _ = base.act(obs[..., :C.N_FEATURES + 4], um, tm, greedy=True)
    assert torch.equal(a, b) and torch.allclose(val, vb, atol=1e-5)
    # sampled overrides at init ~ sigmoid(-8) per live gate
    torch.manual_seed(2)
    many = torch.cat([res.act_residual(obs, um, tm)[1] for _ in range(50)])
    assert many.float().mean() < 5e-3


def test_evaluate_matches_act_logp():
    torch.manual_seed(3)
    res = Fs1Policy(Fs1Config(n_extra_in=2, residual=True, override_bias=0.0, **SMALL)).eval()
    obs, um, tm = _inputs(5, extra=2)
    sopen = torch.tensor([True, False, True, False, True])
    a, gate, lp, _, _ = res.act_residual(obs, um, tm, strategy_open=sopen)
    assert not gate[~sopen][:, C.MAX_UNITS:C.MAX_UNITS + 12].any()
    lp2, ent, _, p_ov = res.evaluate_residual(obs, um, tm, a, gate, strategy_open=sopen)
    assert torch.allclose(lp, lp2, atol=1e-4)
    assert torch.all(lp2[~sopen, 1] == 0)
    assert p_ov.shape == (5, 38) and torch.all(p_ov[:, 12:C.MAX_UNITS] == 0)   # absent units carry no gate


def test_gae_day_limits():
    T, N = 48, 2
    torch.manual_seed(4)
    r = torch.randn(T, N)
    v = torch.randn(T, N)
    done = torch.zeros(T, N, dtype=torch.bool)
    done[29, 1] = True
    hour0 = torch.zeros(T, N, dtype=torch.bool)
    hour0[::24] = True
    last = torch.randn(N)
    adv1, _ = compute_gae_day(r, v, done, hour0, last, 1.0)
    # lambda 1: Monte-Carlo return to the episode end / rollout end minus V
    for n in range(N):
        for t in (0, 10, 30, 47):
            end = 30 if (n == 1 and t <= 29) else T
            boot = 0.0 if (n == 1 and t <= 29) else float(last[n])
            g = float(r[t:end, n].sum()) + boot
            assert abs(float(adv1[t, n]) - (g - float(v[t, n]))) < 1e-4
    adv0, _ = compute_gae_day(r, v, done, hour0, last, 0.0)
    # lambda 0: rewards to the next day boundary + V(boundary) - V
    assert abs(float(adv0[3, 0]) - (float(r[3:24, 0].sum()) + float(v[24, 0]) - float(v[3, 0]))) < 1e-4


def _rollout(res, T=6, n_seat=4, extra=2):
    ro = ResidualRollout(T, n_seat, C.N_FEATURES + 4 + extra, res.cfg.d_model, "cpu")
    for t in range(T):
        obs, um, tm = _inputs(n_seat, extra=extra, seed=10 + t)
        hour0 = torch.full((n_seat,), t % 3 == 0)
        a, gate, lp, val, tok = res.act_residual(obs, um, tm, strategy_open=hour0)
        ro.add_residual(t, obs.half(), um, tm, a, gate, lp, val, tok.half(), hour0, hour0)
        ro.add_outcome(t, torch.randn(n_seat), torch.zeros(n_seat), torch.zeros(n_seat, dtype=torch.bool),
                       torch.ones(n_seat, dtype=torch.bool))
    return ro


def test_ppo_update_residual_and_critic_warmup():
    torch.manual_seed(5)
    res = Fs1Policy(Fs1Config(n_extra_in=2, residual=True, override_bias=-1.0, **SMALL))
    ro = _rollout(res)
    adv_s = torch.randn(ro.T, ro.N)
    adv_d = torch.randn(ro.T, ro.N)
    ret = torch.randn(ro.T, ro.N)
    cfg = PPOConfig(minibatch=8, grad_accum=1)
    frozen = copy.deepcopy(res)
    opt = torch.optim.AdamW(frozen.parameters(), lr=1e-2, weight_decay=0.0)
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    st = ppo_update_residual(frozen, opt, ro, adv_s, adv_d, ret, cfg, torch.tensor(1.0), 0.0, 0.1, policy_frozen=True)
    after = frozen.state_dict()
    for k in before:
        changed = not torch.equal(before[k], after[k])
        assert changed == k.startswith("value_head"), k
    opt = torch.optim.AdamW(res.parameters(), lr=1e-2)
    g0 = float(res.unit_gate.bias)
    st = ppo_update_residual(res, opt, ro, adv_s, adv_d, ret, cfg, torch.tensor(1.0), 0.0, 1.0)
    assert np.isfinite(st["loss"]) and st["p_override"] > 0
    assert float(res.unit_gate.bias) != g0


def test_sell_only_gates():
    torch.manual_seed(6)
    res = Fs1Policy(Fs1Config(n_extra_in=2, residual=True, override_bias=0.0, open_gates="sell", **SMALL)).eval()
    obs, um, tm = _inputs(6, extra=2)
    a, gate, lp, _, _ = res.act_residual(obs, um, tm)
    assert not gate[:, :C.MAX_UNITS + 12].any() and gate[:, C.MAX_UNITS + 12:].any()
    assert torch.all(lp[:, 1] == 0)                          # no strategy stream in v1
    lp2, _, _, p_ov = res.evaluate_residual(obs, um, tm, a, gate)
    assert torch.allclose(lp, lp2, atol=1e-4) and torch.all(p_ov[:, :C.MAX_UNITS + 12] == 0)


def test_split_logp_sums_to_total_and_two_stream_update():
    from src.fs1.ppo import Rollout, ppo_update
    torch.manual_seed(7)
    m = Fs1Policy(Fs1Config(**SMALL))
    obs, um, tm = _inputs(4)
    a, lp, _, _, _ = m.act(obs, um, tm)
    m.split_logp = True
    lp_s, _, _ = m.evaluate(obs, um, tm, a)
    assert lp_s.shape == (4, 2) and torch.allclose(lp_s.sum(-1), lp, atol=1e-4)
    T, N = 4, 4
    ro = Rollout(T, N, C.N_FEATURES + 4, 32, "cpu", with_lb=False, split=True)
    for t in range(T):
        o, u, tt = _inputs(N, seed=20 + t)
        a, lp2, _, val, tok = m.act(o, u, tt)
        assert lp2.shape == (N, 2)
        ro.add(t, o.half(), u, tt, a, lp2, val, tok.half())
        ro.add_outcome(t, torch.randn(N), torch.zeros(N), torch.zeros(N, dtype=torch.bool), torch.ones(N, dtype=torch.bool))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    st = ppo_update(m, opt, ro, torch.randn(T, N), torch.randn(T, N), torch.zeros(T, N), PPOConfig(minibatch=8, grad_accum=1),
                    torch.tensor(1.0), 0.0, adv_day=torch.randn(T, N))
    assert np.isfinite(st["loss"]) and "adv_day_std" in st


def test_bundle_numpy_relax_matches_torch():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "fs1_numpy_policy_t", pathlib.Path(__file__).resolve().parents[1] / "submission" / "fs1_numpy_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    um = rng.random((5, 17, 38)) < 0.3
    um[:, 10:] = False
    tm = rng.random((5, 100, 38)) < 0.05
    assert np.array_equal(mod.relax_unit_mask(um, tm), relax_unit_mask(um, tm))
    assert np.array_equal(mod.relax_unit_mask(um, tm), relax_unit_mask(torch.from_numpy(um), torch.from_numpy(tm)).numpy())
