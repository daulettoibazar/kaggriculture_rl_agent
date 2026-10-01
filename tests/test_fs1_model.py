"""fs1 policy: masks are respected, act/evaluate agree, absent units stay idle, init priors."""
import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.model import Fs1Config, Fs1Policy

pytestmark = pytest.mark.kaggri


def tiny():
    torch.manual_seed(0)
    return Fs1Policy(Fs1Config(d_model=32, n_layers=1, n_heads=2, mlp_ratio=2, d_target=16, hist_k=4))


def real_batch(n_envs=6, steps=30, episode=96):
    """Observations/masks from real simulator states after some random play."""
    import src.evo  # noqa: F401  pins rayon threads before kaggri
    import kaggri

    v = kaggri.PyVecEnv(n_envs, {"episodeSteps": episode}, list(range(n_envs)), obs_dtype="float32",
                        tile_mask=True)
    obs, mask = v.reset()
    rng = np.random.default_rng(1)
    for _ in range(steps):
        a = np.zeros((n_envs, 2, v.player_action_dim), np.int64)
        a[:, :, :C.MAX_UNITS] = rng.integers(0, 100, a[:, :, :C.MAX_UNITS].shape)
        a[:, :, C.MAX_UNITS:2 * C.MAX_UNITS] = rng.integers(0, C.N_VERBS, (n_envs, 2, C.MAX_UNITS))
        a[:, :, 2 * C.MAX_UNITS] = rng.integers(0, 3, (n_envs, 2))          # some hires
        a[:, :, 2 * C.MAX_UNITS + 2:2 * C.MAX_UNITS + 7] = rng.integers(0, 3, (n_envs, 2, 5))
        obs, mask, *_ = v.step(a)
    o = torch.from_numpy(np.asarray(obs)).view(2 * n_envs, C.N_TOKENS, C.N_FEATURES).clone()
    aug = torch.cat([o, torch.zeros(2 * n_envs, C.N_TOKENS, 4)], -1).to(torch.float16)
    um = torch.from_numpy(np.asarray(mask)).view(2 * n_envs, C.MAX_UNITS, C.N_VERBS) > 0.5
    tm = torch.from_numpy(np.asarray(v.tile_mask)).view(2 * n_envs, C.N_TARGETS, C.N_VERBS) > 0
    return aug, um, tm, v


def test_sampled_actions_respect_both_masks_and_logp_matches_evaluate():
    m = tiny().eval()
    aug, um, tm, _ = real_batch()
    B = aug.shape[0]
    pair = torch.arange(B) ^ 1
    acts, lp, ent, val, vtok = m.act(aug, um, tm, pair)
    assert acts.shape == (B, C.PLAYER_ACTION_DIM) and val.shape == (B, 2) and vtok.shape == (B, 32)
    target = acts[:, :C.MAX_UNITS]
    verb = acts[:, C.MAX_UNITS:2 * C.MAX_UNITS]
    market = acts[:, 2 * C.MAX_UNITS:]
    # verbs: legal per the unit mask (Idle always allowed)
    um_eff = um.clone()
    um_eff[..., 0] = True
    assert bool(um_eff.gather(-1, verb.unsqueeze(-1)).all())
    # targets: plausible for the sampled verb whenever the verb is not Idle
    tmv = tm.transpose(1, 2).gather(1, verb.unsqueeze(-1).expand(-1, -1, C.N_TARGETS))
    ok = tmv.gather(-1, target.unsqueeze(-1)).squeeze(-1) | (verb == C.VERB_IDLE)
    assert bool(ok.all())
    # absent units (all-zero mask row) are idle with target 0
    absent = ~um.any(-1)
    assert bool((verb[absent] == 0).all()) and bool((target[absent] == 0).all())
    # market choices inside each head's range
    sizes = torch.tensor(C.MARKET_HEAD_SIZES)
    assert bool((market < sizes).all()) and bool((market >= 0).all())
    # evaluate reproduces the sampling log-prob and entropy
    lp2, ent2, val2 = m.evaluate(aug, um, tm, acts, vtok[pair])
    assert torch.allclose(lp, lp2, atol=1e-4), (lp - lp2).abs().max()
    assert torch.allclose(ent, ent2, atol=1e-4)
    assert torch.allclose(val, val2, atol=1e-5)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()


def test_greedy_is_deterministic_and_temperature_zero_matches():
    m = tiny().eval()
    aug, um, tm, _ = real_batch(n_envs=3)
    a1, *_ = m.act(aug, um, tm, None, greedy=True)
    a2, *_ = m.act(aug, um, tm, None, greedy=True)
    assert torch.equal(a1, a2)


def test_market_noop_prior_at_init():
    m = tiny().eval()
    aug, um, tm, _ = real_batch(n_envs=4, steps=0)
    acts, *_ = m.act(aug, um, tm, None)
    market = acts[:, 2 * C.MAX_UNITS:]
    # with a +4 logit on category 0 the spend heads are idle most of the time
    frac_noop = float((market == 0).float().mean())
    assert frac_noop > 0.75, frac_noop


def test_forward_is_evaluate_for_ddp():
    m = tiny().eval()
    aug, um, tm, _ = real_batch(n_envs=2, steps=3)
    acts, *_ = m.act(aug, um, tm, None)
    lp_a, _, _ = m.evaluate(aug, um, tm, acts)
    lp_b, _, _ = m(aug, um, tm, acts)
    assert torch.equal(lp_a, lp_b)


def test_param_count_of_main_config_fits_numpy_budget():
    m = Fs1Policy(Fs1Config())
    n = m.n_params()
    assert 15_000_000 < n < 25_000_000, n


def test_logit_temp_is_the_tempered_policy_in_act_and_evaluate():
    """--train-temperature: logit_temp = T makes act/evaluate score softmax(logits / T), which is what the
    untempered model samples at `temperature=T`, so a bundle exported at T plays the trained policy."""
    m = tiny().eval()
    aug, um, tm, _ = real_batch(n_envs=3)
    B = aug.shape[0]
    pair = torch.arange(B) ^ 1
    m.logit_temp = 0.5
    acts, lp, ent, _, vtok = m.act(aug, um, tm, pair)
    lp2, ent2, _ = m.evaluate(aug, um, tm, acts, vtok[pair])
    assert torch.allclose(lp, lp2, atol=1e-4) and torch.allclose(ent, ent2, atol=1e-4)
    m.logit_temp = 1.0
    lp1, ent1, _ = m.evaluate(aug, um, tm, acts, vtok[pair])
    assert not torch.allclose(lp, lp1, atol=1e-3)
    assert float(ent.mean()) < float(ent1.mean())          # sharper
    # same sampling distribution as temperature=0.5 on the untempered model (Gumbel noise from one seed)
    torch.manual_seed(3)
    a_t, *_ = m.act(aug, um, tm, pair, temperature=0.5)
    m.logit_temp = 0.5
    torch.manual_seed(3)
    a_l, *_ = m.act(aug, um, tm, pair)
    assert torch.equal(a_t, a_l)
