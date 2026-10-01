"""The scripted wheat opener: legal under both masks, earns money over a full season, and the
market lock forces the locked heads to no-op consistently between act and evaluate."""
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.model import Fs1Config, Fs1Policy
from src.fs1.opener import opener_actions, prefix_length
from test_fs1_model import real_batch

pytestmark = pytest.mark.kaggri


def test_opener_actions_are_legal_under_both_masks():
    aug, um, tm, _ = real_batch(n_envs=6, steps=20)
    rows = opener_actions(aug, um, tm)
    target, verb = rows[:, 0], rows[:, C.MAX_UNITS]
    um_eff = um.clone()
    um_eff[..., 0] = True
    assert bool(um_eff[:, 0].gather(-1, verb.unsqueeze(-1)).all())
    non_idle = verb != 0
    ok = tm[torch.arange(len(rows)), target, verb] | ~non_idle
    assert bool(ok.all())
    # every present unit's verb is legal for THAT unit (hands act too); absent units are idle
    verbs = rows[:, C.MAX_UNITS:2 * C.MAX_UNITS]
    assert bool(um_eff.gather(-1, verbs.unsqueeze(-1)).all())
    absent = ~um.any(-1)
    assert bool((verbs[absent] == 0).all())
    assert bool((rows[:, 2 * C.MAX_UNITS + 12] == 4).all())                        # sell all wheat


def test_opener_earns_money_over_a_season():
    from src.fs1.env import Fs1VecEnv

    n = 4
    env = Fs1VecEnv(n, [101, 102, 103, 104], "cpu", hist_k=4, episode_steps=C.EPISODE_STEPS, stagger=False)
    final = None
    for _ in range(C.EPISODE_STEPS):
        obs, um, tm = env.flat()
        rows = opener_actions(obs, um, tm)
        done, fm, _ = env.step(rows.view(n, 2, -1))
        if bool(done.all()):
            final = fm
            break
    assert final is not None
    # the pilot-2 wheat-only rule made ~$6,600/seat; this one (hands, land, melons) ~$16,000 mean
    assert bool((final > C.STARTING_MONEY + 3000).all()), final
    assert float(final.mean()) > 10_000, final


def test_prefix_length_schedule():
    assert prefix_length(0.0, 240, 0.3) == 240
    assert prefix_length(0.15, 240, 0.3) == 120
    assert prefix_length(0.3, 240, 0.3) == 0
    assert prefix_length(0.9, 240, 0.3) == 0
    assert prefix_length(0.0, 240, 0.0) == 0


def test_market_lock_forces_noop_and_is_consistent():
    torch.manual_seed(0)
    m = Fs1Policy(Fs1Config(d_model=32, n_layers=1, n_heads=2, mlp_ratio=2, d_target=16, hist_k=4,
                            market_noop_bias=0.0, market_sell_noop_bias=0.0)).eval()
    aug, um, tm, _ = real_batch(n_envs=6, steps=5)
    lock = torch.zeros(C.N_MARKET_HEADS, dtype=torch.bool)
    lock[1] = True
    lock[7:10] = True
    acts, lp, ent, _, vtok = m.act(aug, um, tm, None, market_lock=lock)
    market = acts[:, 2 * C.MAX_UNITS:]
    assert bool((market[:, 1] == 0).all()) and bool((market[:, 7:10] == 0).all())
    assert bool((market[:, 2:7] > 0).any())                       # unlocked heads still sample
    lp2, ent2, _ = m.evaluate(aug, um, tm, acts, vtok, market_lock=lock)
    assert torch.allclose(lp, lp2, atol=1e-4) and torch.allclose(ent, ent2, atol=1e-4)
