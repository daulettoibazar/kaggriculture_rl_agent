"""fs1 reward and env wrapper: constants match kaggri, the potential telescopes, validity and
history bookkeeping, Encoder tile-mask parity with the vec-env plane."""
import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.reward import Potential, SCALE, alpha_schedule, shaped_reward

pytestmark = pytest.mark.kaggri


def test_constants_match_kaggri():
    import src.evo  # noqa: F401
    import kaggri

    assert kaggri.OBS_DIM == C.OBS_DIM and kaggri.MASK_DIM == C.MASK_DIM
    assert kaggri.TILE_MASK_DIM == C.TILE_MASK_DIM and kaggri.N_TOKENS == C.N_TOKENS
    assert kaggri.N_FEATURES == C.N_FEATURES and kaggri.MAX_UNITS == C.MAX_UNITS
    assert kaggri.N_TARGETS == C.N_TARGETS and kaggri.N_VERBS == C.N_VERBS
    assert tuple(kaggri.MARKET_HEAD_SIZES) == C.MARKET_HEAD_SIZES
    assert tuple(kaggri.QTY_BUCKETS) == C.QTY_BUCKETS and tuple(kaggri.SELL_FRACS) == C.SELL_FRACS
    assert tuple(kaggri.PRODUCTS) == C.PRODUCTS and tuple(kaggri.CROPS) == C.CROPS
    assert tuple(kaggri.ANIMALS) == C.ANIMALS
    assert [v.upper() for v in kaggri.VERB_NAMES[:3]] == ["IDLE", "WATER", "HARVEST"]
    cfg = kaggri.default_config()
    assert cfg["startingMoney"] == C.STARTING_MONEY and cfg["shedCapacity"] == C.SHED_CAPACITY
    assert cfg["episodeSteps"] == C.EPISODE_STEPS and cfg["turnsPerDay"] == C.TURNS_PER_DAY
    mp = cfg["marketParams"]
    for i, name in enumerate(C.PRODUCTS):
        assert mp[name]["base"] == C.BASE_PRICE[i], name


def test_potential_of_the_initial_state_is_the_starting_money():
    import src.evo  # noqa: F401
    import kaggri

    v = kaggri.PyVecEnv(4, None, [1, 2, 3, 4], obs_dtype="float32", tile_mask=True)
    obs, _ = v.reset()
    o = torch.from_numpy(np.asarray(obs)).view(4, 2, C.N_TOKENS, C.N_FEATURES)
    money = torch.from_numpy(np.asarray(v.money))
    phi = Potential("cpu")(o, money)
    assert torch.allclose(phi, torch.full((4, 2), C.STARTING_MONEY)), phi


def test_buying_seeds_is_neutral_for_the_potential():
    import src.evo  # noqa: F401
    import kaggri

    v = kaggri.PyVecEnv(1, None, [7], obs_dtype="float32", tile_mask=True)
    obs, _ = v.reset()
    pot = Potential("cpu")
    o = torch.from_numpy(np.asarray(obs)).view(1, 2, C.N_TOKENS, C.N_FEATURES)
    phi0 = pot(o, torch.from_numpy(np.asarray(v.money)))
    a = np.zeros((1, 2, v.player_action_dim), np.int64)
    a[0, 0, 2 * C.MAX_UNITS + 2] = 3          # seat 0 buys QTY_BUCKETS[3] = 4 wheat seeds ($10 each)
    obs, mask, rew, done, info = v.step(a)
    o = torch.from_numpy(np.asarray(obs)).view(1, 2, C.N_TOKENS, C.N_FEATURES)
    phi1 = pot(o, torch.from_numpy(np.asarray(info["money"])))
    d = (phi1 - phi0)[0]
    # seat 0: -$40 money, +$40 seed value => $0 (no held goods, so no market drift)
    assert abs(float(d[0])) < 1e-3, d
    assert abs(float(d[1])) < 1e-3, d


def test_shaped_reward_telescopes_and_flips_sign_between_seats():
    phi_prev = torch.tensor([[3000.0, 3000.0]])
    phi_next = torch.tensor([[3500.0, 3200.0]])
    fm = torch.zeros(1, 2)
    done = torch.tensor([False])
    r = shaped_reward(phi_prev, phi_next, fm, done, alpha=1.0)
    assert torch.allclose(r, torch.tensor([[300.0, -300.0]]) / SCALE)
    r0 = shaped_reward(phi_prev, phi_next, fm, done, alpha=0.0)
    assert torch.allclose(r0, torch.tensor([[500.0, 200.0]]) / SCALE)
    # terminal: next potential is money only, plus the win term
    fm = torch.tensor([[4000.0, 3900.0]])
    rt = shaped_reward(phi_prev, phi_next, fm, torch.tensor([True]), alpha=1.0, w_win=1.0)
    # dphi = (4000-3000, 3900-3000) = (1000, 900): margin 100 for seat 0, plus the win term
    assert torch.allclose(rt, torch.tensor([[100.0 / SCALE + 1.0, -100.0 / SCALE - 1.0]]))
    tie = shaped_reward(phi_prev, phi_next, torch.tensor([[4000.0, 4000.0]]), torch.tensor([True]), 1.0)
    assert torch.allclose(tie, torch.zeros(1, 2))
    # alpha = 0 at the terminal: each seat is credited its own full gain, no win term difference
    own = shaped_reward(phi_prev, phi_next, fm, torch.tensor([True]), alpha=0.0, w_win=0.0)
    assert torch.allclose(own, torch.tensor([[1000.0, 900.0]]) / SCALE)
    assert alpha_schedule(0, 100) == 0.0 and alpha_schedule(30, 100) == 1.0 and 0.4 < alpha_schedule(15, 100) < 0.6


def test_env_wrapper_validity_history_and_pairing():
    from src.fs1.env import Fs1VecEnv

    n, steps = 6, 48
    env = Fs1VecEnv(n, list(range(n)), "cpu", hist_k=3, episode_steps=steps, stagger=True, stagger_seed=3)
    assert env.obs_aug.shape == (n, 2, C.N_TOKENS, C.N_FEATURES + 3) and env.obs_aug.dtype == torch.float16
    assert env.unit_mask.shape == (n, 2, C.MAX_UNITS, C.N_VERBS) and env.tile_mask.shape == (n, 2, 100, 38)
    assert (env.offsets > 0).any() and bool((env.invalid.cpu().numpy() == (env.offsets > 0)).all())
    assert torch.allclose(env.phi, torch.full((n, 2), C.STARTING_MONEY))   # nobody acted during warmup
    obs, um, tm = env.flat()
    assert obs.shape[0] == 2 * n and bool((Fs1VecEnv.pair_index(n, "cpu")[::2] == torch.arange(1, 2 * n, 2)).all())

    seen_done = torch.zeros(n, dtype=torch.bool)
    for _ in range(steps + 2):
        was_invalid = env.invalid.clone()
        a = torch.zeros(n, 2, C.PLAYER_ACTION_DIM, dtype=torch.long)
        done, final_money, valid = env.step(a)
        assert torch.equal(valid, ~was_invalid)
        # after a terminal the env is valid, and the history restarts at zero
        if bool(done.any()):
            assert bool((~env.invalid[done]).all())
            assert bool((env.hist[done] == 0).all())
            assert bool((final_money[done] > 0).all())
        seen_done |= done
        # the newest history column is the price delta; on non-done envs it is bounded
        assert torch.isfinite(env.hist).all()
    assert bool(seen_done.all())


def test_encoder_tile_mask_matches_the_vecenv_plane():
    import src.evo  # noqa: F401
    import kaggri
    from src.evo.kernel_opponents import observation_of

    n = 5
    v = kaggri.PyVecEnv(n, None, list(range(10, 10 + n)), obs_dtype="float32", tile_mask=True)
    v.reset()
    rng = np.random.default_rng(5)
    for _ in range(40):
        a = np.zeros((n, 2, v.player_action_dim), np.int64)
        a[:, :, :C.MAX_UNITS] = rng.integers(0, 100, (n, 2, C.MAX_UNITS))
        a[:, :, C.MAX_UNITS:2 * C.MAX_UNITS] = rng.integers(0, C.N_VERBS, (n, 2, C.MAX_UNITS))
        a[:, :, 2 * C.MAX_UNITS + 2:2 * C.MAX_UNITS + 7] = rng.integers(0, 3, (n, 2, 5))
        obs, mask, rew, done, info = v.step(a)
    enc = kaggri.Encoder(v.config() if hasattr(v, "config") else None)
    plane = np.asarray(v.tile_mask).reshape(n, 2, C.TILE_MASK_DIM)
    unit_plane = np.asarray(mask).reshape(n, 2, C.MASK_DIM)
    obs_plane = np.asarray(obs).reshape(n, 2, C.OBS_DIM)
    checked = 0
    for i in range(n):
        sd = v.state_dict(i)
        for p in range(2):
            kobs = observation_of(sd, p, struct=False)
            o, m, tmask = enc.encode_full(kobs)
            assert np.array_equal(np.asarray(tmask), plane[i, p]), (i, p)
            assert np.allclose(np.asarray(m, dtype=np.float32), unit_plane[i, p])
            assert np.allclose(np.asarray(o, dtype=np.float32), obs_plane[i, p], atol=2e-3)
            checked += 1
    assert checked == 2 * n
