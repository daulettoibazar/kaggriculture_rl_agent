"""Public programs as seat-1 opponents: the exact-turn path drives only the public block's seat 1,
finished games are scored and the program instance is renewed per episode."""
import pathlib

import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.public_opponents import PublicOpponents

_CHA22X = pathlib.Path(__file__).resolve().parents[1] / "dist/tape_v21_cha22x/policy.py"
needs_cha22x = pytest.mark.skipif(not _CHA22X.exists(), reason="cha22 agent not vendored; see README")

pytestmark = [pytest.mark.kaggri, needs_cha22x]


def test_public_program_drives_only_seat_one_of_its_env(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv

    monkeypatch.chdir(tmp_path)                       # public programs write into the CWD
    env = Fs1VecEnv(2, [11, 12], "cpu", hist_k=4, episode_steps=C.EPISODE_STEPS, stagger=False)
    pub = PublicOpponents(["cha22x_bundle"], 1, 1, timeout=5.0, seed=0)
    zeros = torch.zeros(2, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
    for _ in range(48):                               # two days
        exact = pub.exact_actions(env.env, 2)
        assert exact[0] == [None, None] and exact[1][0] is None and isinstance(exact[1][1], dict)
        done, fm, _ = env.step(zeros, exact)
        pub.on_done(done.numpy(), fm.numpy())
    money = env.money
    assert float(money[0, 0]) == C.STARTING_MONEY and float(money[0, 1]) == C.STARTING_MONEY   # both PASS
    assert float(money[1, 0]) == C.STARTING_MONEY                                            # learner PASSes
    assert float(money[1, 1]) != C.STARTING_MONEY                                            # the program acts
    assert pub.policies[0].stats.fallbacks == 0
    s = pub.summary()
    assert set(s) == {"win_vs_public", "win_vs_public_ema", "public_games", "public_fallbacks"}
    assert s["public_games"]["cha22x_bundle"] == 0


def test_public_episode_end_is_scored_and_program_renewed(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv

    monkeypatch.chdir(tmp_path)
    steps = 30
    env = Fs1VecEnv(1, [21], "cpu", hist_k=4, episode_steps=steps, stagger=False)
    pub = PublicOpponents(["cha22x_bundle"], 1, 0, timeout=5.0, seed=1)
    first = pub.policies[0]
    zeros = torch.zeros(1, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
    finished = 0
    for _ in range(steps + 2):
        done, fm, _ = env.step(zeros, pub.exact_actions(env.env, 1))
        pub.on_done(done.numpy(), fm.numpy())
        finished += int(done[0])
    assert finished == 1
    s = pub.summary()
    assert s["public_games"]["cha22x_bundle"] == 1.0 and s["win_vs_public"]["cha22x_bundle"] in (0.0, 0.5, 1.0)
    assert pub.policies[0] is not first                                                      # fresh instance
    tot = pub.totals("cpu")
    assert tot.shape == (3, 1) and float(tot[0, 0]) == 1.0


def test_first_partial_episode_of_a_staggered_env_is_not_scored(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv

    monkeypatch.chdir(tmp_path)
    steps = 30
    # stagger=True starts envs at random phases; find a seed whose single env starts mid-game
    env = None
    for seed in range(1, 40):
        cand = Fs1VecEnv(1, [31], "cpu", hist_k=4, episode_steps=steps, stagger=True, stagger_seed=seed)
        if int(cand.env.env_steps()[0]) > 0:
            env = cand
            break
    assert env is not None
    pub = PublicOpponents(["cha22x_bundle"], 1, 0, timeout=5.0, seed=2)
    zeros = torch.zeros(1, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
    finished = 0
    for _ in range(2 * steps + 2):
        done, fm, _ = env.step(zeros, pub.exact_actions(env.env, 1))
        pub.on_done(done.numpy(), fm.numpy())
        finished += int(done[0])
    assert finished == 2                                                # the partial one and one full one
    assert pub.summary()["public_games"]["cha22x_bundle"] == 1.0                  # only the full episode counted
