"""Program-taught prefixes: both seats of a teacher env follow the program for exactly the drawn prefix,
other envs are untouched, prefix flags line up, and a new episode redraws."""
import pathlib

import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.teacher_prefix import TeacherPrefix

_CHA22X = pathlib.Path(__file__).resolve().parents[1] / "dist/tape_v21_cha22x/policy.py"
needs_cha22x = pytest.mark.skipif(not _CHA22X.exists(), reason="cha22 agent not vendored; see README")

pytestmark = [pytest.mark.kaggri, needs_cha22x]


def test_teacher_drives_both_seats_for_the_prefix_only(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv

    monkeypatch.chdir(tmp_path)
    steps = 60
    env = Fs1VecEnv(2, [41, 42], "cpu", hist_k=4, episode_steps=steps, stagger=False)
    teach = TeacherPrefix(["cha22x_bundle"], [0], timeout=5.0, seed=3)
    teach.start_all(env.env, 30)
    teach.prefix[0] = 30                                                # pin the drawn length for the test
    assert teach.prefix_len(2).tolist() == [30, 0]
    zeros = torch.zeros(2, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
    driven = []
    for t in range(steps - 1):                                          # an episode has episode_steps - 1 decisions
        rows = [[None, None] for _ in range(2)]
        driven.append(teach.fill(rows, env.env, np.asarray(env.env.env_steps())))
        if t < 30:
            assert isinstance(rows[0][0], dict) and isinstance(rows[0][1], dict) and rows[1] == [None, None]
        else:
            assert rows == [[None, None], [None, None]], (t, np.asarray(env.env.env_steps()).tolist(), teach.prefix.tolist())
        if t == 29:
            money_after_prefix = env.money[0].clone()
        done, fm, _ = env.step(zeros, rows)
        teach.on_done(done.numpy(), 30)
    assert bool(done[0])                                                # the episode ended on the last decision
    assert driven[:30] == [1] * 30 and driven[30:] == [0] * (steps - 31)
    assert float(money_after_prefix[0]) != C.STARTING_MONEY and float(money_after_prefix[1]) != C.STARTING_MONEY
    assert float(fm[1, 0]) == C.STARTING_MONEY                          # the other env only ever passed
    s = teach.summary()
    assert s["teacher_episodes"] == 1 and s["teacher_fallbacks"] == 0
    assert 0 <= teach.prefix[0] <= 30                                   # redrawn for the new episode


def test_staggered_env_gets_no_prefix_in_its_first_episode(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv

    monkeypatch.chdir(tmp_path)
    env = None
    for seed in range(1, 40):
        cand = Fs1VecEnv(1, [51], "cpu", hist_k=4, episode_steps=30, stagger=True, stagger_seed=seed)
        if int(cand.env.env_steps()[0]) > 0:
            env = cand
            break
    assert env is not None
    teach = TeacherPrefix(["cha22x_bundle"], [0], timeout=5.0, seed=0)
    teach.start_all(env.env, 20)
    assert teach.prefix[0] == 0
