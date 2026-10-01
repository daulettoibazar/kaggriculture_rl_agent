import numpy as np

from src.fs1 import constants as C
from src.fs1.q4 import LAND_HEAD, STEPS_PER_DAY, Q4Intervention


def test_q4_forces_once_per_episode_only_when_legal_and_in_window():
    q = Q4Intervention(n_envs=4, frac=1.0, day_lo=2, day_hi=2, patience=1, seed=1)
    d = STEPS_PER_DAY
    # episode start: every env draws day 2
    assert len(q.select(np.zeros(4, int), np.full(4, 3), np.full(4, 9000.0))) == 0
    # day 1: not yet
    assert len(q.select(np.full(4, d), np.full(4, 3), np.full(4, 9000.0))) == 0
    # day 2: env 0 legal, env 1 has 2 quadrants, env 2 is broke, env 3 already has 4
    idx = q.select(np.full(4, 2 * d), np.array([3, 2, 3, 4]), np.array([9000.0, 9000.0, 100.0, 9000.0]))
    assert idx.tolist() == [0]
    # env 0 is done for this episode; env 2 fires later once it has the cash, within patience
    idx = q.select(np.full(4, 2 * d + 5), np.array([4, 3, 3, 4]), np.array([9000.0, 9000.0, 5000.0, 9000.0]))
    assert idx.tolist() == [1, 2]
    # past patience nothing fires
    assert len(q.select(np.full(4, 5 * d), np.full(4, 3), np.full(4, 9000.0))) == 0
    assert q.forced == 3
    assert LAND_HEAD == 2 * C.MAX_UNITS + 1


def test_q4_frac_zero_never_fires():
    q = Q4Intervention(n_envs=8, frac=0.0, seed=3)
    assert len(q.select(np.zeros(8, int), np.full(8, 3), np.full(8, 9000.0))) == 0
    assert len(q.select(np.full(8, 10 * STEPS_PER_DAY), np.full(8, 3), np.full(8, 9000.0))) == 0
