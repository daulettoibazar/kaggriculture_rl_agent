"""Raw program turns -> macro rows: in-place ops label their tile, walks get the look-ahead label of
the next op, market orders map to head bins, and a real program's labels are mostly legal."""
import numpy as np
import pytest
import torch

from src.fs1 import constants as C
from src.fs1.imitation_labels import LabelBuffer, market_bins, sell_bucket, qty_bucket, unit_verb

pytestmark = pytest.mark.kaggri


def _state(farmer=(4, 4), hands=(), shed=None, day=0):
    return {"day": day, "farms": [{"farmer": list(farmer), "hands": [list(h) for h in hands]}, {}],
            "privates": [{"shed": shed or {}}, {}]}


def test_unit_verbs_and_market_bins():
    assert unit_verb(["WATER"]) == 1 and unit_verb(["PLANT", "MELON"]) == 14 and unit_verb(["PASS"]) == 0
    assert unit_verb(["PICKUP", "COW", 1]) == 24 and unit_verb(["PICKUP", "WHEAT", 8]) == 19
    assert unit_verb(["PLACE", "SHEEP"]) == 17 and unit_verb(["PLACE", "MILK", 3]) == 34 and unit_verb(["DROP"]) == 37
    assert unit_verb(["NORTH"]) is None and unit_verb(["PICKUP", "CARROT", 2]) is None
    assert qty_bucket(5) in (2, 3) and qty_bucket(8) == 4 and qty_bucket(1) == 1
    assert sell_bucket(10, 10) == 4 and sell_bucket(5, 10) == 2 and sell_bucket(3, 10) == 1 and sell_bucket(0, 10) == 0
    bins = market_bins([["HIRE"], ["HIRE"], ["HIRE"], ["BUY_LAND"], ["BUY_SEED", "WHEAT", 8], ["BUY_ANIMAL", "COW", 1],
                        ["BUY_PRODUCT", "FERTILIZER", 2], ["SELL", "MELON", 7], ["SELL", "BOGUS", 3]],
                       {"MELON": 7})
    assert bins[0] == 3 and bins[1] == 1 and bins[2] == 4 and bins[8] == 1 and bins[11] == 2 and bins[12 + 4] == 4
    assert bins.sum() == 3 + 1 + 4 + 1 + 2 + 4


def test_lookahead_labels_a_walk_with_its_destination_op():
    buf = LabelBuffer()
    # farmer walks east twice then waters at (6, 4); hand 0 waters in place at (1, 1) on step 0
    buf.observe(0, _state(farmer=(4, 4), hands=[(1, 1)]), 0, {"farmer": ["EAST"], "hands": [["WATER"]], "market": []})
    buf.observe(1, _state(farmer=(5, 4), hands=[(1, 1)]), 0, {"farmer": ["EAST"], "hands": [["PASS"]], "market": [["HIRE"]]})
    buf.observe(2, _state(farmer=(6, 4), hands=[(1, 1)]), 0, {"farmer": ["WATER"], "hands": [["NORTH"]], "market": []})
    rows = {t: (r, u, m) for t, r, u, m in buf.drain()}
    assert set(rows) == {0, 1, 2}
    for t in (0, 1, 2):                                  # the whole walk is labelled (tile 46, WATER)
        r, u, m = rows[t]
        assert u[0] and r[0] == 4 * 10 + 6 and r[C.MAX_UNITS] == 1
    assert rows[0][1][1] and rows[0][0][1] == 11 and rows[0][0][C.MAX_UNITS + 1] == 1      # hand watered in place
    assert rows[1][1][1] and rows[1][0][C.MAX_UNITS + 1] == 0                              # PASS -> IDLE label
    assert not rows[2][1][1]                                                               # hand's walk still open
    assert rows[1][0][2 * C.MAX_UNITS] == 1 and rows[0][2].all()                          # hire bin, all heads labelled
    assert not buf.drain()                                                                 # nothing new
    # a day boundary drops a hand's unfinished walk but keeps the farmer's
    buf.observe(3, _state(farmer=(6, 4), hands=[(1, 2)], day=1), 0, {"farmer": ["EAST"], "hands": [["HARVEST"]], "market": []})
    buf.observe(4, _state(farmer=(7, 4), hands=[(1, 2)], day=1), 0, {"farmer": ["HARVEST"], "hands": [["PASS"]], "market": []})
    rows = {t: (r, u, m) for t, r, u, m in buf.drain()}
    assert rows[3][1][0] and rows[3][0][C.MAX_UNITS] == 2 and rows[3][0][0] == 47
    assert 2 not in rows                                                                   # the old walk was not relabelled


def _abo_v57_missing():
    from src.rl3 import opponents as O
    return not O.path_of("abo_v57").is_file()


@pytest.mark.skipif(_abo_v57_missing(), reason="public kernel abo_v57 not pulled; see README")
def test_real_program_labels_are_mostly_legal(tmp_path, monkeypatch):
    from src.fs1.env import Fs1VecEnv
    from src.fs1.teacher_prefix import TeacherPrefix

    monkeypatch.chdir(tmp_path)
    env = Fs1VecEnv(1, [77], "cpu", hist_k=4, episode_steps=C.EPISODE_STEPS, stagger=False)
    teach = TeacherPrefix(["abo_v57"], [0], timeout=5.0, seed=0)
    teach.start_all(env.env, 120)
    teach.prefix[0] = 120
    teach.new_rollout()
    zeros = torch.zeros(1, 2, C.PLAYER_ACTION_DIM, dtype=torch.int64)
    masks = {}
    legal = total = 0
    labelled_rows = 0
    market_nonzero = 0
    for t in range(120):
        obs, um, tm = env.flat()
        masks[t] = (um[0].clone(), tm[0].clone())
        rows = [[None, None]]
        teach.fill(rows, env.env, np.asarray(env.env.env_steps()), t)
        for (lt, e, s, row, ulab, mlab) in teach.drain_labels():
            if s != 0:
                continue
            um0, tm0 = masks[lt]
            for u in range(C.MAX_UNITS):
                if not ulab[u]:
                    continue
                v, tgt = int(row[C.MAX_UNITS + u]), int(row[u])
                ok = bool(um0[u, v]) if v == 0 else bool(um0[u, v] and tm0[tgt, v])
                legal += int(ok)
                total += 1
            labelled_rows += 1
            market_nonzero += int((row[2 * C.MAX_UNITS:] > 0).sum())
        env.step(zeros, rows)
    assert total > 100, total
    assert legal / total > 0.7, (legal, total)              # walks labelled with a plausible destination op
    assert market_nonzero > 20                               # the program hires, buys seeds, sells
