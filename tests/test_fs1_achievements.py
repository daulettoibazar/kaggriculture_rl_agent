"""Achievement flags: none at the start state, the opener reaches several over a season, each pays
once per episode, the tracker resets on done, and the tiered flags read the documented slots."""
import pytest
import torch

from src.fs1 import achievements as A
from src.fs1 import constants as C
from src.fs1.opener import opener_actions

pytestmark = pytest.mark.kaggri


def test_names_weights_consistent():
    assert len(A.NAMES) == len(A.WEIGHTS) == A.N == len(set(A.NAMES))
    w = dict(zip(A.NAMES, A.WEIGHTS))
    # the third quadrant's bonus must cover its $2,000 price at the $300 base bonus, the fourth its $4,000
    assert w["quadrant3"] * 300 >= 2000 and w["quadrant4"] * 300 >= 4000
    assert all(x > 0 for x in A.WEIGHTS)


def test_no_achievement_at_the_start_state_and_opener_reaches_several():
    from src.fs1.env import Fs1VecEnv

    n = 2
    env = Fs1VecEnv(n, [301, 302], "cpu", hist_k=4, episode_steps=C.EPISODE_STEPS, stagger=False)
    obs48 = env.obs_aug[..., :C.N_FEATURES]
    assert not bool(A.achievement_flags(obs48).any())
    tracker = A.AchievementTracker(n, "cpu")
    total = torch.zeros(n, 2, A.N)
    for _ in range(C.EPISODE_STEPS):
        obs, um, tm = env.flat()
        rows = opener_actions(obs, um, tm)
        done, fm, _ = env.step(rows.view(n, 2, -1))
        new = tracker.step(env.obs_aug[..., :C.N_FEATURES], done)
        total += new.float()
        if bool(done.all()):
            break
    # each milestone is credited at most once per seat per episode
    assert float(total.max()) <= 1.0
    reached = {A.NAMES[i] for i in range(A.N) if bool(total[..., i].all())}
    for name in ("hired", "land", "plant_wheat", "plant_melon", "melon_in_shed"):
        assert name in reached, (name, reached)
    # the opener keeps 2 quadrants and <= 6 hands and never keeps animals
    for name in ("quadrant3", "quadrant4", "hands_8", "animal_bought", "animal_placed", "animal_fed",
                 "animal_product", "fertilized"):
        assert name not in reached, (name, reached)
    # after the terminal step the tracker has restarted: nothing is marked achieved
    assert not bool(tracker.achieved.any())


def _blank_obs():
    return torch.zeros(1, C.N_TOKENS, C.N_FEATURES)


def _flags(obs):
    f = A.achievement_flags(obs)[0]
    return {A.NAMES[i] for i in range(A.N) if bool(f[i])}


def test_tiered_flags_read_the_documented_slots():
    H = C.HDR
    o = _blank_obs()
    assert _flags(o) == set()
    # quadrants: slot 15 of the GLOBAL token, /4
    o[0, C.TOK_GLOBAL, H + A.G_UNLOCKED] = 3 / 4
    assert {"land", "quadrant3"} <= _flags(o) and "quadrant4" not in _flags(o)
    o[0, C.TOK_GLOBAL, H + A.G_UNLOCKED] = 1.0
    assert "quadrant4" in _flags(o)
    # crew: slot 14, /16
    o[0, C.TOK_GLOBAL, H + A.G_LIVE_HANDS] = 8 / 16
    f = _flags(o)
    assert {"hired", "hands_8"} <= f and "hands_10" not in f
    o[0, C.TOK_GLOBAL, H + A.G_LIVE_HANDS] = 10 / 16
    assert "hands_10" in _flags(o)
    # planted area: tile kind one-hot
    o[0, C.TOK_TILE0:C.TOK_TILE0 + 60, H + C.T_KIND0 + C.T_KIND_PLANT] = 1.0
    f = _flags(o)
    assert {"planted_45", "planted_55"} <= f and "planted_70" not in f
    o[0, C.TOK_TILE0:C.TOK_TILE0 + 72, H + C.T_KIND0 + C.T_KIND_PLANT] = 1.0
    assert "planted_70" in _flags(o)
    # animal chain: bought (private 5..7, /8), carried (unit slots 6+9.., /32), placed (private 25..27, /100)
    o = _blank_obs()
    o[0, C.TOK_PRIVATE, H + A.V_ANIMALS_SHED0 + 1] = 1 / 8
    assert _flags(o) == {"animal_bought"}
    o[0, C.TOK_UNIT0 + 2, H + A.U_CARRY0 + A.I_GOOSE] = 1 / 32
    assert _flags(o) == {"animal_bought", "animal_carried"}
    o[0, C.TOK_PRIVATE, H + A.V_ANIMALS_PLACED0] = 4 / 100
    f = _flags(o)
    assert {"animal_placed", "animals_2", "animals_4", "built_structure"} <= f and "animals_8" not in f
    o[0, C.TOK_PRIVATE, H + A.V_ANIMALS_PLACED0 + 2] = 4 / 100
    assert "animals_8" in _flags(o)
    # fed / cared today on an occupied structure tile
    t = C.TOK_TILE0 + 5
    o[0, t, H + C.T_KIND0 + C.T_KIND_STRUCT] = 1.0
    assert {"animal_fed", "animal_cared"}.isdisjoint(_flags(o))
    o[0, t, H + C.T_WATERED] = 1.0
    o[0, t, H + A.T_CARED] = 1.0
    assert {"animal_fed", "animal_cared"} <= _flags(o)
    # products: carried milk counts as much as milk in the shed
    o[0, C.TOK_UNIT0, H + A.U_CARRY0 + A.P_MILK] = 1 / 32
    assert "animal_product" in _flags(o)
    # fertilizer: held (shed slot 11+8 or carried 6+8), applied (tile slot 28), ten tiles
    o = _blank_obs()
    o[0, C.TOK_PRIVATE, H + A.V_SHED0 + A.P_FERTILIZER] = 0.01
    assert _flags(o) == {"fertilizer_held"}
    o[0, C.TOK_TILE0:C.TOK_TILE0 + 9, H + A.T_FERT_DAYS] = 0.5
    f = _flags(o)
    assert {"fertilized", "fertilized_3"} <= f and "fertilized_10" not in f
    o[0, C.TOK_TILE0:C.TOK_TILE0 + 10, H + A.T_FERT_DAYS] = 0.5
    assert "fertilized_10" in _flags(o)


def test_bonus_schedule():
    assert A.bonus_at(0.0, 0.03, 0.5) == 0.03
    assert abs(A.bonus_at(0.25, 0.03, 0.5) - 0.015) < 1e-9
    assert A.bonus_at(0.5, 0.03, 0.5) == 0.0
    assert A.bonus_at(0.9, 0.03, 0.5) == 0.0
    assert A.bonus_at(0.0, 0.03, 0.0) == 0.0
    assert A.weights_tensor("cpu").shape == (A.N,)
