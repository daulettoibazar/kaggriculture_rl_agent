"""Achievement bonus: a one-time reward the first time in an episode a seat reaches a milestone.

Potential-based shaping only pays a chain once its later links are within reach, so a mechanic
with a long delay before any payoff never gets its first step tried. Pilot 2 (2026-09-22) reached
a $15K wheat routine and never bought an animal, land or a melon seed; the first self-play run with
the first 15 flat milestones reached $52K and plateaued there (2026-09-23 measurement,
`research/fs1/step1_measure_2026-09-23.md`): 2 quadrants with 43 of 50 tiles planted, ~6 hands,
no animals, no fertilizer, while the strongest public program runs 3 quadrants, 9-11 hands,
14-17 animals and ~60 fertilizer applications and earns 3.5x as much.

Why flat milestones were not enough: land and animals are counted AT COST in the potential, so a
purchase is neutral when made, but an unused quadrant or an unplaced animal is a pure terminal loss
($2,000 / $300-500) against a one-time bonus worth $300; and the old `land` flag fired at two
quadrants, already satisfied by the scripted opener, so quadrant 3 earned nothing. The tiers
below therefore carry WEIGHTS sized to the purchase they must cover (7 x $300 > the $2,000 third
quadrant), every link of the animal chain pays separately (buy -> shed, pick up, place, feed with
carried wheat, care, harvest a product, 4 and 8 animals placed), and land only pays again once it
is planted (55 / 70 tiles). The whole bonus anneals to zero by `ramp_frac` of the run so the final
policy optimises money and wins alone. The per-game counts double as the diagnostic of which
mechanics the policy has actually reached.

All flags are read from the seat's own tokens (`rust/OBS_LAYOUT.md` slots), never the opponent's.
"""
from __future__ import annotations

import torch

from . import constants as C

# (name, weight in units of the base bonus)
MILESTONES = (
    # routine for the resumed policy (fires in >80% of games): a constant the critic absorbs, kept at a
    # third of the base bonus as a diagnostic only (Opus review 4, S2)
    ("hired", 0.33), ("land", 0.33),
    ("plant_wheat", 0.33), ("plant_carrot", 0.33), ("plant_tomato", 0.33), ("plant_strawberry", 0.33),
    ("plant_melon", 0.33), ("built_structure", 0.33),
    ("melon_in_shed", 0.33), ("strawberry_in_shed", 0.33), ("tomato_in_shed", 0.33),
    ("hands_8", 0.33), ("hands_10", 0.33), ("fertilizer_held", 0.33),
    # scale: land and planted area, with a reachable first tier (2 quadrants hold at most ~45 planted)
    ("quadrant3", 7.0), ("quadrant4", 14.0),
    ("planted_45", 3.0), ("planted_55", 4.0), ("planted_70", 4.0),
    # the animal chain, one link at a time, then herd size
    ("animal_bought", 2.0), ("animal_carried", 1.0), ("animal_placed", 3.0), ("animal_fed", 2.0),
    ("animal_cared", 1.0), ("animal_product", 4.0), ("animals_2", 3.0), ("animals_4", 4.0), ("animals_8", 6.0),
    # fertilizer applied, with a reachable middle tier
    ("fertilized", 1.0), ("fertilized_3", 2.0), ("fertilized_10", 3.0),
)
NAMES = tuple(m[0] for m in MILESTONES)
WEIGHTS = tuple(m[1] for m in MILESTONES)
N = len(NAMES)

# payload slots (rust/OBS_LAYOUT.md; the chain-mechanics spec of 2026-09-23)
G_LIVE_HANDS, G_UNLOCKED = 14, 15
V_ANIMALS_SHED0 = C.V_ANIMALS0                      # 5..7 bought, unplaced animals in the shed, /8
V_SHED0 = C.V_SHED0                                 # 11..19 products in the shed, /shedCapacity
V_PLANTS0, V_ANIMALS_PLACED0, V_EMPTY_COOPS, V_EMPTY_PASTURES = 20, 25, 30, 31   # placed /100
T_FERT_DAYS, T_CARED = 28, 33
U_CARRY0 = C.U_CARRY0                               # 6..17 carried per SHED_ITEM, /32
P_EGG, P_MILK, P_WOOL, P_MELON, P_STRAWBERRY, P_TOMATO, P_FERTILIZER = 5, 6, 7, 4, 3, 2, 8
I_GOOSE = 9                                         # SHED_ITEMS index of the first animal


def weights_tensor(device) -> torch.Tensor:
    return torch.tensor(WEIGHTS, dtype=torch.float32, device=device)


def achievement_flags(obs: torch.Tensor) -> torch.Tensor:
    """obs [..., 140, 48] float -> bool [..., N]: which milestones hold in this state."""
    H = C.HDR
    o = obs.float()
    glob = o[..., C.TOK_GLOBAL, :]
    priv = o[..., C.TOK_PRIVATE, :]
    tiles = o[..., C.TOK_TILE0:C.TOK_TILE1, :]
    units = o[..., C.TOK_UNIT0:C.TOK_UNIT1, :]
    eps = 1e-6
    plants = priv[..., H + V_PLANTS0:H + V_PLANTS0 + 5]                                     # [..., 5]
    placed = priv[..., H + V_ANIMALS_PLACED0:H + V_ANIMALS_PLACED0 + 3].sum(-1) * 100.0    # animals placed
    in_shed_animals = priv[..., H + V_ANIMALS_SHED0:H + V_ANIMALS_SHED0 + 3].sum(-1) * 8.0
    shed = priv[..., H + V_SHED0:H + V_SHED0 + 9]                                           # [..., 9]
    carry = units[..., H + U_CARRY0:H + U_CARRY0 + C.N_SHED_ITEMS].sum(-2) * C.U_CARRY_SCALE  # [..., 12]
    kind = tiles[..., H + C.T_KIND0:H + C.T_KIND0 + 7]
    is_plant = kind[..., C.T_KIND_PLANT] > 0.5
    is_animal_tile = kind[..., C.T_KIND_STRUCT] > 0.5                                       # occupied coop/pasture
    n_planted = is_plant.sum(-1)
    fed = (is_animal_tile & (tiles[..., H + C.T_WATERED] > 0.5)).any(-1)                   # fed_today shares slot 25
    cared = (is_animal_tile & (tiles[..., H + T_CARED] > 0.5)).any(-1)
    fert_days = tiles[..., H + T_FERT_DAYS] > eps
    quadrants = glob[..., H + G_UNLOCKED] * 4.0
    hands = glob[..., H + G_LIVE_HANDS] * 16.0
    products_animal = (shed[..., P_EGG] + shed[..., P_MILK] + shed[..., P_WOOL]
                       + carry[..., P_EGG] + carry[..., P_MILK] + carry[..., P_WOOL]) > eps
    n_fert = fert_days.sum(-1)
    flags = [
        hands > eps,
        quadrants > 1.5,
        plants[..., 0] > eps, plants[..., 1] > eps, plants[..., 2] > eps, plants[..., 3] > eps, plants[..., 4] > eps,
        (priv[..., H + V_EMPTY_COOPS] + priv[..., H + V_EMPTY_PASTURES] > eps) | (placed > eps),
        shed[..., P_MELON] > eps,
        shed[..., P_STRAWBERRY] > eps,
        shed[..., P_TOMATO] > eps,
        hands > 7.5,
        hands > 9.5,
        (shed[..., P_FERTILIZER] + carry[..., P_FERTILIZER]) > eps,
        quadrants > 2.5,
        quadrants > 3.5,
        n_planted >= 45,
        n_planted >= 55,
        n_planted >= 70,
        in_shed_animals > eps,
        carry[..., I_GOOSE:I_GOOSE + 3].sum(-1) > eps,
        placed > eps,
        fed,
        cared,
        products_animal,
        placed > 1.5,
        placed > 3.5,
        placed > 7.5,
        fert_days.any(-1),
        n_fert >= 3,
        n_fert >= 10,
    ]
    assert len(flags) == N
    return torch.stack(flags, dim=-1)


class AchievementTracker:
    """Per-seat first-time milestone detection across auto-resetting episodes."""

    def __init__(self, n_envs: int, device):
        self.achieved = torch.zeros(n_envs, 2, N, dtype=torch.bool, device=device)

    def step(self, obs_next: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """obs_next [n, 2, 140, 48] is the state after the transition (a fresh episode where done);
        returns the NEW milestones reached by this transition, bool [n, 2, N].

        On a done step the terminal state is not observable (the env already reset), so no
        milestone is credited there and the tracker restarts for the new episode."""
        flags = achievement_flags(obs_next)
        new = flags & ~self.achieved
        d = done[:, None, None]
        new = torch.where(d, torch.zeros_like(new), new)
        self.achieved = torch.where(d, torch.zeros_like(flags), self.achieved | flags)
        return new

    def reset_episode_flags(self, obs: torch.Tensor):
        """Mark what already holds at an episode start (nothing at the default start state)."""
        self.achieved |= achievement_flags(obs)


def bonus_at(progress: float, bonus: float, ramp_frac: float) -> float:
    """Per-milestone base bonus: `bonus` -> 0 linearly over the first `ramp_frac` of the run.
    A milestone pays `bonus_at(...) * WEIGHTS[i]`."""
    if ramp_frac <= 0:
        return 0.0
    return float(bonus * max(0.0, 1.0 - progress / ramp_frac))
