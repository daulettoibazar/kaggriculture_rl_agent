"""Potential-based dense reward for fs1, computed from the observation tensors on the GPU.

Phi_p(s) values everything seat p owns at what it is worth right now:

    money
  + shed holdings at their curve-aware dump revenue   held * price * (1 + depth) / 2
  + items carried by units                            products at 0.8 * price, animals at cost
  + seeds in the shed at seed cost
  + planted tiles at seed cost * (1 + 0.6 * progress to first yield) * (0.5 + 0.5 * watered_today)
  + harvestable units on tiles at 0.8 * price
  + placed animals at cost * (0.5 + 0.5 * fed_today), unplaced animals at cost
  + unlocked land at the cumulative quadrant price paid

The terminal potential is money alone, so with gamma = 1 the per-step differences telescope
to exactly (final money - starting money) over an episode: the shaping is unbiased with
respect to the money margin, and every step of the seed -> plant -> water -> harvest ->
deposit -> sell chain still carries credit. Purchases (seeds, animals, land) are valued AT cost
so the first link of the chain is neutral rather than a loss (a 2-GPU check with a 10% haircut
showed the policy learning "never buy" before it learned to farm); waste is punished by the
terminal potential, where unused assets and unsold inventory are worth nothing.

The daily-need factor (watered_today / fed_today, TILE slot 25) is what makes WATER and FEED
carry immediate credit: a plant is worth half until it is watered today, so plant + water is the
neutral atomic unit, watering an unwatered plant is +50% of its value at once, and letting the
day roll over unwatered costs the same 50% before the plant dies as a weed the next night. A
2-GPU check without this factor showed 34 plantings but under 4 waterings per game.

The observation is the seat's own, player-relative token set (`rust/OBS_LAYOUT.md`), read in
f32 before the trainer quantises it for the policy. Money comes from `info["money"]` (exact
f32) rather than the observation's clipped, scaled slot.
"""
from __future__ import annotations

import torch

from . import constants as C

SCALE = 10_000.0  # dollars per reward unit


def _t(vals, device, dtype=torch.float32):
    return torch.tensor(vals, device=device, dtype=dtype)


class Potential:
    """Callable `phi(obs_f32, money) -> dollars`, with the economics tensors cached per device."""

    def __init__(self, device, shed_capacity: float = C.SHED_CAPACITY):
        self.cap = float(shed_capacity)
        self.seed_cost = _t(C.SEED_COST, device)                 # [5]
        self.first_yield = _t(C.FIRST_YIELD_DAY, device)         # [5]
        self.animal_cost = _t(C.ANIMAL_COST, device)             # [3]
        self.land_paid = _t(C.LAND_PAID_CUM, device)             # [4], by unlocked count - 1

    def __call__(self, obs: torch.Tensor, money: torch.Tensor) -> torch.Tensor:
        """obs: [..., 140, 48] float32 (one seat per leading index); money: [...] float32 dollars."""
        H = C.HDR
        tiles = obs[..., C.TOK_TILE0:C.TOK_TILE1, :]
        units = obs[..., C.TOK_UNIT0:C.TOK_UNIT1, :]
        prods = obs[..., C.TOK_PRODUCT0:C.TOK_PRODUCT1, :]
        priv = obs[..., C.TOK_PRIVATE, :]
        glob = obs[..., C.TOK_GLOBAL, :]

        price = prods[..., H + C.P_PRICE_ABS] * C.P_PRICE_ABS_SCALE          # [..., 9]
        depth = prods[..., H + C.P_DEPTH]                                    # [..., 9] in [0, 1]
        held = prods[..., H + C.P_HELD] * self.cap                           # [..., 9]
        inv_value = (held * price * (1.0 + depth) * 0.5).sum(-1)

        carry = units[..., H + C.U_CARRY0:H + C.U_CARRY0 + C.N_SHED_ITEMS] * C.U_CARRY_SCALE  # [..., 17, 12]
        item_value = torch.cat([0.8 * price, self.animal_cost.expand(*price.shape[:-1], 3)], dim=-1)
        carry_value = (carry * item_value.unsqueeze(-2)).sum((-1, -2))

        seeds = priv[..., H + C.V_SEEDS0:H + C.V_SEEDS0 + 5] * C.V_SEED_SCALE
        seed_value = (seeds * self.seed_cost).sum(-1)
        unplaced = priv[..., H + C.V_ANIMALS0:H + C.V_ANIMALS0 + 3] * C.V_ANIMAL_SCALE
        unplaced_value = (unplaced * self.animal_cost).sum(-1)

        is_plant = tiles[..., H + C.T_KIND0 + C.T_KIND_PLANT]                # [..., 100]
        is_struct = tiles[..., H + C.T_KIND0 + C.T_KIND_STRUCT]
        crop_oh = tiles[..., H + C.T_CROP0:H + C.T_CROP0 + 5]                # [..., 100, 5]
        animal_oh = tiles[..., H + C.T_ANIMAL0:H + C.T_ANIMAL0 + 3]          # [..., 100, 3]
        age = tiles[..., H + C.T_AGE] * C.T_AGE_SCALE
        yield_units = tiles[..., H + C.T_YIELD] * C.T_YIELD_SCALE
        need_met = 0.5 + 0.5 * tiles[..., H + C.T_WATERED]                  # watered_today / fed_today

        crop_seed_cost = (crop_oh * self.seed_cost).sum(-1)
        crop_first = (crop_oh * self.first_yield).sum(-1).clamp_min(1.0)
        progress = (age / crop_first).clamp(0.0, 1.0)
        plant_value = is_plant * crop_seed_cost * (1.0 + 0.6 * progress) * need_met

        crop_price = (crop_oh * price[..., :5].unsqueeze(-2)).sum(-1)
        animal_price = (animal_oh * price[..., 5:8].unsqueeze(-2)).sum(-1)
        yield_value = 0.8 * yield_units * (is_plant * crop_price + is_struct * animal_price)
        animal_asset = is_struct * (animal_oh * self.animal_cost).sum(-1) * need_met
        farm_value = (plant_value + yield_value + animal_asset).sum(-1)

        n_unlocked = (glob[..., H + C.G_UNLOCKED_QUADRANTS] * 4.0).round().long().clamp(1, 4) - 1
        land_value = self.land_paid[n_unlocked]

        return money + inv_value + carry_value + seed_value + unplaced_value + farm_value + land_value


def shaped_reward(
    phi_prev: torch.Tensor,
    phi_next: torch.Tensor,
    final_money: torch.Tensor,
    done: torch.Tensor,
    alpha: float,
    w_win: float = 1.0,
) -> torch.Tensor:
    """Per-seat reward for one transition.

    phi_prev, phi_next, final_money: [B, 2] dollars; done: [B] bool (both seats end together).
    On a done step the next potential is the terminal one (money only), i.e. `final_money`.
    r_p = (dphi_p - alpha * dphi_q) / SCALE + done * w_win * sign(final_p - final_q).
    """
    d = done.unsqueeze(-1)
    dphi = torch.where(d, final_money - phi_prev, phi_next - phi_prev)          # [B, 2]
    dphi_opp = dphi.flip(-1)
    dense = (dphi - alpha * dphi_opp) / SCALE
    margin = final_money - final_money.flip(-1)
    win = torch.sign(margin) * w_win
    return dense + torch.where(d, win, torch.zeros_like(win))


def alpha_schedule(update: int, total_updates: int, ramp_frac: float = 0.3) -> float:
    """0 -> 1 linearly over the first `ramp_frac` of the run, then 1."""
    ramp = max(1, int(total_updates * ramp_frac))
    return float(min(1.0, update / ramp))


def alpha_from_progress(progress: float, ramp_frac: float = 0.3) -> float:
    """Same ramp, driven by the run's progress fraction (updates or wall clock, whichever is further)."""
    return float(min(1.0, progress / max(1e-9, ramp_frac)))


def ramp_with_hold(progress: float, start: float, hold: float, end: float) -> float:
    """`start` until progress `hold`, then linear to 1 at progress `end` (1 afterwards).

    start = hold = 0 reproduces `alpha_from_progress` (0 -> 1 over `end`). Iteration 2 (2026-09-23)
    holds the opponent-relative and win terms at 0.3 while the milestone bonus is high: at full weight a
    lost league game costs the equivalent of $10,000, which made every costly experiment (land, animals)
    a losing proposition against the frozen, non-exploring self and the agent retreated from them."""
    start, hold, end = float(start), float(hold), float(end)
    if progress <= hold:
        return min(1.0, start) if hold > 0 else float(start + (1.0 - start) * min(1.0, progress / max(1e-9, end)))
    if end <= hold:
        return 1.0
    return float(start + (1.0 - start) * min(1.0, (progress - hold) / (end - hold)))
