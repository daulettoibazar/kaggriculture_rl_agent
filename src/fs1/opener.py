"""Scripted opener for the fs1 reverse curriculum.

A short rule, written here and nowhere else (no public program involved), that drives BOTH
seats for the first `prefix_len` steps of an episode during early training. The network takes
over mid-season with hands hired, land bought and crops (wheat and melons) in the ground, so it
first learns to water, harvest and sell, then to plant, then to buy and hire, as the prefix
shrinks to zero. Prefix transitions are excluded from training (like the stagger junk episodes).

Pilot 2 (2026-09-22) used a farmer-only wheat version of this rule and plateaued at ~$15K with
no land, no hands to speak of and no high-value crops; this version demonstrates all three.

Every present unit acts with the same priority (water > harvest > deposit > plant), each on the
nearest tile where the verb is plausible; the resolver walks the unit there and arbitrates
same-turn conflicts. Market: hire two hands at the start of each day while the crew is small,
buy the second quadrant when affordable, keep wheat seeds stocked, buy a few melon seeds while
there is season left for them, and offer the shed's crops for sale every turn.

Vectorised over seats with torch; reads only the seat's own observation and masks.
"""
from __future__ import annotations

import torch

from . import constants as C

VERB_WATER, VERB_HARVEST = 1, 2
VERB_PLANT0 = 10                 # Plant(crop) in CROPS order: wheat 10, carrot 11, tomato 12, strawberry 13, melon 14
VERB_PLANT_WHEAT, VERB_PLANT_MELON = 10, 14
VERB_DEPOSIT0 = 28               # Deposit(product) in PRODUCTS order
HEAD_HIRE, HEAD_LAND = 0, 1
HEAD_SEED0 = 2                   # seed_qty[crop], CROPS order
HEAD_SELL0 = 12                  # sell_frac[product], PRODUCTS order
QTY_BUCKET_4, QTY_BUCKET_8 = 3, 4
SELL_ALL = 4
T_X, T_Y, T_SHED_ACCESS = 15, 16, 17
G_HOUR, G_DAY, G_LIVE_HANDS, G_UNLOCKED = 2, 1, 14, 15
BIG = 1e9
MAX_HANDS_TARGET = 6
MELON_LAST_DAY = 18


def opener_actions(obs_aug: torch.Tensor, unit_mask: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
    """obs_aug [B, 140, F] (any float dtype), unit_mask [B, 17, 38] bool, tile_mask [B, 100, 38] bool
    -> action rows [B, 55] int64 (targets | verbs | market)."""
    B = obs_aug.shape[0]
    dev = obs_aug.device
    H = C.HDR
    o = obs_aug.float()
    tiles = o[:, C.TOK_TILE0:C.TOK_TILE1, :]                                     # [B, 100, F]
    units = o[:, C.TOK_UNIT0:C.TOK_UNIT1, :]                                     # [B, 17, F]
    priv = o[:, C.TOK_PRIVATE, :]
    glob = o[:, C.TOK_GLOBAL, :]
    um = unit_mask.bool()
    tm = tile_mask.bool()

    # distance from every unit to every tile
    ux, uy = units[:, :, H + C.U_X] * 9.0, units[:, :, H + C.U_Y] * 9.0          # [B, 17]
    tx, ty = tiles[:, :, H + T_X] * 9.0, tiles[:, :, H + T_Y] * 9.0                # [B, 100]
    dist = (tx[:, None, :] - ux[:, :, None]).abs() + (ty[:, None, :] - uy[:, :, None]).abs()   # [B, 17, 100]

    def nearest(plausible: torch.Tensor) -> torch.Tensor:
        """plausible [B, 17, 100] or [B, 100] -> nearest tile index per unit [B, 17]."""
        if plausible.dim() == 2:
            plausible = plausible[:, None, :].expand(-1, C.MAX_UNITS, -1)
        d = torch.where(plausible, dist, torch.full_like(dist, BIG))
        return d.argmin(dim=2)

    day = glob[:, H + G_DAY] * 29.0
    verb = torch.zeros(B, C.MAX_UNITS, dtype=torch.long, device=dev)
    target = torch.zeros(B, C.MAX_UNITS, dtype=torch.long, device=dev)

    def assign(cond, v, plaus):
        nonlocal verb, target
        v_t = v if torch.is_tensor(v) else torch.full_like(verb, v)
        verb = torch.where(cond, v_t, verb)
        target = torch.where(cond, nearest(plaus), target)

    # lowest priority first, each later assignment overrides
    # plant: melon while there is season for it, else wheat
    can_wheat = um[:, :, VERB_PLANT_WHEAT] & tm[:, :, VERB_PLANT_WHEAT].any(1, keepdim=True)
    can_melon = um[:, :, VERB_PLANT_MELON] & tm[:, :, VERB_PLANT_MELON].any(1, keepdim=True) \
        & (day <= MELON_LAST_DAY)[:, None]
    assign(can_wheat, VERB_PLANT_WHEAT, tm[:, :, VERB_PLANT_WHEAT])
    assign(can_melon, VERB_PLANT_MELON, tm[:, :, VERB_PLANT_MELON])
    # deposit the product the unit carries most of
    carried = units[:, :, H + C.U_CARRY0:H + C.U_CARRY0 + len(C.PRODUCTS)] * C.U_CARRY_SCALE   # [B, 17, 9]
    top_prod = carried.argmax(dim=2)                                                             # [B, 17]
    dep_verb = VERB_DEPOSIT0 + top_prod
    dep_legal = um.gather(2, dep_verb.unsqueeze(-1)).squeeze(-1) & (carried.max(dim=2).values >= 1.0)
    shed_tiles = tiles[:, :, H + T_SHED_ACCESS].bool()
    assign(dep_legal & shed_tiles.any(1, keepdim=True), dep_verb, shed_tiles)
    # harvest, then water on top
    assign(um[:, :, VERB_HARVEST] & tm[:, :, VERB_HARVEST].any(1, keepdim=True), VERB_HARVEST, tm[:, :, VERB_HARVEST])
    assign(um[:, :, VERB_WATER] & tm[:, :, VERB_WATER].any(1, keepdim=True), VERB_WATER, tm[:, :, VERB_WATER])
    present = um.any(dim=2)
    verb = torch.where(present, verb, torch.zeros_like(verb))
    target = torch.where(present, target, torch.zeros_like(target))

    rows = torch.zeros(B, C.PLAYER_ACTION_DIM, dtype=torch.long, device=dev)
    rows[:, :C.MAX_UNITS] = target
    rows[:, C.MAX_UNITS:2 * C.MAX_UNITS] = verb

    # market
    money = glob[:, H + C.G_MONEY_LIN] * 20000.0
    hour = glob[:, H + G_HOUR] * 23.0
    hands = glob[:, H + G_LIVE_HANDS] * 16.0
    unlocked = glob[:, H + G_UNLOCKED] * 4.0
    seeds = priv[:, H + C.V_SEEDS0:H + C.V_SEEDS0 + 5] * C.V_SEED_SCALE                  # [B, 5]
    mk = 2 * C.MAX_UNITS
    hire = (hour < 0.5) & (hands < MAX_HANDS_TARGET) & (money >= 300.0) & (day <= 20)
    rows[:, mk + HEAD_HIRE] = torch.where(hire, 2, 0)
    land = (unlocked < 1.5) & (money >= 2500.0) & (day <= 20)
    rows[:, mk + HEAD_LAND] = torch.where(land, 1, 0)
    buy_wheat = (seeds[:, 0] < 1.0) & (money >= 100.0)
    rows[:, mk + HEAD_SEED0 + 0] = torch.where(buy_wheat, QTY_BUCKET_8, 0)
    buy_melon = (seeds[:, 4] < 1.0) & (money >= 800.0) & (day <= MELON_LAST_DAY)
    rows[:, mk + HEAD_SEED0 + 4] = torch.where(buy_melon, QTY_BUCKET_4, 0)
    for prod in (0, 1, 2, 3, 4):                                                        # sell all crops held
        rows[:, mk + HEAD_SELL0 + prod] = SELL_ALL
    return rows


def prefix_length(progress: float, max_len: int, ramp_frac: float) -> int:
    """Longest scripted prefix at this progress: max_len -> 0 linearly over `ramp_frac`."""
    if ramp_frac <= 0:
        return 0
    return int(round(max_len * max(0.0, 1.0 - progress / ramp_frac)))
