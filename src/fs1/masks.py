"""Unit-mask relaxation (rl_last F1 fix, research/rl_last/DESIGN.md "Action interface").

The simulator's unit mask makes Plant(c), Fertilize, Feed and PlaceAnimal(a) legal only when the
unit already holds the resource (seed in the shed, fertilizer / wheat / the animal in hand). A
program buys the seed and walks to the tile in the same turn (BUY_SEED settles in the market
phase, after the unit phase), so 19-56% of its plant labels were masked out (review 6, F1). The
relaxed mask makes those verbs legal for every live unit whenever the tile-verb plausibility
plane has a target; the resolver already turns an un-resourced arrival into PASS.

Apply identically in the training env, the BC loader and the export policy. Works on torch
tensors and numpy arrays: unit_mask [..., 17, 38] bool, tile_mask [..., 100, 38] bool.
"""
from __future__ import annotations

RELAXED_VERBS = (3, 7, 10, 11, 12, 13, 14, 15, 16, 17)   # Fertilize, Feed, Plant x5, PlaceAnimal x3


def relax_unit_mask(unit_mask, tile_mask):
    """-> a new relaxed unit mask (inputs untouched)."""
    alive = unit_mask.any(-1)                              # [..., 17]; absent unit rows are all zero
    v = list(RELAXED_VERBS)
    plausible = tile_mask[..., v].any(-2)                  # [..., 10]
    out = unit_mask.clone() if hasattr(unit_mask, "clone") else unit_mask.copy()
    out[..., v] = out[..., v] | (alive[..., :, None] & plausible[..., None, :])
    return out
