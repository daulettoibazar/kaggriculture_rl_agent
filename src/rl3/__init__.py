"""rl3: multi-base sale-timing residual with extended action space and features.

A strict superset of rl2: accepts rl2 checkpoints at init and reproduces rl2 decisions
exactly when the new ops and features are zero-weighted.
"""
__all__ = ["features", "policy", "residual_apply", "reward", "arena",
           "numpy_policy", "observation", "opponents"]
