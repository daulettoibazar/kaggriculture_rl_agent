"""fs1 — self-play PPO from scratch on the exact simulator's token observation.

Modules
-------
constants   token layout, slot indices and game economics used by the reward and the heads
reward      potential-based dense reward computed from the observation tensors on the GPU
model       entity-token transformer actor-critic with verb / target / market heads
env         PyVecEnv wrapper: phase stagger, price history, torch tensors, validity flags
league      frozen opponents: last-best, PFSP pool, promotion gate
ppo         rollout storage, GAE, the clipped update
"""
