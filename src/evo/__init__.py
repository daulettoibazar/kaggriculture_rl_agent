"""Exact-simulator helpers. Importing this package pins RAYON/OMP/MKL threads to 1 before kaggri loads."""
import os as _os

# kaggri.PyVecEnv.step releases the GIL into a rayon pool sized to every logical core;
# pin before anything imports kaggri (setdefault keeps an explicit caller override).
for _k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
