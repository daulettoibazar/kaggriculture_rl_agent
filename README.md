# Kaggriculture

An agent for the Kaggle "Kaggriculture" competition: a 2-player farming simulation of 720 steps
where the most money wins and the players interact only through a shared market (rules:
`docs/RULES.md`). The agent is an 18.5M-parameter entity-token transformer (`src/fs1/model.py`):
140 tokens x 48 features, per-unit verb and target heads for up to 17 workers, 21 market heads, a
centralised critic. It plays every worker action and every market order itself, sampling at
temperature 0.5. It was cloned from public leaderboard replays and then trained by self-play PPO
against a league of past selves, public notebook programs and recorded market streams.

The game engine is `kaggri`, an exact Rust re-implementation of the official simulator (`rust/`,
about 9 ms per game, parity-tested against recorded official traces). The shipped bundle runs the
policy in numpy with the engine bindings for observation encoding, about 200 ms per turn on one thread.

## Repository map

| path | what |
|---|---|
| `src/fs1/` | policy, environment wrapper, PPO, reward, masks, league, phantom seats, 4th-quadrant intervention, numpy export |
| `src/evo/kernel_opponents.py` | runs public notebook programs closed-loop inside the exact simulator |
| `src/rl3/opponents.py` | opponent registry: names to program paths |
| `submission/` | bundle entry point `fs1_main.py` and the numpy policy `fs1_numpy_policy.py` |
| `tools/fs1/` | `make_replay_demos.py`, `bc_pretrain.py`, `fold_style.py`, `train.py`, `export_bundle.py`, `panel_eval.py`, `replay_market_streams.py` |
| `tools/rl_last/g1_pair.py` | paired margins against a control run, with LCB95 |
| `tools/arena.py` | official `kaggle_environments` interpreter harness: strength and per-turn timing |
| `scripts/` | `bc.sh` (replays -> clone) and `rl.sh` (self-play PPO on local GPUs) |
| `configs/learning/` | `bcsp_start.json`, `rl_main.json`, `rl_q4_finetune.json` |
| `rust/` | the `kaggri` engine (`PORT_SPEC.md`, `OBS_LAYOUT.md`) |
| `dist/` | the two final bundles `rl5_ph_u1384`, `rl5_q4_u301` (weights are release assets), the engine wheel |
| `tests/` | `pytest -q` |

## Install

Python 3.11+, with `torch`, `numpy`, `kaggle_environments` and `pytest`. Install the engine wheel,
or build it with maturin:

```
pip install dist/wheels/kaggri_py-0.1.0-*.whl
# or: pip install maturin && cd rust && maturin build --release -m crates/kaggri-py/Cargo.toml
```

Pin `RAYON_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` before importing `kaggri` (importing
`src.evo` does it) and set `PYTHONHASHSEED=0` for every evaluation: some public programs iterate
string sets. The scripts set both.

## Data layout

`data/` and `runs/` are gitignored; make them directories or symlinks to a large disk.

```
data/replays_2026-09-26/        replay JSONs + catalog.json (public leaderboard episodes used for cloning)
data/phantom_streams_top20.json recorded market order streams (tools/fs1/replay_market_streams.py)
data/public_kernels/<author>_<slug>/extracted/main.py   opponent programs (see Third-party programs)
data/fs1_demos/                 generated demo shards
runs/                           checkpoints: put the release assets ckpt_*.pt here
```

## Train

1. Clone the replays into a policy (one GPU; `EPOCHS`, `DEVICE`, `MAX_EPISODES` are optional):

```
scripts/bc.sh
```

   This writes `runs/bc/clone.pt`: demo shards from the replay catalog, behaviour cloning with a
   per-team style embedding initialised from the from-scratch self-play checkpoint, and style 8
   folded into a plain checkpoint.

2. Self-play PPO, in three consecutive stages (each starts from the previous stage's checkpoint;
   one trainer process per GPU):

```
CONFIG=configs/learning/bcsp_start.json   GPUS=2 scripts/rl.sh
CONFIG=configs/learning/rl_main.json      GPUS=2 scripts/rl.sh
CONFIG=configs/learning/rl_q4_finetune.json GPUS=2 scripts/rl.sh
```

   `bcsp_start` warm-starts self-play from the clone (the clone is the KL reference); `rl_main` is the
   long run with the full opponent mix (mirror games, a league of past selves, public programs, phantom
   seats); `rl_q4_finetune` is a short final stage with a forced 4th-quadrant purchase in half the games.
   Each config names its start checkpoint (`init-from`), the KL reference policy, the opponent mix and
   the schedule; extra `train.py` flags can be
   appended to override any key. The trainer writes `latest.pt` to the run directory, checkpoints on
   `SIGUSR1`, and resumes when re-run with the same run directory.

3. Export and evaluate a checkpoint:

```
python3 tools/fs1/export_bundle.py --checkpoint runs/<run>/latest.pt --out dist/<name> --temperature 0.5 --relax-masks --verify
python3 tools/fs1/panel_eval.py --checkpoint runs/<run>/latest.pt --temperature 0.5 --relax-masks --out runs/panel/<name>
python3 tools/arena.py --agent dist/<name>/main.py --opponents dist/rl5_q4_u301/main.py -n 64 --sides both
```

   A bundle is `main.py`, `fs1_numpy_policy.py`, `fs1_policy.npz`, `fs1_bundle.json`,
   `manifest.json`, `kaggri/`, `LICENSE.txt`, `NOTICE.txt`; tar those members to submit.

## Third-party programs

No third-party code ships in the bundles or in this repository. Training and evaluation opponents
are public Kaggle notebooks (Apache 2.0): download them into `data/public_kernels/<author>_<slug>/`
at the paths listed in `src/rl3/opponents.py`. The cha22 control agent belongs in
`dist/tape_v21_cha22x/policy.py` and `dist/tape_v20_cha22/policy.py`; its source is
https://www.kaggle.com/code/abhinav0370/cha22-agent (credits in `dist/tape_v21_cha22x/NOTICE.txt`).
Tests that need a missing program skip.
