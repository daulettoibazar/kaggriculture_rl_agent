# Kaggriculture

Agent for the Kaggle "Kaggriculture" simulation: 2 players, 720 steps, most money wins, interaction
only through a shared market (rules: `docs/RULES.md`). The agent is an 18.5M-parameter entity-token
transformer (`src/fs1/model.py`) cloned from leaderboard replays and trained by self-play PPO inside
`kaggri`, an exact Rust re-implementation of the simulator. Setup, data layout and training: `README.md`.

## Map

| path | what |
|---|---|
| `src/fs1/` | `model.py` policy, `env.py` wrapper, `ppo.py`, `reward.py`, `masks.py`, `league.py`, `phantom.py` (recorded market streams as opponents), `public_pool.py` (public programs as opponents), `q4.py` (forced 4th-quadrant purchase), `export.py` |
| `src/evo/kernel_opponents.py` | runs public notebook programs closed-loop in the exact simulator |
| `src/rl3/opponents.py` | opponent registry (name -> program path) |
| `submission/` | bundle entry point `fs1_main.py`, numpy policy `fs1_numpy_policy.py` |
| `tools/fs1/` | `make_replay_demos.py`, `bc_pretrain.py`, `fold_style.py`, `train.py`, `export_bundle.py`, `panel_eval.py`, `replay_market_streams.py` |
| `tools/rl_last/g1_pair.py`, `tools/arena.py` | paired margin with LCB95; official interpreter harness |
| `scripts/bc.sh`, `scripts/rl.sh` | clone from replays; self-play PPO on local GPUs |
| `configs/learning/` | `bcsp_start.json` -> `rl_main.json` -> `rl_q4_finetune.json` |
| `rust/` | the engine (`PORT_SPEC.md`, `OBS_LAYOUT.md`; parity tests in `crates/kaggri-core/tests/`) |
| `dist/` | final bundles `rl5_ph_u1384`, `rl5_q4_u301`; the engine wheel in `dist/wheels/` |

## Commands

```
pytest -q                                   # tests needing third-party programs skip when absent
scripts/bc.sh                               # replays -> runs/bc/clone.pt
CONFIG=configs/learning/rl_main.json GPUS=2 scripts/rl.sh
python3 tools/fs1/export_bundle.py --checkpoint <ckpt> --out dist/<name> --temperature 0.5 --relax-masks --verify
```

## Working notes

- Pin `RAYON_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` before importing `kaggri` (`import src.evo`
  does it), or one process takes every core. Never import it in a parent that later forks workers.
- `PYTHONHASHSEED=0` in every harness: public programs iterate string sets.
- Executing public programs writes files into the CWD; run sweeps from a scratch directory.
- Never `pkill -f X` in a command whose own line contains X.
- The observation and action layout is fixed (`rust/OBS_LAYOUT.md`); new inputs are appended as
  zero-initialised columns so old checkpoints still load.
