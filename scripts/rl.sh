#!/bin/bash
# Self-play PPO on local GPUs. One trainer process per GPU (torchrun); each process runs the config's n-envs.
# Usage: CONFIG=configs/learning/rl_main.json GPUS=2 RUN_DIR=runs/rl_main scripts/rl.sh [extra tools/fs1/train.py flags]
# Configs: bcsp_start.json (from the BC clone), rl_main.json (main recipe), rl_q4_finetune.json (4th-quadrant fine-tune).
# Stop with a checkpoint: kill -USR1 <trainer pid>; resume by re-running with the same RUN_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/learning/rl_main.json}; GPUS=${GPUS:-1}; RUN_DIR=${RUN_DIR:-runs/$(basename "$CONFIG" .json)}
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FS1_RAYON_THREADS=${FS1_RAYON_THREADS:-4}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FS1_WANDB=${FS1_WANDB:-0}
mkdir -p "$RUN_DIR"
echo "config=$CONFIG gpus=$GPUS run_dir=$RUN_DIR"
if [ "$GPUS" -le 1 ]; then
    exec python3 tools/fs1/train.py --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
else
    exec torchrun --standalone --nproc_per_node="$GPUS" tools/fs1/train.py --config "$CONFIG" --run-dir "$RUN_DIR" "$@"
fi
