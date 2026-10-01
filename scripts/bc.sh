#!/bin/bash
# Behaviour cloning from leaderboard replays on one local GPU:
#   replays (data/replays_2026-09-26) -> demo shards -> BC with per-team styles -> one style folded into a plain checkpoint.
# Usage: scripts/bc.sh            (env: DEVICE=cuda:0 EPOCHS=5 STYLE=8 INIT=runs/ckpt_lineage_1_selfplay_scratch_u924.pt
#                                  OUT=runs/bc MAX_EPISODES= (all) WORKERS=8)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
DEVICE=${DEVICE:-cuda:0}; EPOCHS=${EPOCHS:-5}; STYLE=${STYLE:-8}; OUT=${OUT:-runs/bc}; WORKERS=${WORKERS:-8}
INIT=${INIT:-runs/ckpt_lineage_1_selfplay_scratch_u924.pt}
DEMOS=${DEMOS:-data/fs1_demos/replays}
EXTRA=${MAX_EPISODES:+--max-episodes $MAX_EPISODES}
echo "== 1/3 replays -> demos ($DEMOS)"
python3 tools/fs1/make_replay_demos.py --catalog data/replays_2026-09-26/catalog.json --out "$DEMOS" --max-rank 60 --workers "$WORKERS" $EXTRA
echo "== 2/3 behaviour cloning ($OUT, $EPOCHS epochs, init $INIT)"
python3 tools/fs1/bc_pretrain.py --demos "$DEMOS" --n-styles 62 --relax-masks --init "$INIT" --out "$OUT" --epochs "$EPOCHS" --device "$DEVICE" ${BC_EXTRA:-}
echo "== 3/3 fold style $STYLE -> $OUT/clone.pt"
python3 tools/fs1/fold_style.py --checkpoint "$OUT/latest.pt" --style "$STYLE" --out "$OUT/clone.pt"
echo "done: $OUT/clone.pt (use it as init-from / ref-policy in configs/learning/bcsp_start.json)"
