#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-runs/latest/checkpoints/latest.pt}"
python -m backgammon_rlx.train.evaluate \
    --checkpoint "$CKPT" \
    --games 500 \
    --out "runs/eval_$(date +%Y%m%d_%H%M%S).json"
