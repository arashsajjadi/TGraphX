#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-runs/latest/checkpoints/latest.pt}"
python -m backgammon_rlx.play --checkpoint "$CKPT"
