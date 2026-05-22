#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python -m backgammon_rlx.train.train_ppo --config configs/rtx5080.yaml "$@"
