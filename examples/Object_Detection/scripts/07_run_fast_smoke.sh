#!/usr/bin/env bash
# End-to-end FAST_SMOKE for the TGraphX object-detection graph-fusion pipeline.
# Designed to run in minutes with no real model downloads (synthetic detectors).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

CONFIG="$PROJECT_ROOT/configs/fast_smoke.yaml"

echo "[smoke] config: $CONFIG"
python -m od_graph_fusion.cli --config "$CONFIG"

echo ""
echo "[smoke] OK. Run output:"
ls "$PROJECT_ROOT/runs/" 2>/dev/null
