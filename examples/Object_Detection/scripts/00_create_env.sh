#!/usr/bin/env bash
# Create a clean virtual environment for the TGraphX object-detection
# graph-fusion experiment. Installs TGraphX from PyPI by default.
#
# Usage:
#   bash scripts/00_create_env.sh            # PyPI install (default)
#   bash scripts/00_create_env.sh --dev      # editable TGraphX from repo root
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
VENV="$PROJECT_ROOT/.venv-od-fusion"

DEV_INSTALL=0
if [[ "${1:-}" == "--dev" ]]; then
    DEV_INSTALL=1
fi

echo "[od-fusion] creating venv: $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

python -m pip install -U pip wheel setuptools

# torch / torchvision (CPU build by default; user can replace with CUDA wheel)
echo "[od-fusion] installing torch / torchvision (CPU index by default)"
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Core requirements
python -m pip install -r "$PROJECT_ROOT/requirements-object-detection.txt"

# TGraphX
if [[ $DEV_INSTALL -eq 1 ]]; then
    echo "[od-fusion] installing TGraphX in editable mode from $REPO_ROOT"
    python -m pip install -e "$REPO_ROOT"
else
    echo "[od-fusion] installing TGraphX from PyPI"
    python -m pip install -U tgraphx
fi

# Environment report
python - <<'PY'
import importlib, sys, platform
print("=" * 60)
print("TGraphX Object-Detection Fusion — environment report")
print("=" * 60)
print(f"Python:     {sys.version.split()[0]}")
print(f"Platform:   {platform.platform()}")
for pkg in ("torch", "torchvision", "tgraphx",
            "ultralytics", "transformers", "timm",
            "pycocotools", "cv2", "PIL", "matplotlib", "pandas"):
    try:
        m = importlib.import_module(pkg if pkg != "PIL" else "PIL")
        v = getattr(m, "__version__", "installed")
        print(f"{pkg:14s} {v}")
    except ImportError:
        print(f"{pkg:14s} not installed")
try:
    import torch
    print(f"CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
except Exception:
    pass
print("=" * 60)
PY

echo ""
echo "[od-fusion] done. Activate with:"
echo "  source $VENV/bin/activate"
