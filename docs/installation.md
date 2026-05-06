# Installation

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.13
- torchvision ≥ 0.14
- PyYAML ≥ 5.4 (for YAML config loading)

## From source (recommended during development)

```bash
git clone https://github.com/arashsajjadi/TGraphX.git
cd TGraphX
pip install -e .
```

## CPU-only PyTorch (CI / machines without GPU)

Install PyTorch first to avoid pulling CUDA wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Optional extras

```bash
# Hardware monitoring in the dashboard (CPU %, RAM, GPU utilisation/temp)
pip install "tgraphx[monitoring]"   # installs psutil and pynvml

# Third-party experiment-tracking libraries (use their APIs directly)
pip install "tgraphx[tracking]"     # installs tensorboard and mlflow

# Development tools (pytest, build, twine)
pip install "tgraphx[dev]"
```

> **Note:** `psutil`, `pynvml`, `tensorboard`, and `mlflow` are **never**
> imported at base `import tgraphx` time. They are loaded lazily only when
> you explicitly request them.

## Verify installation

```python
import tgraphx
print(tgraphx.__version__)          # e.g. "0.1.1"

from tgraphx import Graph, TensorGATLayer, build_grid_graph
from tgraphx.performance import env_report
print(env_report())
```

## Conda environment

An `environment.yml` is provided for Conda users:

```bash
conda env create -f environment.yml
conda activate tgraphx
```
