<p align="center">
  <img src="https://raw.githubusercontent.com/arashsajjadi/TGraphX/main/TGRAPHX.png" alt="TGraphX logo" width="220">
</p>

# TGraphX

[![Tests](https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml/badge.svg)](https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/tgraphx.svg)](https://pypi.org/project/tgraphx/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.13+](https://img.shields.io/badge/pytorch-1.13%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

📄 **Preprint:** [TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning](https://arxiv.org/abs/2504.03953) · *Sajjadi & Eramian, arXiv 2025*

TGraphX is a PyTorch library for graph neural networks whose **node features are multi-dimensional tensors** — such as `[C, H, W]` image-patch feature maps — rather than flat vectors. Convolutional message passing operates directly on spatial feature maps at each node, so local structure is never destroyed by flattening.

---

## Colab tutorial

TGraphX includes a hands-on Google Colab notebook that installs the latest PyPI release and walks through the main API features interactively:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1agls1xtqE5WxbWthcG0HEa3Gbk3fvoCD?usp=sharing)

**[Open the TGraphX Colab Tutorial →](https://colab.research.google.com/drive/1agls1xtqE5WxbWthcG0HEa3Gbk3fvoCD?usp=sharing)**

The notebook covers:

- CPU/GPU environment checks
- Vector node classification
- 2-D spatial graph classification using image patches
- 3-D volumetric graph classification
- Graph-level and node-level regression
- Edge prediction / edge classification
- Layer zoo: `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer`, `LinearMessagePassing`, legacy `AttentionMessagePassing`
- Edge weights and edge features
- Dashboard-compatible run files

> **Honesty note:** All tasks in the notebook use controlled synthetic data
> designed to verify installation, API behaviour, device compatibility, and
> gradient flow.  They are not benchmark results and make no real-world
> performance or state-of-the-art claims.

---

## The problem TGraphX solves

Standard GNN frameworks (PyG, DGL) expect flat vector node features. Flattening a `[C, H, W]` feature map into a `[C·H·W]` vector discards the spatial structure that makes CNNs effective. TGraphX keeps each node's representation as a tensor and applies 1×1 convolutions during message passing, so every neighbourhood aggregation step acts like a miniature CNN across neighbouring feature maps.

---

## What is currently implemented

All "tensor-aware" GNN layers in this list operate on spatial node feature
maps `[N, C, H, W]` and preserve the spatial layout through message
passing.  They are *adaptations* of the canonical algorithms — not
drop-in clones of PyTorch Geometric's vector-feature implementations.

| Component | Class | Input node shape | Notes |
|-----------|-------|-----------------|-------|
| Graph data structure | `Graph`, `GraphBatch` | any | Validated; `.to(device)` |
| Tensor-aware GCN-style message passing | `ConvMessagePassing` | `[N, C, H, W]` | `aggr="sum"` or `"mean"`; 1×1 conv messages + deep CNN aggregator |
| Tensor-aware GAT (multi-head) | `TensorGATLayer` | `[N, C, H, W]` | True GAT: softmax over incoming edges per destination, per head; scalar attention per `(edge, head)` |
| Tensor-aware GraphSAGE | `TensorGraphSAGELayer` | `[N, C, H, W]` | Separate self / neighbour 1×1 Conv2d; mean or max aggregation; optional L2 normalise |
| Tensor-aware GIN / GINEConv | `TensorGINLayer` | `[N, C, H, W]` | `(1+ε)·h_j + Σ h_i`; default 1×1 Conv MLP, learnable ε, optional GINEConv edge term |
| Spatial-gating message passing (legacy) | `AttentionMessagePassing` | `[N, C, H, W]` or `[N, D]` | Per-edge sigmoid gating — **not** GAT.  Kept for backward compatibility; use `TensorGATLayer` for true GAT. |
| Vector message passing | `LinearMessagePassing` | `[N, D]` | Base layer with linear projections |
| Custom layer base class | `TensorMessagePassingLayer` | any | Override `message` / `update`; base handles aggregation |
| CNN patch encoder | `CNNEncoder` | `[N, C_in, pH, pW]` | Outputs spatial feature maps `[N, C_out, H', W']` |
| Optional pre-encoder | `PreEncoder` | `[N, C_in, pH, pW]` | Custom or pretrained ResNet-18 |
| Unified CNN-GNN model | `CNN_GNN_Model` | `[N, C, pH, pW]` | Takes **pre-split** patches; user supplies `edge_index` |
| Graph classification | `GraphClassifier` | `[N, C, H, W]` | Mean / sum / max readout |
| Node classification | `NodeClassifier` | `[N, D]` | Vector features only |
| Dataset & loader | `GraphDataset`, `GraphDataLoader` | — | Wraps `torch.utils.data` |
| Utilities | `load_config`, `get_device` | — | YAML/JSON config; CUDA→MPS→CPU |

## What is NOT yet implemented

- **Graph Transformers** (global attention with positional encodings).
- **Per-channel and per-pixel attention** for `TensorGATLayer`. Currently only scalar attention per `(edge, head)` is supported (the default safer choice).
- **Spatial edge features in `TensorGATLayer`** are accepted as `[E, edge_dim, H, W]` and **mean-pooled** to a vector before the per-`(edge, head)` attention bias projection. This keeps GAT in the scalar-attention regime (no per-pixel attention). `TensorGraphSAGELayer` and `TensorGINLayer` use the full spatial tensor directly (selected via `edge_features_kind="spatial"`).
- **Per-voxel and per-pixel attention.** `TensorGATLayer` keeps scalar attention per `(edge, head)` even in 3-D mode. Volumetric edge tensors `[E, C_e, D, H, W]` are mean-pooled over `(D, H, W)` before the bias projection — there is no per-voxel attention.
- **Arbitrary-rank tensor support.** TGraphX supports vector `[N, D]`, 2-D spatial `[N, C, H, W]`, and 3-D volumetric `[N, C, D, H, W]` node features for the listed layers. It does not claim arbitrary-rank tensor support.
- **Heterogeneous and temporal graphs.**
- **MLflowLogger.** Not implemented.  Use the `mlflow` client directly.
  `pip install mlflow`.
- **GAT chunked forward.** Deferred to v0.2.4. GAT's destination-wise
  softmax requires all incoming edge scores simultaneously; a correct two-pass
  implementation is needed. `ConvMessagePassing`, `TensorGraphSAGELayer`, and
  `TensorGINLayer` all support `chunk_size` in their `forward()` call.
- **Hardware-monitoring extras.** CPU/RAM/GPU metrics in the dashboard
  require optional packages: `pip install tgraphx[monitoring]`.
- **torch.compile / AMP.** `torch.compile` is available in PyTorch ≥ 2.0
  and may improve throughput for large graphs; compile overhead dominates
  for small ones.  Some GNN ops (e.g. `index_add_` in GAT) require matching
  dtypes and do not work transparently under float16 autocast; bfloat16 or
  full-precision inference is recommended.
- **Profiling / logging.** No profiling, hardware polling, or file writes
  are enabled by default.  All are opt-in.

---

## Performance

### Environment and hardware report

```python
from tgraphx.performance import env_report, estimate_message_memory, recommended_device

print(env_report())                           # Python/PyTorch/CUDA/MPS info
print(env_report(include_hardware=True))      # + CPU/RAM/CUDA memory (needs psutil)
print(env_report(include_sensors=True))       # + GPU util/temp (needs pynvml)

dev = recommended_device()                    # CUDA > MPS > CPU

# Estimate peak message-buffer memory before running
m = estimate_message_memory(num_edges=1024, out_shape=(64, 8, 8))
print(f"~{m['total_mb']:.1f} MB  ({m['note']})")
```

### Benchmarks

```bash
# Layer throughput (CPU-safe, all flags optional)
python benchmarks/benchmark_layers.py --layer gat --nodes 64 --edges 256 \
    --shape 8,4,4 --device cpu --iters 10

# CUDA + AMP + backward
python benchmarks/benchmark_layers.py --layer conv --nodes 256 --edges 2048 \
    --shape 32,8,8 --device cuda --amp 1 --backward 1

# Save JSON result
python benchmarks/benchmark_layers.py --layer gin --shape 8,4,4 \
    --output results/gin.json

# Graph builder timing
python benchmarks/benchmark_graph_builders.py --small    # CI-safe
python benchmarks/benchmark_graph_builders.py            # full
```

### torch.compile and AMP

```bash
python examples/torch_compile_benchmark.py   # eager vs compiled, correctness check
python examples/mixed_precision_inference.py  # autocast forward demo (finite-output check)
python examples/memory_report.py             # env report + memory estimates
```

**AMP policy (v0.2.2):**

| Backend | Recommended dtype | Status | Notes |
|---------|:-----------------:|:------:|-------|
| CPU | bfloat16 | ⚠️ Best-effort | Tested in CI |
| CUDA | float16 / bfloat16 | ⚠️ Best-effort | bfloat16 needs Ampere+ |
| MPS | — | ❌ Not tested | PyTorch operator coverage varies |

v0.2.2 fixes: `broadcast_edge_weight` casts edge weights to activation dtype;
`TensorGATLayer` casts attention weights before `index_add_`; `edge_softmax`
upcasts to fp32 for numerical stability and casts back. See
[docs/performance.md](docs/performance.md#amp-policy) for full details.

### Optional chunked forward (ConvMessagePassing)

Reduce peak edge-buffer memory by processing edges in chunks:

```python
from tgraphx.layers.conv_message import ConvMessagePassing

layer = ConvMessagePassing(in_shape=(32, 8, 8), out_shape=(32, 8, 8), aggr="sum")
# Same output as unchunked; lower peak memory for large E
out = layer(x, edge_index, chunk_size=512)
```

Supported aggregations: `"sum"` and `"mean"`.  `"max"` falls back to the
standard path with a warning.  GAT, SAGE, and GIN chunking are deferred
(softmax and custom scatter make them more complex).

### Hardware compatibility

| Platform | Forward | AMP | torch.compile | CI coverage | Notes |
|----------|:-------:|:---:|:-------------:|:-----------:|-------|
| CPU | ✅ | ⚠️ bfloat16 only | ✅ | Full CI | Compile overhead may dominate small graphs |
| CUDA | ✅ | ⚠️ float16 (op-dependent) | ✅ | Full CI | `index_add_` ops require dtype match |
| MPS (Apple Silicon) | ✅ | ⚠️ limited | ⚠️ | No CI | Best-effort; some ops may not compile |
| Linux | ✅ | ✅ | ✅ | Full CI (ubuntu-latest) | Primary CI platform |
| Windows | ✅ | ✅ | ✅ | No CI | Best-effort; no automated tests |
| macOS | ✅ | ⚠️ limited | ⚠️ | No CI | MPS support best-effort |

---

## Training utilities

TGraphX includes lightweight training helpers — not a full training framework.
All logging and file writes are **off by default**.

### Training loop helpers

```python
from tgraphx.training import train_epoch, evaluate, fit
import torch.nn.functional as F

# One-line training loop
history = fit(
    model, train_loader, val_loader=val_loader,
    epochs=20,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    loss_fn=F.cross_entropy,
    log_level=1,           # print per-epoch summary
)
# → [{"epoch":0, "train_loss":0.9, "val_loss":0.85}, ...]
```

Supported batch formats: `GraphBatch` (with `graph_labels` / `node_labels`)
and `(Tensor, Tensor)` tuples.

### Standalone helpers

```python
from tgraphx.training import (
    set_seed,            # seeds torch / numpy / random
    count_parameters,    # trainable parameter count
    save_checkpoint,     # torch.save wrapper
    load_checkpoint,     # returns saved epoch number
    accuracy,            # multi-class argmax accuracy
    mean_absolute_error, mean_squared_error,
)
```

### CSV logging (dashboard-compatible)

```python
from tgraphx.tracking import CSVLogger

with CSVLogger("runs/my_run") as logger:
    history = fit(model, train_loader, ..., logger=logger)
# writes runs/my_run/metrics.csv with UTC timestamps
```

### TensorBoard logging (optional)

```python
from tgraphx.tracking import TensorBoardLogger  # lazy import

# pip install tensorboard  or  pip install "tgraphx[tracking]"
with TensorBoardLogger("runs/tb") as tb:
    history = fit(model, train_loader, ..., logger=tb)
```

Nothing is written unless you explicitly pass a logger.
`MLflowLogger` is not implemented — use the `mlflow` client directly.

---

## Dashboard

TGraphX includes a local training dashboard — **off by default**, zero external dependencies, no telemetry.

### Quick start

```bash
# Launch (localhost only, no token needed)
tgraphx-dashboard --logdir runs/demo
# → http://127.0.0.1:8765

# LAN access — explicit token required
tgraphx-dashboard --logdir runs/demo \
  --host 0.0.0.0 --token MY_SECRET_TOKEN

# Auto-generated token
tgraphx-dashboard --logdir runs/demo --host 0.0.0.0 --token auto

# Offline HTML snapshot — no server needed
tgraphx-dashboard --logdir runs/demo --export-html snapshot.html
```

```python
# Python API — non-blocking background thread
from tgraphx.dashboard import launch_dashboard_background, export_dashboard_html
server = launch_dashboard_background("runs/demo", port=8765)
# ... training loop ...
server.shutdown()

# Offline snapshot
export_dashboard_html("runs/demo", "snapshot.html")
```

### Dashboard features

| Section | Contents |
|---------|----------|
| **Overview** | Status chip, epoch progress, live loss, elapsed / ETA |
| **Metrics** | SVG line charts, window selector, EMA smoothing, per-chart CSV/SVG export |
| **Graph** | Graph summary, degree stats, `graph_stats.json` precomputed cards, SVG preview |
| **Hardware** | CPU/RAM/GPU/CUDA/MPS, power draw, thermal status (optional `psutil`/`pynvml`) |
| **Logs** | Scrollable metric table with CSV export |
| **Config** | `run_metadata.json` rendered safely |
| **Tools** | Copy URL, export buttons, refresh controls |
| **TV mode** | Full-screen large-font passive monitoring |

Key features:

- **Incremental updates** — browser requests only new rows via `?since_row=N`
- **Multi-run selector** — point at a parent directory; select runs by name
- **Color-blind-safe palette** — Okabe-Ito toggle, persisted in localStorage
- **Accessible** — skip link, ARIA labels, focus-visible, reduced-motion support
- **Export** — metrics CSV, per-chart CSV/SVG, print/save PDF, offline HTML snapshot
- **Responsive** — phone, tablet, desktop, TV/large-monitor layouts
- **Pause/resume** polling, configurable refresh interval

### Security model

| Scenario | Token required? |
|----------|:-:|
| `--host 127.0.0.1` (default) | No |
| `--host 0.0.0.0` + connecting from localhost | No |
| `--host 0.0.0.0` + connecting from another device | **Yes** |
| Starting LAN mode without `--token` | **Refused at startup** |

Read-only · no external CDN · no telemetry · no token leakage in API responses · path-traversal protected.

### Log files

```
metrics.csv          — epoch,train_loss,val_loss,... (ISO-8601 UTC timestamp column)
run_metadata.json    — run name, status, total_epochs, device, task (free-form dict)
graph_metadata.json  — optional graph summary + edge_index for preview (≤200 nodes)
graph_stats.json     — optional precomputed stats (write with write_graph_stats())
```

```python
from tgraphx import write_graph_stats
write_graph_stats({"num_nodes": 100, "num_edges": 400, "density": 0.04},
                  "runs/demo/graph_stats.json")
```

### Hardware monitoring (optional)

```bash
pip install "tgraphx[monitoring]"   # psutil + pynvml
```

Missing packages show a compact "unavailable" reason per row — no broken charts.

---

## Privacy and local-first behavior

TGraphX is designed to be **entirely local and private**:

| Behavior | Default |
|---|---|
| Telemetry / analytics | None — never |
| Remote calls at import | None |
| Dashboard | Off — launch explicitly |
| CSV metric logging | Off — create `CSVLogger` explicitly |
| TensorBoard logging | Off — create `TensorBoardLogger` explicitly |
| Hardware monitoring | Off — pass `include_hardware=True` to `env_report` |
| Checkpoints | Off — call `save_checkpoint` explicitly |
| Graph serialization | Off — include `edge_index` in JSON manually |
| Background threads | None (unless `launch_dashboard_background` is called) |
| File writes | Only to paths you explicitly provide |
| Reads | Dashboard reads only inside `--logdir` |
| External CDN / assets | None — dashboard is fully self-contained |

No `~/.tgraphx` directory or user-level config is created by default.

---

## Installation

```bash
pip install tgraphx
```

Optional extras:

```bash
pip install "tgraphx[tracking]"    # TensorBoard integration
pip install "tgraphx[monitoring]"  # psutil + pynvml (dashboard hardware panel)
pip install "tgraphx[dev]"         # pytest, build, twine
```

For a specific PyTorch build (e.g. CPU-only or a particular CUDA version), install PyTorch **before** TGraphX:

```bash
# CPU-only example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tgraphx
```

**Install from source:**

```bash
git clone https://github.com/arashsajjadi/TGraphX.git
cd TGraphX
pip install -e ".[dev]"
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for GPU-specific install commands.
A Conda environment file is provided at [`environment.yml`](environment.yml).

---

## Quickstart

**Vector node features** (simplest case):

```python
import torch
from tgraphx import Graph, LinearMessagePassing, build_model, fit

# 8 nodes, 32-dimensional vector features
x = torch.randn(8, 32)
src = torch.arange(8)
edge_index = torch.stack([src, (src + 1) % 8])

g = Graph(x, edge_index)
layer = LinearMessagePassing(in_shape=(32,), out_shape=(64,))
out = layer(g.node_features, g.edge_index)   # [8, 64]
out.sum().backward()
```

**Spatial node features** — `[C, H, W]` preserved through message passing:

```python
import torch
from tgraphx import Graph, ConvMessagePassing

N, C, H, W = 6, 16, 8, 8
node_features = torch.randn(N, C, H, W)
src = torch.arange(N)
edge_index = torch.stack([src, (src + 1) % N])

g = Graph(node_features, edge_index)
layer = ConvMessagePassing(in_shape=(C, H, W), out_shape=(32, H, W))
out = layer(g.node_features, g.edge_index)   # [6, 32, 8, 8]
out.sum().backward()
```

---

## Graph Builders

TGraphX ships **pure-PyTorch graph builders** that return `edge_index`
tensors (`[2, E]`, `dtype=torch.long`) ready for any GNN layer or
`Graph` constructor.  They create **fixed, rule-based** adjacency
structures — they do **not** implement learned adjacency.

### Builder API

| Function | Description | Key params |
|----------|-------------|-----------|
| `build_grid_graph(rows, cols)` | 2-D 4-connected grid | `directed`, `self_loops` |
| `build_grid_graph_3d(depth, rows, cols)` | 3-D 6-connected grid | `directed`, `self_loops` |
| `build_fully_connected_graph(num_nodes)` | Complete graph | `self_loops` |
| `build_knn_graph(coords, k)` | k-nearest-neighbour | `directed`, `self_loops` |
| `build_radius_graph(coords, radius)` | All pairs within radius | `directed`, `self_loops` |
| `build_iou_graph(boxes, threshold)` | Bounding-box IoU ≥ threshold | `directed`, `self_loops` |
| `build_random_graph(num_nodes, num_edges)` | Uniform random sample | `directed`, `self_loops`, `seed` |

> **O(N²) warning:** `build_knn_graph` and `build_radius_graph` call
> `torch.cdist` internally, which requires **O(N²)** time and memory.
> For large graphs (N > 10 000), use approximate-NN libraries instead.
>
> **O(N²) warning:** `build_fully_connected_graph` emits N·(N−1) edges.
> Memory grows quadratically with node count.

### Grid graph quickstart

```python
import torch
from tgraphx import Graph, build_grid_graph
from tgraphx.layers.conv_message import ConvMessagePassing

# 3×3 patch grid — 9 nodes, each with a [4, 8, 8] spatial feature
node_features = torch.randn(9, 4, 8, 8)
edge_index = build_grid_graph(3, 3, directed=False, self_loops=True)
# edge_index: [2, 33]  (24 neighbour + 9 self-loop edges)

g = Graph(node_features, edge_index)

layer = ConvMessagePassing(in_shape=(4, 8, 8), out_shape=(8, 8, 8))
out = layer(g.node_features, g.edge_index)   # [9, 8, 8, 8]
```

### 2-D image patch graph

```python
import torch
from tgraphx import build_grid_graph, image_to_patches
from tgraphx.layers.gat import TensorGATLayer

# Extract 4×4 patches from a [B, C, H, W] image
images = torch.randn(2, 3, 8, 8)
patches = image_to_patches(images, patch_size=4)   # [2, 4, 3, 4, 4]

# Build the patch grid graph
edge_index = build_grid_graph(2, 2, directed=False, self_loops=True)

# Run GAT on one image's patches
x = patches[0]   # [4, 3, 4, 4]
gat = TensorGATLayer(in_channels=3, out_channels=8, num_heads=2, spatial_rank=2)
out = gat(x, edge_index)   # [4, 8, 4, 4]
```

### 3-D volume patch graph

```python
import torch
from tgraphx import build_grid_graph_3d, volume_to_patches
from tgraphx.layers.gat import TensorGATLayer

# Extract 4×4×4 patches from a [B, C, D, H, W] volume
volumes = torch.randn(1, 2, 8, 8, 8)
patches = volume_to_patches(volumes, patch_size=4)   # [1, 8, 2, 4, 4, 4]

# Build the 3-D patch grid graph
edge_index = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)

# Run GAT on one volume's patches
x = patches[0]   # [8, 2, 4, 4, 4]
gat = TensorGATLayer(in_channels=2, out_channels=4, num_heads=2, spatial_rank=3)
out = gat(x, edge_index)   # [8, 4, 4, 4, 4]
```

### Patch helper API

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `patch_grid_shape(H, W, patch_size, stride)` | — | `(n_h, n_w)` | Raises if not exactly covered |
| `image_to_patches(images, patch_size, stride)` | `[B, C, H, W]` | `[B, P, C, ph, pw]` | Row-major; matches grid node order |
| `volume_patch_grid_shape(D, H, W, patch_size, stride)` | — | `(n_d, n_h, n_w)` | Raises if not exactly covered |
| `volume_to_patches(volumes, patch_size, stride)` | `[B, C, D, H, W]` | `[B, P, C, pd, ph, pw]` | Depth-row-col order; matches 3-D grid |

---

## Concept: tensor-aware node features

In a standard GNN a node carries a vector `x_i ∈ ℝ^d`. In TGraphX a node carries a tensor `X_i ∈ ℝ^{C×H×W}`. Every layer follows the standard message-passing template:

```
M_{i→j} = φ( X_i, X_j, E_{i→j} )                   # per-edge messages
A_j     = AGG_{i ∈ N(j)} M_{i→j}                    # permutation-invariant aggregation
X'_j    = ψ( X_j, A_j )                             # update
```

Each layer instantiates this with its own `(φ, AGG, ψ)`:

| Layer | `φ` (message) | `AGG` | `ψ` (update) |
|-------|---------------|-------|--------------|
| `ConvMessagePassing` | `Conv1×1(Concat(X_i, X_j[, E_ij]))` | `sum` / `mean` / `max` | `DeepCNNAggregator(A_j)` (+ optional residual) |
| `TensorGATLayer`     | `α_{ij}^k · W^k X_i` with `α_{ij}^k = softmax_i(LeakyReLU(a_dst·pool(W^k X_j) + a_src·pool(W^k X_i) + b^k(e_ij)))` | sum (weighted) | concat or mean over heads, optional residual |
| `TensorGraphSAGELayer` | `W_neigh(X_i)` (+ optional spatial cat or vector bias from `e_ij`) | `mean` / `max` | `W_self(X_j) + AGG`, optional L2 normalise |
| `TensorGINLayer`       | `X_i` or `ReLU(X_i + φ_e(e_ij))` (GINEConv) | `sum` | `MLP((1+ε)·X_j + Σ_i M_ij)` |
| `LinearMessagePassing` | `Linear(Concat(x_i, x_j[, e_ij]))` | `sum` / `mean` / `max` | identity (override to customise) |

Spatial dimensions `H` and `W` are preserved through every learned transform. All aggregations are permutation-invariant over the order of incoming edges, and every layer is permutation-equivariant over node reindexing (verified by `tests/test_math.py`).

The graph structure — which nodes are connected — is **not learned by the model**. Users supply `edge_index` based on domain knowledge (e.g., spatial proximity of patches, IoU overlap of bounding boxes, kNN on patch centres).

### Edge feature formats per layer

| Layer | Vector `[E, D_e]` | Spatial `[E, C_e, H, W]` |
|-------|:----:|:----:|
| `ConvMessagePassing` | ✗ | ✓ (concatenated along channels; channel count must equal node channel count) |
| `TensorGATLayer` | ✓ (additive attention bias on logits) | ⚠️ accepted; mean-pooled to scalar attention bias (no per-pixel attention) |
| `TensorGraphSAGELayer` | ✓ (additive channel bias post-`W_neigh`) | ✓ (concatenated to source) |
| `TensorGINLayer` | ✓ (broadcast bias before ReLU) | ✓ (1×1 Conv2d projection) |

> **TensorGATLayer spatial edge features:** `[E, C_e, H, W]` (or `[E, C_e, D, H, W]` for 3-D nodes) are
> accepted and mean-pooled over spatial dims before the attention bias projection.  Spatial dims do
> **not** need to match the node spatial dims.  Mismatched rank (e.g. 5-D edges into a 2-D-configured
> GAT) raises `NotImplementedError`.  Use `TensorGraphSAGELayer` or `TensorGINLayer` for full
> spatial edge-feature processing (no pooling).

---

## Factory API

### `make_layer` — create any layer by name

```python
from tgraphx import make_layer

# 2-D spatial GAT with 2 attention heads
layer = make_layer("gat", in_shape=(8, 4, 4), out_shape=(16, 4, 4), heads=2, residual=True)

# Vector linear message passing
layer = make_layer("linear", in_shape=(32,), out_shape=(64,), aggr="mean")
```

| Name | Layer class | Shape support |
|------|-------------|--------------|
| `"conv"` | `ConvMessagePassing` | 2-D / 3-D spatial |
| `"gat"` | `TensorGATLayer` | 2-D / 3-D spatial |
| `"sage"` | `TensorGraphSAGELayer` | 2-D / 3-D spatial |
| `"gin"` | `TensorGINLayer` | 2-D / 3-D spatial |
| `"linear"` | `LinearMessagePassing` | vector only |
| `"legacy_attention"` | `AttentionMessagePassing` | vector / 2-D spatial |

### `build_model` — complete task model

```python
from tgraphx import build_model

# Node classification on vector features
model = build_model(
    task="node_classification",
    layer="linear",
    in_shape=(32,),
    hidden_shape=(64,),
    num_layers=3,
    num_classes=5,
)
out = model(x, edge_index)    # [N, 5]

# Graph classification on 2-D image patches
model = build_model(
    task="graph_classification",
    layer="gat",
    in_shape=(3, 4, 4),       # [C, ph, pw]
    hidden_shape=(8, 4, 4),
    num_layers=2,
    num_classes=10,
    heads=2,
    pooling="mean",
)
out = model(x, edge_index, batch=batch)   # [G, 10]
```

### Task support matrix

| Task | Vector `[N,D]` | 2-D spatial `[N,C,H,W]` | 3-D volumetric `[N,C,D,H,W]` |
|------|:-:|:-:|:-:|
| `node_classification` | ✓ | ✓ | ✓ |
| `node_regression` | ✓ | ✓ | ✓ |
| `graph_classification` | ✓ | ✓ | ✓ |
| `graph_regression` | ✓ | ✓ | ✓ |
| `edge_prediction` | ✓ | ✓ (spatial pool) | ✓ (spatial pool) |
| `link_prediction` | deferred | deferred | deferred |

> **Spatial / volumetric tasks**: after the GNN stack the factory applies
> global spatial average-pooling to flatten `[N, C, *spatial]` → `[N, C]`
> before the linear head.  Spatial resolution is preserved inside every
> GNN layer and only collapsed at the final readout.

### `build_model_from_config` — config-driven construction

```python
from tgraphx import build_model_from_config

# From a Python dict (no eval, no exec)
config = {
    "model": {
        "task": "graph_classification",
        "layer": "gat",
        "in_shape": [8, 4, 4],
        "hidden_shape": [16, 4, 4],
        "num_layers": 2,
        "num_classes": 3,
        "heads": 2,
        "residual": True,
        "dropout": 0.1,
    }
}
model = build_model_from_config(config)

# From a JSON file
model = build_model_from_config("config.json")

# From a YAML file (requires PyYAML)
model = build_model_from_config("config.yaml")
```

---

## Examples

```bash
# Tensor-aware GCN-style spatial message passing
python examples/minimal_spatial_message_passing.py

# Graph classification — short training loop with synthetic data
python examples/minimal_graph_classifier.py

# Tensor-aware multi-head GAT (verifies that attention weights sum to 1
# per destination per head)
python examples/tensor_gat_minimal.py

# Tensor-aware GraphSAGE (mean / max / with-edge-features variants)
python examples/tensor_graphsage_minimal.py

# Custom user-defined message-passing layer subclass
python examples/custom_message_passing.py

# Trainability sanity: tiny overfit on a relational synthetic task with GAT
python examples/tiny_overfit_tensor_gat.py

# Edge-feature dependency: GAT/GIN/SAGE with vector edge features
python examples/tiny_overfit_edge_features.py

# Deep 8-layer stack gradient sanity for every GNN family
python examples/gradient_sanity_stack.py

# Graph builder directedness and self-loop demo
python examples/directed_vs_undirected_graphs.py

# 2-D image → patch → GNN (ConvMP + GAT)
python examples/image_patch_graph.py

# 3-D volume → patch → GNN (ConvMP + GAT)
python examples/volume_patch_graph.py

# All four GNN families on a 2-D and 3-D grid graph
python examples/gnn_family_with_graph_builders.py

# Factory examples (graph builders + model factories)
python examples/01_vector_node_classification.py
python examples/02_spatial_graph_classification.py
python examples/03_volumetric_graph_classification.py
python examples/04_config_based_model.py
python examples/05_edge_prediction.py
```

---

## API reference

### `Graph`

```python
from tgraphx import Graph

g = Graph(
    node_features,                 # torch.Tensor  [N, ...]              required
    edge_index=None,               # torch.LongTensor [2, E] or None     optional
    edge_weight=None,              # torch.Tensor [E]                    optional
    edge_features=None,            # torch.Tensor [E, ...]               optional
    node_labels=None,              # torch.Tensor [N, ...]               optional
    edge_labels=None,              # torch.Tensor [E, ...]               optional
    graph_label=None,              # torch.Tensor (any shape)            optional
    metadata=None,                 # dict                                optional
)
g.clone()                          # deep copy (tensors and metadata)
g.to("cuda", dtype=torch.float32)  # moves all tensor fields; dtype only
                                   #   applies to floating-point tensors
g.cpu(); g.cuda()                  # convenience aliases
g.add_self_loops(); g.remove_self_loops()
g.make_undirected(reduce="mean")   # symmetrize, coalesce duplicates
g.is_undirected()                  # structural check (ignores weights)
g.validate()                       # re-run validation after manual mutation
g.num_nodes, g.num_edges, g.feature_shape, g.edge_feature_shape
g.has_edges, g.has_edge_weight, g.has_edge_features
```

Supported per-node feature layouts: `[N, D]`, `[N, C, H, W]`, and storage-level
`[N, C, D, H, W]`. Edge features mirror this: `[E, D_e]`, `[E, C_e, H, W]`, and
storage-level `[E, C_e, D, H, W]`.

`Graph.__init__` raises immediately on:
- non-Tensor inputs
- `node_features` or `edge_features` with fewer than 2 dimensions
- `edge_index` with wrong shape, wrong dtype (`torch.long` required), or out-of-range indices
- `edge_weight` that is not 1-D, or whose length differs from `E`
- per-edge tensors (`edge_weight` / `edge_features` / `edge_labels`) supplied
  without an `edge_index`
- device mismatch between `node_features` and any other tensor field
- length mismatch between `node_features` / `edge_index` and any per-node /
  per-edge tensor

### `GraphBatch`

```python
from tgraphx import GraphBatch

batch = GraphBatch([g1, g2, g3])
# batch.node_features  [N_total, ...]
# batch.edge_index     [2, E_total]   — indices offset per graph
# batch.edge_weight    [E_total]      — concatenated if every graph has it
# batch.edge_features  [E_total, ...] — concatenated if every graph has it
# batch.node_labels    [N_total, ...] — concatenated if every graph has it
# batch.edge_labels    [E_total, ...] — concatenated if every graph has it
# batch.graph_labels   [B, ...]       — stacked if every graph has it
# batch.metadata       list[Any]      — verbatim, length B (None entries OK)
# batch.batch          [N_total]      — graph membership (dtype=long)
batch.to("cuda")
```

All graphs must share the same per-node feature shape (and per-edge feature
shape, when present). Passing graphs with different spatial sizes raises a
`ValueError` with a descriptive message. Optional per-edge tensors must be
present on every graph that has edges, or none — mixing-some-with-none is
rejected so per-edge data is never silently dropped.

### `ConvMessagePassing`

```python
from tgraphx.layers import ConvMessagePassing

layer = ConvMessagePassing(
    in_shape=(C, H, W),          # tuple: per-node input shape (spatial only)
    out_shape=(C_out, H, W),     # H and W must stay equal to in_shape's H, W
    aggr="sum",                  # "sum" (default) | "mean" | "max"
    use_edge_features=False,     # set True to concatenate edge tensors into messages
    aggregator_params=None,      # dict forwarded to DeepCNNAggregator; e.g.
                                 #   {"num_layers": 2, "dropout_prob": 0.1}
    residual=False,              # add skip connection when in_shape == out_shape
)
out = layer(node_features, edge_index)              # [N, C_out, H, W]
out = layer(node_features, edge_index, edge_features)  # with edge features
```

> **`aggr="max"`** is supported via `scatter_reduce_(reduce='amax')`.  When `chunk_size` is also
> set, `aggr="max"` falls back to the unchunked path with a `warnings.warn`.
> Use `GraphClassifier(pooling="max")` for graph-level max readout.

### `AttentionMessagePassing`

```python
from tgraphx.layers import AttentionMessagePassing

# Spatial path
layer = AttentionMessagePassing(in_shape=(C, H, W), out_shape=(C_out, H, W))

# Vector path (also supported)
layer = AttentionMessagePassing(in_shape=(D,), out_shape=(D_out,))

out = layer(node_features, edge_index)   # [N, C_out, H, W] or [N, D_out]
```

**Important:** This layer computes `attn = sigmoid(q·k / √d)` independently per edge. Attention weights are **not** normalised over each destination node's neighbourhood (no softmax). This differs from standard GAT. **For true GAT, use `TensorGATLayer` below.**

### `TensorGATLayer` (true multi-head GAT)

True GAT-style attention adapted to spatial features. For every destination
node `j` and every head `k`, attention weights satisfy `Σ_i α_ij^k = 1`.

```python
from tgraphx.layers import TensorGATLayer

# 4 heads × 8 channels each = 32 output channels (heads concatenated)
layer = TensorGATLayer(
    in_channels=16,
    out_channels=32,        # divisible by num_heads when concat_heads=True
    num_heads=4,
    concat_heads=True,      # False → average heads, output is per-head channels
    negative_slope=0.2,     # LeakyReLU before edge softmax
    attn_dropout=0.0,       # dropout on attention weights (training only)
    residual=True,          # auto 1×1 projection if in/out channels differ
    bias=True,
    add_self_loops=False,   # True ensures every node has at least 1 in-edge
    use_edge_features=False,  # set True to enable EGAT-style vector edge bias
    edge_dim=None,            # required when use_edge_features=True
)
out = layer(x, edge_index)                                  # [N, 32, H, W]

# Inspect attention weights (e.g. for visualisation or testing):
out, attn = layer(x, edge_index, return_attention=True)
# attn shape: [E, num_heads]; sums to 1 over incoming edges per destination per head.

# EGAT-style vector edge attention bias (e.g. relative box coords, IoU,
# distances, scale ratio):
layer_e = TensorGATLayer(
    in_channels=16, out_channels=32, num_heads=4,
    use_edge_features=True, edge_dim=3,
)
out = layer_e(x, edge_index, edge_features=ef)   # ef: [E, 3]
```

Attention is **scalar per `(edge, head)`** in this implementation: the
projected query and key feature maps are mean-pooled over `H × W` before
being scored, while the value tensors keep their full spatial layout
during aggregation. Per-pixel and per-channel attention modes are not yet
supported.

**Spatial edge features** (`[E, C_e, H, W]` for `spatial_rank=2`;
`[E, C_e, D, H, W]` for `spatial_rank=3`) are accepted: spatial dims are
mean-pooled to a channel vector before the per-`(edge, head)` attention bias
projection (spatial dims need not match node spatial dims).  Use
`TensorGraphSAGELayer` or `TensorGINLayer` for full spatial edge-feature
processing without pooling.

### `TensorGraphSAGELayer`

Tensor-aware GraphSAGE: `h_j' = W_self(h_j) + W_neigh(AGG_i h_i)`.

```python
from tgraphx.layers import TensorGraphSAGELayer

layer = TensorGraphSAGELayer(
    in_channels=16,
    out_channels=32,
    aggr="mean",            # "mean" or "max"
    normalize=False,        # True → L2-normalise output channel vector per pixel
    bias=True,
    residual=False,
    use_edge_features=False,
    edge_dim=None,             # required when use_edge_features=True
    edge_features_kind="spatial",  # or "vector" — see below
)
out = layer(x, edge_index)                                  # [N, 32, H, W]

# Spatial edge features [E, edge_dim, H, W] — concatenated to source.
layer_s = TensorGraphSAGELayer(
    in_channels=16, out_channels=32,
    use_edge_features=True, edge_dim=4, edge_features_kind="spatial",
)
out = layer_s(x, edge_index, edge_features=ef_spatial)

# Vector edge features [E, edge_dim] — projected to channel bias and added
# to W_neigh(h_src) before aggregation.
layer_v = TensorGraphSAGELayer(
    in_channels=16, out_channels=32,
    use_edge_features=True, edge_dim=3, edge_features_kind="vector",
)
out = layer_v(x, edge_index, edge_features=ef_vector)        # ef_vector: [E, 3]
```

Isolated nodes (no incoming edges) receive only the self transform — the
neighbour aggregate is zero.

### `TensorGINLayer`

Tensor-aware GIN / GINEConv: `h_j' = MLP((1+ε)·h_j + Σ_i m_ij)`.

```python
from tgraphx.layers import TensorGINLayer

# Default 1×1 Conv MLP (preserves spatial layout)
layer = TensorGINLayer(
    in_channels=16,
    out_channels=32,
    hidden_channels=24,     # defaults to out_channels
    eps=0.0,
    train_eps=False,        # set True to make ε a learnable scalar parameter
    use_batchnorm=False,
)
out = layer(x, edge_index)                                  # [N, 32, H, W]

# Custom MLP (any nn.Module mapping [N, in_channels, H, W] → [N, out_channels, H, W])
import torch.nn as nn
custom_mlp = nn.Sequential(
    nn.Conv2d(16, 24, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(24, 32, kernel_size=1),
)
layer = TensorGINLayer(in_channels=16, out_channels=32, mlp=custom_mlp)

# GINEConv-style spatial edge inclusion: messages = ReLU(h_src + φ(e_ij))
layer_s = TensorGINLayer(
    in_channels=16, out_channels=32,
    use_edge_features=True, edge_dim=4, edge_features_kind="spatial",
)
out = layer_s(x, edge_index, edge_features=ef_spatial)

# Vector edge features [E, edge_dim] — projected to [E, in_channels, 1, 1]
# and broadcast over H × W before ReLU.
layer_v = TensorGINLayer(
    in_channels=16, out_channels=32,
    use_edge_features=True, edge_dim=3, edge_features_kind="vector",
)
out = layer_v(x, edge_index, edge_features=ef_vector)        # ef_vector: [E, 3]
```

### Custom layers via `TensorMessagePassingLayer`

```python
import torch, torch.nn as nn
from tgraphx.layers import TensorMessagePassingLayer

class MyConv(TensorMessagePassingLayer):
    def __init__(self, c_in, c_out):
        super().__init__(in_shape=(c_in,), out_shape=(c_out,), aggr="mean")
        self.W_g = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.W_v = nn.Conv2d(c_in, c_out, kernel_size=1)

    def message(self, src, dest, edge_attr):
        gate = torch.sigmoid(self.W_g(src + dest))
        return gate * self.W_v(src)

    def update(self, node_feature, aggregated_message):
        return aggregated_message
```

The base class handles per-edge gather and aggregation (`sum` or `mean`)
for arbitrary trailing tensor shapes. See
[`examples/custom_message_passing.py`](examples/custom_message_passing.py).

### `CNNEncoder`

```python
from tgraphx.models import CNNEncoder

enc = CNNEncoder(
    in_channels=3,
    out_features=64,
    num_layers=3,         # total Conv2d blocks
    hidden_channels=64,
    dropout_prob=0.3,
    use_batchnorm=True,
    use_residual=True,    # residual skip in intermediate blocks
    pool_layers=1,        # how many blocks include SafeMaxPool2d(2)
    return_feature_map=True,   # True → [N, out_features, H', W']
                               # False → [N, out_features] (global avg pool)
    pre_encoder=None,     # optional PreEncoder instance
)
features = enc(patches)   # patches: [N, in_channels, patch_H, patch_W]
```

### `GraphClassifier`

```python
from tgraphx.models import GraphClassifier

clf = GraphClassifier(
    in_shape=(C, H, W),
    hidden_shape=(C_hidden, H, W),
    num_classes=5,
    num_layers=2,
    aggr="sum",
    pooling="mean",        # "mean" | "sum" | "max"
)
logits = clf(
    node_features,         # [N, C, H, W]
    edge_index,            # [2, E]
    batch=batch_vector,    # [N] — required for graph-level output
    edge_features=None,    # optional
)                          # → [num_graphs, num_classes]
```

### `NodeClassifier`

```python
from tgraphx.models import NodeClassifier

nc = NodeClassifier(
    in_shape=(64,),        # vector features only
    hidden_shape=(128,),
    num_classes=3,
    num_layers=2,
)
logits = nc(node_features, edge_index)   # [N, num_classes]
```

### `CNN_GNN_Model`

A full CNN → GNN → classify pipeline that accepts **pre-split** node patches.

```python
from tgraphx.models import CNN_GNN_Model

model = CNN_GNN_Model(
    cnn_params=dict(
        in_channels=3,
        out_features=64,
        num_layers=2,
        hidden_channels=64,
        dropout_prob=0.0,
        use_batchnorm=False,
        use_residual=False,
        pool_layers=1,
        return_feature_map=True,
    ),
    gnn_in_dim=(64, 8, 8),      # must match CNN output shape exactly
    gnn_hidden_dim=(64, 8, 8),
    num_classes=10,
    num_gnn_layers=2,
    gnn_dropout=0.3,            # forwarded to DeepCNNAggregator
    residual=True,              # per-layer skip connection
    skip_cnn_to_classifier=False,
)

# raw_patches: pre-split by the user; shape [N, in_channels, pH, pW]
logits = model(raw_patches, edge_index)            # [N, num_classes]  (node-level)
logits = model(raw_patches, edge_index, batch=b)   # [G, num_classes]  (graph-level)
```

### `get_device`

```python
from tgraphx.core.utils import get_device

device = get_device()             # CUDA (if available) → MPS → CPU
device = get_device(device_id=1)  # specific CUDA device
```

---

## Shape conventions

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `node_features` | `[N, D]`, `[N, C, H, W]`, or `[N, C, D, H, W]` | float | vector, 2-D spatial, or 3-D volumetric; N = number of nodes |
| `node_features` (vector) | `[N, D]` | float | For `NodeClassifier` / `LinearMessagePassing` |
| `edge_index` | `[2, E]` | `torch.long` | Row 0 = source nodes, row 1 = destination nodes |
| `edge_features` | `[E, ...]` | float | Optional; length must equal E |
| `batch` | `[N]` | `torch.long` | Maps each node to its graph index |

---

## Device support

| Device | Status |
|--------|--------|
| CPU | Tested (CI) |
| NVIDIA CUDA | Tested (PyTorch 2.10, CUDA 12.8) |
| Apple Silicon MPS | Code path exists in `get_device()`; not yet in CI |
| Multi-GPU | Not supported |

```python
from tgraphx.core.utils import get_device

device = get_device()
model.to(device)
g.to(device)
batch.to(device)
```

---

## Supported Python and PyTorch versions

| Python | PyTorch | Status |
|--------|---------|--------|
| 3.10 | ≥ 1.13 | CI (ubuntu-latest) |
| 3.11 | ≥ 1.13 | CI (ubuntu-latest) |
| 3.12 | ≥ 1.13 | CI (ubuntu-latest) |
| 3.9 | ≥ 1.13 | Should work; not in CI |

---

## Support status

### Legend

| Label | Meaning |
|-------|---------|
| ✅ Stable | Tested in CI; API is stable |
| 🧪 Experimental | Available but not yet guaranteed-stable |
| ⚠️ Best-effort | Works in practice; known constraints documented |
| ⏳ Planned | On roadmap; not yet implemented |
| ❌ Not supported | Out of scope for the current release |
| 🔒 Opt-in | Disabled by default; explicitly enabled by the user |

### Backend support

| Backend | Forward | AMP | torch.compile | CI coverage | Status | Notes |
|---------|:-------:|:---:|:-------------:|:-----------:|:------:|-------|
| CPU | ✅ | ⚠️ bfloat16 | ✅ | Full CI | ✅ Stable | Compile overhead for small graphs |
| CUDA | ✅ | ⚠️ op-dependent | ✅ | Full CI | ✅ Stable | `index_add_` requires dtype match under float16 |
| MPS (Apple Silicon) | ✅ | ⚠️ limited | ⚠️ partial | No CI | ⚠️ Best-effort | PyTorch operator coverage varies |
| Linux | ✅ | ✅ | ✅ | Full CI (ubuntu-latest) | ✅ Stable | Primary CI platform |
| Windows | ✅ | ✅ | ✅ | No CI | ⚠️ Best-effort | Not in CI; known to install correctly |
| macOS | ✅ | ⚠️ limited | ⚠️ | No CI | ⚠️ Best-effort | MPS path; no CI coverage |
| Multi-GPU | ❌ | ❌ | ❌ | No CI | ❌ Not supported | — |

> ⚠️ **Best-effort backend:** MPS support depends on PyTorch operator coverage per release.
> CPU workflows are tested; MPS-specific AMP/compile paths may fall back or be skipped.
>
> ⚠️ **Windows/macOS:** The package installs and runs on Windows and macOS, but automated tests
> run on Ubuntu only.  Regressions on those platforms may not be caught until user reports.

### Feature support

| Feature | Status | Notes |
|---------|:------:|-------|
| Vector node features `[N, D]` | ✅ Stable | `LinearMessagePassing`, `"linear"` factory |
| 2-D spatial node features `[N, C, H, W]` | ✅ Stable | All four spatial layers |
| 3-D volumetric node features `[N, C, D, H, W]` | ✅ Stable | `spatial_rank=3` |
| Arbitrary-rank tensors (rank ≥ 4) | ❌ Not supported | Only vector, 2-D, 3-D |
| Edge weights `[E]` | ✅ Stable | All layers |
| Vector edge features `[E, D_e]` | ✅ Stable | GAT, SAGE, GIN |
| Spatial edge features `[E, C_e, H, W]` | ⚠️ Best-effort | ConvMP (concat); GAT (mean-pooled); SAGE/GIN (full) |
| Volumetric edge features `[E, C_e, D, H, W]` | ⚠️ Best-effort | Same as spatial; `spatial_rank=3` |
| Graph Transformer | ❌ Not supported | ⏳ Planned v0.2.5 feasibility study |
| Heterogeneous graphs | ❌ Not supported | ⏳ Planned v0.2.5+ |
| Temporal graphs | ❌ Not supported | ⏳ Planned v0.2.5+ |
| Learned graph construction | ❌ Not supported | `edge_index` is always user-supplied |
| PyG/DGL converters | ❌ Not supported | ⏳ Planned v0.2.5 |
| MLflowLogger | ❌ Not supported | Use `mlflow` client directly |
| Dashboard | 🔒 Opt-in | Launch explicitly; zero overhead when off |
| Offline dashboard export | ✅ Stable | `--export-html` or `export_dashboard_html()` |
| Multi-run dashboard | ✅ Stable | Point `--logdir` at parent directory |
| Hardware monitoring | 🔒 Opt-in | `pip install "tgraphx[monitoring]"` |
| TensorBoard logging | 🔒 Opt-in | `pip install "tgraphx[tracking]"`; `TensorBoardLogger` |

### Scalability support

| Feature | Status | Notes |
|---------|:------:|-------|
| `ConvMessagePassing` chunked forward | ✅ Stable | `aggr="sum"` / `"mean"`; max falls back with warning |
| `TensorGraphSAGELayer` chunked forward | ✅ Stable | v0.2.3; mean and max; pass `chunk_size=K` to `forward()` |
| `TensorGINLayer` chunked forward | ✅ Stable | v0.2.3; sum aggregation; pass `chunk_size=K` to `forward()` |
| `TensorGATLayer` chunked forward | ⏳ Planned v0.2.4 | Requires two-pass algorithm for destination-wise softmax |
| `build_grid_graph` / `build_grid_graph_3d` | ✅ Stable | O(E) — scales well |
| `build_random_graph` | ✅ Stable | O(E) — scales well |
| `build_knn_graph` / `build_radius_graph` | ⚠️ Best-effort | O(N²) time; `chunk_size=K` reduces peak memory to O(K×N) |
| `build_fully_connected_graph` | ⚠️ Best-effort | O(N²) edges; N > 5 000 emits warning |
| `build_iou_graph` | ⚠️ Best-effort | O(N²) IoU; `chunk_size=K` reduces peak memory to O(K×N) |
| `build_random_graph` | ✅ Stable | `algorithm="sample"` uses O(num_edges) memory for large N |
| Dashboard metrics API | ✅ Stable | Incremental `?since_row=N`; `--max-metric-rows` cap; byte-seek tail-read (v0.2.3) |
| Large `metrics.csv` tail-read | ✅ Stable | v0.2.3: byte-seek on append; full reparse on rotation/truncation |

> ⚠️ **Scalability warning:** `build_knn_graph`, `build_radius_graph`, `build_fully_connected_graph`,
> and `build_iou_graph` use pairwise `torch.cdist` or enumerate all pairs.  Memory and time grow as
> **O(N²)**.  A `warnings.warn` is emitted when node count exceeds the threshold (10 000 for kNN/radius,
> 5 000 for fully-connected/IoU).  For large graphs use an approximate-NN library instead.

### Attention support

| Feature | Status | Notes |
|---------|:------:|-------|
| Scalar attention per `(edge, head)` | ✅ Stable | Default in `TensorGATLayer` |
| Vector edge attention bias | ✅ Stable | `use_edge_features=True, edge_dim=D` |
| Spatial edge attention bias (2-D/3-D) | ⚠️ Best-effort | Accepted; mean-pooled to scalar before projection |
| Per-channel attention | ❌ Not supported | ⏳ Planned v0.2.4 |
| Per-pixel attention | ❌ Not supported | ⏳ Planned v0.2.4 |
| Per-voxel attention | ❌ Not supported | ⏳ Planned v0.2.4 |

---

## Limitations

- **Scope:** TGraphX provides tensor-aware adaptations of GCN-style, GAT, GraphSAGE, and GIN. It is **not** a drop-in PyTorch Geometric replacement: heterogeneous graphs, temporal graphs, graph transformers, and learned graph construction are all out of scope for the current release.
- **`AttentionMessagePassing` is not GAT.** It uses per-edge sigmoid gating without softmax normalisation. Use `TensorGATLayer` for true multi-head GAT.
- **Scalar attention only in `TensorGATLayer`.** Per-channel and per-pixel attention modes are not implemented.
- **GAT edge features (vector or spatial → pooled).** `TensorGATLayer` accepts `[E, edge_dim]` vectors and matching-rank spatial tensors (`[E, edge_dim, H, W]` when `spatial_rank=2`, `[E, edge_dim, D, H, W]` when `spatial_rank=3`). Spatial tensors are mean-pooled over their spatial dims before the per-`(edge, head)` attention bias projection — this keeps the scalar-attention regime stable. Mixed-rank edges (e.g. 5-D into a 2-D-configured GAT, or 4-D into a 3-D-configured GAT) raise `NotImplementedError`. `TensorGraphSAGELayer` and `TensorGINLayer` use matching-rank spatial edge tensors directly without pooling.
- **Node-feature ranks supported.** TGraphX supports vector `[N, D]`, 2-D spatial `[N, C, H, W]`, and 3-D volumetric `[N, C, D, H, W]` node features for the listed layers. It does not claim arbitrary-rank tensor support. Vector node features are handled by `LinearMessagePassing`; 2-D / 3-D by `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, and `TensorGINLayer` (with `spatial_rank=2` or `3` at construction time).
- **Graph builders are included** (`build_grid_graph`, `build_knn_graph`, etc.) but cover common structural patterns only. More complex topology must be supplied as `edge_index` directly.
- **Patch helpers are included** (`image_to_patches`, `volume_to_patches`) for extracting non-overlapping patches; they require exact-divisible dimensions and do not pad automatically.
- **Dashboard** is a local-first lightweight monitor, not a TensorBoard replacement. See [Dashboard](#dashboard) and [docs/dashboard.md](docs/dashboard.md) for scope.
- **Differentiability:** All learned parameters (CNN encoder, message-passing layers, classifier) are end-to-end differentiable. The graph topology (`edge_index`) is user-provided and is not learned by the model.

## GNN family coverage

| GNN family | Implemented? | Tested? | Limitations |
|------------|-------------|---------|------------|
| Tensor-aware GCN-style (Conv message passing) | ✅ `ConvMessagePassing` | ✅ | 2-D `[N, C, H, W]` and 3-D `[N, C, D, H, W]`; edge features must be matching-rank spatial with channel count = node channel count |
| Tensor-aware GAT (multi-head) | ✅ `TensorGATLayer` | ✅ | 2-D and 3-D node features (`spatial_rank=2`/`3`); scalar attention per `(edge, head)`; vector `[E, D_e]` and matching-rank spatial / volumetric edge features (mean-pooled); no per-pixel / per-voxel attention |
| Tensor-aware GraphSAGE | ✅ `TensorGraphSAGELayer` | ✅ | 2-D and 3-D node features (`spatial_rank=2`/`3`); mean / max only; no LSTM aggregator |
| Tensor-aware GIN / GINEConv | ✅ `TensorGINLayer` | ✅ | 2-D and 3-D node features (`spatial_rank=2`/`3`) |
| MPNN-style custom layer | ✅ `TensorMessagePassingLayer` base | ✅ (subclass test + example) | — |
| Edge-conditioned MP (spatial / volumetric) | ✅ `ConvMessagePassing`, `TensorGATLayer` (mean-pooled), `TensorGraphSAGELayer`, `TensorGINLayer` | ✅ | edge features `[E, C_e, H, W]` (2-D) or `[E, C_e, D, H, W]` (3-D, matching the layer's `spatial_rank`) |
| Per-edge `edge_weight` (`[E]`) | ✅ all four message-passing layers, 2-D and 3-D | ✅ | scales messages before aggregation; on GAT applied after softmax-normalised attention |
| 3-D / volumetric node features | ✅ `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer` | ✅ | `[N, C, D, H, W]`; pass `spatial_rank=3` to GAT/SAGE/GIN, or `(C, D, H, W)` `in_shape` to `ConvMessagePassing`. `DeepCNNAggregator` is rank-aware. `LinearMessagePassing` covers vector `[N, D]` and is unaffected. |
| Edge-conditioned MP (vector) | ✅ `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer` | ✅ | edge features `[E, D_e]`; `edge_features_kind="vector"` |
| `aggr="sum"\|"mean"\|"max"` base | ✅ all three modes | ✅ hand-computed + backward | `ConvMessagePassing` `aggr="max"` routes through `scatter_max` |
| Graph Transformer | ❌ | — | not implemented |
| Heterogeneous graphs | ❌ | — | not supported |
| Temporal / spatiotemporal graphs | ❌ | — | not supported |
| Learned graph construction | ❌ | — | `edge_index` is always user-supplied |
| Arbitrary-rank tensor support beyond rank 0 / 2 / 3 | ❌ | — | only vector, 2-D, and 3-D shapes are supported |

---

## Project structure

```
TGraphX/
├── tgraphx/
│   ├── __init__.py          # public API re-exports (Graph, layers, builders, training, …)
│   ├── core/
│   │   ├── graph.py         # Graph, GraphBatch
│   │   ├── dataloader.py    # GraphDataset, GraphDataLoader
│   │   ├── graph_utils.py   # edge topology helpers
│   │   └── utils.py         # load_config, get_device
│   ├── layers/
│   │   ├── base.py             # TensorMessagePassingLayer, LinearMessagePassing
│   │   ├── conv_message.py     # ConvMessagePassing
│   │   ├── attention_message.py# AttentionMessagePassing (legacy sigmoid)
│   │   ├── gat.py              # TensorGATLayer (true multi-head GAT)
│   │   ├── sage.py             # TensorGraphSAGELayer
│   │   ├── gin.py              # TensorGINLayer / GINEConv
│   │   ├── factory.py          # make_layer()
│   │   ├── _scatter.py         # internal: edge_softmax, scatter_*
│   │   ├── aggregator.py       # DeepCNNAggregator
│   │   └── safe_pool.py        # SafeMaxPool2d
│   ├── models/
│   │   ├── factory.py          # build_model(), build_model_from_config()
│   │   ├── edge_predictor.py   # EdgePredictor
│   │   ├── regressors.py       # NodeRegressor, GraphRegressor
│   │   ├── graph_classifier.py # GraphClassifier
│   │   ├── node_classifier.py  # NodeClassifier
│   │   ├── cnn_encoder.py      # CNNEncoder
│   │   ├── cnn_gnn_model.py    # CNN_GNN_Model
│   │   └── pre_encoder.py      # PreEncoder (optional ResNet-18)
│   ├── graph_builders.py    # build_grid_graph, build_knn_graph, image_to_patches, …
│   ├── training.py          # train_epoch, evaluate, fit, set_seed, checkpointing, …
│   ├── tracking.py          # CSVLogger, TensorBoardLogger, write_graph_stats
│   ├── performance.py       # env_report, recommended_device, estimate_message_memory
│   └── dashboard/
│       ├── app.py           # DashboardServer, export_dashboard_html, CLI main()
│       ├── __init__.py      # launch_dashboard, launch_dashboard_background
│       └── static/          # dashboard.css, dashboard.js (packaged in wheel)
├── tests/
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_graph.py
│   ├── test_layers.py
│   ├── test_models.py
│   ├── test_devices.py
│   ├── test_gnn_families.py    # GAT, GraphSAGE, GIN, custom subclass
│   ├── test_math.py            # edge-order invariance, permutation-equivariance, H=W=1, isolated nodes
│   ├── test_gradients.py       # single-layer backward, 8-layer stacks, tiny overfit
│   └── test_edge_features.py   # vector edge features for GAT, SAGE, GIN
├── examples/
│   ├── minimal_spatial_message_passing.py
│   ├── minimal_graph_classifier.py
│   ├── tensor_gat_minimal.py
│   ├── tensor_graphsage_minimal.py
│   ├── custom_message_passing.py
│   ├── tiny_overfit_tensor_gat.py      # trainability check per GNN family
│   ├── tiny_overfit_edge_features.py   # vector edge feature dependency check
│   └── gradient_sanity_stack.py        # 8-layer deep stack gradient norms
├── pyproject.toml
├── requirements.txt
├── environment.yml
└── LICENSE
```

---

## Development

```bash
# Install with dev dependencies (pytest, build, twine)
pip install -e ".[dev]"

# Run the test suite (CPU tests always run; CUDA/MPS skipped if unavailable)
pytest

# Run a specific test file
pytest tests/test_layers.py -v

# Run the examples
python examples/minimal_spatial_message_passing.py
python examples/minimal_graph_classifier.py
```

---

## Authorship

**Software author and maintainer:**
Arash Sajjadi, PhD Candidate in Computer Science, University of Saskatchewan
([arash.sajjadi@usask.ca](mailto:arash.sajjadi@usask.ca))

**Academic supervision:**
Mark Eramian, PhD Supervisor / Academic Advisor, University of Saskatchewan

**Related preprint:**
*TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning*
Arash Sajjadi and Mark Eramian — [arXiv:2504.03953](https://arxiv.org/abs/2504.03953)

> The software package is developed and maintained by Arash Sajjadi.
> Mark Eramian is Arash Sajjadi's PhD supervisor and co-author of the related
> academic preprint.  Software authorship and paper co-authorship are separate
> roles; both are acknowledged accurately above.

---

## Citation

If you use TGraphX in your research, please cite:

```bibtex
@misc{sajjadi2025tgraphxtensorawaregraphneural,
      title={TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning},
      author={Arash Sajjadi and Mark Eramian},
      year={2025},
      eprint={2504.03953},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.03953},
}
```

---

## License

TGraphX is released under the [MIT License](LICENSE).

---

Questions, issues, or contributions are welcome — please open a [GitHub issue](https://github.com/arashsajjadi/TGraphX/issues) or pull request.
