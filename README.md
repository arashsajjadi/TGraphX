# TGraphX

[![Tests](https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml/badge.svg)](https://github.com/arashsajjadi/TGraphX/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.13+](https://img.shields.io/badge/pytorch-1.13%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

📄 **Preprint:** [TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning](https://arxiv.org/abs/2504.03953) · *Sajjadi & Eramian, arXiv 2025*

TGraphX is a PyTorch library for graph neural networks whose **node features are multi-dimensional tensors** — such as `[C, H, W]` image-patch feature maps — rather than flat vectors. Convolutional message passing operates directly on spatial feature maps at each node, so local structure is never destroyed by flattening.

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

- **Image-to-patch splitting.** `CNN_GNN_Model` and other layers expect pre-split node patches supplied by the user. No patch-extraction utility exists yet.
- **Graph construction.** `edge_index` is always user-supplied. There are no built-in graph builders (kNN, grid, radius, IoU, learned adjacency).
- **Graph Transformers** (global attention with positional encodings).
- **Per-channel and per-pixel attention** for `TensorGATLayer`. Currently only scalar attention per `(edge, head)` is supported (the default safer choice).
- **Spatial edge features in `TensorGATLayer`.** GAT supports vector edge features `[E, edge_dim]` as an additive attention bias only. `TensorGraphSAGELayer` and `TensorGINLayer` support **both** spatial `[E, edge_dim, H, W]` and vector `[E, edge_dim]` edge features (selected via `edge_features_kind`).
- **3-D volumetric aggregation.** `DeepCNNAggregator` uses `Conv2d`, so node features of shape `[C, D, H, W]` will fail at the `ConvMessagePassing` aggregation step even though the message function has a `Conv3d` branch. The new `TensorGATLayer`, `TensorGraphSAGELayer`, and `TensorGINLayer` are also 2-D only.
- **Heterogeneous and temporal graphs.**

---

## Installation

TGraphX is **not yet published as a stable release on PyPI**. Install from source:

```bash
git clone https://github.com/arashsajjadi/TGraphX.git
cd TGraphX
pip install -e .
```

Runtime dependencies are installed automatically:

```
torch>=1.13
torchvision>=0.14
pyyaml>=5.4
```

For a specific PyTorch build (e.g., CPU-only or a particular CUDA version), install PyTorch first and then install TGraphX:

```bash
# Example: CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

See [pytorch.org](https://pytorch.org/get-started/locally/) for GPU-specific install commands. A Conda environment file is provided in [`environment.yml`](environment.yml).

---

## Quickstart

```python
import torch
from tgraphx import Graph
from tgraphx.layers import ConvMessagePassing

# 6 nodes, each a 16-channel 8×8 feature map
N, C, H, W = 6, 16, 8, 8
node_features = torch.randn(N, C, H, W)

# Directed cycle: 0→1→2→3→4→5→0
src = torch.arange(N)
edge_index = torch.stack([src, (src + 1) % N])   # [2, N]

g = Graph(node_features, edge_index)   # validates inputs

layer = ConvMessagePassing(
    in_shape=(C, H, W),     # per-node input shape
    out_shape=(32, H, W),   # H and W are preserved; channels expand to 32
)
out = layer(g.node_features, g.edge_index)   # [6, 32, 8, 8]
print(out.shape)

out.sum().backward()   # all learned stages are differentiable
```

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
| `ConvMessagePassing` | ✗ | ✓ (concatenated along channels) |
| `TensorGATLayer` | ✓ (additive attention bias on logits) | ✗ |
| `TensorGraphSAGELayer` | ✓ (additive channel bias post-`W_neigh`) | ✓ (concatenated to source) |
| `TensorGINLayer` | ✓ (broadcast bias before ReLU) | ✓ (1×1 Conv2d projection) |

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
```

---

## API reference

### `Graph`

```python
from tgraphx import Graph

g = Graph(
    node_features,      # torch.Tensor  [N, ...]          required
    edge_index,         # torch.LongTensor [2, E] or None  required
    edge_features,      # torch.Tensor  [E, ...]           optional
)
g.to("cuda")            # moves all tensors in-place; returns self
```

`Graph.__init__` raises immediately on:
- non-Tensor inputs
- `edge_index` with wrong shape, wrong dtype (`torch.long` required), or out-of-range indices
- device mismatch between `node_features` and `edge_index` / `edge_features`
- `edge_features` length not equal to the number of edges

### `GraphBatch`

```python
from tgraphx import GraphBatch

batch = GraphBatch([g1, g2, g3])
# batch.node_features  [N_total, C, H, W]
# batch.edge_index     [2, E_total]   — indices offset per graph
# batch.edge_features  [E_total, ...] — concatenated if present
# batch.batch          [N_total]      — graph membership (dtype=long)
batch.to("cuda")
```

All graphs must share the same per-node feature shape. Passing graphs with different spatial sizes raises a `ValueError` with a descriptive message.

### `ConvMessagePassing`

```python
from tgraphx.layers import ConvMessagePassing

layer = ConvMessagePassing(
    in_shape=(C, H, W),          # tuple: per-node input shape (spatial only)
    out_shape=(C_out, H, W),     # H and W must stay equal to in_shape's H, W
    aggr="sum",                  # "sum" (default) | "mean"
    use_edge_features=False,     # set True to concatenate edge tensors into messages
    aggregator_params=None,      # dict forwarded to DeepCNNAggregator; e.g.
                                 #   {"num_layers": 2, "dropout_prob": 0.1}
    residual=False,              # add skip connection when in_shape == out_shape
)
out = layer(node_features, edge_index)              # [N, C_out, H, W]
out = layer(node_features, edge_index, edge_features)  # with edge features
```

> `aggr="max"` raises `NotImplementedError`. Use `GraphClassifier(pooling="max")` for graph-level max readout.

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
supported. **Spatial** edge feature tensors are not supported by this
layer — use `TensorGraphSAGELayer` or `TensorGINLayer` for those.

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
| `node_features` | `[N, C, H, W]` | float | 2-D spatial; N = number of nodes |
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

## Limitations

- **Scope:** TGraphX provides tensor-aware adaptations of GCN-style, GAT, GraphSAGE, and GIN. It is **not** a drop-in PyTorch Geometric replacement: heterogeneous graphs, temporal graphs, graph transformers, and learned graph construction are all out of scope for the current release.
- **No graph builders:** Users must supply `edge_index`. Common strategies for image-patch graphs include grid connectivity, kNN on patch centres, and IoU-based adjacency.
- **No patch extraction:** Users split images into patches before passing them to the model.
- **`AttentionMessagePassing` is not GAT.** It uses per-edge sigmoid gating without softmax normalisation. Use `TensorGATLayer` for true multi-head GAT.
- **Scalar attention only in `TensorGATLayer`.** Per-channel and per-pixel attention modes are not implemented.
- **GAT edge features are vector-only.** `TensorGATLayer` accepts `[E, edge_dim]` vector edge features as an additive attention bias. Spatial edge tensors `[E, C_e, H, W]` are not supported in GAT (use `TensorGraphSAGELayer` or `TensorGINLayer` for those).
- **2-D aggregation only:** All layers expect node features of shape `[N, C, H, W]`. 3-D / volumetric inputs `[N, C, D, H, W]` are not supported end-to-end.
- **Differentiability:** All learned parameters (CNN encoder, message-passing layers, classifier) are end-to-end differentiable. The graph topology (`edge_index`) is user-provided and is not learned by the model.

## GNN family coverage

| GNN family | Implemented? | Tested? | Limitations |
|------------|-------------|---------|------------|
| Tensor-aware GCN-style (Conv message passing) | ✅ `ConvMessagePassing` | ✅ | 2-D spatial only; edge features must be spatial `[E, C_e, H, W]` |
| Tensor-aware GAT (multi-head) | ✅ `TensorGATLayer` | ✅ | scalar attention per `(edge, head)`; vector edge bias `[E, D_e]` supported; no spatial edge features |
| Tensor-aware GraphSAGE | ✅ `TensorGraphSAGELayer` | ✅ | mean / max only; no LSTM aggregator |
| Tensor-aware GIN / GINEConv | ✅ `TensorGINLayer` | ✅ | 2-D spatial only |
| MPNN-style custom layer | ✅ `TensorMessagePassingLayer` base | ✅ (subclass test + example) | — |
| Edge-conditioned MP (spatial) | ✅ `ConvMessagePassing`, `TensorGraphSAGELayer`, `TensorGINLayer` | ✅ | edge features `[E, C_e, H, W]` |
| Edge-conditioned MP (vector) | ✅ `TensorGATLayer`, `TensorGraphSAGELayer`, `TensorGINLayer` | ✅ | edge features `[E, D_e]`; `edge_features_kind="vector"` |
| `aggr="sum"\|"mean"\|"max"` base | ✅ all three modes | ✅ hand-computed + backward | `ConvMessagePassing` `aggr="max"` routes through `scatter_max` |
| Graph Transformer | ❌ | — | not implemented |
| Heterogeneous graphs | ❌ | — | not supported |
| Temporal / spatiotemporal graphs | ❌ | — | not supported |
| Learned graph construction | ❌ | — | `edge_index` is always user-supplied |
| 3-D / volumetric aggregation | ❌ | — | layers are 2-D only (`Conv2d` aggregators) |

---

## Project structure

```
TGraphX/
├── tgraphx/
│   ├── __init__.py          # public API re-exports
│   ├── core/
│   │   ├── graph.py         # Graph, GraphBatch
│   │   ├── dataloader.py    # GraphDataset, GraphDataLoader
│   │   └── utils.py         # load_config, get_device
│   ├── layers/
│   │   ├── base.py             # TensorMessagePassingLayer, LinearMessagePassing
│   │   ├── conv_message.py     # ConvMessagePassing
│   │   ├── attention_message.py# AttentionMessagePassing (legacy sigmoid)
│   │   ├── gat.py              # TensorGATLayer (true multi-head GAT)
│   │   ├── sage.py             # TensorGraphSAGELayer
│   │   ├── gin.py              # TensorGINLayer / GINEConv
│   │   ├── _scatter.py         # internal: edge_softmax, scatter_*
│   │   ├── aggregator.py       # DeepCNNAggregator
│   │   └── safe_pool.py        # SafeMaxPool2d
│   └── models/
│       ├── cnn_encoder.py      # CNNEncoder
│       ├── cnn_gnn_model.py    # CNN_GNN_Model
│       ├── graph_classifier.py
│       ├── node_classifier.py
│       └── pre_encoder.py      # PreEncoder (optional ResNet-18)
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
