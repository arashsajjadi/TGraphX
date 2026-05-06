"""Minimal example: tensor-aware GraphSAGE.

Demonstrates Hamilton et al. (2017)'s GraphSAGE adapted to spatial node
features ``[N, C, H, W]``:

    h_j' = W_self(h_j) + W_neigh( AGG_{i ∈ N(j)} h_i )

with ``W_self`` and ``W_neigh`` realised as 1×1 Conv2d, and ``AGG`` either
``"mean"`` (default) or ``"max"``.  Optional L2 normalisation along the
channel dimension produces unit-norm channel vectors per spatial position.

Run from the repository root:
    python examples/tensor_graphsage_minimal.py
"""

import torch

from tgraphx import Graph
from tgraphx.layers import TensorGraphSAGELayer

torch.manual_seed(0)

# ── Graph ───────────────────────────────────────────────────────────────── #
N, C_in, H, W = 6, 16, 8, 8
node_features = torch.randn(N, C_in, H, W)
src = torch.arange(N)
edge_index = torch.stack([src, (src + 1) % N])
g = Graph(node_features, edge_index)

# ── Mean-aggregator SAGE with L2 normalisation ──────────────────────────── #
layer_mean = TensorGraphSAGELayer(
    in_channels=C_in,
    out_channels=32,
    aggr="mean",
    normalize=True,         # unit channel-vector per spatial location
    residual=False,
)
out = layer_mean(g.node_features, g.edge_index)
print(f"mean-aggregator output     : {out.shape}")
norm_per_pixel = out.pow(2).sum(dim=1).sqrt()
print(f"L2 norm of channel vector  : {norm_per_pixel.mean().item():.4f} "
      f"(should be 1.0 because normalize=True)")

# ── Max-aggregator SAGE ─────────────────────────────────────────────────── #
layer_max = TensorGraphSAGELayer(
    in_channels=C_in,
    out_channels=32,
    aggr="max",
    normalize=False,
)
out_max = layer_max(g.node_features, g.edge_index)
print(f"max-aggregator output      : {out_max.shape}")

# ── SAGE with spatial edge features ─────────────────────────────────────── #
edge_dim = 4
edge_feat = torch.randn(g.edge_index.size(1), edge_dim, H, W)
layer_with_edges = TensorGraphSAGELayer(
    in_channels=C_in, out_channels=32,
    aggr="mean",
    use_edge_features=True, edge_dim=edge_dim,
)
out_e = layer_with_edges(g.node_features, g.edge_index, edge_features=edge_feat)
print(f"with-edge-features output  : {out_e.shape}")

# ── Backward ────────────────────────────────────────────────────────────── #
out_max.sum().backward()
print()
print("Backward pass: OK")
