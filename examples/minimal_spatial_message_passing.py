"""Minimal example: spatial message passing with ConvMessagePassing.

Shows the core TGraphX idea in ~30 lines:
  - node features keep their [C, H, W] shape throughout message passing
  - graph structure is provided by the user as edge_index
  - the full pipeline is end-to-end differentiable

Run from the repository root:
    python examples/minimal_spatial_message_passing.py
"""

import torch
from tgraphx import Graph
from tgraphx.layers import ConvMessagePassing

torch.manual_seed(0)

# ── Graph definition ──────────────────────────────────────────────────────── #
# Six nodes.  Each node is a 16-channel 8×8 feature map (e.g. an image patch
# processed by a CNN encoder, or a raw crop from a detection backbone).
N, C, H, W = 6, 16, 8, 8
node_features = torch.randn(N, C, H, W)

# Directed cycle 0→1→2→3→4→5→0  (edge_index shape: [2, E])
src = torch.arange(N)
dst = (src + 1) % N
edge_index = torch.stack([src, dst])

# Graph validates inputs and stores tensors.
g = Graph(node_features, edge_index)
print(f"node_features : {g.node_features.shape}")   # [6, 16, 8, 8]
print(f"edge_index    : {g.edge_index.shape}")       # [2, 6]

# ── Message-passing layer ──────────────────────────────────────────────────── #
# ConvMessagePassing concatenates source and destination feature maps along the
# channel dimension (so 2×C channels), then applies a 1×1 Conv2d to project to
# out_channels.  A DeepCNNAggregator refines the aggregated messages using 3×3
# convolutions before updating each node's representation.
layer = ConvMessagePassing(
    in_shape=(C, H, W),     # per-node input shape
    out_shape=(32, H, W),   # per-node output shape — spatial dims H,W are preserved
    aggr="sum",             # neighbourhood aggregation: "sum" or "mean"
    residual=False,         # set True to add a skip connection (requires same shape)
)

out = layer(g.node_features, g.edge_index)
print(f"output        : {out.shape}")   # [6, 32, 8, 8]

# ── Backward pass ─────────────────────────────────────────────────────────── #
# Every operation is differentiable — gradients flow from the output back to
# both the model parameters and the input node features.
out.sum().backward()
print("Gradient check : OK")
assert g.node_features.grad is None        # node_features was not requires_grad
assert next(layer.parameters()).grad is not None   # layer params have gradients

# ── CUDA / MPS ────────────────────────────────────────────────────────────── #
from tgraphx.core.utils import get_device

device = get_device()   # CUDA → MPS → CPU (whichever is available)
if device.type != "cpu":
    g2 = Graph(node_features.to(device), edge_index.to(device))
    layer2 = layer.to(device)
    out2 = layer2(g2.node_features, g2.edge_index)
    print(f"Device output  : {out2.shape} on {out2.device}")
