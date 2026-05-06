"""Custom tensor-aware message-passing layer.

This example shows how to write a new GNN layer for spatial features by
subclassing ``TensorMessagePassingLayer`` and overriding ``message`` (and
optionally ``update``).  The base class handles aggregation (sum or mean)
for you, including correct broadcasting for ``[N, C, H, W]`` tensors.

The custom layer below implements a simple sigmoid-gated message:

    m_ij = σ( W_g(h_i + h_j) ) ⊙ W_v(h_i)
    h_j' = mean_i  m_ij                       (base-class aggregation)

Run from the repository root:
    python examples/custom_message_passing.py
"""

import torch
import torch.nn as nn

from tgraphx import Graph
from tgraphx.layers import TensorMessagePassingLayer


class GatedConvMessagePassing(TensorMessagePassingLayer):
    """User-defined sigmoid-gated 1x1 convolutional message passing.

    Subclasses only need to:
      1. Pass ``in_shape`` / ``out_shape`` to the base for bookkeeping.
      2. Override ``message`` with the desired per-edge computation.
      3. Optionally override ``update`` if the base behaviour
         (BatchNorm + dropout + residual) is not desired.

    The base class handles per-edge gather, aggregation, and orchestration.
    """

    def __init__(self, in_channels: int, out_channels: int):
        # in_shape / out_shape are stored on self by the base class but only
        # consulted when use_batchnorm=True; here we keep them 1-D tuples
        # for simplicity.
        super().__init__(
            in_shape=(in_channels,),
            out_shape=(out_channels,),
            aggr="mean",
        )
        self.W_gate = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.W_value = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def message(self, src, dest, edge_attr):
        # src, dest: [E, C_in, H, W]
        gate = torch.sigmoid(self.W_gate(src + dest))   # [E, C_out, H, W]
        return gate * self.W_value(src)

    def update(self, node_feature, aggregated_message):
        # Use the aggregated message directly (base class would otherwise
        # apply BN/dropout/residual, but we disabled them by not setting
        # those flags).
        return aggregated_message


# ── Demo ───────────────────────────────────────────────────────────────── #
torch.manual_seed(0)

N, C_in, H, W = 6, 16, 8, 8
x = torch.randn(N, C_in, H, W)
src = torch.arange(N)
edge_index = torch.stack([src, (src + 1) % N])
g = Graph(x, edge_index)

layer = GatedConvMessagePassing(in_channels=C_in, out_channels=32)
out = layer(g.node_features, g.edge_index)
print(f"input  : {g.node_features.shape}")     # [6, 16, 8, 8]
print(f"output : {out.shape}")                  # [6, 32, 8, 8]

# Backward pass through the custom layer.
out.sum().backward()
print("Backward pass: OK")
print(f"# trainable parameters: "
      f"{sum(p.numel() for p in layer.parameters() if p.requires_grad)}")
