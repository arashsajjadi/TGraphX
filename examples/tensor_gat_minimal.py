"""Minimal example: tensor-aware multi-head GAT.

Demonstrates the canonical GAT (Veličković et al. 2018) adapted to spatial
node feature maps ``[N, C, H, W]``:

  * a learned linear projection per head (1x1 Conv2d)
  * scalar attention score per (edge, head) computed from spatially-pooled
    queries and keys
  * **softmax over each destination's incoming edges, per head** — so
    attention weights sum to 1
  * weighted sum of full-resolution value tensors

Run from the repository root:
    python examples/tensor_gat_minimal.py
"""

import torch

from tgraphx import Graph
from tgraphx.layers import TensorGATLayer

torch.manual_seed(0)

# ── Synthetic graph ─────────────────────────────────────────────────────── #
# 6 nodes, each carrying a 16-channel 8×8 feature map (e.g. an image patch
# already encoded by a CNN).
N, C_in, H, W = 6, 16, 8, 8
node_features = torch.randn(N, C_in, H, W)

# Directed cycle 0→1→2→3→4→5→0
src = torch.arange(N)
edge_index = torch.stack([src, (src + 1) % N])

g = Graph(node_features, edge_index)

# ── Multi-head GAT ──────────────────────────────────────────────────────── #
# 4 heads × 8 channels each = 32 output channels (concatenated heads).
layer = TensorGATLayer(
    in_channels=C_in,
    out_channels=32,            # 4 heads × 8 channels
    num_heads=4,
    concat_heads=True,
    add_self_loops=True,        # ensures every node has at least one in-edge
    attn_dropout=0.1,
    residual=True,              # auto-projects 16 → 32
    bias=True,
)
layer.eval()                    # disable attention dropout for the demo

# Forward pass; ask the layer to also return raw attention weights.
out, attn = layer(g.node_features, g.edge_index, return_attention=True)
print(f"input  : {g.node_features.shape}")     # [6, 16, 8, 8]
print(f"output : {out.shape}")                  # [6, 32, 8, 8]
print(f"attn   : {attn.shape}  (E_eff, num_heads)")
#                with self-loops, E_eff = E + N = 12

# ── Verify the GAT correctness invariant ─────────────────────────────────── #
# For every destination j and head h, attention weights over its incoming
# edges must sum to 1.  We rebuild the destination index used internally.
loop = torch.arange(N)
dst_with_loops = torch.cat([edge_index[1], loop])
sums = torch.zeros(N, layer.num_heads).index_add_(0, dst_with_loops, attn)
print()
print(f"Σ attention per (destination, head)  : {sums.mean().item():.4f} "
      f"(should be 1.0)")
print(f"max abs deviation from 1            : {(sums - 1).abs().max().item():.2e}")

# ── Backward pass ───────────────────────────────────────────────────────── #
out.sum().backward()
print()
print("Backward pass: OK")
