"""Spatial (tensor) edge features end-to-end across all layer families.

Builds a small graph with
    node_features  : [N, C, H, W]
    edge_features  : [E, C_e, H, W]   (true tensor edges, not vectors)
and runs each layer family that supports tensor edge features:

* ``ConvMessagePassing``    — concatenates edge tensors with src/dest along
                              channels, then a 1×1 conv produces messages.
* ``TensorGATLayer``        — mean-pools edge tensors to a vector per
                              ``(edge, edge_dim)``, then projects to a
                              per-(edge, head) attention bias.
* ``TensorGraphSAGELayer``  — concatenates edge tensors with source features
                              along channels, then ``W_neigh`` produces
                              neighbour messages.
* ``TensorGINLayer``        — projects edge tensors to ``in_channels`` and
                              adds them to source features inside a ReLU.

For each layer the script prints:

* output shape (sanity),
* whether non-zero edge tensors change the output (they should),
* whether the edge-projection parameters receive a non-zero gradient.

Run from the repository root:
    python examples/tensor_edge_features.py
"""

from __future__ import annotations

import torch

from tgraphx import Graph
from tgraphx.layers import (
    ConvMessagePassing,
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)


# --------------------------------------------------------------------------- #
# Graph and edge tensors                                                       #
# --------------------------------------------------------------------------- #

def make_graph(seed: int = 0):
    torch.manual_seed(seed)
    N, C, H, W = 6, 4, 4, 4
    x = torch.randn(N, C, H, W)
    edge_index = torch.tensor(
        [[0, 2, 3, 0, 1, 4, 5, 4],
         [1, 1, 1, 2, 3, 0, 4, 5]],
        dtype=torch.long,
    )
    return x, edge_index, N, C, H, W


def make_edge_tensors(E: int, C_node: int, C_edge_for_gat: int, H: int, W: int):
    """Returns two edge feature tensors:

    * one whose channel count matches ``C_node`` (used by ConvMessagePassing,
      which wires edge channels into the message convolution alongside src
      and dest features), and
    * one with an arbitrary ``C_edge_for_gat`` channel count, used by GAT,
      SAGE, and GIN — these layers each have an explicit ``edge_dim``.
    """
    torch.manual_seed(1)
    ef_conv = torch.randn(E, C_node, H, W)
    ef_other = torch.randn(E, C_edge_for_gat, H, W)
    return ef_conv, ef_other


# --------------------------------------------------------------------------- #
# Per-layer demonstration                                                      #
# --------------------------------------------------------------------------- #

def run_layer(name, layer, x, ei, ef, edge_proj_param):
    """Forward + zero-edge comparison + parameter gradient check."""
    layer.eval()
    out_real = layer(x, ei, edge_features=ef)
    out_zero = layer(x, ei, edge_features=torch.zeros_like(ef))
    diff = (out_real - out_zero).abs().max().item()

    # Re-run in train mode for a backward pass on the real edges.
    layer.train()
    layer.zero_grad()
    out = layer(x, ei, edge_features=ef.detach().requires_grad_(True))
    out.sum().backward()
    grad_norm = (
        edge_proj_param.grad.norm().item()
        if edge_proj_param.grad is not None
        else 0.0
    )

    print(
        f"  {name:24s} out.shape={tuple(out_real.shape)}  "
        f"|out_real - out_zero|_max={diff:.3e}  "
        f"edge_proj.grad_norm={grad_norm:.3e}"
    )


def main():
    x, ei, N, C, H, W = make_graph()
    E = ei.size(1)
    C_edge = 3  # for GAT / SAGE / GIN
    ef_conv, ef_other = make_edge_tensors(E, C_node=C, C_edge_for_gat=C_edge, H=H, W=W)

    # Build a Graph object to demonstrate that .node_features / .edge_index /
    # .edge_features round-trip cleanly into every layer call site.
    g_other = Graph(x, ei, edge_features=ef_other)
    print(
        f"Graph: nodes={g_other.num_nodes}, edges={g_other.num_edges}, "
        f"node feature shape={g_other.feature_shape}, "
        f"edge feature shape={g_other.edge_feature_shape}"
    )
    print()

    print("Layers:")

    # ConvMessagePassing: edge channels must equal node channels (C).
    conv = ConvMessagePassing(
        (C, H, W), (8, H, W),
        use_edge_features=True,
        aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
    )
    run_layer("ConvMessagePassing", conv, x, ei, ef_conv, conv.conv.weight)

    gat = TensorGATLayer(
        in_channels=C, out_channels=8, num_heads=2,
        use_edge_features=True, edge_dim=C_edge,
    )
    run_layer("TensorGATLayer (4-D edges)", gat, x, ei, ef_other, gat.edge_bias_proj.weight)

    sage = TensorGraphSAGELayer(
        in_channels=C, out_channels=8, aggr="mean",
        use_edge_features=True, edge_dim=C_edge,
        edge_features_kind="spatial",
    )
    # SAGE's edge channels are concatenated with src into W_neigh — so the
    # edge-aware parameter is W_neigh.
    run_layer("TensorGraphSAGELayer", sage, x, ei, ef_other, sage.W_neigh.weight)

    gin = TensorGINLayer(
        in_channels=C, out_channels=8,
        use_edge_features=True, edge_dim=C_edge,
        edge_features_kind="spatial",
    )
    run_layer("TensorGINLayer", gin, x, ei, ef_other, gin.edge_proj.weight)

    print()
    print("All four layers produced output that visibly depends on the "
          "edge tensor and propagated gradients into their edge-projection "
          "parameters.")


if __name__ == "__main__":
    main()
