"""Per-edge ``edge_weight`` scaling across all layer families.

For every layer family we demonstrate that:

* ``edge_weight=None`` and ``edge_weight=ones(E)`` produce identical output
  (round-trip identity),
* a non-trivial ``edge_weight`` changes the output (the layer actually uses
  it),
* gradients flow back into ``edge_weight`` when it has ``requires_grad=True``,
* ``edge_weight`` and ``edge_features`` compose: both can be passed together
  in the same call and influence the output independently.

Run from the repository root:
    python examples/weighted_edges.py
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


def _check_layer(name, layer, x, ei, weight):
    """Run identity / variation / gradient checks on one layer."""
    layer.eval()
    E = ei.size(1)
    out_none = layer(x, ei)
    out_ones = layer(x, ei, edge_weight=torch.ones(E))
    out_real = layer(x, ei, edge_weight=weight)
    identity_ok = torch.allclose(out_none, out_ones, atol=1e-5)
    diff = (out_real - out_none).abs().max().item()

    # Gradient flow into edge_weight.
    layer.train()
    layer.zero_grad()
    w_g = weight.detach().clone().requires_grad_(True)
    out = layer(x, ei, edge_weight=w_g)
    out.sum().backward()
    grad_norm = w_g.grad.norm().item() if w_g.grad is not None else 0.0

    print(
        f"  {name:24s} ones==no_weight: {str(identity_ok):5s}  "
        f"|out_w - out_no|_max={diff:.3e}  "
        f"weight.grad_norm={grad_norm:.3e}"
    )


def main():
    x, ei, N, C, H, W = make_graph()
    E = ei.size(1)
    weight = torch.linspace(0.5, 2.0, E)

    # Build Graph to round-trip edge_weight through the data structure.
    g = Graph(x, ei, edge_weight=weight)
    print(
        f"Graph: nodes={g.num_nodes}, edges={g.num_edges}, "
        f"has_edge_weight={g.has_edge_weight}"
    )
    print()

    print("edge_weight scaling on every layer family:")
    _check_layer(
        "ConvMessagePassing",
        ConvMessagePassing(
            (C, H, W), (8, H, W),
            aggregator_params={"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0},
        ),
        x, ei, weight,
    )
    _check_layer(
        "TensorGATLayer",
        TensorGATLayer(in_channels=C, out_channels=8, num_heads=2, bias=False),
        x, ei, weight,
    )
    _check_layer(
        "TensorGraphSAGELayer",
        TensorGraphSAGELayer(in_channels=C, out_channels=8, aggr="mean"),
        x, ei, weight,
    )
    _check_layer(
        "TensorGINLayer",
        TensorGINLayer(in_channels=C, out_channels=8, train_eps=True),
        x, ei, weight,
    )

    # Compose: edge_weight + edge_features together, fed via Graph.
    print()
    print("Composing edge_weight with vector edge_features (TensorGATLayer):")
    edge_dim = 3
    ef = torch.randn(E, edge_dim)
    g_full = Graph(x, ei, edge_weight=weight, edge_features=ef)
    layer = TensorGATLayer(
        in_channels=C, out_channels=8, num_heads=2,
        use_edge_features=True, edge_dim=edge_dim, bias=False,
    )
    layer.eval()
    out_both = layer(
        g_full.node_features, g_full.edge_index,
        edge_features=g_full.edge_features,
        edge_weight=g_full.edge_weight,
    )
    out_ef = layer(g_full.node_features, g_full.edge_index, edge_features=g_full.edge_features)
    out_w = layer(g_full.node_features, g_full.edge_index, edge_weight=g_full.edge_weight) \
        if not layer.use_edge_features else None
    print(
        f"  out shape: {tuple(out_both.shape)}  "
        f"|out_both - out_ef|_max={(out_both - out_ef).abs().max().item():.3e}"
    )

    # Spatial edges + weight on SAGE.
    print()
    print("Spatial edge_features + edge_weight (TensorGraphSAGELayer):")
    ef_spatial = torch.randn(E, edge_dim, H, W)
    sage = TensorGraphSAGELayer(
        in_channels=C, out_channels=8,
        use_edge_features=True, edge_dim=edge_dim,
        edge_features_kind="spatial",
    )
    sage.eval()
    g_spatial = Graph(x, ei, edge_weight=weight, edge_features=ef_spatial)
    out = sage(
        g_spatial.node_features, g_spatial.edge_index,
        edge_features=g_spatial.edge_features,
        edge_weight=g_spatial.edge_weight,
    )
    print(f"  out shape: {tuple(out.shape)}, finite: {bool(torch.isfinite(out).all())}")


if __name__ == "__main__":
    main()
