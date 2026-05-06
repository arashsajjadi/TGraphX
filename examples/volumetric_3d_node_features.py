"""Volumetric 3-D node features end-to-end across all four layer families.

Demonstrates that ``[N, C, D, H, W]`` node tensors flow cleanly through
``ConvMessagePassing``, ``TensorGATLayer``, ``TensorGraphSAGELayer``, and
``TensorGINLayer`` with ``spatial_rank=3``.

For each layer the script:

* runs a forward pass and prints the output shape,
* attaches ``edge_weight`` and prints the resulting `|out_w - out_no|_max`,
* attaches volumetric ``edge_features`` (where the layer supports them) and
  prints the same diff,
* runs a backward pass and reports the input-gradient norm to confirm
  finite gradients.

Run from the repository root:
    python examples/volumetric_3d_node_features.py
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
from tgraphx.models import GraphClassifier


N, C, D, H, W = 6, 3, 3, 4, 4
EDGE_DIM = 3


def make_graph(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(N, C, D, H, W)
    edge_index = torch.tensor(
        [[0, 2, 3, 0, 1, 4, 5, 4],
         [1, 1, 1, 2, 3, 0, 4, 5]],
        dtype=torch.long,
    )
    return x, edge_index


def report(name, layer, x, ei, *, ef=None):
    """One-row summary for a single layer in 3-D mode."""
    layer.eval()
    E = ei.size(1)
    w = torch.linspace(0.5, 2.0, E)
    out_no = layer(x, ei, edge_features=ef) if ef is not None else layer(x, ei)
    out_w = (
        layer(x, ei, edge_features=ef, edge_weight=w)
        if ef is not None
        else layer(x, ei, edge_weight=w)
    )
    diff = (out_w - out_no).abs().max().item()

    # Backward sanity in train mode.
    layer.train()
    layer.zero_grad()
    x_g = x.detach().clone().requires_grad_(True)
    out = (
        layer(x_g, ei, edge_features=ef, edge_weight=w)
        if ef is not None
        else layer(x_g, ei, edge_weight=w)
    )
    out.sum().backward()
    grad_norm = x_g.grad.norm().item() if x_g.grad is not None else 0.0
    finite = bool(torch.isfinite(out_no).all() and torch.isfinite(x_g.grad).all())

    edges_label = "vol" if ef is not None else "—  "
    print(
        f"  {name:24s} edges={edges_label}  "
        f"out.shape={tuple(out_no.shape)}  "
        f"|out_w - out_no|_max={diff:.3e}  "
        f"x.grad_norm={grad_norm:.3e}  finite={finite}"
    )


def main():
    x, ei = make_graph()
    E = ei.size(1)

    # Volumetric edge tensors for each layer family.  ConvMessagePassing
    # requires edge_channels == node_channels (= C).  GAT/SAGE/GIN have an
    # explicit ``edge_dim`` so we use ``EDGE_DIM`` channels there.
    ef_conv = torch.randn(E, C, D, H, W)
    ef_other = torch.randn(E, EDGE_DIM, D, H, W)

    g = Graph(x, ei, edge_features=ef_other)
    print(
        f"Graph (3-D): nodes={g.num_nodes}, edges={g.num_edges}, "
        f"node feature shape={g.feature_shape}, "
        f"edge feature shape={g.edge_feature_shape}"
    )
    print()

    print("3-D layer scan (no edge features, then with volumetric edges):")
    fast_agg = {"num_layers": 1, "use_batchnorm": False, "dropout_prob": 0.0}

    conv = ConvMessagePassing(
        (C, D, H, W), (8, D, H, W),
        aggregator_params=fast_agg,
    )
    report("ConvMessagePassing  (plain)", conv, x, ei)

    conv_e = ConvMessagePassing(
        (C, D, H, W), (8, D, H, W),
        use_edge_features=True, aggregator_params=fast_agg,
    )
    report("ConvMessagePassing  (edges)", conv_e, x, ei, ef=ef_conv)

    gat = TensorGATLayer(
        in_channels=C, out_channels=8, num_heads=2, spatial_rank=3,
    )
    report("TensorGATLayer       (plain)", gat, x, ei)

    gat_e = TensorGATLayer(
        in_channels=C, out_channels=8, num_heads=2, spatial_rank=3,
        use_edge_features=True, edge_dim=EDGE_DIM,
    )
    report("TensorGATLayer       (vol→pool)", gat_e, x, ei, ef=ef_other)

    sage = TensorGraphSAGELayer(
        in_channels=C, out_channels=8, aggr="mean", spatial_rank=3,
    )
    report("TensorGraphSAGELayer (plain)", sage, x, ei)

    sage_e = TensorGraphSAGELayer(
        in_channels=C, out_channels=8, aggr="mean", spatial_rank=3,
        use_edge_features=True, edge_dim=EDGE_DIM, edge_features_kind="spatial",
    )
    report("TensorGraphSAGELayer (edges)", sage_e, x, ei, ef=ef_other)

    gin = TensorGINLayer(
        in_channels=C, out_channels=8, spatial_rank=3, train_eps=True,
    )
    report("TensorGINLayer       (plain)", gin, x, ei)

    gin_e = TensorGINLayer(
        in_channels=C, out_channels=8, spatial_rank=3, train_eps=True,
        use_edge_features=True, edge_dim=EDGE_DIM, edge_features_kind="spatial",
    )
    report("TensorGINLayer       (edges)", gin_e, x, ei, ef=ef_other)

    print()
    print("GraphClassifier 3-D smoke (two graphs, mean pooling, 3 classes):")
    x_b = torch.cat([x, x], dim=0)
    ei_b = torch.cat([ei, ei + N], dim=1)
    batch = torch.cat([torch.zeros(N, dtype=torch.long),
                       torch.ones(N, dtype=torch.long)])
    model = GraphClassifier(
        in_shape=(C, D, H, W), hidden_shape=(8, D, H, W),
        num_classes=3, num_layers=2, aggr="sum", pooling="mean",
    )
    logits = model(x_b, ei_b, batch=batch)
    print(f"  logits.shape={tuple(logits.shape)}, "
          f"finite={bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    main()
