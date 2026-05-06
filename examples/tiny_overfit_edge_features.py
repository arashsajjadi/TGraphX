"""Tiny overfit script demonstrating that vector edge features matter.

Builds a synthetic relational task whose label per node depends on a
per-edge scalar (e.g. "relation strength"), then trains a tiny model and
verifies that:

* loss decreases over iterations,
* the edge projection parameters receive non-zero gradients,
* zeroing the edge features at inference produces a different output.

Three layers are demonstrated: ``TensorGATLayer`` (vector edge attention
bias), ``TensorGINLayer`` (vector GINEConv), and ``TensorGraphSAGELayer``
(vector channel bias).

Run from the repository root:
    python examples/tiny_overfit_edge_features.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim

from tgraphx.layers import (
    TensorGATLayer,
    TensorGINLayer,
    TensorGraphSAGELayer,
)


def make_task(seed: int = 7):
    torch.manual_seed(seed)
    N, C, H, W = 8, 4, 3, 3
    x = torch.randn(N, C, H, W)
    # Dense edge set so the GAT softmax has > 1 incoming edge per dest.
    src = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7,
                        1, 2, 3, 4, 5, 6, 7, 0])
    dst = torch.tensor([1, 2, 3, 0, 4, 0, 5, 0, 6, 0, 7, 1, 2, 3,
                        4, 5, 6, 7, 0, 1, 2, 3])
    ei = torch.stack([src, dst])
    EDGE_DIM = 3
    edge_features = torch.randn(ei.size(1), EDGE_DIM)
    # Construct labels using a closed-form combination of neighbour features
    # weighted by an edge-attribute-dependent scalar.  Any layer that
    # *ignores* edge features will not be able to fit this perfectly.
    weight = (edge_features.sum(dim=1) > 0).float() * 2.0 - 1.0  # ±1 per edge
    with torch.no_grad():
        agg = torch.zeros_like(x)
        agg.index_add_(
            0, ei[1], x[ei[0]] * weight.view(-1, 1, 1, 1)
        )
        labels = (agg.flatten(1).mean(dim=1) > 0).long()
    return x, ei, edge_features, labels, EDGE_DIM


def train(layer_factory, x, ei, edge_features, labels, *, steps=60, lr=0.05):
    torch.manual_seed(0)
    N, C, H, W = x.shape
    gnn1 = layer_factory(C, 8)
    gnn2 = layer_factory(8, 8)
    head = nn.Linear(8, 2)
    params = list(gnn1.parameters()) + list(gnn2.parameters()) + list(head.parameters())
    opt = optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    first = None
    last = None
    for _ in range(steps):
        opt.zero_grad()
        h = torch.relu(gnn1(x, ei, edge_features=edge_features))
        h = gnn2(h, ei, edge_features=edge_features)
        logits = head(h.mean(dim=(-2, -1)))
        loss = loss_fn(logits, labels)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
        last = loss.item()

    # Probe: does zeroing edge features change the inference output?
    with torch.no_grad():
        h_real = gnn1(x, ei, edge_features=edge_features)
        h_zero = gnn1(x, ei, edge_features=torch.zeros_like(edge_features))
        diff = (h_real - h_zero).abs().mean().item()

    # Probe: did the edge projection parameters receive gradients?
    edge_proj_grad_norm = 0.0
    for name, p in gnn1.named_parameters():
        if "edge" in name and p.grad is not None:
            edge_proj_grad_norm = max(edge_proj_grad_norm, p.grad.norm().item())

    return first, last, diff, edge_proj_grad_norm


def main() -> None:
    x, ei, ef, labels, edge_dim = make_task()

    print(f"{'layer':<25}{'init':>10}{'final':>10}{'Δ(zeroed)':>14}{'edge ‖∇‖':>12}")
    print("-" * 71)
    rows = []
    for name, factory in [
        ("TensorGATLayer",
         lambda c_in, c_out: TensorGATLayer(
             in_channels=c_in, out_channels=c_out, num_heads=2,
             use_edge_features=True, edge_dim=edge_dim, add_self_loops=True,
         )),
        ("TensorGINLayer (vector)",
         lambda c_in, c_out: TensorGINLayer(
             in_channels=c_in, out_channels=c_out,
             use_edge_features=True, edge_dim=edge_dim,
             edge_features_kind="vector", train_eps=True,
         )),
        ("TensorGraphSAGELayer (vec)",
         lambda c_in, c_out: TensorGraphSAGELayer(
             in_channels=c_in, out_channels=c_out, aggr="mean",
             use_edge_features=True, edge_dim=edge_dim,
             edge_features_kind="vector",
         )),
    ]:
        first, last, diff, gnorm = train(factory, x, ei, ef, labels)
        rows.append((name, first, last, diff, gnorm))
        print(f"{name:<25}{first:>10.4f}{last:>10.4f}{diff:>14.4e}{gnorm:>12.3e}")
        assert math.isfinite(last)
        assert last < first - 0.05, f"{name} did not converge: {first:.4f} -> {last:.4f}"
        assert diff > 1e-3, f"{name} ignores edge features (Δ={diff:.2e})"
        assert gnorm > 0.0, f"{name} edge-projection params have zero gradient"
    print()
    print("All edge-feature checks: PASSED")


if __name__ == "__main__":
    main()
