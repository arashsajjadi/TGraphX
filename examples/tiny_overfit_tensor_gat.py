"""Tiny overfit sanity script for TensorGATLayer.

Trains a 2-layer ``TensorGATLayer`` model on a deterministic synthetic
node-classification task and prints initial vs final loss.  This is *not*
a benchmark — the goal is to verify trainability and gradient health.

Run from the repository root:
    python examples/tiny_overfit_tensor_gat.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.optim as optim

from tgraphx.layers import TensorGATLayer


def main() -> None:
    torch.manual_seed(0)

    # ── synthetic graph + labels ───────────────────────────────────────── #
    N, C, H, W = 8, 4, 3, 3
    x = torch.randn(N, C, H, W)
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7,
                        1, 2, 3, 4, 5, 6, 7, 0,
                        0, 4])
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        4, 0])
    edge_index = torch.stack([src, dst])

    # Label depends on the *aggregated neighbour features*, so a model
    # that ignores edges cannot solve the task.
    with torch.no_grad():
        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_index[1], x[edge_index[0]])
        labels = (agg.flatten(1).mean(dim=1) > 0).long()

    # ── 2-layer GAT model with a per-node linear classifier ────────────── #
    gnn1 = TensorGATLayer(in_channels=C, out_channels=8,
                          num_heads=2, add_self_loops=True)
    gnn2 = TensorGATLayer(in_channels=8, out_channels=8,
                          num_heads=2, add_self_loops=True)
    head = nn.Linear(8, 2)
    params = list(gnn1.parameters()) + list(gnn2.parameters()) + list(head.parameters())

    opt = optim.Adam(params, lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None
    print(f"{'step':>5}  {'loss':>10}  {'acc':>6}")
    for step in range(1, 41):
        opt.zero_grad()
        h = torch.relu(gnn1(x, edge_index))
        h = gnn2(h, edge_index)
        pooled = h.mean(dim=(-2, -1))     # [N, 8] global avg pool per node
        logits = head(pooled)             # [N, 2]
        loss = loss_fn(logits, labels)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()
        if step in (1, 5, 10, 20, 40):
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"{step:>5}  {final_loss:>10.4f}  {acc:>6.2f}")

    print()
    print(f"initial loss : {initial_loss:.4f}")
    print(f"final   loss : {final_loss:.4f}")
    assert math.isfinite(final_loss)
    assert final_loss < initial_loss - 0.05, "loss did not decrease meaningfully"
    print("Trainability check: PASSED")


if __name__ == "__main__":
    main()
