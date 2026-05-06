"""Minimal example: graph-level classification with a short training loop.

Demonstrates:
  - Building two synthetic graphs with [N, C, H, W] node features.
  - Batching them into a single GraphBatch.
  - Running GraphClassifier for graph-level prediction.
  - A five-step training loop with cross-entropy loss.

Run from the repository root:
    python examples/minimal_graph_classifier.py

Note: this example uses small synthetic data.  Substitute real patch features
and a real edge_index computed from your domain (kNN, IoU, grid, …) for actual
experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from tgraphx import Graph, GraphBatch
from tgraphx.models import GraphClassifier

torch.manual_seed(42)

# ── Synthetic dataset ─────────────────────────────────────────────────────── #
# Two classes, four graphs each.  Node features: [N, C, H, W].
C, H, W = 8, 4, 4
NUM_CLASSES = 2
GRAPHS_PER_CLASS = 4


def make_random_graph(N: int, label: int):
    """Return a (Graph, label) pair with a random directed cycle structure."""
    x = torch.randn(N, C, H, W)
    src = torch.arange(N)
    ei = torch.stack([src, (src + 1) % N])   # directed cycle
    return Graph(x, ei), label


dataset = (
    [make_random_graph(N=torch.randint(3, 7, ()).item(), label=0)
     for _ in range(GRAPHS_PER_CLASS)]
    + [make_random_graph(N=torch.randint(3, 7, ()).item(), label=1)
       for _ in range(GRAPHS_PER_CLASS)]
)

# ── Model ─────────────────────────────────────────────────────────────────── #
clf = GraphClassifier(
    in_shape=(C, H, W),
    hidden_shape=(16, H, W),
    num_classes=NUM_CLASSES,
    num_layers=2,
    aggr="sum",
    pooling="mean",   # "mean" | "sum" | "max"
)

optimizer = optim.Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────────── #
# We batch all graphs in one step for simplicity.  In practice use GraphDataLoader.
graphs, labels = zip(*dataset)
batch = GraphBatch(list(graphs))
labels_t = torch.tensor(list(labels), dtype=torch.long)

print(f"Batched node_features : {batch.node_features.shape}")
print(f"Batch vector          : {batch.batch.tolist()}")
print()

NUM_STEPS = 5
clf.train()
for step in range(1, NUM_STEPS + 1):
    optimizer.zero_grad()

    logits = clf(
        batch.node_features,    # [N_total, C, H, W]
        batch.edge_index,       # [2, E_total]
        batch=batch.batch,      # [N_total]  — graph membership
        edge_features=None,     # not used here
    )                           # → [num_graphs, NUM_CLASSES]

    loss = loss_fn(logits, labels_t)
    loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=1)
    acc = (preds == labels_t).float().mean().item()
    print(f"Step {step:2d} | loss={loss.item():.4f} | acc={acc:.2f}")

print()
print("Logits shape :", logits.shape)   # [8, 2]
print("Done.")
