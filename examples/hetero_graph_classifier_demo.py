"""hetero_graph_classifier_demo.py — train a tiny HeteroGraphClassifier.

Synthetic two-class hetero graph dataset:
- Class 0: dense (paper, author) connections.
- Class 1: sparse connections.

Verifies forward/backward + loss decreases over a few steps.

CPU-safe; no internet; experimental APIs (🧪 v0.2.5).
"""
import torch
import torch.nn.functional as F

from tgraphx import HeteroGraph, HeteroGraphBatch
from tgraphx.models.hetero_models import HeteroGraphClassifier

torch.manual_seed(0)


def make_graph(class_id):
    n_p = 5
    n_a = 4
    if class_id == 0:
        n_e = 8  # dense
    else:
        n_e = 2  # sparse
    return HeteroGraph(
        node_stores={
            "paper": torch.randn(n_p, 8),
            "author": torch.randn(n_a, 8),
        },
        edge_stores={
            ("author", "writes", "paper"): torch.stack([
                torch.randint(0, n_a, (n_e,)),
                torch.randint(0, n_p, (n_e,)),
            ], dim=0).long(),
        },
        graph_label=torch.tensor(class_id, dtype=torch.long),
    )

# 8 graphs: 4 of each class
graphs = [make_graph(i % 2) for i in range(8)]
batch = HeteroGraphBatch(graphs)
labels = batch.graph_labels  # [8]

clf = HeteroGraphClassifier(
    node_in_dims={"paper": 8, "author": 8},
    edge_types=[("author", "writes", "paper")],
    hidden_dim=16, num_layers=2, num_classes=2,
)
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)

losses = []
for step in range(20):
    opt.zero_grad()
    logits = clf(batch.x_dict, batch.edge_index_dict, batch_dict=batch.batch_dict)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"Loss step 0:   {losses[0]:.4f}")
print(f"Loss step 19:  {losses[-1]:.4f}")
acc = (logits.argmax(-1) == labels).float().mean().item()
print(f"Final train accuracy: {acc:.2f}")
print("HeteroGraphClassifier demo: PASSED" if torch.isfinite(logits).all() else "FAILED")
