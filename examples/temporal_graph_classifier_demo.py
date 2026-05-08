"""temporal_graph_classifier_demo.py — TemporalGraphClassifier on synthetic temporal task.

Trains a tiny TemporalGraphClassifier where:
- Class 0 sequences have increasing mean node features over time.
- Class 1 sequences have decreasing mean node features over time.
- The "last" readout extracts the final snapshot embedding.

CPU-safe; experimental APIs (🧪 v0.2.5).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph, TemporalGraphSequence, TemporalGraphBatch
from tgraphx.models.temporal_models import TemporalGraphClassifier

torch.manual_seed(0)


class MeanPoolBase(nn.Module):
    """Tiny stateless graph encoder: linear + mean pool per graph."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, gb):
        x = self.lin(gb.node_features)
        out = torch.zeros(gb.num_graphs, x.size(1))
        out = out.index_add(0, gb.batch, x)
        cnt = torch.zeros(gb.num_graphs).index_add(
            0, gb.batch, torch.ones(x.size(0))
        )
        return out / cnt.unsqueeze(1).clamp_min(1.0)


def make_seq(class_id, length=3, n_nodes=4, dim=8):
    direction = +0.5 if class_id == 0 else -0.5
    base = torch.randn(n_nodes, dim)
    graphs = [
        Graph(base + direction * t * torch.ones(n_nodes, dim), None)
        for t in range(length)
    ]
    return TemporalGraphSequence(graphs)


# 6 sequences: 3 each class.
seqs = [make_seq(i % 2) for i in range(6)]
labels = torch.tensor([i % 2 for i in range(6)], dtype=torch.long)
batch = TemporalGraphBatch(seqs)

base = MeanPoolBase(in_dim=8, out_dim=16)
clf = TemporalGraphClassifier(base, feature_dim=16, num_classes=2, readout="last")

opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
for step in range(40):
    opt.zero_grad()
    logits = clf(batch)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    opt.step()

acc = (logits.argmax(-1) == labels).float().mean().item()
print(f"Final loss:    {loss.item():.4f}")
print(f"Train accuracy:{acc:.2f}")
assert torch.isfinite(logits).all()
print("TemporalGraphClassifier demo: PASSED")
