"""Tensor node classification with NeighborLoader — standalone demo.

This is a concise demo of the canonical TGraphX workflow:
- Graph with y= labels
- NeighborLoader yielding GraphMiniBatch
- ConvMessagePassing
- Seed-node loss via batch.seed_logits() and batch.seed_y

For the full tutorial with explanations, see:
    tutorials/tensor_node_classification_neighbor_loader.py

For the zero-boilerplate version, see:
    examples/easy_tensor_node_classification_no_torch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from tgraphx import Graph, ConvMessagePassing, NeighborLoader
from tgraphx.reproducibility import set_seed

set_seed(42)

N, C, H, W = 300, 8, 6, 6
x = torch.randn(N, C, H, W)
edge_index = torch.randint(0, N, (2, 1500))
y = torch.randint(0, 4, (N,))

g = Graph(node_features=x, edge_index=edge_index, y=y)
print(f"Graph: {g}")

loader = NeighborLoader(g, fanouts=[10, 5], batch_size=32, shuffle=True, seed=42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvMessagePassing(in_shape=(C, H, W), out_shape=(16, H, W))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 4)

    def forward(self, x, ei):
        z = self.conv(x, ei).relu()
        return self.head(self.pool(z).flatten(1))


model = Model()
opt = Adam(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        logits = model(batch.node_features, batch.edge_index)
        loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.detach().item() * batch.batch_size
        n += batch.batch_size
    print(f"Epoch {epoch+1}  loss={total_loss/n:.4f}")

print("Demo PASSED")
