"""set_transformer_demo.py — learned implicit relations over tensor nodes.

SetTransformerModel (v1.5.0) infers pairwise relations from node content
by global self-attention instead of consuming a supplied edge_index
(topology source "learned_implicit").  CPU-safe, runs in seconds.
"""
import warnings

import torch
import torch.nn as nn

from tgraphx import (
    SetTransformerModel,
    TopologyIgnoredWarning,
    build_model,
    topology_source_of,
)

torch.manual_seed(0)

print("--- Direct construction: tensor-valued nodes, variable set sizes ---")
model = SetTransformerModel(
    task="graph_classification",
    in_shape=(3, 8, 8),      # each node is a [3, 8, 8] tensor
    embed_dim=32,
    num_layers=2,
    num_heads=4,
    dropout=0.0,             # explicit — TGraphX never hides regularization
    num_classes=4,
)
x = torch.randn(9, 3, 8, 8)                      # 9 nodes across 2 graphs
batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
logits = model(x, None, batch=batch)
print(f"  logits shape: {tuple(logits.shape)}  (2 graphs, 4 classes)")
print(f"  topology_source: {model.topology_source}")

print("\n--- Permutation invariance of the graph-level output ---")
model.eval()
perm = torch.randperm(9)
delta = (model(x, None, batch=batch) -
         model(x[perm], None, batch=batch[perm])).abs().max().item()
print(f"  max |out - out_permuted| = {delta:.2e}")

print("\n--- A supplied edge_index is ignored (with a warning) ---")
edge_index = torch.tensor([[0, 1], [1, 0]])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model(x, edge_index, batch=batch)
topo = [w for w in caught if issubclass(w.category, TopologyIgnoredWarning)]
print(f"  TopologyIgnoredWarning emitted: {len(topo) == 1}")

print("\n--- Factory construction (family= is an alias for layer=) ---")
fmodel = build_model(
    task="graph_classification", family="set_transformer",
    in_shape=(16,), hidden_shape=(32,), num_layers=1,
    num_classes=3, heads=2, dropout=0.0, on_edge_index="ignore",
)
xv = torch.randn(7, 16)
bv = torch.tensor([0, 0, 0, 1, 1, 1, 1])
print(f"  factory logits: {tuple(fmodel(xv, None, batch=bv).shape)}")
print(f"  topology_source_of('set_transformer') = "
      f"{topology_source_of('set_transformer')!r}")
print(f"  topology_source_of('conv')            = "
      f"{topology_source_of('conv')!r}")

print("\n--- Config round trip (deterministic reconstruction) ---")
cfg = fmodel.config()
clone = SetTransformerModel.from_config(cfg)
clone.load_state_dict(fmodel.state_dict())
clone.eval()
fmodel.eval()
same = torch.equal(fmodel(xv, None, batch=bv), clone(xv, None, batch=bv))
print(f"  reconstructed model output identical: {same}")

print("\n--- One optimization step (sanity, not a benchmark) ---")
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
loss = nn.CrossEntropyLoss()(model(x, None, batch=batch), torch.tensor([0, 2]))
loss.backward()
opt.step()
print(f"  loss: {loss.item():.4f}  (finite: {torch.isfinite(loss).item()})")

print("\nset_transformer_demo: OK")
