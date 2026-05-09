"""NeighborLoader and FeatureStore demo for scalable node classification.

Demonstrates the NeighborLoader for mini-batch node classification
and InMemoryFeatureStore for on-demand feature fetching.
No downloads required.
"""
import torch
from tgraphx import Graph
from tgraphx.loaders import make_neighbor_loader, make_graph_loader
from tgraphx.feature_store import InMemoryFeatureStore
from tgraphx.mining import stochastic_block_model_graph

print("=" * 60)
print("NeighborLoader + FeatureStore Demo (TGraphX v0.5.0)")
print("=" * 60)
torch.manual_seed(0)

# Build a 100-node SBM graph.
ei, N, labels = stochastic_block_model_graph([25, 25, 25, 25], p_in=0.4, p_out=0.02, seed=0)
x = torch.randn(N, 16)
print(f"\nGraph: {N} nodes, {ei.size(1)} edges, 4 communities")

# ── Feature store ─────────────────────────────────────────────────────────────
store = InMemoryFeatureStore()
store.put("x", x)
store.put("label", labels.float())
print(f"\nFeature store: {store.summary()['num_features']} features, "
      f"{store.memory_estimate_bytes() // 1024} KB")

# ── NeighborLoader for node classification ────────────────────────────────────
g = Graph(x, ei)
train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[:60] = True

loader = make_neighbor_loader(g, fanouts=[10, 5], mask=train_mask, batch_size=16, shuffle=True, seed=42)
print(f"\nNeighborLoader batches: {len(loader)}")
print("(fanouts=[10,5] → 2-hop neighborhood per seed)")

for i, (subg, seeds) in enumerate(loader):
    # Fetch features from the feature store.
    x_batch = store.get("x", ids=seeds)
    y_batch = store.get("label", ids=seeds)
    print(f"\nBatch {i+1}: seeds={seeds.size(0)}, subgraph_nodes={subg.num_nodes}, "
          f"subgraph_edges={subg.edge_index.size(1)}, x_batch={tuple(x_batch.shape)}")
    if i >= 1:
        print("  (showing first 2 batches only)")
        break

# ── GraphLoader for graph classification ─────────────────────────────────────
graphs_list = [
    Graph(torch.randn(n, 8), torch.zeros((2, 0), dtype=torch.long))
    for n in range(4, 12)
]
g_loader = make_graph_loader(graphs_list, batch_size=3, shuffle=False, seed=0)
print(f"\nGraphLoader for {len(graphs_list)} graphs → {len(g_loader)} batches")
for batch in g_loader:
    print(f"  Batch node_features shape: {batch.node_features.shape}")
    break

print("\nDemo complete.")
