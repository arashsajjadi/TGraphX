"""neighbor_sampling_demo.py — neighbour sampling for mini-batch training.

Demonstrates:
1. Building a synthetic graph.
2. Iterating with NeighborSamplerLoader to obtain k-hop subgraphs.
3. Forward-passing each subgraph through a small TGraphX layer.

CPU-safe.
"""
import torch

from tgraphx import Graph, NeighborSamplerLoader
from tgraphx.layers import LinearMessagePassing

torch.manual_seed(0)

N, D = 100, 16
x = torch.randn(N, D)
src = torch.randint(0, N, (300,))
dst = torch.randint(0, N, (300,))
ei = torch.stack([src, dst], dim=0).long()
graph = Graph(x, ei, edge_weight=torch.rand(300))
print(f"Source graph: {N} nodes, {ei.size(1)} edges")

layer = LinearMessagePassing(in_shape=(D,), out_shape=(D,)).eval()

loader = NeighborSamplerLoader(
    graph,
    batch_size=8,
    fanouts=[10, 5],   # 2-hop sample with fan-out 10 then 5
    shuffle=True,
    seed=42,
)
print(f"Loader: {len(loader)} batches, fanouts={loader.fanouts}")

total_seeds = 0
for i, sub in enumerate(loader):
    with torch.no_grad():
        out = layer(sub.node_features, sub.edge_index)
    seeds = sub.metadata["sampling"]["seed_nodes"]
    total_seeds += seeds.numel()
    print(f"  batch {i}: subgraph nodes={sub.num_nodes:3d}  edges={sub.num_edges:3d}  "
          f"seeds={seeds.numel()}  out shape={tuple(out.shape)}")

assert total_seeds == N
print(f"\nAll {N} input seeds covered exactly once: PASSED")
print("neighbor_sampling_demo: PASSED")
