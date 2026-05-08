"""Graph path algorithms demo: BFS/DFS, Dijkstra, MST, cuts.

All algorithms are deterministic and require no downloads.
"""
import torch
from tgraphx.mining import (
    bfs_order, dfs_order, multi_source_bfs,
    dijkstra_shortest_path, reconstruct_path,
    all_pairs_shortest_path_length,
    minimum_spanning_tree, maximum_spanning_tree,
    cut_size, conductance, normalized_cut, boundary_edges,
)

print("=" * 60)
print("Graph Path Algorithms Demo (TGraphX v0.4.3)")
print("=" * 60)

# Weighted graph: 0-1-2-3 chain with a shortcut 0-3 via weight 4.
edges = [(0,1,1.0),(1,2,1.0),(2,3,1.0),(0,3,4.0)]
src = [u for u,v,_ in edges] + [v for u,v,_ in edges]
dst = [v for u,v,_ in edges] + [u for u,v,_ in edges]
w   = [w for _,_,w in edges] + [w for _,_,w in edges]
ei  = torch.tensor([src, dst], dtype=torch.long)
wt  = torch.tensor(w, dtype=torch.float)
N   = 4

print(f"\nGraph: 4-node weighted (chain + shortcut 0→3, weight 4)")
print(f"Edges: {edges}")

# BFS / DFS.
print(f"\nBFS from node 0: {bfs_order(ei, 0, N).tolist()}")
print(f"DFS from node 0: {dfs_order(ei, 0, N).tolist()}")

# Dijkstra.
dist, pred = dijkstra_shortest_path(ei, 0, N, edge_weight=wt)
print(f"\nDijkstra from 0: {[round(x, 2) for x in dist.tolist()]}")
for t in range(1, N):
    path = reconstruct_path(0, t, pred)
    print(f"  Path 0→{t}: {path}  (dist {dist[t]:.1f})")

# All-pairs.
D = all_pairs_shortest_path_length(ei, N, edge_weight=wt)
print(f"\nAll-pairs shortest path matrix:\n{D.numpy().round(1)}")

# MST.
mst_ei, mst_w, total = minimum_spanning_tree(ei, N, edge_weight=wt)
print(f"\nMST total weight: {total:.1f} (chain 0-1-2-3 = 3.0; shortcut 0-3 = 4.0)")
print(f"MST has {mst_ei.size(1)//2} undirected edges")

# Cuts.
subset = torch.tensor([0, 1])
print(f"\nPartition S={{{subset.tolist()}}}, complement={{2,3}}:")
print(f"  cut_size: {cut_size(ei, N, subset)}")
print(f"  conductance: {conductance(ei, N, subset):.4f}")

labels = torch.tensor([0, 0, 1, 1])
nc = normalized_cut(ei, N, labels)
print(f"  normalized_cut with labels {labels.tolist()}: {nc:.4f}")

print("\nDemo complete.")
