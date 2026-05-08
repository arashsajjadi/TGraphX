"""Graph algorithms demo — v0.3.2 beta primitives.

Demonstrates:
- connected_components / is_connected / number_connected_components.
- bfs_layers / bfs_edges.
- shortest_path_length.
- degree / degree_features.
"""
from __future__ import annotations

import torch
from tgraphx.algorithms import (
    bfs_edges,
    bfs_layers,
    connected_components,
    degree,
    degree_features,
    is_connected,
    number_connected_components,
    shortest_path_length,
)

print("=" * 60)
print("Graph Algorithms Demo (TGraphX v0.3.2 beta)")
print("=" * 60)

# ── Graph A: two components {0,1,2} and {3,4} ─────────────────────────────────
ei_a = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)
N_a = 5

labels = connected_components(ei_a, num_nodes=N_a)
K = number_connected_components(ei_a, N_a)
print(f"\nGraph A edges: {ei_a.T.tolist()}")
print(f"  Component labels: {labels.tolist()}")
print(f"  Number of components: {K}")
print(f"  Is connected: {is_connected(ei_a, N_a)}")

# ── Graph B: connected path 0 → 1 → 2 → 3 ────────────────────────────────────
ei_b = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
N_b = 4

print(f"\nGraph B edges: {ei_b.T.tolist()}")
print(f"  Is connected (as undirected): {is_connected(ei_b, N_b)}")

layers = bfs_layers(ei_b, source=0, num_nodes=N_b)
print(f"  BFS layers from 0: {[l.tolist() for l in layers]}")

bfs = bfs_edges(ei_b, source=0, num_nodes=N_b)
print(f"  BFS tree edges: {bfs.T.tolist()}")

dist = shortest_path_length(ei_b, source=0, num_nodes=N_b)
print(f"  Shortest path lengths from 0: {dist.tolist()}")

# ── Degree utilities ──────────────────────────────────────────────────────────
print(f"\nDegree utilities on graph B:")
print(f"  Out-degree:   {degree(ei_b, N_b, mode='out').tolist()}")
print(f"  In-degree:    {degree(ei_b, N_b, mode='in').tolist()}")
print(f"  Total-degree: {degree(ei_b, N_b, mode='both').tolist()}")

feats = degree_features(ei_b, num_nodes=N_b)
print(f"  degree_features (out|in|total) per node:")
for i, row in enumerate(feats.tolist()):
    print(f"    node {i}: {row}")

feats_log = degree_features(ei_b, num_nodes=N_b, log_scale=True)
print(f"  degree_features (log_scale=True) per node:")
for i, row in enumerate(feats_log.tolist()):
    print(f"    node {i}: {[f'{x:.3f}' for x in row]}")

print("\nDemo complete.")
