"""Graph mining — WL kernel and similarity demo."""
import torch
from tgraphx.mining import (
    weisfeiler_lehman_labels, wl_graph_features, wl_kernel_matrix,
    pairwise_graph_similarity, degree_histogram_distance,
)

print("=" * 60)
print("Weisfeiler-Lehman Kernel Demo")
print("=" * 60)

# Three graphs: two identical chains, one star.
def chain_graph(N):
    src = list(range(N-1)) + list(range(1,N))
    dst = list(range(1,N)) + list(range(N-1))
    return {"edge_index": torch.tensor([src,dst], dtype=torch.long), "num_nodes": N}

def star_graph(N):
    src = [0]*(N-1) + list(range(1,N))
    dst = list(range(1,N)) + [0]*(N-1)
    return {"edge_index": torch.tensor([src,dst], dtype=torch.long), "num_nodes": N}

g1 = chain_graph(4)  # identical
g2 = chain_graph(4)  # identical
g3 = star_graph(4)   # different structure

graphs = [g1, g2, g3]
labels = ["chain-1", "chain-2", "star"]

feat, vocab = wl_graph_features(graphs, num_iterations=2)
print(f"\nWL feature matrix shape: {tuple(feat.shape)}")
print(f"Vocabulary size: {len(vocab)}")

K = wl_kernel_matrix(graphs, normalize=True)
print(f"\nWL kernel matrix (normalised):")
for i, li in enumerate(labels):
    row = [f"{float(K[i,j]):.3f}" for j in range(len(graphs))]
    print(f"  {li:10s}: {row}")

print(f"\nExpected: chain-1 vs chain-2 ≈ 1.0, chain vs star < 1.0")
print(f"  chain-1 vs chain-2: {float(K[0,1]):.3f}")
print(f"  chain-1 vs star:    {float(K[0,2]):.3f}")

# Degree histogram distance.
d = degree_histogram_distance(g1["edge_index"], g1["num_nodes"],
                               g3["edge_index"], g3["num_nodes"])
print(f"\nDegree histogram L1 distance (chain vs star): {d:.4f}")

print("\nDemo complete.")
