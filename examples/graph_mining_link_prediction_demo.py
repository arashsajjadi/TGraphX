"""Graph mining — classical link prediction scoring demo."""
import torch
from tgraphx.mining import (
    common_neighbors_score, jaccard_score, adamic_adar_score,
    resource_allocation_score, preferential_attachment_score,
)

print("=" * 60)
print("Link Prediction Scoring Demo")
print("=" * 60)

# 5-node graph: triangle 0-1-2 plus node 3 connected to 1.
edges = [(0,1),(1,2),(0,2),(1,3)]
src = [u for u,v in edges] + [v for u,v in edges]
dst = [v for u,v in edges] + [u for u,v in edges]
ei = torch.tensor([src, dst], dtype=torch.long)
N = 4

# Candidate pairs to score.
pairs = torch.tensor(
    [[0,0,2,3],[3,4,3,4]], dtype=torch.long,  # (0,3), (0,4), (2,3), (3,4)
)

print(f"\nGraph edges: {edges}")
print(f"Candidate pairs: {list(zip(pairs[0].tolist(), pairs[1].tolist()))}")
print()

scorers = {
    "Common Neighbors": common_neighbors_score(ei, pairs, num_nodes=5),
    "Jaccard":          jaccard_score(ei, pairs, num_nodes=5),
    "Adamic-Adar":      adamic_adar_score(ei, pairs, num_nodes=5),
    "Resource Alloc":   resource_allocation_score(ei, pairs, num_nodes=5),
    "Pref. Attachment": preferential_attachment_score(ei, pairs, num_nodes=5),
}

for name, scores in scorers.items():
    s = [f"{float(v):.3f}" for v in scores.tolist()]
    print(f"  {name:20s}: {s}")

print("\nDemo complete.")
