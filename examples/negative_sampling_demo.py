"""Negative sampling demo — v0.3.2 beta primitives.

Demonstrates:
- negative_sampling on a small directed graph.
- structured_negative_sampling (triplets).
- batched_negative_sampling with a GraphBatch.
- hard_negative_sampling with a toy embedding.
"""
from __future__ import annotations

import torch
from tgraphx import (
    negative_sampling,
    structured_negative_sampling,
    batched_negative_sampling,
    hard_negative_sampling,
)

print("=" * 60)
print("Negative Sampling Demo (TGraphX v0.3.2 beta)")
print("=" * 60)

# ── Graph: 5-node directed path 0→1→2→3→4 ────────────────────────────────────
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
N = 5
print(f"\nPositive edges: {edge_index.T.tolist()}")

# 1. Uniform negative sampling -------------------------------------------
neg = negative_sampling(edge_index, num_nodes=N, num_neg_samples=4, seed=0)
print(f"\nnegative_sampling (4 edges):\n  {neg.T.tolist()}")
assert neg.size(0) == 2

# 2. Structured -----------------------------------------------------------
i, j, k = structured_negative_sampling(edge_index, num_nodes=N, seed=0)
print(f"\nstructured_negative_sampling triplets (i, j, k) (first 4):")
pos_set = {(int(edge_index[0,c]), int(edge_index[1,c])) for c in range(edge_index.size(1))}
for c in range(min(4, len(i))):
    src, dst, neg_k = int(i[c]), int(j[c]), int(k[c])
    valid = (src, neg_k) not in pos_set
    print(f"  ({src}, {dst}) → k={neg_k}  valid={valid}")

# 3. Batched (two graphs) -------------------------------------------------
ei_b = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
neg_b = batched_negative_sampling(ei_b, batch, num_neg_samples=2, seed=0)
print(f"\nbatched_negative_sampling:")
for c in range(neg_b.size(1)):
    u, v = int(neg_b[0, c]), int(neg_b[1, c])
    print(f"  ({u}, {v})  graph={int(batch[u])} (no cross-graph)")

# 4. Hard negatives -------------------------------------------------------
emb = torch.zeros(N, 4)
for i_ in range(N):
    emb[i_, 0] = float(i_) / N  # nearby nodes have similar embeddings

neg_hard = hard_negative_sampling(
    edge_index, emb, num_nodes=N, num_neg_samples=4,
    candidate_pool_size=64, method="cosine", seed=0,
)
neg_rand = negative_sampling(edge_index, N, num_neg_samples=4, seed=0)

emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
def mean_sim(ne):
    if ne.size(1) == 0: return 0.0
    return float((emb_norm[ne[0]] * emb_norm[ne[1]]).sum(dim=1).mean().item())

print(f"\nhard_negative_sampling:")
print(f"  Hard  negatives: {neg_hard.T.tolist()}  mean_cosine_sim={mean_sim(neg_hard):.3f}")
print(f"  Rand  negatives: {neg_rand.T.tolist()}  mean_cosine_sim={mean_sim(neg_rand):.3f}")
print(f"  Hard negatives are harder (higher similarity) than random: "
      f"{mean_sim(neg_hard) >= mean_sim(neg_rand) - 0.05}")

print("\nDemo complete.")
