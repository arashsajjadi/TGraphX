"""Node2Vec / DeepWalk unsupervised graph embeddings demo.

No downloads required — uses a synthetic graph.
"""
import torch
from tgraphx.mining import (
    node2vec_walks, deepwalk_walks, generate_skipgram_pairs,
    Node2VecEmbedding, train_node2vec_step, extract_node2vec_embeddings,
    stochastic_block_model_graph,
)

print("=" * 60)
print("Node2Vec / DeepWalk Embedding Demo (TGraphX v0.4.4)")
print("=" * 60)
torch.manual_seed(0)

# Two-community SBM graph.
ei, N, labels = stochastic_block_model_graph([8, 8], p_in=0.6, p_out=0.05, seed=0)
print(f"\nGraph: {N} nodes, {ei.size(1)} edges, 2 communities")

# Generate walks.
walks = node2vec_walks(ei, N, walk_length=20, walks_per_node=10, p=1.0, q=1.0, seed=0)
print(f"Generated {walks.shape[0]} walks of length {walks.shape[1]}")

# Generate skip-gram pairs.
centers, contexts, negatives = generate_skipgram_pairs(
    walks, window_size=5, negative_ratio=5, num_nodes=N, seed=0,
)
print(f"Skip-gram pairs: {centers.size(0)} positive, {negatives.size(0)} negative")

# Train embedding model.
model = Node2VecEmbedding(num_nodes=N, embedding_dim=8)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
losses = []
for _ in range(20):
    losses.append(train_node2vec_step(model, opt, centers, contexts, negatives))
print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f} (↓ {losses[0] > losses[-1]})")

# Extract and evaluate.
emb = extract_node2vec_embeddings(model)  # [N, 8] L2-normalised
print(f"Embedding shape: {emb.shape}")

# Check intra-community similarity vs inter-community.
# Community 0: nodes 0-7, Community 1: nodes 8-15.
sim = emb @ emb.t()  # cosine similarity (emb is normalised)
intra_sim = float((sim[:8, :8].mean() + sim[8:, 8:].mean()) / 2)
inter_sim = float(sim[:8, 8:].mean())
print(f"Intra-community mean cosine sim: {intra_sim:.3f}")
print(f"Inter-community mean cosine sim: {inter_sim:.3f}")

print("\nDemo complete.")
