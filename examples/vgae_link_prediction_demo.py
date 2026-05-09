"""VGAE / Graph Autoencoder link prediction demo.

Trains a VGAE on a synthetic graph to reconstruct edges.
No downloads required.
"""
import torch
from tgraphx.mining import (
    VGAE, GraphAutoencoder, VGAEGCNEncoder, DotProductDecoder,
    train_gae_step, evaluate_link_prediction,
    stochastic_block_model_graph,
)

print("=" * 60)
print("VGAE Link Prediction Demo (TGraphX v0.5.0)")
print("=" * 60)
torch.manual_seed(0)

# Synthetic 2-community graph.
ei, N, labels = stochastic_block_model_graph([10, 10], p_in=0.5, p_out=0.05, seed=0)
x = torch.randn(N, 8)
print(f"\nGraph: {N} nodes, {ei.size(1)} edges, 2 communities")

# Split edges: 80% train, 20% test.
E = ei.size(1)
perm = torch.randperm(E, generator=torch.Generator().manual_seed(0))
n_train = int(0.8 * E)
train_ei = ei[:, perm[:n_train]]
test_pos = ei[:, perm[n_train:]]
# Simple random negatives.
neg_src = torch.randint(N, (test_pos.size(1),), generator=torch.Generator().manual_seed(1))
neg_dst = torch.randint(N, (test_pos.size(1),), generator=torch.Generator().manual_seed(2))
test_neg = torch.stack([neg_src, neg_dst])

# GAE.
enc = VGAEGCNEncoder(in_dim=8, hidden_dim=32, out_dim=16)
gae = GraphAutoencoder(enc)
opt = torch.optim.Adam(gae.parameters(), lr=5e-3)
print("\n--- Training GAE ---")
losses = []
for ep in range(50):
    neg = torch.stack([torch.randint(N, (n_train,)), torch.randint(N, (n_train,))])
    loss = train_gae_step(gae, opt, x, train_ei, train_ei, neg)
    losses.append(loss)
print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
gae_metrics = evaluate_link_prediction(gae, x, train_ei, test_pos, test_neg)
print(f"  Test AUROC: {gae_metrics['auroc']:.4f}")

# VGAE.
enc2 = VGAEGCNEncoder(in_dim=8, hidden_dim=32, out_dim=16)
vgae = VGAE(enc2)
opt2 = torch.optim.Adam(vgae.parameters(), lr=5e-3)
print("\n--- Training VGAE ---")
losses2 = []
for ep in range(50):
    neg = torch.stack([torch.randint(N, (n_train,)), torch.randint(N, (n_train,))])
    loss = train_gae_step(vgae, opt2, x, train_ei, train_ei, neg)
    losses2.append(loss)
print(f"  Loss: {losses2[0]:.4f} → {losses2[-1]:.4f}")
vgae_metrics = evaluate_link_prediction(vgae, x, train_ei, test_pos, test_neg)
print(f"  Test AUROC: {vgae_metrics['auroc']:.4f}")

print("\nDemo complete.")
