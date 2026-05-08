"""Self-supervised graph learning demo: contrastive losses, augmentations, DGI.

All operations are deterministic and require no downloads.
"""
import torch
import torch.nn as nn
from tgraphx.mining import (
    contrastive_loss, supervised_contrastive_loss, triplet_loss, bpr_loss,
    drop_edges, mask_node_features,
    DGIObjective,
    degree_encoding, centrality_encoding, attach_structural_encodings,
    GraphSequenceClassifier, bfs_sequence_encode, pad_sequences,
    erdos_renyi_graph, stochastic_block_model_graph,
)

print("=" * 60)
print("Self-Supervised Graph Learning Demo (TGraphX v0.4.3)")
print("=" * 60)

torch.manual_seed(0)
ei, N = erdos_renyi_graph(10, p=0.3, seed=0)
x = torch.randn(N, 8)

# ── Augmentations ─────────────────────────────────────────────────────────────
print("\n1. Graph augmentations:")
ei_drop, _ = drop_edges(ei, p=0.2, seed=0)
x_mask, mask = mask_node_features(x, p=0.15, seed=0)
print(f"   Original edges: {ei.size(1)} → dropped: {ei_drop.size(1)}")
print(f"   Masked features: {int(mask.sum())} of {x.numel()} elements")

# ── Contrastive loss ──────────────────────────────────────────────────────────
print("\n2. Contrastive loss (NT-Xent):")
z1 = torch.randn(8, 16, requires_grad=True)
z2 = torch.randn(8, 16, requires_grad=True)
cl = contrastive_loss(z1, z2, temperature=0.07)
print(f"   NT-Xent loss: {float(cl):.4f}")

print("\n3. Supervised contrastive loss:")
emb = torch.randn(12, 16, requires_grad=True)
lbs = torch.tensor([0,0,0,1,1,1,2,2,2,0,1,2])
scl = supervised_contrastive_loss(emb, lbs)
print(f"   SupCon loss: {float(scl):.4f}")

print("\n4. Triplet loss:")
tl = triplet_loss(z1[:4], z2[:4], torch.randn(4, 16))
print(f"   Triplet loss: {float(tl):.4f}")

print("\n5. BPR loss:")
bpr = bpr_loss(torch.randn(8), torch.randn(8))
print(f"   BPR loss: {float(bpr):.4f}")

# ── DGI objective ─────────────────────────────────────────────────────────────
print("\n6. DGI-style objective:")
dgi = DGIObjective(embed_dim=8, summary_dim=8)
pos_emb = torch.randn(N, 8, requires_grad=True)
neg_emb = torch.randn(N, 8, requires_grad=True)
dgi_loss = dgi(pos_emb, neg_emb)
print(f"   DGI loss: {float(dgi_loss):.4f}")

# ── Structural encodings ──────────────────────────────────────────────────────
print("\n7. Structural encodings:")
enc_deg = degree_encoding(ei, N)
print(f"   Degree encoding: {tuple(enc_deg.shape)}")
enc_cent = centrality_encoding(ei, N, include=["degree", "pagerank"])
print(f"   Centrality encoding: {tuple(enc_cent.shape)}")
x_aug = attach_structural_encodings(x, enc_deg)
print(f"   Augmented features: {x.shape} → {x_aug.shape}")

# ── Graph sequence classifier ─────────────────────────────────────────────────
print("\n8. Graph sequence classifier (SBM patterns):")
_, N_sbm, labels_sbm = stochastic_block_model_graph([5,5], 0.8, 0.05, seed=0)
ei_sbm, _, _ = stochastic_block_model_graph([5,5], 0.8, 0.05, seed=0)
# Build BFS sequences for two graphs.
x_sbm = torch.randn(10, 4)
seq1 = bfs_sequence_encode(ei_sbm, 10, node_features=x_sbm, start=0)
seq2 = bfs_sequence_encode(ei_sbm, 10, node_features=x_sbm, start=5)
padded, lengths = pad_sequences([seq1, seq2])
clf = GraphSequenceClassifier(input_dim=4, hidden_dim=16, num_classes=2)
logits = clf(padded, lengths)
print(f"   Sequence logits shape: {tuple(logits.shape)}")

print("\nDemo complete.")
