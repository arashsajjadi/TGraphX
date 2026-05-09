"""Knowledge graph embedding demo: TransE and DistMult.

No downloads required — uses a tiny synthetic knowledge graph.
"""
import torch
from tgraphx.mining import (
    KnowledgeGraph, negative_triple_sampling,
    TransE, DistMult, train_kg_step,
)

print("=" * 60)
print("Knowledge Graph Embedding Demo (TGraphX v0.4.4)")
print("=" * 60)
torch.manual_seed(0)

# Tiny synthetic KG: 8 entities, 3 relations, 12 triples.
# Encoding: subject → predicate → object.
triples = torch.tensor([
    [0, 0, 1], [1, 0, 2], [2, 0, 3], [3, 0, 0],  # relation 0: chain
    [0, 1, 4], [1, 1, 5], [2, 1, 6], [3, 1, 7],  # relation 1: cross
    [4, 2, 1], [5, 2, 2], [6, 2, 3], [7, 2, 0],  # relation 2: back
], dtype=torch.long)

kg = KnowledgeGraph(triples)
print(f"\nKG: {kg.num_entities} entities, {kg.num_relations} relations, {len(kg)} triples")
train_kg, val_kg, test_kg = kg.train_val_test_split(ratios=(0.7, 0.15, 0.15), seed=0)
print(f"Split: {len(train_kg)} train, {len(val_kg)} val, {len(test_kg)} test")

# Train TransE.
print("\n--- Training TransE ---")
transe = TransE(kg.num_entities, kg.num_relations, embedding_dim=16, margin=1.0)
opt_te = torch.optim.Adam(transe.parameters(), lr=0.01)
losses_te = []
for epoch in range(30):
    neg = negative_triple_sampling(train_kg.triples, kg.num_entities, num_neg=2, seed=epoch)
    loss = train_kg_step(transe, opt_te, train_kg.triples, neg[:len(train_kg.triples)])
    losses_te.append(loss)
print(f"  TransE loss: {losses_te[0]:.4f} → {losses_te[-1]:.4f} (↓ {losses_te[0] > losses_te[-1]})")

# Train DistMult.
print("\n--- Training DistMult ---")
distmult = DistMult(kg.num_entities, kg.num_relations, embedding_dim=16)
opt_dm = torch.optim.Adam(distmult.parameters(), lr=0.01)
losses_dm = []
for epoch in range(30):
    neg = negative_triple_sampling(train_kg.triples, kg.num_entities, num_neg=2, seed=epoch)
    loss = train_kg_step(distmult, opt_dm, train_kg.triples, neg[:len(train_kg.triples)])
    losses_dm.append(loss)
print(f"  DistMult loss: {losses_dm[0]:.4f} → {losses_dm[-1]:.4f} (↓ {losses_dm[0] > losses_dm[-1]})")

# Score a query triple.
print("\n--- TransE scores for (entity 0, relation 0, ?) ---")
with torch.no_grad():
    h = torch.tensor([0] * kg.num_entities)
    r = torch.tensor([0] * kg.num_entities)
    t = torch.arange(kg.num_entities)
    scores = transe.score(h, r, t)
    top_pred = int(scores.argmax().item())
    print(f"  Top predicted tail: {top_pred} (true: entity 1)")

print("\nDemo complete.")
