"""Neural prototype graph membership demo.

Trains a PrototypeMembershipScorer to decide whether a query graph
belongs to a class prototype.  Uses a fully synthetic 2-class dataset
of path vs. cycle graphs with class-specific node features.
"""
import torch
from tgraphx.mining import (
    ClassGraphBuilder,
    CandidateGraphBuilder,
    PrototypeMembershipScorer,
    create_synthetic_pattern_dataset,
    train_prototype_membership_step,
    cosine_graph_membership_baseline,
)

print("=" * 60)
print("Neural Prototype Membership Demo")
print("=" * 60)

torch.manual_seed(0)
D, N, C = 8, 6, 3

# Build class graphs from a synthetic dataset.
ds = create_synthetic_pattern_dataset(
    num_graphs_per_class=20, num_nodes=N, in_dim=D, seed=0, noise_std=0.1,
)
support_items = [g for g in ds[:30]]  # 10 per class
query_items = [g for g in ds[30:]]    # 10 per class hold-out

# Build class support graphs.
support_feats = torch.stack([g["node_features"] for g in support_items])  # [30, N, D] → need flat
support_feats_flat = support_feats.view(-1, D)  # [30*N, D]
support_labels = torch.tensor([g["label"] for g in support_items]).repeat_interleave(N)

# For topology, use per-graph mean embedding.
support_embs = torch.stack([g["node_features"].mean(0) for g in support_items])
# Assign each embedding to its correct index (one per sample).
support_feats_per_sample = torch.stack([g["node_features"].mean(0) for g in support_items])
support_labels_per_sample = torch.tensor([g["label"] for g in support_items])

builder = ClassGraphBuilder(k_support=4, max_neighbor_fraction=0.5)
builder.fit(support_feats_per_sample, support_labels_per_sample, embeddings=support_embs)
print("\nClass graph summary:")
for cls, info in builder.report().items():
    print(f"  Class {cls}: nodes={info['num_nodes']}, edges={info['num_edges']}")

# Build candidate graphs and targets.
cand_builder = CandidateGraphBuilder(top_k_query=3)
model = PrototypeMembershipScorer(in_dim=D, hidden_dim=32, out_dim=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

def build_candidates_and_targets(query_items, builder, cand_builder, num_candidates_per_query=3):
    """For each query, make one positive and two negative candidates."""
    candidates, targets = [], []
    classes = sorted(builder.class_graphs_.keys())
    for qi, g in enumerate(query_items[:6]):  # use first 6 queries
        qf = g["node_features"].mean(0)
        true_cls = g["label"]
        for cls in classes:
            cg = builder.get_class_graph(cls)
            cand, q_idx = cand_builder.build(cg, qf, qf)
            candidates.append(cand)
            targets.append(1.0 if cls == true_cls else 0.0)
    return candidates, torch.tensor(targets)

# Train.
candidates, targets = build_candidates_and_targets(query_items, builder, cand_builder)
print(f"\nTraining on {len(candidates)} candidates (positive/negative)...")
losses = []
for epoch in range(30):
    loss = train_prototype_membership_step(model, opt, candidates, targets)
    losses.append(loss)
print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} ({'↓ decreased' if losses[-1] < losses[0] else '?'})")

# Cosine baseline comparison.
print("\nCosine baseline vs neural model (first 3 queries):")
model.eval()
classes = sorted(builder.class_graphs_.keys())
for qi, g in enumerate(query_items[:3]):
    qf = g["node_features"].mean(0)
    true_cls = g["label"]
    baseline = cosine_graph_membership_baseline(qf, builder)
    baseline_pred = max(baseline, key=baseline.get)
    neural_scores = {}
    with torch.no_grad():
        for cls in classes:
            cg = builder.get_class_graph(cls)
            cand, q_idx = cand_builder.build(cg, qf, qf)
            score = float(model(cand["node_features"], cand["edge_index"], q_idx).item())
            neural_scores[cls] = score
    neural_pred = max(neural_scores, key=neural_scores.get)
    print(f"  Query {qi} (true={true_cls}): "
          f"baseline_pred={baseline_pred}  neural_pred={neural_pred}")

print("\nDemo complete.")
