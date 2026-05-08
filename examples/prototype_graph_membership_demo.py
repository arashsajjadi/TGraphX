"""Prototype graph membership demo.

Demonstrates how to build class graphs from labelled embeddings,
build candidate graphs for queries, and evaluate membership scores.

This is the TGraphX-native class-graph pattern recognition paradigm.
"""
import torch
from tgraphx.mining import (
    ClassGraphBuilder,
    CandidateGraphBuilder,
    MembershipEvaluator,
    cosine_graph_membership_baseline,
    write_prototype_membership_report,
)
import tempfile, os, json

print("=" * 60)
print("Prototype Graph Membership Demo")
print("=" * 60)

# ── Synthetic dataset: 3 classes, 8-dim embeddings ─────────────────────────
torch.manual_seed(0)
N_support = 30   # support samples per class (in total)
N_query = 9      # 3 queries per class
D = 8
C = 3

# Support embeddings: class c is clustered around a unique prototype.
protos = torch.randn(C, D)
protos = protos / protos.norm(dim=1, keepdim=True)

support_embs = torch.cat([
    protos[c].unsqueeze(0) + 0.3 * torch.randn(N_support // C, D)
    for c in range(C)
])
support_labels = torch.tensor([c for c in range(C) for _ in range(N_support // C)])
support_feats = support_embs.clone()  # use embeddings as features here

query_embs = torch.cat([
    protos[c].unsqueeze(0) + 0.3 * torch.randn(N_query // C, D)
    for c in range(C)
])
query_labels = torch.tensor([c for c in range(C) for _ in range(N_query // C)])
query_feats = query_embs.clone()

# ── Build class graphs ──────────────────────────────────────────────────────
builder = ClassGraphBuilder(
    k_support=3, max_neighbor_fraction=0.6, ensure_connected=True,
)
builder.fit(support_feats, support_labels, embeddings=support_embs)
print("\nClass graph summary:")
for cls, info in builder.report().items():
    print(f"  Class {cls}: nodes={info['num_nodes']}, "
          f"edges={info['num_edges']}, density={info['density']:.3f}")

# ── Cosine baseline ─────────────────────────────────────────────────────────
print("\nCosine baseline (query 0, true class 0):")
q0 = query_embs[0]
baseline = cosine_graph_membership_baseline(q0, builder)
print(f"  Class scores: { {k: round(v,3) for k,v in baseline.items()} }")

# ── Candidate graph membership ──────────────────────────────────────────────
cand_builder = CandidateGraphBuilder(top_k_query=3)
cg_0 = builder.get_class_graph(0)
cand, q_idx = cand_builder.build(cg_0, query_feats[0], query_embs[0])
print(f"\nCandidate graph (query 0 vs class 0):")
print(f"  Nodes: {cand['num_nodes']} (query idx = {q_idx})")
print(f"  Edges: {cand['edge_index'].size(1)}")

# ── Evaluator with a simple mean-similarity scorer ─────────────────────────
def mean_query_similarity_score(candidate: dict) -> float:
    """Score by mean cosine sim from query to support nodes."""
    feats = candidate["node_features"].float()
    q_idx = candidate["query_idx"]
    q_emb = feats[q_idx]
    support_embs = torch.cat([feats[:q_idx], feats[q_idx+1:]], dim=0)
    if support_embs.numel() == 0:
        return 0.0
    q_norm = q_emb / (q_emb.norm().clamp(min=1e-8))
    s_norm = support_embs / (support_embs.norm(dim=1, keepdim=True).clamp(min=1e-8))
    return float((q_norm * s_norm).sum(dim=1).mean().item())

result = MembershipEvaluator.evaluate(
    score_fn=mean_query_similarity_score,
    query_features=query_feats,
    query_labels=query_labels,
    class_builder=builder,
    candidate_builder=cand_builder,
    query_embeddings=query_embs,
)
print(f"\nMembership evaluation:")
print(f"  Accuracy:          {result['accuracy']:.3f}")
print(f"  Balanced accuracy: {result['balanced_accuracy']:.3f}")
print(f"  Per-class F1:")
for cls, m in result["classification_report"].items():
    print(f"    class {cls}: F1={m['f1']:.3f}")

# ── Write report ────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    p = write_prototype_membership_report(
        os.path.join(tmp, "prototype_membership_report.json"), result,
    )
    print(f"\nReport written to: {p}")

print("\nDemo complete.")
