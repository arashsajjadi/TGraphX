"""Graph mining — anomaly detection demo."""
import torch
from tgraphx.mining import DegreeAnomalyScorer, graph_level_anomaly_scores, write_anomaly_summary
import tempfile, os, json

print("=" * 60)
print("Graph Anomaly Detection Demo")
print("=" * 60)

# Sparse graph with one hub (node 0 connects to all others).
src = [0]*8 + list(range(1,9)) + [1,2,3,4,5,2,3,4,3,4,4]
dst = list(range(1,9)) + [0]*8 + [2,3,4,5,6,3,4,5,4,5,5]
ei = torch.tensor([src+dst, dst+src], dtype=torch.long).unique(dim=1)
N = 9

scorer = DegreeAnomalyScorer().fit(ei, N)
scores = scorer.score_nodes(ei, N)
print(f"\nNode anomaly scores (hub = node 0):")
for i, s in enumerate(scores.tolist()):
    label = " ← HUB" if i == 0 else ""
    print(f"  Node {i}: {s:.3f}{label}")
top = scorer.top_k_anomalous(ei, N, k=3)
print(f"\nTop-3 anomalous nodes: {list(zip(top['node_ids'], top['scores']))}")

# Graph-level anomaly (find the outlier graph in a collection).
def make_graph(N, dense=False):
    if dense:
        src_ = [u for u in range(N) for v in range(N) if u != v]
        dst_ = [v for u in range(N) for v in range(N) if u != v]
    else:
        src_ = list(range(N-1)); dst_ = list(range(1,N))
    return {"edge_index": torch.tensor([src_+dst_, dst_+src_], dtype=torch.long),
            "num_nodes": N}

graphs = [make_graph(4, False)] * 5 + [make_graph(4, True)]  # last is dense outlier
g_scores = graph_level_anomaly_scores(graphs, method="degree_histogram")
print(f"\nGraph-level anomaly scores:")
for i, s in enumerate(g_scores.tolist()):
    label = " ← DENSE OUTLIER" if i == 5 else ""
    print(f"  Graph {i}: {s:.4f}{label}")

# Write report.
with tempfile.TemporaryDirectory() as tmp:
    p = write_anomaly_summary(
        os.path.join(tmp, "anomaly.json"), "degree_zscore",
        scores, top_k=3, threshold=2.0,
    )
    report = json.loads(open(p).read())
    print(f"\nAnomaly report: {list(report.keys())}")

print("\nDemo complete.")
