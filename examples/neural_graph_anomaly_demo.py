"""Neural graph anomaly detection demo.

Trains a GraphAutoencoderAnomalyDetector on a normal graph, then
scores a graph with injected anomalous nodes.
"""
import torch
from tgraphx.mining import (
    GraphAutoencoderAnomalyDetector,
    train_anomaly_autoencoder_step,
    write_anomaly_summary,
)
import tempfile, os, json

print("=" * 60)
print("Neural Graph Anomaly Detection Demo")
print("=" * 60)

torch.manual_seed(0)
N, D = 12, 8

# Normal data: all nodes follow a Gaussian near zero.
ei = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,1,2,3,4,5,6,7,8,9,10,11,0],
                   [1,2,3,4,5,6,7,8,9,10,11,0,0,1,2,3,4,5,6,7,8,9,10,11]], dtype=torch.long)
x_normal = torch.randn(N, D) * 0.3

# Train AE.
ae = GraphAutoencoderAnomalyDetector(in_dim=D, latent_dim=8, hidden_dim=16)
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
losses = []
for _ in range(50):
    loss = train_anomaly_autoencoder_step(ae, opt, x_normal, ei, N)
    losses.append(loss)
print(f"\nTraining loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# Inject anomaly at node 4.
x_anomalous = x_normal.clone()
x_anomalous[4] = x_anomalous[4] + 6.0  # large shift
x_anomalous[9] = x_anomalous[9] + 4.0  # moderate shift

# Score nodes.
scores = ae.node_anomaly_scores(x_anomalous, ei, N)
print("\nNode anomaly scores (higher = more anomalous):")
for i, s in enumerate(scores.tolist()):
    marker = " ← INJECTED" if i in (4, 9) else ""
    print(f"  Node {i:2d}: {s:.4f}{marker}")

# Verify injected nodes are highest.
top2 = scores.topk(2).indices.tolist()
print(f"\nTop-2 anomalous nodes: {top2}")
assert 4 in top2 or 9 in top2, "Expected injected nodes in top-2"
print("Sanity check: at least one injected node in top-2 OK")

# Write anomaly summary.
with tempfile.TemporaryDirectory() as tmp:
    p = write_anomaly_summary(
        os.path.join(tmp, "anomaly_summary.json"),
        "graph_autoencoder_reconstruction",
        scores, top_k=5, threshold=1.0,
    )
    report = json.loads(open(p).read())
    print(f"\nAnomaly report written: {list(report.keys())}")

print("\nDemo complete.")
