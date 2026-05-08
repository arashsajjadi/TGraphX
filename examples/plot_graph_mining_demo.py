"""Graph mining visualization demo (headless / save-to-file).

Demonstrates all major plotting utilities in tgraphx.plotting.
All figures are saved to a temp directory; no display required.
"""
import os, tempfile
import matplotlib
matplotlib.use("Agg")
import torch

from tgraphx.plotting import (
    plot_graph, plot_degree_distribution, plot_adjacency_matrix,
    plot_motif_summary, plot_graph_mining_summary,
    plot_link_prediction_score_distribution,
    plot_graph_similarity_heatmap, plot_anomaly_scores,
    plot_prototype_membership_scores, plot_confusion_matrix,
    plot_training_curves, plot_community_assignments,
    save_figure,
)
from tgraphx.mining import (
    graph_summary, motif_counts, wl_kernel_matrix,
    label_propagation_communities,
    DegreeAnomalyScorer,
)

print("=" * 60)
print("Graph Mining Visualization Demo")
print("=" * 60)

# Build a small test graph (two triangles sharing a node).
edges = [(0,1),(1,2),(0,2),(2,3),(3,4),(2,4)]
src = [u for u,v in edges] + [v for u,v in edges]
dst = [v for u,v in edges] + [u for u,v in edges]
ei = torch.tensor([src, dst], dtype=torch.long)
N = 5

with tempfile.TemporaryDirectory() as tmp:
    def save(fig, name):
        paths = save_figure(fig, os.path.join(tmp, name), formats=("png",))
        print(f"  Saved: {os.path.basename(paths[0])}")
        import matplotlib.pyplot as plt; plt.close(fig)

    print("\nPlotting graph structure...")
    fig, _ = plot_graph(ei, N, layout="spring", seed=42, title="Two triangles")
    save(fig, "graph")

    fig, _ = plot_degree_distribution(ei, N, title="Degree Distribution")
    save(fig, "degree_dist")

    fig, _ = plot_adjacency_matrix(ei, N, title="Adjacency Matrix")
    save(fig, "adjacency")

    print("\nPlotting mining summaries...")
    summary = graph_summary(ei, N)
    fig, _ = plot_graph_mining_summary(summary, title="Graph Summary")
    save(fig, "mining_summary")

    mc = motif_counts(ei, N, directed=False)
    fig, _ = plot_motif_summary(mc, title="Motif Counts")
    save(fig, "motifs")

    print("\nPlotting link prediction scores...")
    from tgraphx.mining import common_neighbors_score, jaccard_score
    pairs = torch.tensor([[0,1,2],[2,3,4]], dtype=torch.long)
    scores = {"common_neighbors": common_neighbors_score(ei, pairs, N),
              "jaccard": jaccard_score(ei, pairs, N)}
    fig, _ = plot_link_prediction_score_distribution(scores, title="Link Prediction Scores")
    save(fig, "link_pred_scores")

    print("\nPlotting similarity heatmap...")
    def _g(n): return {"edge_index": ei, "num_nodes": N}
    K = wl_kernel_matrix([_g(i) for i in range(4)], normalize=True)
    fig, _ = plot_graph_similarity_heatmap(K, labels=[f"G{i}" for i in range(4)])
    save(fig, "similarity_heatmap")

    print("\nPlotting anomaly scores...")
    scorer = DegreeAnomalyScorer().fit(ei, N)
    a_scores = scorer.score_nodes(ei, N)
    fig, _ = plot_anomaly_scores(a_scores, top_k=N, title="Anomaly Scores")
    save(fig, "anomaly_scores")

    print("\nPlotting prototype membership scores...")
    member_scores = {0: 0.85, 1: 0.25, 2: 0.10}
    fig, _ = plot_prototype_membership_scores(member_scores, true_label=0)
    save(fig, "membership_scores")

    print("\nPlotting confusion matrix...")
    M = [[8, 1, 1], [2, 7, 1], [0, 2, 8]]
    fig, _ = plot_confusion_matrix(M, class_names=["path", "star", "cycle"])
    save(fig, "confusion_matrix")

    print("\nPlotting training curves...")
    history = [{"train_loss": 0.9 - i*0.02, "val_loss": 0.95 - i*0.018}
               for i in range(20)]
    fig, _ = plot_training_curves(history)
    save(fig, "training_curves")

    print("\nPlotting community assignments...")
    comms = label_propagation_communities(ei, N, seed=0)
    fig, _ = plot_community_assignments(ei, N, comms, title="Communities")
    save(fig, "communities")

    print(f"\nAll plots saved to: {tmp}")
    saved = sorted(os.listdir(tmp))
    print(f"Files: {saved}")

print("\nDemo complete.")
