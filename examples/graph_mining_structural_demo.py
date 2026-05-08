"""Graph mining structural features demo.

Demonstrates: graph_density, degree_statistics, graph_summary,
structural_features, graph motifs.
"""
import torch
from tgraphx.mining import (
    graph_density, degree_statistics, graph_summary,
    structural_features, motif_counts, motif_features,
)

print("=" * 60)
print("Graph Mining — Structural Features Demo")
print("=" * 60)

# ── Triangle (K3) ──────────────────────────────────────────────────────────
ei_k3 = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
N_k3 = 3
print(f"\nK3 graph (3 nodes, 6 directed edges):")
print(f"  Density: {graph_density(ei_k3, N_k3):.3f}")
stats = degree_statistics(ei_k3, N_k3)
print(f"  Mean out-degree: {stats['mean_out_degree']:.2f}")

motifs = motif_counts(ei_k3, N_k3, directed=False)
print(f"  Triangles: {motifs['triangles']}")
print(f"  Wedges: {motifs['wedges']}")
print(f"  Mean clustering coeff: {motifs['mean_clustering_coefficient']:.3f}")

# ── Path graph ─────────────────────────────────────────────────────────────
ei_path = torch.tensor([[0,1,2,3,1,2,3,4],[1,2,3,4,0,1,2,3]], dtype=torch.long)
N_path = 5
print(f"\nPath graph (5 nodes):")
print(f"  Density: {graph_density(ei_path, N_path):.4f}")
motifs_p = motif_counts(ei_path, N_path, directed=False)
print(f"  Triangles: {motifs_p['triangles']}")  # expected 0
print(f"  Wedges: {motifs_p['wedges']}")

# ── Structural features ────────────────────────────────────────────────────
print(f"\nStructural features for path graph:")
sf = structural_features(ei_path, N_path, features=("degree", "log_degree", "norm_degree"))
print(f"  Shape: {tuple(sf.shape)}")
for i in range(N_path):
    print(f"  Node {i}: degree={sf[i,0]:.0f}  log_deg={sf[i,1]:.3f}  norm={sf[i,2]:.3f}")

# ── Graph summary ──────────────────────────────────────────────────────────
print(f"\nGraph summary (K3):")
s = graph_summary(ei_k3, N_k3)
for k, v in s.items():
    if k != "warnings":
        print(f"  {k}: {v}")

print("\nDemo complete.")
