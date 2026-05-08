"""TGraphX graph mining and pattern recognition subsystem (v0.3.2+).

This package provides tensor-aware graph mining utilities for:

- structural feature extraction (density, degree statistics, summaries),
- classical link prediction scoring (common neighbours, Jaccard, Adamic-Adar, …),
- motif and graphlet counting (triangles, wedges, clustering coefficients),
- Weisfeiler-Lehman graph features and kernels,
- graph similarity and distance measures,
- community detection (label-propagation foundation),
- random walk generation,
- graph anomaly detection (degree-based baselines),
- prototype graph membership / class-graph utilities,
- small-pattern matching foundations,
- frequent pattern primitives,
- temporal graph mining utilities,
- heterogeneous graph mining utilities,
- report writers and dashboard artifact helpers.

All algorithms are:
  - pure PyTorch by default (no mandatory external dependency),
  - compatible with ``tgraphx.Graph`` / ``tgraphx.GraphBatch``,
  - tensor-aware (node features may be ``[N,D]``, ``[N,C,H,W]``, or ``[N,C,D,H,W]``),
  - device-portable (CPU, optional CUDA),
  - deterministic for a given ``seed``,
  - labelled with an explicit stability level.

TGraphX does **not** claim to replace NetworkX, PyG, DGL, cuGraph,
gSpan, or any full graph-analytics library.  These utilities are
GNN-training-oriented graph mining building blocks.

Stability levels:
- ``Beta`` — API stable within v0.3.x; may evolve before v0.4.0.
- ``Experimental`` — API or semantics may change in minor releases.

Quickstart::

    from tgraphx.mining import (
        graph_density,
        degree_statistics,
        triangle_count,
        local_clustering_coefficient,
        common_neighbors_score,
        wl_graph_features,
        label_propagation_communities,
        random_walks,
        ClassGraphBuilder,
        DegreeAnomalyScorer,
    )
"""
from __future__ import annotations

# ── Level 1: structural ────────────────────────────────────────────────────
from .structural import (
    graph_density,
    degree_statistics,
    graph_summary,
    structural_features,
    add_structural_features,
)

# ── Level 1: link prediction scores ──────────────────────────────────────────
from .link_prediction import (
    common_neighbors_score,
    jaccard_score,
    adamic_adar_score,
    resource_allocation_score,
    preferential_attachment_score,
)

# ── Level 1: motifs ─────────────────────────────────────────────────────────
from .motifs import (
    triangle_count,
    wedge_count,
    local_clustering_coefficient,
    motif_counts,
    motif_features,
)

# ── Level 2: WL kernels ──────────────────────────────────────────────────────
from .kernels import (
    weisfeiler_lehman_labels,
    wl_feature_histogram,
    wl_graph_features,
    wl_kernel_matrix,
    degree_histogram_features,
)

# ── Level 2: similarity ──────────────────────────────────────────────────────
from .similarity import (
    degree_histogram_distance,
    wl_feature_similarity,
    pairwise_graph_similarity,
    graph_feature_cosine_similarity,
)

# ── Level 2: communities ─────────────────────────────────────────────────────
from .communities import (
    label_propagation_communities,
    modularity,
    community_summary,
)

# ── Level 2: random walks ────────────────────────────────────────────────────
from .random_walk import (
    random_walks,
    generate_random_walks,
)

# ── Level 2: anomaly detection ───────────────────────────────────────────────
from .anomaly import (
    DegreeAnomalyScorer,
    EgoDensityAnomalyScorer,
    graph_level_anomaly_scores,
)

# ── Level 2: prototype graph membership ─────────────────────────────────────
from .prototype import (
    ClassGraphBuilder,
    CandidateGraphBuilder,
    GraphMembershipDataset,
    MembershipEvaluator,
    cosine_graph_membership_baseline,
)

# ── Level 3: patterns ────────────────────────────────────────────────────────
from .patterns import (
    path_pattern_count,
    star_pattern_count,
    contains_triangle,
    small_pattern_counts,
)

# ── Level 3: frequent patterns ───────────────────────────────────────────────
from .frequent import (
    frequent_node_labels,
    frequent_degree_bins,
    support_count,
)

# ── Level 3: temporal mining ─────────────────────────────────────────────────
from .temporal import (
    temporal_degree,
    sliding_window_edges,
    temporal_chronological_split,
    burst_score,
)

# ── Level 3: hetero mining ───────────────────────────────────────────────────
from .hetero import (
    typed_degree_features,
    relation_frequency_features,
)

# ── Neural mining (Experimental) ─────────────────────────────────────────────
from .neural import (
    PrototypeMembershipScorer,
    GraphAutoencoderAnomalyDetector,
    GraphPatternClassifier,
    create_synthetic_pattern_dataset,
    train_prototype_membership_step,
    train_anomaly_autoencoder_step,
    train_graph_pattern_classifier_step,
)

# ── Reports ───────────────────────────────────────────────────────────────────
from .reports import (
    write_graph_mining_summary,
    write_motif_summary,
    write_link_prediction_summary,
    write_anomaly_summary,
    write_prototype_membership_report,
)

__all__ = [
    # Structural (Beta)
    "graph_density",
    "degree_statistics",
    "graph_summary",
    "structural_features",
    "add_structural_features",
    # Link prediction (Beta)
    "common_neighbors_score",
    "jaccard_score",
    "adamic_adar_score",
    "resource_allocation_score",
    "preferential_attachment_score",
    # Motifs (Beta)
    "triangle_count",
    "wedge_count",
    "local_clustering_coefficient",
    "motif_counts",
    "motif_features",
    # Kernels (Beta)
    "weisfeiler_lehman_labels",
    "wl_feature_histogram",
    "wl_graph_features",
    "wl_kernel_matrix",
    "degree_histogram_features",
    # Similarity (Beta)
    "degree_histogram_distance",
    "wl_feature_similarity",
    "pairwise_graph_similarity",
    "graph_feature_cosine_similarity",
    # Communities (Beta)
    "label_propagation_communities",
    "modularity",
    "community_summary",
    # Random walks (Beta)
    "random_walks",
    "generate_random_walks",
    # Anomaly (Experimental)
    "DegreeAnomalyScorer",
    "EgoDensityAnomalyScorer",
    "graph_level_anomaly_scores",
    # Prototype (Experimental)
    "ClassGraphBuilder",
    "CandidateGraphBuilder",
    "GraphMembershipDataset",
    "MembershipEvaluator",
    "cosine_graph_membership_baseline",
    # Patterns (Experimental)
    "path_pattern_count",
    "star_pattern_count",
    "contains_triangle",
    "small_pattern_counts",
    # Frequent (Experimental)
    "frequent_node_labels",
    "frequent_degree_bins",
    "support_count",
    # Temporal mining (Experimental)
    "temporal_degree",
    "sliding_window_edges",
    "temporal_chronological_split",
    "burst_score",
    # Hetero mining (Experimental)
    "typed_degree_features",
    "relation_frequency_features",
    # Reports (Beta)
    "write_graph_mining_summary",
    "write_motif_summary",
    "write_link_prediction_summary",
    "write_anomaly_summary",
    "write_prototype_membership_report",
    # Neural mining (Experimental)
    "PrototypeMembershipScorer",
    "GraphAutoencoderAnomalyDetector",
    "GraphPatternClassifier",
    "create_synthetic_pattern_dataset",
    "train_prototype_membership_step",
    "train_anomaly_autoencoder_step",
    "train_graph_pattern_classifier_step",
]
