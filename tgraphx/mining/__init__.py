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

# Alias: motif_profile is the user-facing name for motif_counts (v1.3.4+).
motif_profile = motif_counts

# ── Level 2: WL kernels ──────────────────────────────────────────────────────
from .kernels import (
    weisfeiler_lehman_labels,
    wl_feature_histogram,
    wl_graph_features,
    wl_kernel_matrix,
    degree_histogram_features,
)


def wl_subtree_kernel(
    edge_index_a: "torch.Tensor",
    num_nodes_a: int,
    edge_index_b: "torch.Tensor",
    num_nodes_b: int,
    h: int = 2,
    node_labels_a=None,
    node_labels_b=None,
    normalize: bool = False,
) -> float:
    """WL subtree kernel between two graphs.

    Computes the inner product of WL feature histograms across ``h`` refinement
    rounds.  Unlabelled graphs are initialised by degree (default).

    Args:
        edge_index_a: ``LongTensor[2, E_a]`` — first graph edges.
        num_nodes_a:  Node count for the first graph.
        edge_index_b: ``LongTensor[2, E_b]`` — second graph edges.
        num_nodes_b:  Node count for the second graph.
        h: Number of WL refinement iterations (default 2).
        node_labels_a: Optional initial integer labels for graph A.
        node_labels_b: Optional initial integer labels for graph B.
        normalize: If ``True``, normalise by ``sqrt(K(A,A) * K(B,B))``.

    Returns:
        Float kernel value (higher = more similar).

    Stability: Beta (v1.3.4+).

    Example::

        from tgraphx.mining import wl_subtree_kernel
        import torch
        ei_ring5 = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        ei_ring3 = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        k = wl_subtree_kernel(ei_ring5, 5, ei_ring3, 3, h=3)
    """
    import torch as _torch

    graphs = [
        {"edge_index": edge_index_a, "num_nodes": num_nodes_a,
         **({"node_labels": node_labels_a} if node_labels_a is not None else {})},
        {"edge_index": edge_index_b, "num_nodes": num_nodes_b,
         **({"node_labels": node_labels_b} if node_labels_b is not None else {})},
    ]
    K = wl_kernel_matrix(graphs, num_iterations=h, normalize=False)
    k_ab = float(K[0, 1].item())
    if normalize:
        k_aa = float(K[0, 0].item())
        k_bb = float(K[1, 1].item())
        denom = (k_aa * k_bb) ** 0.5
        return k_ab / denom if denom > 0.0 else 0.0
    return k_ab

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

# ── Centrality (Beta) ────────────────────────────────────────────────────────
from .centrality import (
    degree_centrality,
    in_degree_centrality,
    out_degree_centrality,
    pagerank,
    personalized_pagerank,
    hits,
    katz_centrality,
    closeness_centrality,
    harmonic_centrality,
    betweenness_centrality,
    eigenvector_centrality,
    k_core_numbers,
)


def centrality_summary(
    edge_index: "torch.Tensor",
    num_nodes: int,
    directed: bool = False,
    top_k: int = 5,
) -> dict:
    """Return a summary of centrality metrics for a graph.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: Treat graph as directed (default ``False``).
        top_k: Number of top nodes to report per metric.

    Returns:
        JSON-serialisable dict with keys:

        - ``num_nodes`` — node count.
        - ``num_edges`` — edge count.
        - ``degree_centrality`` — tensor of per-node degree centrality.
        - ``top_degree_nodes`` — list of (node, value) for top-k by degree centrality.

    Stability: Beta (v1.3.4+).

    Example::

        from tgraphx.mining import centrality_summary
        import torch
        star_ei = torch.tensor([[0,0,0,0,1,2,3,4],[1,2,3,4,0,0,0,0]])
        s = centrality_summary(star_ei, num_nodes=5)
        # s["top_degree_nodes"][0][0] == 0  (star center has highest degree)
    """
    import torch as _torch
    from .centrality import degree_centrality as _dc

    dc = _dc(edge_index, num_nodes, directed=directed)
    num_edges = int(edge_index.size(1))

    # Top-k nodes by degree centrality.
    top_k_ = min(top_k, num_nodes)
    vals, idxs = _torch.topk(dc, top_k_)
    top_nodes = [(int(idxs[i].item()), float(vals[i].item())) for i in range(top_k_)]

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "degree_centrality": dc,
        "top_degree_nodes": top_nodes,
    }

# ── Graph generators (Beta) ──────────────────────────────────────────────────
from .generators import (
    erdos_renyi_graph,
    barabasi_albert_graph,
    stochastic_block_model_graph,
    watts_strogatz_graph,
    random_geometric_graph,
    planted_partition_graph,
    grid_2d_graph,
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
    karate_club_graph,
    synthetic_anomaly_graph,
    motif_injected_graph,
)

# ── Spectral analysis (Beta) ─────────────────────────────────────────────────
from .spectral import (
    graph_laplacian,
    normalized_laplacian,
    laplacian_eigenvalues,
    fiedler_vector,
    algebraic_connectivity,
    laplacian_eigvec_positional_encoding,
    spectral_clustering,
    spectral_distance,
    dirichlet_energy,
)

# ── Semi-supervised learning (Beta) ─────────────────────────────────────────
from .label_prop import (
    label_propagation,
    LabelPropagationClassifier,
)

# ── Embeddings (Beta) ────────────────────────────────────────────────────────
from .embeddings import (
    extract_node_embeddings,
    extract_graph_embeddings,
    embedding_similarity_matrix,
    embedding_pairwise_distances,
    embedding_nearest_neighbors,
)

# ── High-level API (Beta) ────────────────────────────────────────────────────
from .api import (
    analyze_graph,
    graph_mining_report,
    run_link_prediction_baseline,
)

# ── Graph paths and algorithms (Beta) ────────────────────────────────────────
from .paths import (
    bfs_order,
    dfs_order,
    multi_source_bfs,
    reachable_nodes,
    dijkstra_shortest_path,
    batched_shortest_path_length,
    all_pairs_shortest_path_length,
    reconstruct_path,
    minimum_spanning_tree,
    maximum_spanning_tree,
    cut_size,
    normalized_cut,
    conductance,
    volume,
    boundary_edges,
    write_path_summary,
)

# ── Graph learning utilities (Experimental) ───────────────────────────────────
from .graph_learning import (
    contrastive_loss,
    supervised_contrastive_loss,
    triplet_loss,
    bpr_loss,
    reconstruction_loss,
    drop_edges,
    drop_nodes,
    mask_node_features,
    add_random_edges,
    subgraph_sampling,
    DGIObjective,
    GraphCLObjective,
    create_negative_pairs,
    create_positive_pairs_from_batch,
)

# ── Structural encodings (Beta) ───────────────────────────────────────────────
from .structural_encodings import (
    degree_encoding,
    random_walk_structural_encoding,
    shortest_path_anchor_encoding,
    centrality_encoding,
    community_encoding,
    StructuralEncodingModule,
    attach_structural_encodings,
)

# ── Graph sequence models (Experimental) ─────────────────────────────────────
from .sequence_models import (
    GraphSequenceEncoder,
    GraphSequenceClassifier,
    GraphRNNEdgeGenerator,
    bfs_sequence_encode,
    random_walk_sequence_encode,
    pad_sequences,
)

# ── Matching, coloring, clique, max-flow (Beta/Experimental) ─────────────────
from .matching_coloring import (
    greedy_maximal_matching,
    bipartite_greedy_matching,
    greedy_coloring,
    welsh_powell_coloring,
    greedy_maximal_independent_set,
    enumerate_maximal_cliques,
    edmonds_karp_max_flow,
    min_cut_from_max_flow,
    wl_isomorphism_test,
    write_algorithm_report,
)

# ── Node2Vec / DeepWalk (Experimental) ───────────────────────────────────────
from .node2vec import (
    node2vec_walks,
    deepwalk_walks,
    generate_skipgram_pairs,
    Node2VecEmbedding,
    train_node2vec_step,
    extract_node2vec_embeddings,
)

# ── Knowledge graphs (Experimental) ─────────────────────────────────────────
from .knowledge_graph import (
    KnowledgeGraph,
    negative_triple_sampling,
    filtered_ranking_metrics,
    TransE,
    DistMult,
    train_kg_step,
)

# ── Hypergraphs (Experimental) ───────────────────────────────────────────────
from .hypergraph import (
    Hypergraph,
    incidence_to_bipartite_graph,
    clique_expansion,
    star_expansion,
    hypergraph_density,
)

# ── Graph IO (Beta) ───────────────────────────────────────────────────────────
from .graph_io import (
    read_edge_list_csv,
    write_edge_list_csv,
    read_graph_json,
    write_graph_json,
    save_graph_npz,
    load_graph_npz,
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

# ── VGAE / Graph Autoencoder (Experimental) ─────────────────────────────────
from .vgae import (
    GraphAutoencoder,
    VGAE,
    DotProductDecoder,
    MLPEdgeDecoder,
    GCNEncoder as VGAEGCNEncoder,
    train_gae_step,
    evaluate_link_prediction,
)

# ── Reports ───────────────────────────────────────────────────────────────────
from .reports import (
    write_graph_mining_summary,
    write_motif_summary,
    write_link_prediction_summary,
    write_anomaly_summary,
    write_prototype_membership_report,
    write_kg_summary,
    write_kg_metrics_report,
    write_hypergraph_summary,
    write_vgae_report,
    write_loader_summary,
    write_feature_store_summary,
    write_graphsaint_sampler_report,
    write_cluster_partition_report,
    write_hetero_summary,
    write_temporal_summary,
    write_ogb_tgb_report,
    write_estimator_report,
    write_pipeline_report,
    write_sparse_backend_report,
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
    "motif_profile",        # alias for motif_counts (v1.3.4+)
    # Kernels (Beta)
    "weisfeiler_lehman_labels",
    "wl_feature_histogram",
    "wl_graph_features",
    "wl_kernel_matrix",
    "degree_histogram_features",
    "wl_subtree_kernel",    # two-graph WL kernel (v1.3.4+)
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
    # Centrality (Beta)
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "pagerank",
    "personalized_pagerank",
    "hits",
    "katz_centrality",
    "closeness_centrality",
    "harmonic_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "k_core_numbers",
    "centrality_summary",   # convenience wrapper (v1.3.4+)
    # Generators (Beta)
    "erdos_renyi_graph",
    "barabasi_albert_graph",
    "stochastic_block_model_graph",
    "watts_strogatz_graph",
    "random_geometric_graph",
    "planted_partition_graph",
    "grid_2d_graph",
    "complete_graph",
    "cycle_graph",
    "path_graph",
    "star_graph",
    "karate_club_graph",
    "synthetic_anomaly_graph",
    "motif_injected_graph",
    # Spectral (Beta)
    "graph_laplacian",
    "normalized_laplacian",
    "laplacian_eigenvalues",
    "fiedler_vector",
    "algebraic_connectivity",
    "laplacian_eigvec_positional_encoding",
    "spectral_clustering",
    "spectral_distance",
    "dirichlet_energy",
    # Semi-supervised (Beta)
    "label_propagation",
    "LabelPropagationClassifier",
    # Embeddings (Beta)
    "extract_node_embeddings",
    "extract_graph_embeddings",
    "embedding_similarity_matrix",
    "embedding_pairwise_distances",
    "embedding_nearest_neighbors",
    # High-level API (Beta)
    "analyze_graph",
    "graph_mining_report",
    "run_link_prediction_baseline",
    # Neural mining (Experimental)
    "PrototypeMembershipScorer",
    "GraphAutoencoderAnomalyDetector",
    "GraphPatternClassifier",
    "create_synthetic_pattern_dataset",
    "train_prototype_membership_step",
    "train_anomaly_autoencoder_step",
    "train_graph_pattern_classifier_step",
]
