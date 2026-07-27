"""Public API stability registry."""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional


# Maintained by hand; expanded each minor release.
_STABILITY: Dict[str, str] = {
    # v1.5.0 configuration-transition helpers
    "DropoutDefaultChangeWarning": "stable",
    "LEGACY_CNN_DROPOUT_PROB": "stable",
    # Stable core
    "Graph": "stable",
    "GraphBatch": "stable",
    "GraphDataset": "stable",
    "GraphDataLoader": "stable",
    "set_seed": "stable",
    "ConvMessagePassing": "stable",
    "LinearMessagePassing": "stable",
    "TensorGATLayer": "stable",
    "TensorGraphSAGELayer": "stable",
    "TensorGINLayer": "stable",
    "GCNConv": "stable",
    "GATv2Conv": "stable",
    "APPNP": "stable",
    "global_mean_pool": "stable",
    "global_max_pool": "stable",
    "global_sum_pool": "stable",
    "build_grid_graph": "stable",
    "build_knn_graph": "stable",
    "build_radius_graph": "stable",
    "count_parameters": "stable",
    "fit": "stable",
    "train_epoch": "stable",
    "evaluate": "stable",
    "CSVLogger": "stable",
    "TensorBoardLogger": "stable",
    "write_run_metadata": "stable",
    "write_metrics_summary": "stable",
    "write_graph_stats": "stable",
    "write_dataset_metadata": "stable",
    "write_sampling_metadata": "stable",
    # Beta
    "GraphMiniBatch": "beta",
    "NeighborLoader": "beta",
    "LinkNeighborLoader": "beta",
    "GraphLoader": "beta",
    "InMemoryFeatureStore": "beta",
    "MemmapFeatureStore": "beta",
    "easy": "beta",
    "KnowledgeGraph": "beta",
    "KGTrainer": "beta",
    "KGTrainingConfig": "beta",
    "TransEModel": "beta",
    "DistMultModel": "beta",
    "RESCALModel": "beta",
    "SimplEModel": "beta",
    "KGEvaluator": "beta",
    "PyGPlanetoidDataset": "beta",
    "PyGTUDatasetAdapter": "beta",
    "MNISTPatchGraphDataset": "beta",
    "CIFAR10PatchGraphDataset": "beta",
    "graph_summary": "beta",
    "motif_profile": "beta",
    "degree_statistics": "beta",
    "triangle_count": "beta",
    # v1.4.0 ergonomics layer (beta — APIs may evolve)
    "ux": "beta",
    "validate_graph": "beta",
    "describe": "beta",
    "summary": "beta",
    "reproducible": "beta",
    "check_leakage": "beta",
    "save": "beta",
    "load": "beta",
    "knn_graph": "beta",
    "build_class_prototypes": "beta",
    "build_prototype_graph": "beta",
    "image_to_patch_graph": "beta",
    "audit_run_dir": "beta",
    "dashboard_audit": "beta",
    "workflow": "beta",
    "compare": "beta",
    "load_dataset": "beta",
    # v1.4.1 one-call helpers (beta)
    "classify_nodes": "beta",
    "node_classification": "beta",
    "fit_node_classifier": "beta",
    "train_node_classifier": "beta",
    "kg_completion": "beta",
    "fit_kg": "beta",
    "train_kg": "beta",
    "make_graph": "beta",
    "build_graph": "beta",
    "explain_error": "beta",
    "troubleshoot_error": "beta",
    "debug_batch": "beta",
    "batch_summary": "beta",
    "assert_batch_consistent": "beta",
    "dataset_card": "beta",
    "model_card": "beta",
    "benchmark_card": "beta",
    "audit_package_readiness": "beta",
    "WorkflowResult": "beta",
    "generate_graph": "experimental",
    "graph_generator": "experimental",
    "generate": "experimental",
    "evaluate_generated_graphs": "experimental",
    "graph_generation_report": "experimental",
    "compare_generated_graphs": "experimental",
    "generation_metrics": "experimental",
    "optimize_graph": "experimental",
    "evolve_graph": "experimental",
    "graph_evolution": "experimental",
    "run_evolution": "experimental",
    "train_graph_rl": "experimental",
    "graph_rl": "experimental",
    "run_rl": "experimental",
    "audit_generation_run": "beta",
    "audit_evolution_run": "beta",
    "audit_rl_run": "beta",
    # Experimental
    "ComplExModel": "experimental",
    "RotatEModel": "experimental",
    "run_kg_hpo": "experimental",
    "KGRGCNModel": "experimental",
    "HeteroGraph": "experimental",
    "TemporalGraphSequence": "experimental",
    "VGAEGraphGenerator": "experimental",
    "AutoregressiveEdgeGenerator": "experimental",
    "GeneticAlgorithmOptimizer": "experimental",
    "NSGAIIOptimizer": "experimental",
    "PPOAgent": "experimental",
    "DQNAgent": "experimental",
}


# Aliases recognized by various entry points (for discovery / LLM hints)
_ALIASES: Dict[str, List[str]] = {
    "Graph": ["x", "y", "labels", "node_labels", "edge_attr", "edge_features",
              "graph_label", "from_edges", "from_adjacency", "from_networkx"],
    "KnowledgeGraph": ["from_hrt", "from_triples", "triples",
                        "num_entities", "num_relations"],
    "NeighborLoader": ["fanouts", "num_neighbors", "fanout"],
    "ConvMessagePassing": ["in_shape", "out_shape"],
    "workflow": ["run_workflow", "pipeline"],
    "describe": ["summary", "inspect", "info"],
    "save": ["save_tgraphx", "save_graph"],
    "load": ["load_tgraphx", "load_graph"],
    "reproducible": ["seeded"],
    # v1.4.1 aliases
    "make_graph": ["build_graph", "graph"],
    "classify_nodes": ["node_classification", "fit_node_classifier",
                        "train_node_classifier"],
    "kg_completion": ["fit_kg", "train_kg"],
    "explain_error": ["troubleshoot_error"],
    "debug_batch": ["batch_summary", "assert_batch_consistent"],
    "generate_graph": ["graph_generator", "generate"],
    "evaluate_generated_graphs": ["graph_generation_report",
                                    "compare_generated_graphs",
                                    "generation_metrics"],
    "optimize_graph": ["evolve_graph", "graph_evolution", "run_evolution"],
    "train_graph_rl": ["graph_rl", "run_rl"],
}


def public_api() -> Dict[str, List[str]]:
    """Return all public TGraphX APIs grouped by stability level."""
    grouped: Dict[str, List[str]] = {
        "stable": [], "beta": [], "experimental": [], "deprecated": [],
    }
    for name, level in _STABILITY.items():
        grouped.setdefault(level, []).append(name)
    for k in grouped:
        grouped[k].sort()
    return grouped


def api_status(name: str) -> str:
    """Return the stability level of a public TGraphX API.

    Raises a helpful error with the closest match if the name is unknown.
    """
    if name in _STABILITY:
        return _STABILITY[name]
    # Try aliases
    for canonical, aliases in _ALIASES.items():
        if name in aliases:
            return f"{_STABILITY.get(canonical, 'unknown')} (alias of {canonical})"
    suggestion = difflib.get_close_matches(name, list(_STABILITY.keys()), n=1)
    hint = f" Closest match: {suggestion[0]!r}." if suggestion else ""
    raise KeyError(f"Unknown TGraphX API {name!r}.{hint}")


def list_aliases(canonical: str) -> List[str]:
    """Return known aliases for a canonical API name (may be empty)."""
    return list(_ALIASES.get(canonical, []))
