"""Public API stability registry."""
from __future__ import annotations

import difflib
from typing import Dict, List, Optional


# Maintained by hand; expanded each minor release.
_STABILITY: Dict[str, str] = {
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
