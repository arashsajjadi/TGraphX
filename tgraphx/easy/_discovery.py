"""Discovery functions and capability registry for TGraphX easy mode."""
from __future__ import annotations

from typing import Dict, Optional

from ._exceptions import TGraphXUnknownNameError

_TASKS: Dict[str, str] = {
    "node_classification": "Predict a label for each node in the graph.",
    "graph_classification": "Predict a label for each whole graph.",
    "link_prediction": "Predict whether edges exist between node pairs.",
    "node_regression": "Predict a continuous value per node.",
    "graph_regression": "Predict a continuous value per graph.",
    "knowledge_graph_completion": "Complete missing (h, r, t) triples in a KG.",
    "graph_generation": "Generate new graphs with desired properties.",
    "evolutionary_optimization": "Optimize graph structure via evolutionary search.",
    "graph_rl": "Train RL agents that act on graph environments.",
    "graph_mining": "Structural analysis: communities, centrality, anomalies.",
}

_MODELS: Dict[str, Dict[str, str]] = {
    "node_classification": {
        "tensor_gcn": "Two-layer ConvMessagePassing for [N, C, H, W] node features.",
        "vector_gcn": "Two-layer GCNConv for [N, D] vector node features.",
        "tensor_sage": "Two-layer TensorGraphSAGELayer for tensor features.",
        "auto": "Automatically choose based on node feature shape.",
    },
    "graph_classification": {
        "tensor_gcn_pool": "ConvMessagePassing + global pool for graph classification.",
        "vector_gcn_pool": "GCNConv + global pool for vector features.",
        "auto": "Automatically choose based on node feature shape.",
    },
    "link_prediction": {
        "dot_product": "Dot-product link predictor.",
        "bilinear": "Bilinear link predictor.",
    },
}

_SAMPLERS: Dict[str, str] = {
    "neighbor": "NeighborLoader — multi-hop neighbor sampling (best for large graphs).",
    "full": "Use all nodes as a single batch (best for small graphs).",
    "graphsaint_node": "GraphSAINT node sampler.",
    "graphsaint_edge": "GraphSAINT edge sampler.",
    "graphsaint_rw": "GraphSAINT random-walk sampler.",
    "cluster": "Cluster-GCN partitioned loader.",
}

_WORKFLOWS: Dict[str, str] = {
    "train_node_classifier": "Train a GNN for node classification.",
    "fit_node_classifier": "Alias for train_node_classifier.",
    "synthetic_tensor_node_classification": "Generate synthetic tensor node classification data.",
    "synthetic_vector_node_classification": "Generate synthetic vector node classification data.",
    "doctor": "Check TGraphX installation and dependencies.",
    "show_capabilities": "Show all TGraphX capabilities.",
}


def list_tasks() -> Dict[str, str]:
    """Return a dict of task names and their descriptions."""
    return dict(_TASKS)


def list_models(task: Optional[str] = None) -> Dict[str, str]:
    """Return available model names for a given task.

    Args:
        task: Task name (e.g. ``"node_classification"``).  When ``None``,
            return all models for all tasks.

    Returns:
        Dict mapping model name to description.
    """
    if task is None:
        result: Dict[str, str] = {}
        for models in _MODELS.values():
            result.update(models)
        return result
    if task not in _MODELS:
        available = list(_MODELS)
        raise TGraphXUnknownNameError(
            f"Unknown task '{task}'. Available tasks: {available}."
        )
    return dict(_MODELS[task])


def list_samplers() -> Dict[str, str]:
    """Return available sampler names and their descriptions."""
    return dict(_SAMPLERS)


def list_workflows() -> Dict[str, str]:
    """Return available high-level workflow function names."""
    return dict(_WORKFLOWS)


def explain_workflow(name: str) -> str:
    """Print and return a description of a workflow function."""
    if name not in _WORKFLOWS:
        raise TGraphXUnknownNameError(
            f"Unknown workflow '{name}'.  Available: {list(_WORKFLOWS)}."
        )
    desc = _WORKFLOWS[name]
    print(f"{name}: {desc}")
    return desc
