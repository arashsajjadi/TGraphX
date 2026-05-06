"""Core data structures and utilities for TGraphX."""

from .graph import Graph, GraphBatch
from .dataloader import GraphDataset, GraphDataLoader
from .utils import load_config, get_device
from .graph_utils import (
    add_self_loops,
    coalesce_edges,
    is_undirected,
    make_undirected,
    remove_self_loops,
    validate_edge_features,
    validate_edge_index,
    validate_edge_weight,
)

__all__ = [
    "Graph",
    "GraphBatch",
    "GraphDataset",
    "GraphDataLoader",
    "load_config",
    "get_device",
    "add_self_loops",
    "coalesce_edges",
    "is_undirected",
    "make_undirected",
    "remove_self_loops",
    "validate_edge_features",
    "validate_edge_index",
    "validate_edge_weight",
]
