"""Heterogeneous graph mining utilities.

Utilities for mining typed-node / typed-edge (heterogeneous) graphs.

Stability: Experimental (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "typed_degree_features",
    "relation_frequency_features",
]


def typed_degree_features(
    hetero_graph,
    mode: str = "out",
) -> Dict[str, torch.Tensor]:
    """Compute per-type degree features for a heterogeneous graph.

    For each node type, the degree is the count of edges of any relation
    that connect to a node of that type (outgoing for ``mode="out"``,
    incoming for ``mode="in"``, both for ``mode="both"``).

    Args:
        hetero_graph: A TGraphX :class:`~tgraphx.HeteroGraph`.
        mode: ``"out"``, ``"in"``, or ``"both"``.

    Returns:
        Dict ``{node_type: FloatTensor[N_type]}``.
    """
    if mode not in ("out", "in", "both"):
        raise ValueError(f"mode must be 'out', 'in', or 'both'; got {mode!r}")

    node_counts: Dict[str, int] = {}
    for nt in hetero_graph.node_types:
        feats = hetero_graph.node_feature_stores.get(nt)
        if feats is not None:
            node_counts[nt] = int(feats.size(0))
        else:
            node_counts[nt] = 0

    deg: Dict[str, torch.Tensor] = {
        nt: torch.zeros(nc, dtype=torch.long)
        for nt, nc in node_counts.items()
    }

    for et in hetero_graph.edge_types:
        src_type, rel_type, dst_type = et
        ei = hetero_graph.edge_index(et)
        if ei is None or ei.numel() == 0:
            continue
        n_src = node_counts.get(src_type, 0)
        n_dst = node_counts.get(dst_type, 0)
        ones = torch.ones(ei.size(1), dtype=torch.long)
        if mode in ("out", "both") and n_src > 0:
            deg[src_type].scatter_add_(0, ei[0].clamp(0, n_src - 1), ones)
        if mode in ("in", "both") and n_dst > 0:
            deg[dst_type].scatter_add_(0, ei[1].clamp(0, n_dst - 1), ones)

    return {nt: d.float() for nt, d in deg.items()}


def relation_frequency_features(
    hetero_graph,
) -> Dict[str, int]:
    """Count the number of edges per relation type.

    Args:
        hetero_graph: A TGraphX :class:`~tgraphx.HeteroGraph`.

    Returns:
        Dict ``{relation_type_str: edge_count}`` (JSON-serializable).
    """
    result: Dict[str, int] = {}
    for et in hetero_graph.edge_types:
        src_type, rel_type, dst_type = et
        key = f"{src_type}__{rel_type}__{dst_type}"
        ei = hetero_graph.edge_index(et)
        result[key] = int(ei.size(1)) if ei is not None else 0
    return result
