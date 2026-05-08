"""Structural graph mining features in pure PyTorch.

Functions in this module compute graph-level and node-level structural
properties that are commonly used as input features for graph mining
and GNN training.

Stability: Beta (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "graph_density",
    "degree_statistics",
    "graph_summary",
    "structural_features",
    "add_structural_features",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _validate(edge_index: torch.Tensor, num_nodes: int, tag: str = "edge_index") -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"{tag} must have shape [2, E]; got {tuple(edge_index.shape)}"
        )
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and int(edge_index.max().item()) >= num_nodes:
        raise ValueError(
            f"{tag} max node id {int(edge_index.max())} >= num_nodes={num_nodes}"
        )


def _degrees(
    edge_index: torch.Tensor, num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (out_degree, in_degree) LongTensors of shape [num_nodes]."""
    device = edge_index.device
    ones = torch.ones(edge_index.size(1), dtype=torch.long, device=device)
    out_deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    in_deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_index.numel():
        out_deg.scatter_add_(0, edge_index[0], ones)
        in_deg.scatter_add_(0, edge_index[1], ones)
    return out_deg, in_deg


def _self_loop_count(edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0:
        return 0
    return int((edge_index[0] == edge_index[1]).sum().item())


# ── Public API ───────────────────────────────────────────────────────────────


def graph_density(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = True,
    exclude_self_loops: bool = True,
) -> float:
    """Return the density of the graph.

    For a **directed** graph (``directed=True``) without self-loops:

        density = E / (N * (N - 1))

    For an **undirected** graph (``directed=False``) without self-loops:

        density = E_undirected / (N * (N - 1) / 2)

    where E_undirected counts each undirected edge once (i.e. if both
    ``(u,v)`` and ``(v,u)`` are present they count as one edge).

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count ``N``.
        directed: When ``True`` (default), treat the graph as directed.
        exclude_self_loops: When ``True`` (default), self-loops are not
            counted in the numerator or denominator.

    Returns:
        Float in ``[0.0, 1.0]``; ``0.0`` when ``num_nodes <= 1``.
    """
    _validate(edge_index, num_nodes)
    if num_nodes <= 1:
        return 0.0

    E = int(edge_index.size(1))
    if exclude_self_loops:
        E -= _self_loop_count(edge_index)

    max_edges: int
    if directed:
        max_edges = num_nodes * (num_nodes - 1)
    else:
        # Count unique undirected edges: sort each pair and deduplicate.
        if E > 0 and exclude_self_loops:
            ei = edge_index[:, edge_index[0] != edge_index[1]]
        elif E > 0:
            ei = edge_index
        else:
            ei = edge_index
        if ei.numel() > 0:
            # Convert to unique undirected pairs.
            pairs = torch.stack([
                torch.minimum(ei[0], ei[1]),
                torch.maximum(ei[0], ei[1]),
            ], dim=0)
            unique_pairs = torch.unique(pairs, dim=1)
            E = int(unique_pairs.size(1))
        max_edges = num_nodes * (num_nodes - 1) // 2

    if max_edges == 0:
        return 0.0
    return float(E) / float(max_edges)


def degree_statistics(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = True,
) -> Dict[str, Any]:
    """Return a dict of degree statistics for the graph.

    Keys:
        - ``min_out_degree``, ``max_out_degree``, ``mean_out_degree``
        - ``min_in_degree``, ``max_in_degree``, ``mean_in_degree``
        - ``min_total_degree``, ``max_total_degree``, ``mean_total_degree``
        - ``isolated_node_count``: nodes with total degree 0
        - ``density``: directed density (excludes self-loops)

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False``, in and out are swapped / equal.

    Returns:
        Plain Python dict of floats and ints (JSON-serializable).
    """
    _validate(edge_index, num_nodes)
    if num_nodes == 0:
        return {
            "min_out_degree": 0, "max_out_degree": 0, "mean_out_degree": 0.0,
            "min_in_degree": 0, "max_in_degree": 0, "mean_in_degree": 0.0,
            "min_total_degree": 0, "max_total_degree": 0, "mean_total_degree": 0.0,
            "isolated_node_count": 0,
            "density": 0.0,
        }

    out_deg, in_deg = _degrees(edge_index, num_nodes)
    total = out_deg + in_deg

    def _stats(t: torch.Tensor) -> Tuple[int, int, float]:
        return int(t.min().item()), int(t.max().item()), float(t.float().mean().item())

    out_min, out_max, out_mean = _stats(out_deg)
    in_min, in_max, in_mean = _stats(in_deg)
    tot_min, tot_max, tot_mean = _stats(total)
    isolated = int((total == 0).sum().item())

    return {
        "min_out_degree": out_min,
        "max_out_degree": out_max,
        "mean_out_degree": round(out_mean, 6),
        "min_in_degree": in_min,
        "max_in_degree": in_max,
        "mean_in_degree": round(in_mean, 6),
        "min_total_degree": tot_min,
        "max_total_degree": tot_max,
        "mean_total_degree": round(tot_mean, 6),
        "isolated_node_count": isolated,
        "density": round(graph_density(edge_index, num_nodes, directed=directed), 8),
    }


def graph_summary(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = True,
    include_components: bool = True,
) -> Dict[str, Any]:
    """Return a JSON-serializable dict summarising the graph.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: Whether to treat the graph as directed.
        include_components: When ``True`` (default), includes connected
            component count from :mod:`tgraphx.algorithms.connectivity`.

    Returns:
        Plain Python dict (JSON-serializable).
    """
    _validate(edge_index, num_nodes)
    num_edges = int(edge_index.size(1))
    self_loops = _self_loop_count(edge_index)

    summary: Dict[str, Any] = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_self_loops": self_loops,
        "directed": directed,
        **degree_statistics(edge_index, num_nodes, directed=directed),
    }

    if include_components and num_nodes > 0:
        try:
            from tgraphx.algorithms.connectivity import number_connected_components
            summary["num_connected_components"] = number_connected_components(
                edge_index, num_nodes
            )
        except Exception:  # pragma: no cover
            summary["num_connected_components"] = None

    warnings: List[str] = []
    if num_nodes > 10_000 and num_edges > num_nodes * 100:
        warnings.append("Dense large graph: some mining operations may be slow.")
    if summary.get("density", 0) > 0.5:
        warnings.append("High-density graph (>50%).")
    summary["warnings"] = warnings
    return summary


def structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    features: Tuple[str, ...] = ("degree", "in_degree", "out_degree", "log_degree"),
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return a structural feature matrix ``[num_nodes, F]``.

    Available features:

    - ``"degree"`` — total degree (in + out).
    - ``"in_degree"`` — in-degree.
    - ``"out_degree"`` — out-degree.
    - ``"log_degree"`` — ``log1p(total_degree)``.
    - ``"log_in_degree"`` — ``log1p(in_degree)``.
    - ``"log_out_degree"`` — ``log1p(out_degree)``.
    - ``"norm_degree"`` — degree / max_degree (0 if no edges).

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        features: Tuple of feature names to include.
        dtype: Output dtype; defaults to ``torch.float32``.

    Returns:
        ``FloatTensor[num_nodes, len(features)]``.
    """
    _validate(edge_index, num_nodes)
    out_dtype = dtype or torch.float32
    out_deg, in_deg = _degrees(edge_index, num_nodes)
    total = (out_deg + in_deg).float()
    out_f = out_deg.float()
    in_f = in_deg.float()
    max_deg = float(total.max().item()) if total.numel() else 0.0

    _MAP = {
        "degree": lambda: total,
        "in_degree": lambda: in_f,
        "out_degree": lambda: out_f,
        "log_degree": lambda: total.log1p(),
        "log_in_degree": lambda: in_f.log1p(),
        "log_out_degree": lambda: out_f.log1p(),
        "norm_degree": lambda: total / max(max_deg, 1.0),
    }

    cols = []
    for feat in features:
        if feat not in _MAP:
            raise ValueError(
                f"Unknown feature {feat!r}; available: {sorted(_MAP)}"
            )
        cols.append(_MAP[feat]())

    if not cols:
        return torch.zeros(num_nodes, 0, dtype=out_dtype, device=edge_index.device)
    return torch.stack(cols, dim=1).to(out_dtype)


def add_structural_features(
    graph,
    features: Tuple[str, ...] = ("log_degree",),
    key: str = "structural",
) -> Any:
    """Append structural features to a TGraphX :class:`~tgraphx.Graph`.

    For graphs with **vector** node features ``[N, D]``, this function
    concatenates the structural features to produce ``[N, D+F]``.

    For graphs with **tensor** node features ``[N, C, H, W]`` or
    ``[N, C, D, H, W]``, the structural features are **not** appended
    to node_features (which would silently destroy the spatial layout).
    Instead, they are stored in ``graph.metadata[key]`` as a tensor.

    Args:
        graph: A :class:`~tgraphx.Graph`.
        features: Feature names for :func:`structural_features`.
        key: Metadata key for the structural tensor in non-vector mode.

    Returns:
        A new :class:`~tgraphx.Graph` with features augmented or metadata set.
    """
    import copy
    from tgraphx.core.graph import Graph

    N = graph.num_nodes
    edge_index = graph.edge_index
    if edge_index is None:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    sf = structural_features(edge_index, N, features=features)
    g = copy.copy(graph)

    # Vector case: safe to concatenate.
    if graph.node_features is not None and graph.node_features.dim() == 2:
        g.node_features = torch.cat(
            [graph.node_features.float(), sf.to(graph.node_features.device)], dim=1,
        )
    else:
        # Tensor or volumetric: store in metadata.
        meta = dict(graph.metadata) if graph.metadata else {}
        meta[key] = sf
        g.metadata = meta

    return g
