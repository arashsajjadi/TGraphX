"""Small graph pattern counting utilities.

These functions count occurrences of small fixed patterns in a graph.
They are useful for structural feature extraction and pattern recognition.

**This is not full subgraph isomorphism** — only small, specific patterns
(paths, stars, triangles) are supported with exact polynomial-time algorithms.
For general subgraph isomorphism see dedicated libraries.

Stability: Experimental (v0.3.2+).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

__all__ = [
    "path_pattern_count",
    "star_pattern_count",
    "contains_triangle",
    "small_pattern_counts",
]


def _build_adj(edge_index: torch.Tensor, num_nodes: int, directed: bool) -> list:
    adj: list = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    for u, v in zip(src, dst):
        if u != v:
            adj[u].add(v)
            if not directed:
                adj[v].add(u)
    return adj


def path_pattern_count(
    edge_index: torch.Tensor,
    num_nodes: int,
    length: int = 2,
    directed: bool = False,
) -> int:
    """Count the number of directed/undirected paths of given length.

    A path of length 2 is a triple ``(u, v, w)`` where ``(u,v)`` and
    ``(v,w)`` are edges and ``u ≠ w``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        length: Path length (currently supports 2 and 3).
        directed: When ``False`` (default), treat as undirected.

    Returns:
        Non-negative integer.

    Notes:
        Paths are counted as ordered tuples (start, …, end).
        For length > 3 a ``NotImplementedError`` is raised.
    """
    if length not in (2, 3):
        raise NotImplementedError(
            f"path_pattern_count supports length 2 or 3; got {length}. "
            "Longer paths are not yet implemented."
        )
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    adj = _build_adj(edge_index, num_nodes, directed)
    count = 0
    if length == 2:
        for v in range(num_nodes):
            nbrs = list(adj[v])
            d = len(nbrs)
            for i in range(d):
                for j in range(d):
                    if i != j:
                        count += 1
    elif length == 3:
        for v in range(num_nodes):
            for u in adj[v]:
                for w in adj[u]:
                    if w != v:
                        for x in adj[w]:
                            if x != u:
                                count += 1
    return count


def star_pattern_count(
    edge_index: torch.Tensor,
    num_nodes: int,
    center_degree: int = 3,
    directed: bool = False,
) -> int:
    """Count nodes with exactly ``center_degree`` or more neighbours.

    A "star of degree k" is a node with at least k outgoing edges.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        center_degree: Minimum hub degree.
        directed: When ``False``, uses total undirected degree.

    Returns:
        Count of hub nodes.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    adj = _build_adj(edge_index, num_nodes, directed)
    return sum(1 for v in range(num_nodes) if len(adj[v]) >= center_degree)


def contains_triangle(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> bool:
    """Return ``True`` if the graph contains at least one triangle.

    Args:
        edge_index: ``LongTensor[2, E]`` (treated as undirected).
        num_nodes: Node count.

    Returns:
        ``bool``.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    from .motifs import triangle_count
    no_loops = edge_index[:, edge_index[0] != edge_index[1]] if edge_index.numel() else edge_index
    return triangle_count(no_loops, num_nodes, directed=False) > 0


def small_pattern_counts(
    edge_index: torch.Tensor,
    num_nodes: int,
    directed: bool = False,
) -> Dict[str, Any]:
    """Return a dict of small-pattern counts.

    Keys: ``triangles``, ``paths_len2``, ``stars_deg3``, ``stars_deg5``.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        directed: When ``False`` (default), treat as undirected.

    Returns:
        Plain Python dict (JSON-serializable).
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    no_loops = edge_index[:, edge_index[0] != edge_index[1]] if edge_index.numel() else edge_index

    from .motifs import triangle_count as tri_count
    triangles = int(tri_count(no_loops, num_nodes, directed=directed))

    try:
        p2 = path_pattern_count(no_loops, num_nodes, length=2, directed=directed)
    except Exception:  # pragma: no cover
        p2 = -1

    s3 = star_pattern_count(no_loops, num_nodes, center_degree=3, directed=directed)
    s5 = star_pattern_count(no_loops, num_nodes, center_degree=5, directed=directed)

    return {
        "triangles": triangles,
        "paths_len2": p2,
        "stars_deg3": s3,
        "stars_deg5": s5,
    }
