"""Connected-component algorithms in pure PyTorch.

The implementations use iterative label propagation:

    label[v] ← min(label[v], min over edges (u, v): label[u])

This converges in O(diameter) iterations and works on CPU and GPU
without any NetworkX or SciPy dependency.  Memory is O(N + E).  For
graphs with hundreds of thousands of nodes this is fast; for billion-edge
graphs use a dedicated graph-analytics library.
"""
from __future__ import annotations

from typing import List, Optional

import torch

__all__ = [
    "connected_components",
    "weakly_connected_components",
    "is_connected",
    "number_connected_components",
]


def _normalize_inputs(
    edge_index: torch.Tensor,
    num_nodes: Optional[int],
) -> tuple[torch.Tensor, int]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}"
        )
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
    ):
        raise ValueError(
            f"edge_index out of range for num_nodes={num_nodes}: "
            f"min={int(edge_index.min())}, max={int(edge_index.max())}"
        )
    return edge_index.to(torch.long), int(num_nodes)


def _undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Return ``edge_index ∪ edge_index.flip(0)`` (no deduplication needed
    — duplicates do not change the label-propagation fixed point)."""
    if edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _label_propagate(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iters: Optional[int] = None,
) -> torch.Tensor:
    """Iterative ``min``-label propagation; returns labels in [0, K)."""
    device = edge_index.device
    if num_nodes == 0:
        return torch.zeros((0,), dtype=torch.long, device=device)

    labels = torch.arange(num_nodes, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        return _compact_labels(labels)

    src = edge_index[0]
    dst = edge_index[1]
    iter_cap = max_iters if max_iters is not None else max(2 * num_nodes, 4)

    for _ in range(iter_cap):
        # Push: each destination takes min(its label, source label).
        new_labels = labels.clone()
        new_labels.scatter_reduce_(
            0, dst, labels[src], reduce="amin", include_self=True,
        )
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    return _compact_labels(labels)


def _compact_labels(labels: torch.Tensor) -> torch.Tensor:
    """Map distinct labels in ``labels`` to contiguous integers [0, K).

    The component containing the smallest original label gets id 0,
    the next one 1, and so on (deterministic).
    """
    if labels.numel() == 0:
        return labels
    unique, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    return inverse


# ── Public API ───────────────────────────────────────────────────────────────


def connected_components(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Return per-node component labels assuming an **undirected** graph.

    Args:
        edge_index: ``LongTensor[2, E]``.  Treated as undirected: an edge
            ``(u, v)`` connects ``u`` and ``v`` even if its reverse is
            absent from ``edge_index``.
        num_nodes: Optional node count.  When ``None``, inferred from
            ``edge_index.max() + 1``; isolated nodes with id higher than
            the maximum endpoint will not appear in the labelling.  Pass
            ``num_nodes`` explicitly to include all nodes.

    Returns:
        ``LongTensor[num_nodes]`` of component labels in ``[0, K)`` where
        ``K`` is the number of components.  Component ids are
        deterministic: the component containing the smallest node id is
        labelled 0.

    Notes:
        For directed graphs use :func:`weakly_connected_components`
        explicitly; the implementation is identical, but the function
        name documents intent.
    """
    edge_index, num_nodes = _normalize_inputs(edge_index, num_nodes)
    return _label_propagate(_undirected(edge_index), num_nodes)


def weakly_connected_components(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    """Return weakly-connected-component labels for a **directed** graph.

    Two nodes are in the same weakly-connected component if they are
    connected when edge directions are ignored.

    Args:
        edge_index: ``LongTensor[2, E]``; direction is ignored.
        num_nodes: Optional node count.  When ``None``, inferred.

    Returns:
        ``LongTensor[num_nodes]`` of component labels in ``[0, K)``.
    """
    edge_index, num_nodes = _normalize_inputs(edge_index, num_nodes)
    return _label_propagate(_undirected(edge_index), num_nodes)


def is_connected(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> bool:
    """Return ``True`` iff the (undirected) graph has exactly one
    connected component.

    By convention the empty graph (``num_nodes == 0``) returns ``False``;
    a single isolated node returns ``True``.
    """
    edge_index, num_nodes = _normalize_inputs(edge_index, num_nodes)
    if num_nodes == 0:
        return False
    if num_nodes == 1:
        return True
    labels = _label_propagate(_undirected(edge_index), num_nodes)
    return int(labels.max().item()) == 0


def number_connected_components(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> int:
    """Return the number of connected components in the (undirected) graph."""
    edge_index, num_nodes = _normalize_inputs(edge_index, num_nodes)
    if num_nodes == 0:
        return 0
    labels = _label_propagate(_undirected(edge_index), num_nodes)
    return int(labels.max().item()) + 1
