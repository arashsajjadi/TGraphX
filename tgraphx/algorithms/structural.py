"""Structural feature helpers in pure PyTorch.

These utilities compute structural graph properties that are commonly
used as node features or input to GNN layers.

Stability: Beta (see ``docs/api_stability.md``).
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "degree",
    "degree_features",
]


def _validate(
    edge_index: torch.Tensor,
    num_nodes: Optional[int],
    tag: str = "edge_index",
) -> tuple[torch.Tensor, int]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"{tag} must have shape [2, E]; got {tuple(edge_index.shape)}"
        )
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative; got {num_nodes}")
    if edge_index.numel() and edge_index.max().item() >= num_nodes:
        raise ValueError(
            f"{tag} max node id {int(edge_index.max())} >= num_nodes={num_nodes}"
        )
    return edge_index.to(torch.long), int(num_nodes)


def degree(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    mode: str = "out",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Return the degree of each node.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count; inferred from ``edge_index.max() + 1`` when
            ``None``.
        mode: ``"out"`` (default) — count outgoing edges; ``"in"`` —
            count incoming; ``"both"`` — sum of in + out (for undirected
            graphs this equals the count of edges touching each node,
            including both directions).
        dtype: Output dtype; defaults to ``torch.long``.

    Returns:
        ``Tensor[num_nodes]`` of non-negative integers (or ``dtype`` if
        specified).  Isolated nodes have degree 0.

    Notes:
        - Self-loops are counted once in both ``"out"`` and ``"in"``
          modes (they contribute 1 to out-degree and 1 to in-degree).
        - O(E) time and O(N) memory.
    """
    if mode not in ("out", "in", "both"):
        raise ValueError(f"mode must be 'out', 'in', or 'both'; got {mode!r}")
    edge_index, num_nodes = _validate(edge_index, num_nodes)
    device = edge_index.device
    out_dtype = dtype if dtype is not None else torch.long
    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    if edge_index.numel():
        if mode in ("out", "both"):
            deg.scatter_add_(
                0,
                edge_index[0],
                torch.ones(edge_index.size(1), dtype=torch.long, device=device),
            )
        if mode in ("in", "both"):
            deg.scatter_add_(
                0,
                edge_index[1],
                torch.ones(edge_index.size(1), dtype=torch.long, device=device),
            )
    return deg.to(out_dtype)


def degree_features(
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    log_scale: bool = False,
) -> torch.Tensor:
    """Return a ``[num_nodes, 3]`` feature matrix [out_deg, in_deg, total_deg].

    Useful for concatenating structural features to existing node
    features before or between GNN layers.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count; inferred when ``None``.
        log_scale: When ``True``, returns ``log1p(deg)`` to reduce the
            dynamic range for imbalanced graphs.  dtype becomes
            ``float32``.

    Returns:
        ``Tensor[num_nodes, 3]`` — columns: out-degree, in-degree,
        total-degree (as float32 when ``log_scale=True``, long otherwise).
    """
    edge_index, num_nodes = _validate(edge_index, num_nodes)
    out_d = degree(edge_index, num_nodes, mode="out")
    in_d = degree(edge_index, num_nodes, mode="in")
    total = out_d + in_d
    feats = torch.stack([out_d, in_d, total], dim=1)
    if log_scale:
        return feats.to(torch.float32).log1p()
    return feats
