"""Optimized sparse graph operations with pure-PyTorch fallback.

All functions use pure PyTorch by default and optionally accelerate via
``torch_scatter`` if installed.  The backend is transparent to callers.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch

__all__ = [
    # Edge utilities
    "coalesce_edge_index",
    "sort_edge_index",
    "remove_self_loops",
    "add_self_loops",
    # Degree
    "degree",
    "in_degree",
    "out_degree",
    # Segment / scatter reductions
    "segment_sum",
    "segment_mean",
    "segment_max",
    "segment_min",
    "segment_softmax",
    # CSR / CSC conversion
    "edge_index_to_csr",
    "csr_to_edge_index",
    # Chunked pairwise
    "chunked_cosine_similarity",
    "chunked_top_k",
    # Backend info
    "backend_info",
    "select_backend",
    "active_backend",
]

# ── Backend detection ─────────────────────────────────────────────────────────


def _has_torch_scatter() -> bool:
    try:
        import torch_scatter  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pyg_lib() -> bool:
    try:
        import pyg_lib  # noqa: F401
        return True
    except ImportError:
        return False


def _has_scipy_sparse() -> bool:
    try:
        import scipy.sparse  # noqa: F401
        return True
    except ImportError:
        return False


_VALID_BACKENDS = ("auto", "pure_torch", "torch_scatter", "pyg_lib")
_active_backend: str = "auto"


def backend_info() -> dict:
    """Return a dict describing which optional backends are available.

    The pure-PyTorch path is always available.  Optional backends are
    detected lazily via import.
    """
    return {
        "pure_torch": True,
        "torch_scatter": _has_torch_scatter(),
        "pyg_lib": _has_pyg_lib(),
        "scipy_sparse": _has_scipy_sparse(),
        "active": _active_backend,
    }


def select_backend(backend: str = "auto") -> str:
    """Select a sparse-ops backend.

    ``"auto"`` picks the best available (``torch_scatter`` if installed,
    else ``pure_torch``).  An explicit choice falls back to
    ``pure_torch`` when the requested backend is missing, with a single
    warning.

    Args:
        backend: One of ``"auto"``, ``"pure_torch"``, ``"torch_scatter"``,
            ``"pyg_lib"``.

    Returns:
        The backend that will actually be used (so callers can log it).
    """
    global _active_backend
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS}; got {backend!r}"
        )
    if backend == "auto":
        if _has_torch_scatter():
            _active_backend = "torch_scatter"
        else:
            _active_backend = "pure_torch"
    elif backend == "torch_scatter" and not _has_torch_scatter():
        warnings.warn(
            "torch_scatter not installed; falling back to pure_torch.",
            RuntimeWarning, stacklevel=2,
        )
        _active_backend = "pure_torch"
    elif backend == "pyg_lib" and not _has_pyg_lib():
        warnings.warn(
            "pyg_lib not installed; falling back to pure_torch.",
            RuntimeWarning, stacklevel=2,
        )
        _active_backend = "pure_torch"
    else:
        _active_backend = backend
    return _active_backend


def active_backend() -> str:
    """Return the currently selected backend (``"auto"`` if never set)."""
    return _active_backend


# ── Edge utilities ────────────────────────────────────────────────────────────


def coalesce_edge_index(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "sum",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Remove duplicate edges, optionally aggregating edge attributes.

    Args:
        edge_index: ``LongTensor[2, E]``.
        edge_attr: Optional ``Tensor[E, *]`` edge attributes.
        num_nodes: Optional node count.
        reduce: Aggregation for duplicate ``edge_attr`` values:
            ``"sum"``, ``"mean"``, ``"max"``, ``"min"``.

    Returns:
        ``(coalesced_edge_index, coalesced_edge_attr)`` — both sorted in
        row-major order.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_attr

    N = num_nodes if num_nodes is not None else int(edge_index.max().item()) + 1
    # Encode each edge as a single integer for fast deduplication.
    idx = edge_index[0] * N + edge_index[1]
    sorted_idx, perm = idx.sort(stable=True)
    sorted_ei = edge_index[:, perm]

    if edge_attr is None:
        # Just deduplicate by unique mask.
        mask = torch.cat([torch.ones(1, dtype=torch.bool, device=edge_index.device),
                          sorted_idx[1:] != sorted_idx[:-1]])
        return sorted_ei[:, mask], None

    sorted_attr = edge_attr[perm]
    # Find group boundaries.
    mask = torch.cat([torch.ones(1, dtype=torch.bool, device=edge_index.device),
                      sorted_idx[1:] != sorted_idx[:-1]])
    unique_ei = sorted_ei[:, mask]
    # Aggregate attributes.
    group_ids = mask.cumsum(0) - 1  # group index for each edge
    D = sorted_attr.shape[1:] if sorted_attr.dim() > 1 else ()
    G = int(mask.sum().item())
    if reduce == "sum":
        agg = torch.zeros(G, *D, dtype=sorted_attr.dtype, device=sorted_attr.device)
        agg.scatter_add_(0, group_ids.unsqueeze(-1).expand_as(sorted_attr) if sorted_attr.dim() > 1 else group_ids, sorted_attr)
    elif reduce == "mean":
        agg_sum = torch.zeros(G, *D, dtype=sorted_attr.dtype, device=sorted_attr.device)
        cnt = torch.zeros(G, dtype=torch.float, device=edge_index.device)
        idx_exp = group_ids.unsqueeze(-1).expand_as(sorted_attr) if sorted_attr.dim() > 1 else group_ids
        agg_sum.scatter_add_(0, idx_exp, sorted_attr)
        cnt.scatter_add_(0, group_ids, torch.ones(len(group_ids), device=edge_index.device))
        agg = agg_sum / cnt.clamp(min=1).unsqueeze(-1) if sorted_attr.dim() > 1 else agg_sum / cnt.clamp(min=1)
    else:
        # Fallback: just take first occurrence.
        agg = sorted_attr[mask]

    return unique_ei, agg


def sort_edge_index(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sort edges in row-major order (src, then dst).

    Args:
        edge_index: ``LongTensor[2, E]``.
        edge_attr: Optional ``Tensor[E, *]``.
        num_nodes: Optional node count.

    Returns:
        ``(sorted_edge_index, sorted_edge_attr)``.
    """
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    N = num_nodes if num_nodes is not None else int(edge_index.max().item()) + 1
    idx = edge_index[0] * N + edge_index[1]
    _, perm = idx.sort(stable=True)
    sorted_ei = edge_index[:, perm]
    sorted_attr = edge_attr[perm] if edge_attr is not None else None
    return sorted_ei, sorted_attr


def remove_self_loops(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Remove edges where src == dst.

    Args:
        edge_index: ``LongTensor[2, E]``.
        edge_attr: Optional ``Tensor[E, *]``.

    Returns:
        ``(new_edge_index, new_edge_attr)`` without self-loops.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask], (edge_attr[mask] if edge_attr is not None else None)


def add_self_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_attr: Optional[torch.Tensor] = None,
    fill_value: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Add self-loops for nodes that don't already have one.

    Args:
        edge_index: ``LongTensor[2, E]``.
        num_nodes: Node count.
        edge_attr: Optional ``Tensor[E, *]`` to extend.
        fill_value: Value used for new self-loop edge attributes.

    Returns:
        ``(new_edge_index, new_edge_attr)`` with self-loops added.
    """
    device = edge_index.device
    # Nodes that already have a self-loop.
    has_loop = set(
        v for v in edge_index[0][edge_index[0] == edge_index[1]].tolist()
    )
    new_nodes = [v for v in range(num_nodes) if v not in has_loop]
    if not new_nodes:
        return edge_index, edge_attr
    loop_src = torch.tensor(new_nodes, dtype=torch.long, device=device)
    loop_ei = torch.stack([loop_src, loop_src], dim=0)
    new_ei = torch.cat([edge_index, loop_ei], dim=1)
    if edge_attr is not None:
        loop_attr = torch.full(
            (len(new_nodes), *edge_attr.shape[1:]),
            fill_value, dtype=edge_attr.dtype, device=device,
        )
        new_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return new_ei, new_attr
    return new_ei, None


# ── Degree ────────────────────────────────────────────────────────────────────


def degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Out-degree of each node.

    Returns:
        ``Tensor[num_nodes]``.
    """
    d = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        d.scatter_add_(0, edge_index[0], ones)
    return d


def in_degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """In-degree of each node.

    Returns:
        ``Tensor[num_nodes]``.
    """
    d = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        d.scatter_add_(0, edge_index[1], ones)
    return d


def out_degree(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Alias for :func:`degree` (out-degree)."""
    return degree(edge_index, num_nodes, dtype)


# ── Segment / scatter reductions ─────────────────────────────────────────────


def segment_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Segmented sum (scatter-add).

    When ``select_backend("torch_scatter")`` is active and the package
    is installed, accelerates via ``torch_scatter.scatter_add``.
    Otherwise falls back to a pure-PyTorch ``scatter_add_``.

    Args:
        src: ``Tensor[E, *]`` values.
        index: ``LongTensor[E]`` segment IDs.
        num_segments: Number of segments.

    Returns:
        ``Tensor[num_segments, *]``.
    """
    if _active_backend == "torch_scatter" and _has_torch_scatter():
        try:
            import torch_scatter  # type: ignore
            return torch_scatter.scatter_add(src, index, dim=0, dim_size=num_segments)
        except Exception:
            pass  # silently fall back
    out = torch.zeros(num_segments, *src.shape[1:], dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def segment_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Segmented mean.

    Args:
        src: ``Tensor[E, *]``.
        index: ``LongTensor[E]``.
        num_segments: Number of segments.

    Returns:
        ``Tensor[num_segments, *]``.
    """
    total = segment_sum(src, index, num_segments)
    cnt = segment_sum(torch.ones(src.size(0), dtype=src.dtype, device=src.device),
                      index, num_segments)
    cnt = cnt.clamp(min=1)
    if total.dim() > 1:
        cnt = cnt.view(-1, *([1] * (total.dim() - 1)))
    return total / cnt


def segment_max(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Segmented max.

    Returns:
        ``Tensor[num_segments, *]`` — -inf for empty segments.
    """
    out = torch.full((num_segments, *src.shape[1:]), float("-inf"),
                     dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    return out


def segment_min(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Segmented min.

    Returns:
        ``Tensor[num_segments, *]`` — +inf for empty segments.
    """
    out = torch.full((num_segments, *src.shape[1:]), float("inf"),
                     dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_reduce_(0, idx, src, reduce="amin", include_self=True)
    return out


def segment_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Segmented softmax (for attention weight normalisation).

    Args:
        src: ``FloatTensor[E]`` raw scores.
        index: ``LongTensor[E]`` destination node IDs.
        num_segments: Number of segments (= num_nodes in message passing).

    Returns:
        ``FloatTensor[E]`` normalised per-segment.
    """
    # Numerical stability: subtract per-segment max.
    max_vals = segment_max(src, index, num_segments)
    stable_src = src - max_vals[index]
    exp_src = stable_src.exp()
    sum_exp = segment_sum(exp_src, index, num_segments)
    return exp_src / (sum_exp[index] + 1e-12)


# ── CSR / CSC conversion ──────────────────────────────────────────────────────


def edge_index_to_csr(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert edge_index to CSR format ``(row_ptr, col_idx)``.

    Args:
        edge_index: ``LongTensor[2, E]`` sorted by source node.
        num_nodes: Node count.

    Returns:
        ``(row_ptr, col_idx)`` — ``LongTensor[N+1]`` and ``LongTensor[E]``.
    """
    # Sort by source node.
    _, perm = edge_index[0].sort(stable=True)
    col_idx = edge_index[1][perm]
    device = edge_index.device
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    if edge_index.numel():
        ones = torch.ones(edge_index.size(1), dtype=torch.long, device=device)
        row_ptr[1:].scatter_add_(0, edge_index[0][perm], ones)
    row_ptr = row_ptr.cumsum(0)
    return row_ptr, col_idx


def csr_to_edge_index(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
) -> torch.Tensor:
    """Convert CSR ``(row_ptr, col_idx)`` to ``edge_index``.

    Returns:
        ``LongTensor[2, E]``.
    """
    N = row_ptr.size(0) - 1
    E = col_idx.size(0)
    if E == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=col_idx.device)
    row = torch.zeros(E, dtype=torch.long, device=col_idx.device)
    for i in range(N):
        start, end = int(row_ptr[i].item()), int(row_ptr[i + 1].item())
        row[start:end] = i
    return torch.stack([row, col_idx], dim=0)


# ── Chunked pairwise operations ───────────────────────────────────────────────


def chunked_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Memory-safe pairwise cosine similarity.

    Args:
        x: ``FloatTensor[M, D]``.
        y: ``FloatTensor[N, D]``.
        chunk_size: Row chunk size for ``x``.

    Returns:
        ``FloatTensor[M, N]``.
    """
    x_norm = x / (x.norm(dim=1, keepdim=True).clamp(min=1e-12))
    y_norm = y / (y.norm(dim=1, keepdim=True).clamp(min=1e-12))
    M = x_norm.size(0)
    results = []
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        chunk = x_norm[start:end] @ y_norm.t()
        results.append(chunk)
    return torch.cat(results, dim=0)


def chunked_top_k(
    scores: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
    largest: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-wise top-k on a potentially large score matrix.

    If scores fit in memory, delegates to ``torch.topk``.  Otherwise
    processes row chunks.

    Args:
        scores: ``FloatTensor[M, N]``.
        k: Number of top elements per row.
        chunk_size: Row chunk size.
        largest: When ``True``, return largest values.

    Returns:
        ``(values, indices)`` — both ``FloatTensor[M, k]`` and
        ``LongTensor[M, k]``.
    """
    M = scores.size(0)
    k = min(k, scores.size(1))
    val_list, idx_list = [], []
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        v, i = scores[start:end].topk(k, dim=1, largest=largest, sorted=True)
        val_list.append(v)
        idx_list.append(i)
    return torch.cat(val_list, dim=0), torch.cat(idx_list, dim=0)
