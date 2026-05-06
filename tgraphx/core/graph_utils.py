"""Graph manipulation helpers used by :class:`Graph` and friends.

These functions are pure: they take tensors and return tensors. They do not
mutate :class:`Graph` instances. Validation helpers raise ``TypeError`` /
``ValueError`` with descriptive messages so users never see a cryptic
torch shape-mismatch error from deep inside aggregation kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #

def validate_edge_index(
    edge_index: Optional[torch.Tensor],
    num_nodes: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Validate ``edge_index`` shape, dtype, device, and index range."""
    if edge_index is None:
        return None
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(
            f"edge_index must be a torch.Tensor or None, "
            f"got {type(edge_index).__name__}"
        )
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
        )
    if edge_index.dtype != torch.long:
        raise TypeError(
            f"edge_index must have dtype torch.long, got {edge_index.dtype}"
        )
    if device is not None and edge_index.device != device:
        raise ValueError(
            f"edge_index device ({edge_index.device}) must match "
            f"node_features device ({device})"
        )
    if edge_index.numel() > 0:
        lo = int(edge_index.min())
        hi = int(edge_index.max())
        if lo < 0 or hi >= num_nodes:
            raise ValueError(
                f"edge_index contains out-of-range indices for {num_nodes} nodes "
                f"(found min={lo}, max={hi}; valid range [0, {num_nodes - 1}])"
            )
    return edge_index


def validate_edge_weight(
    edge_weight: Optional[torch.Tensor],
    num_edges: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Validate ``edge_weight`` is a 1-D tensor of shape ``[E]``."""
    if edge_weight is None:
        return None
    if not isinstance(edge_weight, torch.Tensor):
        raise TypeError(
            f"edge_weight must be a torch.Tensor or None, "
            f"got {type(edge_weight).__name__}"
        )
    if edge_weight.dim() != 1:
        raise ValueError(
            f"edge_weight must be a 1-D tensor of shape [E], "
            f"got shape {tuple(edge_weight.shape)}"
        )
    if edge_weight.size(0) != num_edges:
        raise ValueError(
            f"edge_weight has {edge_weight.size(0)} entries but "
            f"edge_index has {num_edges} edges"
        )
    if device is not None and edge_weight.device != device:
        raise ValueError(
            f"edge_weight device ({edge_weight.device}) must match "
            f"node_features device ({device})"
        )
    return edge_weight


def validate_edge_features(
    edge_features: Optional[torch.Tensor],
    num_edges: int,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Validate ``edge_features`` is at least 2-D with leading dim ``E``."""
    if edge_features is None:
        return None
    if not isinstance(edge_features, torch.Tensor):
        raise TypeError(
            f"edge_features must be a torch.Tensor or None, "
            f"got {type(edge_features).__name__}"
        )
    if edge_features.dim() < 2:
        raise ValueError(
            f"edge_features must have at least 2 dimensions [E, ...] "
            f"(vector or spatial per-edge feature), "
            f"got shape {tuple(edge_features.shape)}"
        )
    if edge_features.size(0) != num_edges:
        raise ValueError(
            f"edge_features has {edge_features.size(0)} entries but "
            f"edge_index has {num_edges} edges"
        )
    if device is not None and edge_features.device != device:
        raise ValueError(
            f"edge_features device ({edge_features.device}) must match "
            f"node_features device ({device})"
        )
    return edge_features


# --------------------------------------------------------------------------- #
# Topology                                                                     #
# --------------------------------------------------------------------------- #

def is_undirected(
    edge_index: Optional[torch.Tensor],
    num_nodes: Optional[int] = None,
) -> bool:
    """Return True iff every edge ``(u, v)`` has a matching reverse ``(v, u)``.

    The check ignores edge weights and edge features (it is purely structural).
    For empty/None ``edge_index`` the graph is considered undirected.
    """
    if edge_index is None or edge_index.numel() == 0:
        return True
    src = edge_index[0].long()
    dst = edge_index[1].long()
    if num_nodes is None:
        num_nodes = int(torch.max(torch.maximum(src, dst))) + 1
    if num_nodes <= 0:
        return True
    f_key = src * num_nodes + dst
    r_key = dst * num_nodes + src
    f_sorted, _ = torch.sort(f_key)
    r_sorted, _ = torch.sort(r_key)
    return torch.equal(f_sorted, r_sorted)


def add_self_loops(
    edge_index: Optional[torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    num_nodes: int = 0,
    fill_value: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Add a self-loop ``i->i`` for every node that does not already have one.

    Existing self-loops are preserved; their weights/features are kept as-is.
    For nodes that did not have a self-loop, ``fill_value`` is broadcast into
    the new ``edge_weight`` and ``edge_features`` rows.
    """
    if device is None:
        if edge_index is not None:
            device = edge_index.device
        elif edge_weight is not None:
            device = edge_weight.device
        elif edge_features is not None:
            device = edge_features.device
        else:
            device = torch.device("cpu")

    self_idx = torch.arange(num_nodes, device=device, dtype=torch.long)

    if edge_index is not None and edge_index.size(1) > 0:
        existing_self_mask = edge_index[0] == edge_index[1]
        already = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        already[edge_index[0][existing_self_mask]] = True
        new_nodes = self_idx[~already]
    else:
        new_nodes = self_idx

    new_loops = torch.stack([new_nodes, new_nodes], dim=0)  # [2, n_new]

    if edge_index is None:
        new_edge_index = new_loops
    else:
        new_edge_index = torch.cat([edge_index, new_loops], dim=1)

    new_edge_weight = edge_weight
    if edge_weight is not None:
        pad = torch.full(
            (new_nodes.size(0),),
            float(fill_value),
            dtype=edge_weight.dtype,
            device=edge_weight.device,
        )
        new_edge_weight = torch.cat([edge_weight, pad], dim=0)

    new_edge_features = edge_features
    if edge_features is not None:
        pad_shape = (new_nodes.size(0),) + tuple(edge_features.shape[1:])
        pad = torch.full(
            pad_shape,
            float(fill_value),
            dtype=edge_features.dtype,
            device=edge_features.device,
        )
        new_edge_features = torch.cat([edge_features, pad], dim=0)

    return new_edge_index, new_edge_weight, new_edge_features


def remove_self_loops(
    edge_index: Optional[torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    edge_labels: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Remove self-loops and propagate the keep-mask to per-edge tensors."""
    if edge_index is None or edge_index.size(1) == 0:
        return edge_index, edge_weight, edge_features, edge_labels
    mask = edge_index[0] != edge_index[1]
    new_ei = edge_index[:, mask]
    new_w = edge_weight[mask] if edge_weight is not None else None
    new_ef = edge_features[mask] if edge_features is not None else None
    new_el = edge_labels[mask] if edge_labels is not None else None
    return new_ei, new_w, new_ef, new_el


def make_undirected(
    edge_index: Optional[torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "mean",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Symmetrize: append the reverse of every edge then coalesce duplicates.

    For nodes ``(u, v)`` that already have a forward and reverse edge, the
    coalesce step merges the duplicates using ``reduce`` (``"mean"`` or
    ``"sum"``) on ``edge_weight`` and ``edge_features``. Plain edge sets
    without weights/features are simply deduplicated.
    """
    if edge_index is None or edge_index.size(1) == 0:
        return edge_index, edge_weight, edge_features

    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    full_ei = torch.cat([edge_index, rev], dim=1)
    full_w = (
        torch.cat([edge_weight, edge_weight], dim=0)
        if edge_weight is not None
        else None
    )
    full_f = (
        torch.cat([edge_features, edge_features], dim=0)
        if edge_features is not None
        else None
    )
    return coalesce_edges(
        full_ei, full_w, full_f, num_nodes=num_nodes, reduce=reduce
    )


def coalesce_edges(
    edge_index: Optional[torch.Tensor],
    edge_weight: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "mean",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sort edges lexicographically by ``(src, dst)`` and merge duplicates.

    Supported reductions: ``"mean"``, ``"sum"``. ``"mean"`` divides by the
    duplicate count per group. The reduction applies to ``edge_weight`` and
    to ``edge_features`` along the leading edge dimension.
    """
    if reduce not in ("mean", "sum"):
        raise ValueError(
            f"coalesce_edges: unsupported reduce={reduce!r}; use 'mean' or 'sum'"
        )
    if edge_index is None or edge_index.size(1) == 0:
        return edge_index, edge_weight, edge_features

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    src = edge_index[0].long()
    dst = edge_index[1].long()
    keys = src * int(num_nodes) + dst
    sorted_keys, perm = torch.sort(keys, stable=True)
    sorted_ei = edge_index[:, perm]

    if sorted_keys.numel() == 1:
        unique_mask = torch.ones(1, dtype=torch.bool, device=sorted_keys.device)
    else:
        unique_mask = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=sorted_keys.device),
                sorted_keys[1:] != sorted_keys[:-1],
            ]
        )

    group_id = unique_mask.long().cumsum(0) - 1
    n_groups = int(group_id[-1]) + 1
    new_ei = sorted_ei[:, unique_mask]

    new_w = None
    if edge_weight is not None:
        sorted_w = edge_weight[perm]
        if not sorted_w.is_floating_point():
            sorted_w = sorted_w.to(torch.float32)
        out = torch.zeros(n_groups, dtype=sorted_w.dtype, device=sorted_w.device)
        out.scatter_add_(0, group_id, sorted_w)
        if reduce == "mean":
            counts = torch.zeros(n_groups, dtype=sorted_w.dtype, device=sorted_w.device)
            counts.scatter_add_(
                0, group_id, torch.ones_like(sorted_w, dtype=sorted_w.dtype)
            )
            out = out / counts.clamp_min(1)
        new_w = out

    new_f = None
    if edge_features is not None:
        sorted_f = edge_features[perm]
        feat_shape = sorted_f.shape[1:]
        flat = sorted_f.reshape(sorted_f.size(0), -1)
        if not flat.is_floating_point():
            flat = flat.to(torch.float32)
        F = flat.size(1)
        idx = group_id.unsqueeze(1).expand(-1, F)
        out = torch.zeros(n_groups, F, dtype=flat.dtype, device=flat.device)
        out.scatter_add_(0, idx, flat)
        if reduce == "mean":
            counts = torch.zeros(n_groups, dtype=flat.dtype, device=flat.device)
            counts.scatter_add_(
                0,
                group_id,
                torch.ones(group_id.size(0), dtype=flat.dtype, device=flat.device),
            )
            out = out / counts.clamp_min(1).unsqueeze(1)
        new_f = out.reshape(n_groups, *feat_shape)

    return new_ei, new_w, new_f


__all__ = [
    "validate_edge_index",
    "validate_edge_weight",
    "validate_edge_features",
    "is_undirected",
    "add_self_loops",
    "remove_self_loops",
    "make_undirected",
    "coalesce_edges",
]
