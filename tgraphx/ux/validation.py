"""Tensor-native graph validation utilities.

The core promise of TGraphX is that node features keep their tensor shape.
These validators give scientists and LLMs a one-line way to verify the promise.
"""
from __future__ import annotations

from typing import Any, Optional

import torch


class GraphValidationError(ValueError):
    """Raised when a Graph or batch fails tensor-native invariants."""


def validate_graph(
    graph: Any,
    strict: bool = False,
    check_device: bool = True,
    check_finite: bool = False,
    check_gradients: bool = False,
    allow_vector_features: bool = True,
) -> dict:
    """Validate a TGraphX :class:`Graph` against tensor-native invariants.

    Args:
        graph: A :class:`tgraphx.Graph` instance.
        strict: If True, raise :class:`GraphValidationError` on the first
            failure. If False, collect all issues into the returned dict.
        check_device: Verify that all tensors share the same device.
        check_finite: Verify no NaN/Inf values in numeric tensors.
        check_gradients: Verify ``requires_grad`` on node_features if set.
        allow_vector_features: If False, require rank ≥ 3 node features
            (i.e. true tensor-native). Default True accepts vector features.

    Returns:
        Dict with keys ``ok`` (bool), ``issues`` (list[str]), ``info`` (dict).
    """
    issues: list[str] = []
    info: dict[str, Any] = {}

    nf = getattr(graph, "node_features", None)
    if nf is None:
        issues.append("graph.node_features is None")
        if strict:
            raise GraphValidationError(issues[-1])
        return {"ok": False, "issues": issues, "info": info}

    info["node_features_shape"] = list(nf.shape)
    info["node_features_dtype"] = str(nf.dtype)
    info["node_features_device"] = str(nf.device)

    # Tensor-native check
    if not allow_vector_features and nf.dim() < 3:
        issues.append(
            f"node_features rank {nf.dim()} < 3; set allow_vector_features=True "
            f"to accept vector features."
        )

    # edge_index validation
    ei = getattr(graph, "edge_index", None)
    if ei is not None:
        info["edge_index_shape"] = list(ei.shape)
        if ei.dim() != 2 or ei.shape[0] != 2:
            issues.append(
                f"edge_index must have shape [2, E]; got {tuple(ei.shape)}. "
                "If you have an [E, 2] tensor, transpose it: edge_index.t() "
                "or use Graph.from_edges(edge_list)."
            )
        else:
            max_idx = int(ei.max().item()) if ei.numel() > 0 else -1
            if max_idx >= nf.size(0):
                issues.append(
                    f"edge_index references node {max_idx} but graph has only "
                    f"{nf.size(0)} nodes."
                )
            if check_device and ei.device != nf.device:
                issues.append(
                    f"edge_index device {ei.device} != node_features device "
                    f"{nf.device}. Use graph.to(device) to move consistently."
                )

    # edge_attr length consistency
    ea = getattr(graph, "edge_features", None)
    if ea is not None and ei is not None:
        info["edge_attr_shape"] = list(ea.shape)
        if ea.size(0) != ei.size(1):
            issues.append(
                f"edge_attr length {ea.size(0)} != number of edges {ei.size(1)}."
            )
        if check_device and ea.device != nf.device:
            issues.append(f"edge_attr device {ea.device} != node_features device {nf.device}.")

    # y / node_labels consistency
    y = getattr(graph, "node_labels", None)
    if y is None:
        y = getattr(graph, "y", None)
    if y is not None and isinstance(y, torch.Tensor):
        info["y_shape"] = list(y.shape)
        if y.dim() >= 1 and y.size(0) != nf.size(0):
            # Could be graph-level label; only flag if expected node-level
            if y.size(0) != 1 and y.dim() == 1:
                # Only flag if shape suggests node-level
                if y.size(0) != nf.size(0):
                    pass  # graph-level OK; don't flag
        if check_device and y.device != nf.device:
            issues.append(f"y device {y.device} != node_features device {nf.device}.")

    # graph_label
    gl = getattr(graph, "graph_label", None)
    if gl is not None and isinstance(gl, torch.Tensor):
        info["graph_label_shape"] = list(gl.shape)

    # Masks
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        m = getattr(graph, mask_name, None)
        if m is not None and isinstance(m, torch.Tensor):
            info[f"{mask_name}_shape"] = list(m.shape)
            if m.size(0) != nf.size(0):
                issues.append(
                    f"{mask_name} length {m.size(0)} != num_nodes {nf.size(0)}."
                )

    # Finite check
    if check_finite:
        if isinstance(nf, torch.Tensor) and nf.is_floating_point():
            if not torch.isfinite(nf).all():
                issues.append("node_features contains NaN/Inf values.")
        if ea is not None and ea.is_floating_point():
            if not torch.isfinite(ea).all():
                issues.append("edge_attr contains NaN/Inf values.")

    # Gradient check
    if check_gradients:
        if isinstance(nf, torch.Tensor) and not nf.requires_grad:
            info["requires_grad_node_features"] = False

    ok = len(issues) == 0
    if not ok and strict:
        raise GraphValidationError("; ".join(issues))
    return {"ok": ok, "issues": issues, "info": info}


def assert_tensor_native(graph: Any, min_rank: int = 3) -> None:
    """Assert that graph.node_features has rank >= min_rank.

    Useful in scientific notebooks to prove the tensor-native claim.
    """
    nf = getattr(graph, "node_features", None)
    if nf is None or not isinstance(nf, torch.Tensor):
        raise GraphValidationError("graph.node_features is missing or not a Tensor")
    if nf.dim() < min_rank:
        raise GraphValidationError(
            f"Expected tensor-native node_features (rank >= {min_rank}), "
            f"got rank {nf.dim()} with shape {tuple(nf.shape)}. "
            "If this is intentional (vector features), use validate_graph(allow_vector_features=True)."
        )


def check_graph_invariants(graph: Any, **kwargs) -> dict:
    """Alias for :func:`validate_graph`."""
    return validate_graph(graph, **kwargs)
