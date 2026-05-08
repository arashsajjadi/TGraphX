"""Vanilla gradient saliency.

Produces ``|∂y/∂x|`` for the node feature tensor of a graph.  Works
with vector, 2-D spatial, and 3-D volumetric node features.
"""
from __future__ import annotations

from typing import Any

import torch

from .utils import call_model, select_target_logit


def node_feature_saliency(
    model: torch.nn.Module,
    graph,
    target: Any = 0,
    abs_value: bool = True,
) -> torch.Tensor:
    """Return saliency w.r.t. ``graph.node_features``.

    Args:
        model: A ``nn.Module`` whose forward accepts
            ``(node_features, edge_index)`` (and optionally other kwargs
            that exist on the graph).
        graph: A TGraphX :class:`Graph` (any feature rank).
        target: Class index to backprop from for classification models;
            ignored when output is a scalar.
        abs_value: When ``True`` (default), return ``|∂y/∂x|``;
            otherwise return raw signed gradients.

    Returns:
        Tensor with the same shape as ``graph.node_features``.
    """
    model_was_training = model.training
    model.eval()
    try:
        x = graph.node_features.detach().clone().requires_grad_(True)
        graph_proxy = _wrap_graph(graph, node_features=x)
        logits = call_model(model, graph_proxy)
        scalar = select_target_logit(logits, target)
        grad, = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)
    finally:
        if model_was_training:
            model.train()
    out = grad.detach()
    return out.abs() if abs_value else out


class _GraphProxy:
    """Tiny stand-in that exposes the same attributes a forward path uses."""

    __slots__ = ("node_features", "edge_index", "edge_weight", "edge_features",
                 "node_labels", "edge_labels", "graph_label", "metadata", "batch")

    def __init__(self, **fields) -> None:
        for k in self.__slots__:
            setattr(self, k, fields.get(k))


def _wrap_graph(graph, node_features: torch.Tensor) -> _GraphProxy:
    """Return a lightweight proxy with ``node_features`` swapped in."""
    return _GraphProxy(
        node_features=node_features,
        edge_index=graph.edge_index,
        edge_weight=getattr(graph, "edge_weight", None),
        edge_features=getattr(graph, "edge_features", None),
        node_labels=getattr(graph, "node_labels", None),
        edge_labels=getattr(graph, "edge_labels", None),
        graph_label=getattr(graph, "graph_label", None),
        metadata=getattr(graph, "metadata", None),
        batch=getattr(graph, "batch", None),
    )
