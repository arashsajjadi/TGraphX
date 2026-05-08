"""Edge-level attribution.

Two methods are provided:

* :func:`edge_gradient_attribution` — gradient of the target logit
  w.r.t. ``graph.edge_weight``.  Requires the model to actually use
  edge_weight; otherwise the gradient is zero (and a warning is
  emitted).

* :func:`edge_perturbation_attribution` — drop each edge in turn and
  measure the change in the target logit.  Slow but model-agnostic.
"""
from __future__ import annotations

import warnings
from typing import Any

import torch

from .saliency import _wrap_graph
from .utils import call_model, select_target_logit


def edge_gradient_attribution(
    model: torch.nn.Module,
    graph,
    target: Any = 0,
    abs_value: bool = True,
) -> torch.Tensor:
    """Return ``|∂y/∂edge_weight|``.

    If ``graph.edge_weight`` is ``None``, a tensor of ones is used and
    its gradient is computed (this captures sensitivity to edge
    presence under additive scaling).
    """
    if graph.edge_index is None or graph.num_edges == 0:
        return torch.zeros(0, device=graph.node_features.device,
                           dtype=graph.node_features.dtype)

    base_w = graph.edge_weight
    if base_w is None:
        base_w = torch.ones(
            graph.num_edges,
            device=graph.node_features.device,
            dtype=graph.node_features.dtype,
        )
    w = base_w.detach().clone().requires_grad_(True)

    model_was_training = model.training
    model.eval()
    try:
        proxy = _wrap_graph(graph, node_features=graph.node_features)
        proxy.edge_weight = w
        logits = call_model(model, proxy)
        scalar = select_target_logit(logits, target)
        grads = torch.autograd.grad(scalar, w, retain_graph=False, create_graph=False,
                                    allow_unused=True)
        grad = grads[0]
        if grad is None:
            warnings.warn(
                "edge_gradient_attribution: model does not depend on edge_weight; "
                "returning zeros.  Use edge_perturbation_attribution for a "
                "model-agnostic alternative.",
                stacklevel=2,
            )
            grad = torch.zeros_like(w)
    finally:
        if model_was_training:
            model.train()

    out = grad.detach()
    return out.abs() if abs_value else out


def edge_perturbation_attribution(
    model: torch.nn.Module,
    graph,
    target: Any = 0,
    max_edges: int = 256,
) -> torch.Tensor:
    """Drop each edge and measure ``Δlogit`` (positive ⇒ the edge supports the prediction).

    The returned tensor has shape ``[num_edges]``.  For graphs with more
    than ``max_edges`` edges, only the first ``max_edges`` are scored
    (this keeps the helper CI-safe; users can call repeatedly with
    different slices for large graphs).
    """
    if graph.edge_index is None or graph.num_edges == 0:
        return torch.zeros(0, device=graph.node_features.device,
                           dtype=graph.node_features.dtype)

    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            base = call_model(model, graph)
            base_scalar = float(select_target_logit(base, target).item())

        E = min(int(max_edges), graph.num_edges)
        scores = torch.zeros(E, device=graph.node_features.device,
                             dtype=graph.node_features.dtype)
        with torch.no_grad():
            for i in range(E):
                mask = torch.ones(graph.num_edges, dtype=torch.bool,
                                  device=graph.edge_index.device)
                mask[i] = False
                ei = graph.edge_index[:, mask]
                ew = graph.edge_weight[mask] if graph.edge_weight is not None else None
                ef = graph.edge_features[mask] if graph.edge_features is not None else None
                proxy = _wrap_graph(graph, node_features=graph.node_features)
                proxy.edge_index = ei
                proxy.edge_weight = ew
                proxy.edge_features = ef
                out = call_model(model, proxy)
                drop_scalar = float(select_target_logit(out, target).item())
                scores[i] = base_scalar - drop_scalar
    finally:
        if model_was_training:
            model.train()

    return scores
