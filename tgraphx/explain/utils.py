"""Internal helpers used by the explainability suite."""
from __future__ import annotations

from typing import Any, Optional

import torch


def select_target_logit(logits: torch.Tensor, target: Any) -> torch.Tensor:
    """Pick the scalar logit corresponding to ``target`` for backprop.

    * If ``logits`` is ``[C]`` (single-graph classification): returns
      ``logits[target]``.
    * If ``logits`` is ``[N, C]`` (batched / node classification):
      returns ``logits[:, target].sum()``.
    * If ``logits`` is ``[1]`` or ``[]``: returns the single value.
    """
    if logits.dim() == 0:
        return logits
    if logits.dim() == 1:
        if logits.numel() == 1:
            return logits.squeeze()
        return logits[int(target)]
    if logits.dim() == 2:
        return logits[:, int(target)].sum()
    raise ValueError(
        f"select_target_logit: unsupported logits shape {tuple(logits.shape)}; "
        f"only 0-D / 1-D / 2-D logits are supported."
    )


def call_model(model, graph, **extra) -> torch.Tensor:
    """Call ``model`` on a Graph, picking the right argument signature.

    The helper auto-supplies ``batch`` for graph-level models that need
    it: when the model expects per-graph readout but the input is a
    single Graph, we synthesise a zero-vector ``batch`` (every node
    belongs to graph 0).
    """
    kwargs: dict = {}
    if getattr(graph, "edge_features", None) is not None:
        kwargs["edge_features"] = graph.edge_features
    if getattr(graph, "edge_weight", None) is not None:
        kwargs["edge_weight"] = graph.edge_weight
    # Always supply batch when the graph proxy already carries one.
    if getattr(graph, "batch", None) is not None:
        kwargs["batch"] = graph.batch
    else:
        # Best-effort: synthesise a zero batch for graph-level models.
        n = graph.node_features.size(0)
        kwargs["batch"] = torch.zeros(
            n, dtype=torch.long, device=graph.node_features.device,
        )
    kwargs.update(extra)
    try:
        return model(graph.node_features, graph.edge_index, **kwargs)
    except TypeError:
        # Model doesn't accept batch / edge_features; retry without them.
        kwargs.pop("batch", None)
        kwargs.pop("edge_features", None)
        kwargs.pop("edge_weight", None)
        return model(graph.node_features, graph.edge_index, **kwargs)
