"""Node and graph regression models for TGraphX.

Both classes stack ``LinearMessagePassing`` layers (vector features) and
finish with a linear head that maps to ``out_dim`` real-valued outputs.
For graph-level tasks a scatter readout pools node embeddings per graph.

Note
----
These classes mirror the existing ``NodeClassifier`` / ``GraphClassifier``
pattern but target regression (continuous outputs) and are included as
standalone public classes.  For a flexible, layer-agnostic alternative
see :func:`tgraphx.models.factory.build_model` with
``task="node_regression"`` or ``task="graph_regression"``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.base import LinearMessagePassing


def _graph_readout(
    x: torch.Tensor,
    batch: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Scatter node vectors to graph-level representation."""
    num_graphs = int(batch.max()) + 1
    if pooling == "mean":
        out = x.new_zeros(num_graphs, x.size(1))
        count = x.new_zeros(num_graphs)
        out.index_add_(0, batch, x)
        count.index_add_(0, batch, x.new_ones(x.size(0)))
        return out / count.clamp(min=1).unsqueeze(1)
    if pooling == "sum":
        out = x.new_zeros(num_graphs, x.size(1))
        out.index_add_(0, batch, x)
        return out
    if pooling == "max":
        idx = batch.unsqueeze(1).expand_as(x)
        out = x.new_full((num_graphs, x.size(1)), float("-inf"))
        out.scatter_reduce_(0, idx, x, reduce="amax", include_self=True)
        out = out.masked_fill(torch.isinf(out) & (out < 0), 0.0)
        return out
    raise ValueError(f"pooling must be 'mean', 'sum', or 'max'; got {pooling!r}")


class NodeRegressor(nn.Module):
    """Node-level regression with stacked ``LinearMessagePassing`` layers.

    Args:
        in_shape: Input per-node feature shape, e.g. ``(D,)``.
        hidden_shape: Hidden per-node feature shape, e.g. ``(64,)``.
        out_dim: Number of real-valued outputs per node.
        num_layers: Total GNN layers (>= 1).
        aggr: Aggregation mode (``"sum"`` / ``"mean"`` / ``"max"``).

    Forward::

        out = model(node_features, edge_index)  # [N, out_dim]
    """

    def __init__(
        self,
        in_shape,
        hidden_shape,
        out_dim: int,
        num_layers: int = 2,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        in_shape = tuple(in_shape)
        hidden_shape = tuple(hidden_shape)

        layers: list[nn.Module] = []
        layers.append(LinearMessagePassing(in_shape, hidden_shape, aggr=aggr))
        for _ in range(num_layers - 1):
            layers.append(LinearMessagePassing(hidden_shape, hidden_shape, aggr=aggr))
        self.gnn_layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden_shape[0], out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features=None,
        edge_weight=None,
    ) -> torch.Tensor:
        x = node_features
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_features, edge_weight))
        return self.head(x)


class GraphRegressor(nn.Module):
    """Graph-level regression with stacked ``LinearMessagePassing`` layers.

    Args:
        in_shape: Input per-node feature shape, e.g. ``(D,)``.
        hidden_shape: Hidden per-node feature shape, e.g. ``(64,)``.
        out_dim: Number of real-valued outputs per graph.
        num_layers: Total GNN layers (>= 1).
        aggr: Message aggregation mode (``"sum"`` / ``"mean"`` / ``"max"``).
        pooling: Graph readout mode (``"mean"`` / ``"sum"`` / ``"max"``).

    Forward::

        out = model(node_features, edge_index, batch=batch)  # [G, out_dim]
    """

    def __init__(
        self,
        in_shape,
        hidden_shape,
        out_dim: int,
        num_layers: int = 2,
        aggr: str = "sum",
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        in_shape = tuple(in_shape)
        hidden_shape = tuple(hidden_shape)

        layers: list[nn.Module] = []
        layers.append(LinearMessagePassing(in_shape, hidden_shape, aggr=aggr))
        for _ in range(num_layers - 1):
            layers.append(LinearMessagePassing(hidden_shape, hidden_shape, aggr=aggr))
        self.gnn_layers = nn.ModuleList(layers)
        self.pooling = pooling
        self.head = nn.Linear(hidden_shape[0], out_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features=None,
        edge_weight=None,
        batch=None,
    ) -> torch.Tensor:
        if batch is None:
            raise ValueError(
                "batch ([N] mapping each node to its graph) is required for "
                "GraphRegressor. Pass batch=... to forward()."
            )
        x = node_features
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_features, edge_weight))
        x = _graph_readout(x, batch, self.pooling)
        return self.head(x)


__all__ = ["NodeRegressor", "GraphRegressor"]
