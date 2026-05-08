"""Structural graph transforms.

Every transform here is *non-mutating by default*: it returns a new
:class:`~tgraphx.Graph` whose tensors are either shared (when no
modification is needed) or freshly allocated.  Set ``inplace=True`` on
a per-class basis when supported.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ..core.graph import Graph
from ..core.graph_utils import (
    add_self_loops as _gu_add_self_loops,
    coalesce_edges as _gu_coalesce_edges,
    make_undirected as _gu_make_undirected,
    remove_self_loops as _gu_remove_self_loops,
)


def _shallow_copy(graph: Graph) -> Graph:
    """Return a new :class:`Graph` referencing the same tensors."""
    return Graph(
        node_features=graph.node_features,
        edge_index=graph.edge_index.clone() if graph.edge_index is not None else None,
        edge_weight=graph.edge_weight,
        edge_features=graph.edge_features,
        node_labels=graph.node_labels,
        edge_labels=graph.edge_labels,
        graph_label=graph.graph_label,
        metadata=dict(graph.metadata) if isinstance(graph.metadata, dict) else None,
    )


class AddSelfLoops:
    """Add a self-loop ``i -> i`` for every node that does not already have one."""

    def __init__(self, fill_value: Optional[float] = None) -> None:
        self.fill_value = fill_value

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.edge_index is None:
            N = new.num_nodes
            ei = torch.arange(N, device=new.node_features.device, dtype=torch.long)
            new.edge_index = torch.stack([ei, ei], dim=0)
        else:
            new.edge_index, edge_weight, edge_features = _gu_add_self_loops(
                new.edge_index, num_nodes=new.num_nodes,
                edge_weight=new.edge_weight,
                edge_features=new.edge_features,
                fill_value=self.fill_value if self.fill_value is not None else 1.0,
            )
            new.edge_weight = edge_weight
            new.edge_features = edge_features
        return new


class RemoveSelfLoops:
    """Drop every self-loop edge."""

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_index is None:
            return _shallow_copy(graph)
        new = _shallow_copy(graph)
        ei, ew, ef, el = _gu_remove_self_loops(
            new.edge_index,
            edge_weight=new.edge_weight,
            edge_features=new.edge_features,
            edge_labels=new.edge_labels,
        )
        new.edge_index = ei
        new.edge_weight = ew
        new.edge_features = ef
        new.edge_labels = el
        return new


class ToUndirected:
    """Symmetrise the edge set so every edge has a reverse counterpart.

    ``reduce`` controls how parallel edges are coalesced when the graph
    has ``edge_weight``; defaults to ``"mean"``.
    """

    def __init__(self, reduce: str = "mean") -> None:
        if reduce not in ("mean", "sum", "min", "max", "amax"):
            raise ValueError(
                f"reduce must be one of mean/sum/min/max/amax; got {reduce!r}"
            )
        self.reduce = reduce

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_index is None:
            return _shallow_copy(graph)
        new = _shallow_copy(graph)
        ei, ew, ef = _gu_make_undirected(
            new.edge_index,
            edge_weight=new.edge_weight,
            edge_features=new.edge_features,
            num_nodes=new.num_nodes,
            reduce=self.reduce,
        )
        new.edge_index = ei
        new.edge_weight = ew
        new.edge_features = ef
        # edge_labels cannot survive coalesce unambiguously.
        if new.edge_labels is not None:
            new.edge_labels = None
        return new


class CoalesceEdges:
    """Sort + deduplicate edges; keeps ``edge_weight`` with reduce policy."""

    def __init__(self, reduce: str = "mean") -> None:
        self.reduce = reduce

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_index is None:
            return _shallow_copy(graph)
        new = _shallow_copy(graph)
        ei, ew, ef = _gu_coalesce_edges(
            new.edge_index,
            edge_weight=new.edge_weight,
            edge_features=new.edge_features,
            num_nodes=new.num_nodes,
            reduce=self.reduce,
        )
        new.edge_index = ei
        new.edge_weight = ew
        new.edge_features = ef
        return new


class DropEdges:
    """Randomly drop a fraction of the edges (Bernoulli per edge).

    Edge weights / features / labels for the surviving edges are
    re-aligned correctly.
    """

    def __init__(self, p: float = 0.1, seed: int | None = None) -> None:
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1); got {p}")
        self.p = float(p)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_index is None or graph.num_edges == 0 or self.p == 0.0:
            return _shallow_copy(graph)
        new = _shallow_copy(graph)
        keep = torch.rand(new.num_edges, generator=self._gen) >= self.p
        idx = torch.where(keep)[0].to(new.edge_index.device)
        new.edge_index = new.edge_index.index_select(1, idx)
        if new.edge_weight is not None:
            new.edge_weight = new.edge_weight.index_select(0, idx)
        if new.edge_features is not None:
            new.edge_features = new.edge_features.index_select(0, idx)
        if new.edge_labels is not None:
            new.edge_labels = new.edge_labels.index_select(0, idx)
        return new
