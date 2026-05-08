"""Experimental HeteroGraphBatch — disjoint batching of HeteroGraph objects.

.. experimental::
    🧪 The API is experimental.  Common patterns (offsetting edge indices
    per node type, preserving edge weights/features and node/graph labels)
    are stable; advanced patterns (mixed feature shapes per type) may be
    refined in v0.2.6+.

Layout
------
For a batch of B :class:`HeteroGraph` objects:

* ``node_stores[t]`` is the concatenation along dim 0 of every graph's
  ``node_features[t]``.  Shape: ``[sum_b N_b^t, *feature]``.
* ``edge_stores[(s, r, d)]`` is the concatenation along dim 1 of every
  graph's edge index for that relation, with **per-graph offsets** added
  to source row ``[0]`` (offset by ``cum_N_b^s``) and destination row
  ``[1]`` (offset by ``cum_N_b^d``).  Shape: ``[2, sum_b E_b^{s,r,d}]``.
* ``batch_dict[t]`` is a 1-D LongTensor mapping each node of type ``t``
  to its source-graph index in ``[0, B)``.  Shape: ``[sum_b N_b^t]``.
* Per-relation ``edge_weight`` / ``edge_features`` / per-type
  ``node_labels`` are concatenated when **all** graphs provide them.
  If any graph has the relation but lacks the attribute, an explicit
  ``ValueError`` is raised — silent dropping is never allowed.

Example
-------
.. code-block:: python

    from tgraphx.core.hetero_graph import HeteroGraph
    from tgraphx.core.hetero_batch import HeteroGraphBatch

    g1 = HeteroGraph(...)
    g2 = HeteroGraph(...)
    batch = HeteroGraphBatch([g1, g2])
    print(batch.num_nodes_dict)
    print(batch.batch_dict["paper"].shape)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .hetero_graph import EdgeType, HeteroGraph, NodeType

__all__ = ["HeteroGraphBatch"]


class HeteroGraphBatch:
    """🧪 Experimental: disjoint batch of :class:`HeteroGraph` objects."""

    def __init__(self, graphs: List[HeteroGraph]) -> None:
        if not isinstance(graphs, list) or len(graphs) == 0:
            raise ValueError(
                "HeteroGraphBatch requires a non-empty list of HeteroGraph instances."
            )
        for i, g in enumerate(graphs):
            if not isinstance(g, HeteroGraph):
                raise TypeError(
                    f"graphs[{i}] is not a HeteroGraph instance; got {type(g)}"
                )

        self._graphs = list(graphs)
        # Use the first graph's node and edge type schema as the reference.
        ref_node_types = set(graphs[0].node_types)
        ref_edge_types = set(graphs[0].edge_types)
        for i, g in enumerate(graphs[1:], start=1):
            if set(g.node_types) != ref_node_types:
                raise ValueError(
                    f"graphs[{i}] has node types {sorted(g.node_types)} "
                    f"but graphs[0] has {sorted(ref_node_types)}.  "
                    f"All graphs in a batch must share the same node-type schema."
                )
            if set(g.edge_types) != ref_edge_types:
                raise ValueError(
                    f"graphs[{i}] has edge types {sorted(g.edge_types)} "
                    f"but graphs[0] has {sorted(ref_edge_types)}.  "
                    f"All graphs in a batch must share the same edge-type schema."
                )

        # Stable type ordering (insertion order from the first graph).
        self._node_types: List[NodeType] = list(graphs[0].node_types)
        self._edge_types: List[EdgeType] = list(graphs[0].edge_types)

        # ── Concatenate node features per type and build batch vectors ─────────
        node_stores: Dict[NodeType, torch.Tensor] = {}
        batch_dict: Dict[NodeType, torch.Tensor] = {}
        cum_offsets: Dict[NodeType, List[int]] = {t: [0] for t in self._node_types}

        for ntype in self._node_types:
            feats = [g.node_features(ntype) for g in graphs]
            # Validate shape compatibility (all but dim 0 must match).
            ref_shape = feats[0].shape[1:]
            for i, f in enumerate(feats[1:], start=1):
                if f.shape[1:] != ref_shape:
                    raise ValueError(
                        f"Node-type {ntype!r}: graph {i} feature shape "
                        f"{tuple(f.shape)} does not match graph 0 shape "
                        f"{tuple(feats[0].shape)} (excluding batch dim)."
                    )
            node_stores[ntype] = torch.cat(feats, dim=0)
            batch_vec_parts = []
            for b, f in enumerate(feats):
                cum_offsets[ntype].append(cum_offsets[ntype][-1] + f.size(0))
                batch_vec_parts.append(
                    torch.full((f.size(0),), b, dtype=torch.long,
                               device=f.device)
                )
            batch_dict[ntype] = torch.cat(batch_vec_parts, dim=0)

        # ── Concatenate edge indices with per-type offsets ─────────────────────
        edge_stores: Dict[EdgeType, torch.Tensor] = {}
        edge_weight_stores: Dict[EdgeType, torch.Tensor] = {}
        edge_feature_stores: Dict[EdgeType, torch.Tensor] = {}

        for etype in self._edge_types:
            src_type, _, dst_type = etype
            ei_parts = []
            for b, g in enumerate(graphs):
                ei = g.edge_index(etype)
                src_off = cum_offsets[src_type][b]
                dst_off = cum_offsets[dst_type][b]
                offset = ei.new_tensor([[src_off], [dst_off]])
                ei_parts.append(ei + offset)
            edge_stores[etype] = torch.cat(ei_parts, dim=1)

            # Edge weights: all-or-none per relation across graphs.
            present = [g.has_edge_weight(etype) for g in graphs]
            if all(present):
                ews = [g.edge_weight(etype) for g in graphs]
                edge_weight_stores[etype] = torch.cat(ews, dim=0)
            elif any(present):
                raise ValueError(
                    f"Edge type {etype!r}: edge_weight is provided in some "
                    f"graphs but not all.  Either provide it everywhere or "
                    f"nowhere — silent dropping is never allowed."
                )

            present = [g.has_edge_features(etype) for g in graphs]
            if all(present):
                efs = [g.edge_features(etype) for g in graphs]
                ref_shape = efs[0].shape[1:]
                for i, ef in enumerate(efs[1:], start=1):
                    if ef.shape[1:] != ref_shape:
                        raise ValueError(
                            f"Edge type {etype!r}: graph {i} edge_features "
                            f"shape {tuple(ef.shape)} does not match graph 0 "
                            f"shape {tuple(efs[0].shape)}."
                        )
                edge_feature_stores[etype] = torch.cat(efs, dim=0)
            elif any(present):
                raise ValueError(
                    f"Edge type {etype!r}: edge_features is provided in some "
                    f"graphs but not all.  Either provide it everywhere or "
                    f"nowhere — silent dropping is never allowed."
                )

        # Per-type node labels — all or none.
        node_label_stores: Dict[NodeType, torch.Tensor] = {}
        for ntype in self._node_types:
            present = [g.has_node_labels(ntype) for g in graphs]
            if all(present):
                node_label_stores[ntype] = torch.cat(
                    [g.node_labels(ntype) for g in graphs], dim=0
                )
            elif any(present):
                raise ValueError(
                    f"Node type {ntype!r}: node_labels is provided in some "
                    f"graphs but not all."
                )

        # Graph labels — stack if all graphs provide one.
        glabels = [g.graph_label for g in graphs]
        if all(gl is not None for gl in glabels):
            graph_labels = torch.stack(glabels, dim=0)
        elif any(gl is not None for gl in glabels):
            raise ValueError(
                "graph_label is provided in some graphs but not all."
            )
        else:
            graph_labels = None

        self._node_stores = node_stores
        self._edge_stores = edge_stores
        self._edge_weight_stores = edge_weight_stores
        self._edge_feature_stores = edge_feature_stores
        self._node_label_stores = node_label_stores
        self._batch_dict = batch_dict
        self._cum_offsets = cum_offsets
        self.graph_labels = graph_labels  # [B, ...]
        self.metadata: List[Any] = [g.metadata for g in graphs]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_graphs(self) -> int:
        return len(self._graphs)

    @property
    def node_types(self) -> List[NodeType]:
        return list(self._node_types)

    @property
    def edge_types(self) -> List[EdgeType]:
        return list(self._edge_types)

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        return {t: v.size(0) for t, v in self._node_stores.items()}

    @property
    def num_edges_dict(self) -> Dict[EdgeType, int]:
        return {t: v.size(1) for t, v in self._edge_stores.items()}

    @property
    def x_dict(self) -> Dict[NodeType, torch.Tensor]:
        return dict(self._node_stores)

    @property
    def edge_index_dict(self) -> Dict[EdgeType, torch.Tensor]:
        return dict(self._edge_stores)

    @property
    def edge_weight_dict(self) -> Dict[EdgeType, torch.Tensor]:
        return dict(self._edge_weight_stores)

    @property
    def edge_features_dict(self) -> Dict[EdgeType, torch.Tensor]:
        return dict(self._edge_feature_stores)

    @property
    def node_labels_dict(self) -> Dict[NodeType, torch.Tensor]:
        return dict(self._node_label_stores)

    @property
    def batch_dict(self) -> Dict[NodeType, torch.Tensor]:
        return dict(self._batch_dict)

    @property
    def device(self) -> torch.device:
        return next(iter(self._node_stores.values())).device

    # ── Accessors ─────────────────────────────────────────────────────────────

    def node_features(self, node_type: NodeType) -> torch.Tensor:
        return self._node_stores[node_type]

    def edge_index(self, edge_type: EdgeType) -> torch.Tensor:
        return self._edge_stores[edge_type]

    def edge_weight(self, edge_type: EdgeType) -> Optional[torch.Tensor]:
        return self._edge_weight_stores.get(edge_type)

    def edge_features(self, edge_type: EdgeType) -> Optional[torch.Tensor]:
        return self._edge_feature_stores.get(edge_type)

    def node_labels(self, node_type: NodeType) -> Optional[torch.Tensor]:
        return self._node_label_stores.get(node_type)

    def batch(self, node_type: NodeType) -> torch.Tensor:
        """Per-node graph index tensor for ``node_type``."""
        return self._batch_dict[node_type]

    # ── Device movement ───────────────────────────────────────────────────────

    def to(self, device, dtype: torch.dtype | None = None) -> "HeteroGraphBatch":
        """Return a new batch with all tensors moved to ``device``."""
        def _move(t: torch.Tensor) -> torch.Tensor:
            if dtype is not None and t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        new_graphs = [g.to(device, dtype=dtype) for g in self._graphs]
        return HeteroGraphBatch(new_graphs)

    def cpu(self) -> "HeteroGraphBatch":
        return self.to("cpu")

    def cuda(self, device_id: int | None = None) -> "HeteroGraphBatch":
        d = "cuda" if device_id is None else f"cuda:{device_id}"
        return self.to(d)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.num_graphs

    def __repr__(self) -> str:
        return (
            f"HeteroGraphBatch("
            f"num_graphs={self.num_graphs}, "
            f"node_types={self._node_types}, "
            f"num_edges_dict={self.num_edges_dict}"
            f")  [🧪 Experimental]"
        )
