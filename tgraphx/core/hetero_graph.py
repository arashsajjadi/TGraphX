"""Experimental lightweight HeteroGraph container.

.. experimental::
    This API is **🧪 Experimental**.  The container stores heterogeneous
    (multi-type) node and edge data but does NOT implement any
    heterogeneous GNN layers.  API may change in future releases.

A ``HeteroGraph`` stores typed node feature tensors and typed edge indices:

.. code-block:: python

    from tgraphx.core.hetero_graph import HeteroGraph

    g = HeteroGraph(
        node_stores={
            "paper": paper_features,   # [N_paper, D]
            "author": author_features, # [N_author, D]
        },
        edge_stores={
            ("author", "writes", "paper"): writes_edge_index,  # [2, E]
            ("paper", "cites", "paper"): cites_edge_index,
        },
    )
    g.to("cuda")
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


NodeType = str
EdgeType = Tuple[str, str, str]  # (src_type, relation, dst_type)


class HeteroGraph:
    """🧪 Experimental: lightweight container for heterogeneous graph data.

    This is a **data container only** — it does not implement any GNN
    message-passing logic.  Use it to organise typed node/edge data before
    feeding into custom hetero-aware code.

    Args:
        node_stores: Dict mapping node-type strings to ``[N, *feature]``
            tensors.  All tensors must be on the same device.
        edge_stores: Dict mapping ``(src_type, relation, dst_type)`` tuples
            to ``[2, E]`` LongTensors of edge indices.  Edge indices for
            relation ``(src, rel, dst)`` must be valid w.r.t.
            ``node_stores[src]`` and ``node_stores[dst]``.
        edge_weight_stores: Optional dict mapping edge types to ``[E]``
            float edge weights.
        edge_feature_stores: Optional dict mapping edge types to
            ``[E, *feature]`` edge feature tensors.
        metadata: Optional dict of run/experiment metadata.
    """

    def __init__(
        self,
        node_stores: Dict[NodeType, torch.Tensor],
        edge_stores: Dict[EdgeType, torch.Tensor],
        edge_weight_stores: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_feature_stores: Optional[Dict[EdgeType, torch.Tensor]] = None,
        node_label_stores: Optional[Dict[NodeType, torch.Tensor]] = None,
        graph_label: Optional[torch.Tensor] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        if not isinstance(node_stores, dict) or not node_stores:
            raise ValueError("node_stores must be a non-empty dict of {str: Tensor}.")
        if not isinstance(edge_stores, dict):
            raise ValueError("edge_stores must be a dict of {(str,str,str): Tensor}.")
        for ntype, feat in node_stores.items():
            if not isinstance(ntype, str):
                raise ValueError(f"Node-type keys must be strings; got {type(ntype)}")
            if not isinstance(feat, torch.Tensor) or feat.dim() < 2:
                raise ValueError(
                    f"Node features for type {ntype!r} must be a Tensor with "
                    f"at least 2 dimensions [N, ...]; got {type(feat)}"
                )
        for etype, ei in edge_stores.items():
            if not (isinstance(etype, tuple) and len(etype) == 3):
                raise ValueError(
                    f"Edge-type keys must be 3-tuples (src, rel, dst); got {etype!r}"
                )
            src_type, rel, dst_type = etype
            if src_type not in node_stores:
                raise ValueError(
                    f"Edge type {etype!r}: src_type {src_type!r} not in node_stores."
                )
            if dst_type not in node_stores:
                raise ValueError(
                    f"Edge type {etype!r}: dst_type {dst_type!r} not in node_stores."
                )
            if not isinstance(ei, torch.Tensor) or ei.dim() != 2 or ei.size(0) != 2:
                raise ValueError(
                    f"Edge index for type {etype!r} must have shape [2, E]; "
                    f"got {type(ei)} {getattr(ei, 'shape', '?')}"
                )
            if ei.dtype != torch.long:
                raise TypeError(
                    f"Edge index for type {etype!r} must be dtype torch.long; "
                    f"got {ei.dtype}."
                )

        # Validate edge_weight stores
        if edge_weight_stores:
            for etype, ew in edge_weight_stores.items():
                if etype not in edge_stores:
                    raise ValueError(
                        f"edge_weight_stores key {etype!r} not found in edge_stores"
                    )
                if not isinstance(ew, torch.Tensor) or ew.dim() != 1:
                    raise ValueError(
                        f"edge_weight for {etype!r} must be a 1-D Tensor; "
                        f"got shape {getattr(ew, 'shape', '?')}"
                    )
                if ew.size(0) != edge_stores[etype].size(1):
                    raise ValueError(
                        f"edge_weight for {etype!r} has length {ew.size(0)} "
                        f"but edge_index has {edge_stores[etype].size(1)} edges."
                    )

        # Validate edge_feature stores
        if edge_feature_stores:
            for etype, ef in edge_feature_stores.items():
                if etype not in edge_stores:
                    raise ValueError(
                        f"edge_feature_stores key {etype!r} not found in edge_stores"
                    )
                if not isinstance(ef, torch.Tensor) or ef.dim() < 2:
                    raise ValueError(
                        f"edge_features for {etype!r} must be at least 2-D; "
                        f"got shape {getattr(ef, 'shape', '?')}"
                    )
                if ef.size(0) != edge_stores[etype].size(1):
                    raise ValueError(
                        f"edge_features for {etype!r} has {ef.size(0)} rows "
                        f"but edge_index has {edge_stores[etype].size(1)} edges."
                    )

        # Validate node_label stores
        if node_label_stores:
            for ntype, nl in node_label_stores.items():
                if ntype not in node_stores:
                    raise ValueError(
                        f"node_label_stores key {ntype!r} not found in node_stores"
                    )
                if not isinstance(nl, torch.Tensor):
                    raise ValueError(
                        f"node_labels for {ntype!r} must be a Tensor; got {type(nl)}"
                    )
                if nl.size(0) != node_stores[ntype].size(0):
                    raise ValueError(
                        f"node_labels for {ntype!r} has {nl.size(0)} rows but "
                        f"node features has {node_stores[ntype].size(0)} rows."
                    )

        self._node_stores: Dict[NodeType, torch.Tensor] = dict(node_stores)
        self._edge_stores: Dict[EdgeType, torch.Tensor] = dict(edge_stores)
        self._edge_weight_stores: Dict[EdgeType, torch.Tensor] = (
            dict(edge_weight_stores) if edge_weight_stores else {}
        )
        self._edge_feature_stores: Dict[EdgeType, torch.Tensor] = (
            dict(edge_feature_stores) if edge_feature_stores else {}
        )
        self._node_label_stores: Dict[NodeType, torch.Tensor] = (
            dict(node_label_stores) if node_label_stores else {}
        )
        self.graph_label = graph_label
        self.metadata = metadata

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def node_types(self) -> list[NodeType]:
        return list(self._node_stores.keys())

    @property
    def edge_types(self) -> list[EdgeType]:
        return list(self._edge_stores.keys())

    def node_features(self, node_type: NodeType) -> torch.Tensor:
        if node_type not in self._node_stores:
            raise KeyError(f"Unknown node type {node_type!r}. "
                           f"Available: {self.node_types}")
        return self._node_stores[node_type]

    def edge_index(self, edge_type: EdgeType) -> torch.Tensor:
        if edge_type not in self._edge_stores:
            raise KeyError(f"Unknown edge type {edge_type!r}. "
                           f"Available: {self.edge_types}")
        return self._edge_stores[edge_type]

    def num_nodes(self, node_type: NodeType) -> int:
        return self._node_stores[node_type].size(0)

    def num_edges(self, edge_type: EdgeType) -> int:
        return self._edge_stores[edge_type].size(1)

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        """Mapping ``node_type -> num_nodes``."""
        return {k: v.size(0) for k, v in self._node_stores.items()}

    @property
    def num_edges_dict(self) -> Dict[EdgeType, int]:
        """Mapping ``edge_type -> num_edges``."""
        return {k: v.size(1) for k, v in self._edge_stores.items()}

    def edge_weight(self, edge_type: EdgeType) -> Optional[torch.Tensor]:
        """Return per-edge weight tensor for ``edge_type`` or ``None``."""
        return self._edge_weight_stores.get(edge_type)

    def edge_features(self, edge_type: EdgeType) -> Optional[torch.Tensor]:
        """Return per-edge feature tensor for ``edge_type`` or ``None``."""
        return self._edge_feature_stores.get(edge_type)

    def has_edge_weight(self, edge_type: EdgeType) -> bool:
        return edge_type in self._edge_weight_stores

    def has_edge_features(self, edge_type: EdgeType) -> bool:
        return edge_type in self._edge_feature_stores

    @property
    def device(self) -> torch.device:
        """Device of the first node-feature tensor (all stores share device)."""
        return next(iter(self._node_stores.values())).device

    @property
    def x_dict(self) -> Dict[NodeType, torch.Tensor]:
        """Dict of node features keyed by node type (compat alias)."""
        return dict(self._node_stores)

    @property
    def edge_index_dict(self) -> Dict[EdgeType, torch.Tensor]:
        """Dict of edge indices keyed by edge type."""
        return dict(self._edge_stores)

    @property
    def edge_weight_dict(self) -> Dict[EdgeType, torch.Tensor]:
        """Dict of edge weights keyed by edge type (relations without weights are absent)."""
        return dict(self._edge_weight_stores)

    @property
    def edge_features_dict(self) -> Dict[EdgeType, torch.Tensor]:
        """Dict of edge features keyed by edge type."""
        return dict(self._edge_feature_stores)

    # ── Device movement ───────────────────────────────────────────────────────

    def to(self, device, dtype: torch.dtype | None = None) -> "HeteroGraph":
        """Move all tensors to ``device`` (and optionally cast floating dtype)."""
        def _move(t: torch.Tensor) -> torch.Tensor:
            if dtype is not None and t.is_floating_point():
                return t.to(device=device, dtype=dtype)
            return t.to(device=device)

        node_stores = {k: _move(v) for k, v in self._node_stores.items()}
        edge_stores = {k: v.to(device=device) for k, v in self._edge_stores.items()}
        ew = {k: _move(v) for k, v in self._edge_weight_stores.items()}
        ef = {k: _move(v) for k, v in self._edge_feature_stores.items()}
        nl = {k: v.to(device=device) for k, v in self._node_label_stores.items()}
        gl = self.graph_label.to(device=device) if self.graph_label is not None else None
        return HeteroGraph(
            node_stores, edge_stores, ew or None, ef or None,
            nl or None, gl, self.metadata,
        )

    def node_labels(self, node_type: NodeType) -> Optional[torch.Tensor]:
        return self._node_label_stores.get(node_type)

    def has_node_labels(self, node_type: NodeType) -> bool:
        return node_type in self._node_label_stores

    def cpu(self) -> "HeteroGraph":
        return self.to("cpu")

    def cuda(self, device_id: int | None = None) -> "HeteroGraph":
        d = "cuda" if device_id is None else f"cuda:{device_id}"
        return self.to(d)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        node_summary = {k: tuple(v.shape) for k, v in self._node_stores.items()}
        edge_summary = {str(k): v.size(1) for k, v in self._edge_stores.items()}
        return (
            f"HeteroGraph(\n"
            f"  node_types={node_summary},\n"
            f"  edge_types_edge_counts={edge_summary}\n"
            f")  [🧪 Experimental]"
        )
