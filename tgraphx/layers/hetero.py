"""Experimental heterogeneous message-passing wrapper.

.. experimental::
    🧪 ``HeteroConv`` is an experimental dispatch wrapper.  It is **not**
    a full hetero-GNN framework: it applies a user-supplied
    relation-specific module to each ``(src, rel, dst)`` edge type and
    aggregates the per-relation outputs at the destination node type.

Design
------
    out_dict[dst_type] = AGG_{rel into dst_type}( module_{(s,r,dst)}(x_s, ei) )

* Per-relation modules can be any TGraphX layer that exposes a
  ``forward(x_dst_type_features, edge_index, ...)``-like signature.
* When the relation source type differs from the destination type
  (e.g., ``("author", "writes", "paper")``), the wrapper passes the
  **destination type's feature tensor sized to fit edge_index's index
  range** to the inner layer; this requires that the inner layer accept
  the source features at row ``[0]`` of edge_index and update only the
  destinations at row ``[1]``.  We therefore pass a stacked ``[N_src + N_dst, *]``
  tensor and remap edge indices so dst rows point into the dst sub-block.
  See ``_dispatch_relation`` for the exact mechanics.

Limitations
-----------
* Vector node features are fully supported.  2-D / 3-D spatial node
  features are supported when **all** relations write into a destination
  type whose inner module preserves the spatial layout.
* No relation-specific edge_features are required; you may pass
  ``edge_features_dict`` and ``edge_weight_dict``.
* Self-loops are the user's responsibility — the wrapper does not insert
  any.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.hetero_graph import EdgeType, NodeType

__all__ = ["HeteroConv"]

_AGGREGATIONS = ("sum", "mean", "max")


class HeteroConv(nn.Module):
    """🧪 Experimental: relation-dispatch wrapper for heterogeneous message passing.

    Args:
        convs: Dict mapping ``(src_type, relation, dst_type)`` tuples to
            ``nn.Module`` instances.  Each module must accept
            ``(x, edge_index)`` and optionally ``edge_weight`` /
            ``edge_features`` keyword arguments and return an updated
            feature tensor with the same number of rows as ``x``.
        aggr: Cross-relation aggregation when multiple relations write
            into the same destination type.  One of ``"sum"`` (default),
            ``"mean"``, or ``"max"``.

    Example::

        from tgraphx.layers.hetero import HeteroConv
        from tgraphx.layers import LinearMessagePassing

        conv = HeteroConv({
            ("author", "writes", "paper"): LinearMessagePassing((D,), (D,)),
            ("paper", "cites", "paper"): LinearMessagePassing((D,), (D,)),
        }, aggr="sum")
        out_dict = conv(
            x_dict={"author": x_a, "paper": x_p},
            edge_index_dict={
                ("author", "writes", "paper"): ei_aw,
                ("paper", "cites", "paper"): ei_pc,
            },
        )
    """

    def __init__(
        self,
        convs: Dict[EdgeType, nn.Module],
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if aggr not in _AGGREGATIONS:
            raise ValueError(
                f"aggr must be one of {_AGGREGATIONS}; got {aggr!r}"
            )
        if not isinstance(convs, dict) or not convs:
            raise ValueError("convs must be a non-empty dict of {EdgeType: nn.Module}.")
        for etype, module in convs.items():
            if not (isinstance(etype, tuple) and len(etype) == 3):
                raise ValueError(
                    f"Conv key must be (src, rel, dst) tuple; got {etype!r}"
                )
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"Conv value for {etype!r} must be an nn.Module; got {type(module)}"
                )

        self.aggr = aggr
        self._edge_types: List[EdgeType] = list(convs.keys())
        # Use ModuleDict with safe string keys (joined with '__'); keep the
        # original tuple in self._edge_types for ordering.
        self.convs = nn.ModuleDict(
            {self._key_to_str(k): v for k, v in convs.items()}
        )

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _key_to_str(etype: EdgeType) -> str:
        s, r, d = etype
        return f"{s}__{r}__{d}"

    @staticmethod
    def _dispatch_relation(
        module: nn.Module,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run a per-relation module and return the **destination** rows.

        The trick: most TGraphX layers expect a single ``[N, *]`` feature
        tensor with edge_index pointing into it.  For a heterogeneous
        relation ``(s, r, d)``, we stack ``[x_src; x_dst]`` and remap
        ``edge_index[1]`` (destination) to point into the second block.
        After the layer runs, we slice the destination block as the
        per-relation output.

        Args:
            module: The relation-specific message-passing layer.
            x_src: Source node features ``[N_src, *]``.
            x_dst: Destination node features ``[N_dst, *]`` (used as the
                "base" for the destination subblock; identity is correct
                only for layers that update via aggregation, not
                transformation of the destination).
            edge_index: ``[2, E]`` original edge indices.
            edge_weight: Optional ``[E]`` per-edge scalars.
            edge_features: Optional per-edge feature tensor.
        """
        if x_src is x_dst:
            # Same node type — no remapping needed.
            kwargs: Dict[str, Any] = {}
            if edge_weight is not None:
                kwargs["edge_weight"] = edge_weight
            if edge_features is not None:
                kwargs["edge_features"] = edge_features
            out = module(x_dst, edge_index, **kwargs)
            return out

        N_src = x_src.size(0)
        N_dst = x_dst.size(0)
        x_combined = torch.cat([x_src, x_dst], dim=0)
        # Remap destinations to the second block.
        ei_remap = edge_index.clone()
        ei_remap[1] = ei_remap[1] + N_src
        kwargs = {}
        if edge_weight is not None:
            kwargs["edge_weight"] = edge_weight
        if edge_features is not None:
            kwargs["edge_features"] = edge_features
        out_combined = module(x_combined, ei_remap, **kwargs)
        # Slice the destination block.
        return out_combined[N_src:]

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_weight_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_features_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> Dict[NodeType, torch.Tensor]:
        """Apply each per-relation module and aggregate per destination type.

        Returns a dict ``out[dst_type] = AGG_{rel into dst_type}(...)``.
        Destination types that receive no messages from any relation in
        ``self.convs`` are returned with their original ``x_dict[dst]``
        (i.e., self-loop fall-through).
        """
        if edge_weight_dict is None:
            edge_weight_dict = {}
        if edge_features_dict is None:
            edge_features_dict = {}

        # Group per-relation outputs by destination type.
        per_dst: Dict[NodeType, List[torch.Tensor]] = {}

        for etype in self._edge_types:
            src_t, _, dst_t = etype
            if src_t not in x_dict:
                raise KeyError(
                    f"x_dict missing source type {src_t!r} required by relation {etype!r}"
                )
            if dst_t not in x_dict:
                raise KeyError(
                    f"x_dict missing destination type {dst_t!r} required by relation {etype!r}"
                )
            if etype not in edge_index_dict:
                raise KeyError(
                    f"edge_index_dict missing relation {etype!r}"
                )
            module = self.convs[self._key_to_str(etype)]
            out = self._dispatch_relation(
                module,
                x_dict[src_t],
                x_dict[dst_t],
                edge_index_dict[etype],
                edge_weight_dict.get(etype),
                edge_features_dict.get(etype),
            )
            per_dst.setdefault(dst_t, []).append(out)

        # Aggregate per destination type.
        out_dict: Dict[NodeType, torch.Tensor] = {}
        for dst_t in x_dict:
            outs = per_dst.get(dst_t)
            if outs is None:
                # No relation writes into this destination type — pass through.
                out_dict[dst_t] = x_dict[dst_t]
                continue
            if len(outs) == 1:
                out_dict[dst_t] = outs[0]
            else:
                stacked = torch.stack(outs, dim=0)  # [n_relations, N_dst, *]
                if self.aggr == "sum":
                    out_dict[dst_t] = stacked.sum(dim=0)
                elif self.aggr == "mean":
                    out_dict[dst_t] = stacked.mean(dim=0)
                else:  # "max"
                    out_dict[dst_t] = stacked.max(dim=0).values

        return out_dict

    def extra_repr(self) -> str:
        return f"aggr={self.aggr!r}, edge_types={len(self._edge_types)}"
