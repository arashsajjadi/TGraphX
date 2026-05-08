"""Experimental heterogeneous graph models.

.. experimental::
    🧪 ``HeteroGraphClassifier`` and ``HeteroNodeClassifier`` are
    experimental compositions of :class:`tgraphx.layers.hetero.HeteroConv`
    blocks with hetero readouts.  They are functional and tested at the
    smoke level but the API may change in v0.2.6+.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..core.hetero_graph import EdgeType, NodeType
from ..layers.base import LinearMessagePassing
from ..layers.hetero import HeteroConv
from ..layers.hetero_readout import hetero_concat_pool, hetero_mean_pool

__all__ = ["HeteroGraphClassifier", "HeteroNodeClassifier"]


def _make_block(
    edge_types: List[EdgeType],
    in_dim: int,
    hidden_dim: int,
    aggr: str,
) -> HeteroConv:
    """Build a HeteroConv block where every relation expects ``in_dim`` features."""
    convs: Dict[EdgeType, nn.Module] = {}
    for etype in edge_types:
        convs[etype] = LinearMessagePassing(
            in_shape=(in_dim,), out_shape=(hidden_dim,), aggr=aggr,
        )
    return HeteroConv(convs, aggr="sum")


class HeteroGraphClassifier(nn.Module):
    """🧪 Experimental: hetero graph classifier with vector node features.

    Stacks ``num_layers`` of vector ``HeteroConv`` blocks then concatenates
    a stable-ordered per-type mean-pool readout into a final MLP head.

    Args:
        node_in_dims: Dict mapping node type to input feature dimension.
        edge_types: List of relation tuples ``(src, rel, dst)``.
        hidden_dim: Hidden feature dimension for all node types after the
            first block.
        num_layers: Number of HeteroConv blocks (>= 1).
        num_classes: Output classification dimension.
        type_order: Optional stable ordering of node types for concat.
        readout: ``"mean"`` (default), ``"sum"``, or ``"max"``.
        dropout: Dropout applied after each block (default 0.0).
        aggr: Per-relation aggregation passed to LinearMessagePassing.
    """

    def __init__(
        self,
        node_in_dims: Dict[NodeType, int],
        edge_types: List[EdgeType],
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        type_order: Optional[List[NodeType]] = None,
        readout: str = "mean",
        dropout: float = 0.0,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if not isinstance(node_in_dims, dict) or not node_in_dims:
            raise ValueError("node_in_dims must be a non-empty dict")
        self.node_types = (
            list(type_order) if type_order is not None else sorted(node_in_dims.keys())
        )
        self.edge_types = list(edge_types)
        self.readout = readout

        # Per-type input projection to a common hidden_dim so that all
        # subsequent HeteroConv blocks operate on a uniform feature size.
        # This is necessary because HeteroConv only updates destination types
        # of supplied relations; types that are never destinations would
        # otherwise keep their original input dim across blocks.
        self.input_proj = nn.ModuleDict({
            t: nn.Linear(in_d, hidden_dim) for t, in_d in node_in_dims.items()
        })

        # Stack of HeteroConv blocks (all operate at hidden_dim).
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = _make_block(self.edge_types, hidden_dim, hidden_dim, aggr)
            self.blocks.append(block)

        # Out-of-place ReLU to avoid in-place autograd version conflicts when
        # the same tensor is reused as src and dst across relations.
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Final MLP head over the concatenated readout.
        total_dim = hidden_dim * len(self.node_types)
        self.head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        batch_dict: Optional[Dict[NodeType, torch.Tensor]] = None,
        edge_weight_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_features_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Project per-type inputs to common hidden_dim.
        h_dict = {t: self.act(self.input_proj[t](x)) for t, x in x_dict.items()}
        for block in self.blocks:
            h_dict = block(
                h_dict, edge_index_dict,
                edge_weight_dict=edge_weight_dict,
                edge_features_dict=edge_features_dict,
            )
            h_dict = {t: self.dropout(self.act(h)) for t, h in h_dict.items()}
        pooled = hetero_concat_pool(
            h_dict, batch_dict=batch_dict, type_order=self.node_types,
            mode=self.readout,
        )
        return self.head(pooled)


class HeteroNodeClassifier(nn.Module):
    """🧪 Experimental: hetero node classifier on a single target node type.

    Stacks ``num_layers`` of vector ``HeteroConv`` blocks and projects the
    target type's representation to ``num_classes``.

    Args:
        node_in_dims: Dict ``node_type -> input dim``.
        edge_types: List of relations.
        hidden_dim: Hidden feature dimension after the first block.
        num_layers: Number of HeteroConv blocks (>= 1).
        num_classes: Output classification dimension for the target type.
        target_type: The node type whose representations are classified.
        dropout: Dropout applied after each block.
        aggr: Per-relation aggregation passed to LinearMessagePassing.
    """

    def __init__(
        self,
        node_in_dims: Dict[NodeType, int],
        edge_types: List[EdgeType],
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        target_type: NodeType,
        dropout: float = 0.0,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if target_type not in node_in_dims:
            raise ValueError(
                f"target_type {target_type!r} not in node_in_dims keys {list(node_in_dims.keys())}"
            )
        self.target_type = target_type

        self.input_proj = nn.ModuleDict({
            t: nn.Linear(in_d, hidden_dim) for t, in_d in node_in_dims.items()
        })

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = _make_block(list(edge_types), hidden_dim, hidden_dim, aggr)
            self.blocks.append(block)

        # Out-of-place ReLU to avoid in-place autograd version conflicts when
        # the same tensor is reused as src and dst across relations.
        self.act = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x_dict: Dict[NodeType, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_weight_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        edge_features_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h_dict = {t: self.act(self.input_proj[t](x)) for t, x in x_dict.items()}
        for block in self.blocks:
            h_dict = block(
                h_dict, edge_index_dict,
                edge_weight_dict=edge_weight_dict,
                edge_features_dict=edge_features_dict,
            )
            h_dict = {t: self.dropout(self.act(h)) for t, h in h_dict.items()}
        return self.head(h_dict[self.target_type])
