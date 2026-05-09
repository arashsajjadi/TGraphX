"""Graph genome data structure for evolutionary graph optimization.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

__all__ = ["GraphGenome"]


@dataclass
class GraphGenome:
    """A graph represented as a genome for evolutionary algorithms.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional Tensor [N, *].
        edge_features: Optional Tensor [E, *].
        node_types: Optional LongTensor [N].
        edge_types: Optional LongTensor [E].
        graph_features: Optional graph-level feature tensor.
        valid_node_mask: Optional BoolTensor [N].
        valid_edge_mask: Optional BoolTensor [E].
        constraints: Dict of constraints for validation.
        metadata: Arbitrary metadata.
    """

    edge_index: torch.Tensor
    num_nodes: int
    node_features: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None
    node_types: Optional[torch.Tensor] = None
    edge_types: Optional[torch.Tensor] = None
    graph_features: Optional[torch.Tensor] = None
    valid_node_mask: Optional[torch.Tensor] = None
    valid_edge_mask: Optional[torch.Tensor] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate genome consistency.

        Raises:
            ValueError: If any constraint is violated.
        """
        if not isinstance(self.edge_index, torch.Tensor):
            raise ValueError("edge_index must be a torch.Tensor")
        if self.edge_index.dim() != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E] but got {list(self.edge_index.shape)}"
            )
        if self.num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative but got {self.num_nodes}")
        num_edges = self.edge_index.shape[1]
        if num_edges > 0:
            if int(self.edge_index.min().item()) < 0:
                raise ValueError("edge_index contains negative node IDs")
            max_id = int(self.edge_index.max().item())
            if max_id >= self.num_nodes:
                raise ValueError(
                    f"edge_index contains node ID {max_id} >= num_nodes={self.num_nodes}"
                )
        if self.node_features is not None and self.node_features.shape[0] != self.num_nodes:
            raise ValueError(
                f"node_features.shape[0]={self.node_features.shape[0]} "
                f"!= num_nodes={self.num_nodes}"
            )
        if self.node_types is not None and self.node_types.shape[0] != self.num_nodes:
            raise ValueError(
                f"node_types.shape[0]={self.node_types.shape[0]} "
                f"!= num_nodes={self.num_nodes}"
            )
        if self.edge_features is not None and self.edge_features.shape[0] != num_edges:
            raise ValueError(
                f"edge_features.shape[0]={self.edge_features.shape[0]} "
                f"!= num_edges={num_edges}"
            )
        if self.edge_types is not None and self.edge_types.shape[0] != num_edges:
            raise ValueError(
                f"edge_types.shape[0]={self.edge_types.shape[0]} "
                f"!= num_edges={num_edges}"
            )

    def to(self, device: torch.device) -> "GraphGenome":
        """Move all tensors to device."""
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return GraphGenome(
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            node_features=_mv(self.node_features),
            edge_features=_mv(self.edge_features),
            node_types=_mv(self.node_types),
            edge_types=_mv(self.edge_types),
            graph_features=_mv(self.graph_features),
            valid_node_mask=_mv(self.valid_node_mask),
            valid_edge_mask=_mv(self.valid_edge_mask),
            constraints=copy.deepcopy(self.constraints),
            metadata=copy.deepcopy(self.metadata),
        )

    def clone(self) -> "GraphGenome":
        """Deep clone."""
        def _cl(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.clone() if t is not None else None

        return GraphGenome(
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            node_features=_cl(self.node_features),
            edge_features=_cl(self.edge_features),
            node_types=_cl(self.node_types),
            edge_types=_cl(self.edge_types),
            graph_features=_cl(self.graph_features),
            valid_node_mask=_cl(self.valid_node_mask),
            valid_edge_mask=_cl(self.valid_edge_mask),
            constraints=copy.deepcopy(self.constraints),
            metadata=copy.deepcopy(self.metadata),
        )

    def to_generated_graph(self) -> "tgraphx.generation.GeneratedGraph":  # type: ignore
        """Convert to GeneratedGraph."""
        from tgraphx.generation.data_model import GeneratedGraph
        return GeneratedGraph(
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            node_features=self.node_features.clone() if self.node_features is not None else None,
            edge_features=self.edge_features.clone() if self.edge_features is not None else None,
            node_types=self.node_types.clone() if self.node_types is not None else None,
            edge_types=self.edge_types.clone() if self.edge_types is not None else None,
            graph_features=self.graph_features.clone() if self.graph_features is not None else None,
            metadata=copy.deepcopy(self.metadata),
        )

    @classmethod
    def from_graph(cls, g: Any) -> "GraphGenome":
        """Create GraphGenome from a GeneratedGraph or tgraphx.Graph.

        Args:
            g: GeneratedGraph or tgraphx.core.graph.Graph.
        """
        ei = g.edge_index
        nf = getattr(g, "node_features", None)
        ef = getattr(g, "edge_features", None)
        nt = getattr(g, "node_types", None)
        et = getattr(g, "edge_types", None)
        nn_val = getattr(g, "num_nodes", None)
        if nn_val is None:
            nn_val = int(ei.max().item()) + 1 if ei.numel() > 0 else 0

        return cls(
            edge_index=ei.clone(),
            num_nodes=nn_val,
            node_features=nf.clone() if nf is not None else None,
            edge_features=ef.clone() if ef is not None else None,
            node_types=nt.clone() if nt is not None else None,
            edge_types=et.clone() if et is not None else None,
            metadata=dict(getattr(g, "metadata", {}) or {}),
        )

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def device(self) -> torch.device:
        return self.edge_index.device
