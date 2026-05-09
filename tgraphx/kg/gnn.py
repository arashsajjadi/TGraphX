"""KG + GNN integration: RGCN-based KG completion.

Converts a KnowledgeGraph into a relational edge_index suitable for
RGCNConv, then wraps it in a full KG-completion model.

KG-to-edge_index mapping:
  edge_index[0, i] = head_i
  edge_index[1, i] = tail_i
  edge_type[i]     = relation_i

Edge attributes from triple_features['edge_attr'] if available.
Edge weights from kg.edge_weight if available.

Stability: Experimental.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import KnowledgeGraph
from .models import KGScoringModel, DistMultModel

__all__ = [
    "kg_to_edge_index",
    "KGRGCNModel",
]


def kg_to_edge_index(
    kg: KnowledgeGraph,
    include_edge_attr: bool = True,
    attr_key: str = "edge_attr",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a KG to relational edge_index format.

    Returns:
        edge_index: ``LongTensor[2, N_t]``
        edge_type: ``LongTensor[N_t]``
        edge_attr: ``FloatTensor[N_t, F]`` from triple_features or None
        edge_weight: ``FloatTensor[N_t]`` or None
    """
    ei, et = kg.to_edge_index()
    ea = kg.triple_features.get(attr_key) if include_edge_attr else None
    ew = kg.edge_weight
    return ei, et, ea, ew


class KGRGCNModel(KGScoringModel):
    """RGCN-based KG completion model.

    Uses :class:`tgraphx.layers.rgcn.RGCNConv` for entity representation
    learning via relation-aware message passing, then decodes with
    DistMult or a user-provided decoder.

    Architecture:
      1. Input: entity features or learned entity embeddings.
      2. RGCN layers: relation-specific message passing.
      3. Decoder: DistMult scoring on entity embeddings.

    Args:
        num_entities: N_e.
        num_relations: N_r.
        in_dim: Input entity feature dimension.  When None, a learnable
            entity embedding of size ``embedding_dim`` is used as input.
        embedding_dim: Hidden/output embedding dimension.
        num_rgcn_layers: Number of RGCN layers (1 or 2).
        num_bases: Number of basis matrices for RGCN.  0 = full.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        in_dim: Optional[int] = None,
        embedding_dim: int = 64,
        num_rgcn_layers: int = 1,
        num_bases: int = 0,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)

        # Input representation: learnable if no input features.
        if in_dim is None:
            self.entity_emb: Optional[nn.Embedding] = nn.Embedding(num_entities, embedding_dim)
            nn.init.xavier_uniform_(self.entity_emb.weight)
            _in_dim = embedding_dim
        else:
            self.entity_emb = None
            _in_dim = int(in_dim)

        # RGCN layers.
        from tgraphx.layers.rgcn import RGCNConv
        self.rgcn_layers = nn.ModuleList()
        layer_in = _in_dim
        for i in range(int(num_rgcn_layers)):
            layer_out = embedding_dim
            self.rgcn_layers.append(
                RGCNConv(layer_in, layer_out, num_relations, num_bases=num_bases)
            )
            layer_in = layer_out

        # Decoder: DistMult-style bilinear relation embedding.
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def encode(
        self,
        edge_index_by_rel: Dict[int, torch.Tensor],
        entity_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute entity representations via RGCN.

        Args:
            edge_index_by_rel: ``{rel_id: LongTensor[2, E_r]}``.
            entity_features: Optional ``FloatTensor[N_e, in_dim]``.

        Returns:
            ``FloatTensor[N_e, embedding_dim]``.
        """
        if entity_features is not None:
            x = entity_features.float()
        elif self.entity_emb is not None:
            x = self.entity_emb.weight
        else:
            raise ValueError("Either entity_features or learnable entity_emb must be set")

        for layer in self.rgcn_layers:
            x = F.relu(layer(x, edge_index_by_rel, self.num_entities))
        return x

    def score_triples(
        self,
        triples: torch.Tensor,
        entity_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DistMult decode using pre-computed entity embeddings.

        When ``entity_embs`` is None, falls back to the learnable embedding.
        """
        h_idx, r_idx, t_idx = triples[:, 0], triples[:, 1], triples[:, 2]
        if entity_embs is None:
            entity_embs = self.entity_emb.weight if self.entity_emb is not None else None
        if entity_embs is None:
            raise ValueError("Pass entity_embs from encode() or set learnable entity_emb")
        h = entity_embs[h_idx]
        r = self.relation_emb(r_idx)
        t = entity_embs[t_idx]
        return (h * r * t).sum(dim=-1)

    def forward_kg(
        self,
        kg: KnowledgeGraph,
        triples: torch.Tensor,
        entity_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode from KG then score triples.

        Args:
            kg: Source :class:`~tgraphx.kg.KnowledgeGraph`.
            triples: ``LongTensor[B, 3]`` to score.
            entity_features: Optional pre-computed entity feature vectors.

        Returns:
            ``FloatTensor[B]`` scores.
        """
        ei, et, _, _ = kg_to_edge_index(kg, include_edge_attr=False)
        # Build relation-indexed edge dict.
        N_r = kg.num_relations
        edge_index_by_rel: Dict[int, torch.Tensor] = {}
        for r in range(N_r):
            mask = et == r
            if mask.any():
                edge_index_by_rel[r] = ei[:, mask]
        embs = self.encode(edge_index_by_rel, entity_features)
        return self.score_triples(triples, embs)
