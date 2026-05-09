"""Feature-aware multimodal KG scoring model.

MultimodalKGModel combines:
  1. A MultimodalEntityFusion module that projects modality-specific entity
     features (image, text, user, vector, …) into a common entity embedding.
  2. A relation embedding lookup.
  3. A DistMult-style scoring decoder.

Score formula (DistMult):
  f(h, r, t) = <z_h, e_r, z_t>

where z_h and z_t are multimodal entity embeddings and e_r is the
relation embedding.

The model is fully differentiable — gradients flow back through all
projector branches whose features are provided, enabling the model to
learn from image, text, user, and vector entity features simultaneously.

Stability: Experimental (v0.6.0+).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import KnowledgeGraph
from .models import KGScoringModel
from .projectors import MultimodalEntityFusion, _BaseProjector

__all__ = ["MultimodalKGModel"]


class MultimodalKGModel(KGScoringModel):
    """Feature-aware KG scoring model for multimodal entity features.

    Scores triples via DistMult on entity embeddings produced by a
    :class:`MultimodalEntityFusion`.  Entities without any feature are
    handled by a learned fallback embedding.

    Args:
        num_entities: N_e.
        num_relations: N_r.
        out_dim: Common entity/relation embedding dimension.
        projectors: ``{modality_name: _BaseProjector}`` mapping each modality
            to a projector whose output is ``FloatTensor[N_e, out_dim]``.
        fusion_mode: ``"add"`` | ``"concat_project"`` | ``"gated"``.

    Usage::

        from tgraphx.kg.projectors import ImageEntityProjector, VectorEntityProjector

        model = MultimodalKGModel(
            num_entities=N_e, num_relations=N_r, out_dim=32,
            projectors={
                "image": ImageEntityProjector(in_channels=3, out_dim=32),
                "user":  VectorEntityProjector(in_dim=16, out_dim=32),
                "text":  VectorEntityProjector(in_dim=8, out_dim=32),
            },
            fusion_mode="gated",
        )
        # Forward using features from a KG:
        scores = model.score_from_kg(kg, triples)

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        out_dim: int,
        projectors: Dict[str, _BaseProjector],
        fusion_mode: str = "gated",
        add_learnable_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.out_dim = int(out_dim)
        self.fusion = MultimodalEntityFusion(
            projectors=projectors,
            out_dim=out_dim,
            num_entities=num_entities,
            fusion_mode=fusion_mode,
            add_learnable_bias=add_learnable_bias,
        )
        self.relation_emb = nn.Embedding(num_relations, out_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        # Cache last computed entity embeddings to avoid recomputing for every triple.
        self._cached_embs: Optional[torch.Tensor] = None

    def encode_entities(
        self,
        entity_features: Dict[str, torch.Tensor],
        entity_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute ``FloatTensor[N_e, out_dim]`` entity embeddings."""
        return self.fusion(entity_features, entity_feature_masks)

    def score_triples(
        self,
        triples: torch.Tensor,
        entity_embs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DistMult score using pre-computed entity embeddings.

        Args:
            triples: ``LongTensor[B, 3]``.
            entity_embs: ``FloatTensor[N_e, out_dim]`` from ``encode_entities``.
                When None, uses cached embeddings (if available).

        Returns:
            ``FloatTensor[B]``.
        """
        if entity_embs is None:
            entity_embs = self._cached_embs
        if entity_embs is None:
            raise ValueError(
                "Pass entity_embs from encode_entities() or call score_from_kg()."
            )
        h = entity_embs[triples[:, 0]]
        r = self.relation_emb(triples[:, 1])
        t = entity_embs[triples[:, 2]]
        return (h * r * t).sum(dim=-1)

    def score_from_kg(
        self,
        kg: "KnowledgeGraph",
        triples: torch.Tensor,
    ) -> torch.Tensor:
        """Encode entities from a KG and score ``triples``.

        Convenience wrapper: encodes entities from ``kg.entity_features``
        and ``kg.entity_feature_masks``, then calls :meth:`score_triples`.

        Args:
            kg: Source :class:`~tgraphx.kg.KnowledgeGraph`.
            triples: ``LongTensor[B, 3]``.

        Returns:
            ``FloatTensor[B]``.
        """
        embs = self.encode_entities(kg.entity_features, kg.entity_feature_masks)
        self._cached_embs = embs
        return self.score_triples(triples, embs)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Alias for score_triples (uses cached embs or raises)."""
        return self.score_triples(triples)

    def named_modality_params(self):
        """Yield (name, param) for all projector parameters (for gradient inspection)."""
        for name, mod in self.fusion.projectors.items():
            for pname, param in mod.named_parameters():
                yield f"projector.{name}.{pname}", param
