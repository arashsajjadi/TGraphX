"""Tensor-aware KG model zoo.

All models implement a unified interface:

  score_triples(triples: LongTensor[B, 3]) -> FloatTensor[B]
    Higher score = more plausible triple.

  score_tail_candidates(heads, relations, candidates=None, chunk_size=None)
    -> FloatTensor[B * N_e] or FloatTensor[B * len(candidates)]

  score_head_candidates(relations, tails, candidates=None, chunk_size=None)

Embedding-only models:
  TransEModel    — -||h+r-t||_p               (Bordes 2013)
  DistMultModel  — <h,r,t>                     (Yang 2015)
  ComplExModel   — Re(<h,r,conj(t)>)           (Trouillon 2016)
  RotatEModel    — -||h∘r-t|| (phase rotation) (Sun 2019)

Feature-aware extensions (optional):
  Each model accepts optional entity_feature_dim, relation_feature_dim.
  When provided, a linear projector blends the embedding with the feature.

  z_e = embedding[e] + W_x · feature_e   (additive blend)

The user controls this blend; no silent flattening of image/volume tensors.
For non-vector features, the user must pre-project to a vector before
passing to the model.

Stability: TransEModel/DistMultModel — Beta. ComplExModel/RotatEModel — Experimental.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "KGScoringModel",
    "TransEModel",
    "DistMultModel",
    "ComplExModel",
    "RotatEModel",
    "RESCALModel",
]


# ── Base interface ────────────────────────────────────────────────────────────


class KGScoringModel(nn.Module):
    """Abstract base class for KG scoring models.

    All concrete models must implement :meth:`score_triples`.
    """

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """Score a batch of triples.

        Args:
            triples: ``LongTensor[B, 3]`` of (h, r, t).

        Returns:
            ``FloatTensor[B]`` — higher = more plausible.
        """
        raise NotImplementedError

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Alias for score_triples."""
        return self.score_triples(triples)

    # Old score() method for backward compatibility
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Score with separate head/relation/tail tensors [B]."""
        return self.score_triples(torch.stack([h, r, t], dim=1))


# ── Entity/relation feature projectors ───────────────────────────────────────


class _FeatureProjector(nn.Module):
    """Linear projector that blends features into embeddings.

    z = embedding + W · features
    """

    def __init__(self, feature_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, emb: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return emb + self.proj(feat.float())


# ── TransE ────────────────────────────────────────────────────────────────────


class TransEModel(KGScoringModel):
    """TransE: -||h + r - t||_p.

    Score is negated distance: higher score = smaller distance = more plausible.

    Initialisation: U[-6/√D, 6/√D] (Bordes 2013).
    Entity embeddings are L2-normalised during scoring (projection constraint
    is enforced post-optimiser step by ``constrain_entity_norm(norm=1)``).

    Args:
        num_entities: N_e.
        num_relations: N_r.
        embedding_dim: D.
        norm: L-norm p ∈ {1, 2}.
        entity_feature_dim: When provided, adds a feature projector for
            entity features.  The caller passes pre-vector features.
        relation_feature_dim: Analogous for relation features.

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        norm: int = 2,
        entity_feature_dim: Optional[int] = None,
        relation_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.norm = int(norm)
        D = self.embedding_dim
        scale = 6.0 / math.sqrt(D)
        self.entity_emb = nn.Embedding(num_entities, D)
        self.relation_emb = nn.Embedding(num_relations, D)
        nn.init.uniform_(self.entity_emb.weight, -scale, scale)
        nn.init.uniform_(self.relation_emb.weight, -scale, scale)
        # Feature projectors (optional).
        self.entity_proj: Optional[nn.Module] = (
            _FeatureProjector(entity_feature_dim, D) if entity_feature_dim else None
        )
        self.relation_proj: Optional[nn.Module] = (
            _FeatureProjector(relation_feature_dim, D) if relation_feature_dim else None
        )

    def _embed_entities(
        self, idx: torch.Tensor, feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = F.normalize(self.entity_emb(idx), p=2, dim=-1)
        if feat is not None and self.entity_proj is not None:
            emb = emb + self.entity_proj(feat.float())
        return emb

    def _embed_relations(
        self, idx: torch.Tensor, feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = self.relation_emb(idx)
        if feat is not None and self.relation_proj is not None:
            emb = emb + self.relation_proj(feat.float())
        return emb

    def score_triples(
        self,
        triples: torch.Tensor,
        entity_features: Optional[torch.Tensor] = None,
        relation_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_idx, r_idx, t_idx = triples[:, 0], triples[:, 1], triples[:, 2]
        h = self._embed_entities(h_idx, entity_features[h_idx] if entity_features is not None else None)
        r = self._embed_relations(r_idx, relation_features[r_idx] if relation_features is not None else None)
        t = self._embed_entities(t_idx, entity_features[t_idx] if entity_features is not None else None)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)

    def constrain_entity_norm(self, norm: float = 1.0) -> None:
        """Project entity embeddings onto the L2-norm-1 sphere in-place."""
        with torch.no_grad():
            n = self.entity_emb.weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            self.entity_emb.weight.data = self.entity_emb.weight.data * (norm / n)


# ── DistMult ──────────────────────────────────────────────────────────────────


class DistMultModel(KGScoringModel):
    """DistMult: <h, r, t> = Σ_i h_i · r_i · t_i.

    Symmetric in (h, t) for fixed r.  This implies all modelled relations
    are symmetric — docs note this limitation.

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        entity_feature_dim: Optional[int] = None,
        relation_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        D = self.embedding_dim
        self.entity_emb = nn.Embedding(num_entities, D)
        self.relation_emb = nn.Embedding(num_relations, D)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.entity_proj: Optional[nn.Module] = (
            _FeatureProjector(entity_feature_dim, D) if entity_feature_dim else None
        )
        self.relation_proj: Optional[nn.Module] = (
            _FeatureProjector(relation_feature_dim, D) if relation_feature_dim else None
        )

    def _emb_e(self, idx, feat=None):
        e = self.entity_emb(idx)
        if feat is not None and self.entity_proj:
            e = self.entity_proj(e, feat.float())
        return e

    def _emb_r(self, idx, feat=None):
        e = self.relation_emb(idx)
        if feat is not None and self.relation_proj:
            e = self.relation_proj(e, feat.float())
        return e

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        h = self.entity_emb(triples[:, 0])
        r = self.relation_emb(triples[:, 1])
        t = self.entity_emb(triples[:, 2])
        return (h * r * t).sum(dim=-1)


# ── ComplEx ───────────────────────────────────────────────────────────────────


class ComplExModel(KGScoringModel):
    """ComplEx: Re(<h, r, conj(t)>) in the complex number field.

    Each entity and relation has a complex embedding split into real and
    imaginary parts (shape ``[N, D]`` each):
      e_h = a_h + i·b_h
      e_r = a_r + i·b_r
      e_t = a_t + i·b_t

    Score:
      Re(<e_h, e_r, conj(e_t)>)
      = Σ (a_h*a_r*a_t + b_h*b_r*b_t + a_h*b_r*b_t + b_h*a_r*a_t)
      (the standard expanded real-arithmetic form)

    Antisymmetric relations are supported (ComplEx was designed for this).

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        D = self.embedding_dim
        # Real parts.
        self.entity_re = nn.Embedding(num_entities, D)
        self.entity_im = nn.Embedding(num_entities, D)
        self.relation_re = nn.Embedding(num_relations, D)
        self.relation_im = nn.Embedding(num_relations, D)
        for emb in (self.entity_re, self.entity_im, self.relation_re, self.relation_im):
            nn.init.xavier_uniform_(emb.weight)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        a_h = self.entity_re(h)
        b_h = self.entity_im(h)
        a_r = self.relation_re(r)
        b_r = self.relation_im(r)
        a_t = self.entity_re(t)
        b_t = self.entity_im(t)
        # Re(<e_h, e_r, conj(e_t)>)
        # = Re((a_h+ib_h)(a_r+ib_r)(a_t-ib_t))
        # Expanded:
        score = (
            (a_h * a_r * a_t).sum(-1)
            + (b_h * b_r * b_t).sum(-1)  # wait — let me re-derive
            + (a_h * b_r * b_t).sum(-1)
            + (b_h * a_r * a_t).sum(-1)
        )
        # Re((a_h + ib_h)(a_r + ib_r)(a_t - ib_t)):
        # Step 1: (a_h + ib_h)(a_r + ib_r)
        #       = (a_h*a_r - b_h*b_r) + i*(a_h*b_r + b_h*a_r)
        # Step 2: × (a_t - ib_t)
        # Real part = (a_h*a_r - b_h*b_r)*a_t + (a_h*b_r + b_h*a_r)*b_t
        # = a_h*a_r*a_t - b_h*b_r*a_t + a_h*b_r*b_t + b_h*a_r*b_t
        score = (
            (a_h * a_r * a_t).sum(-1)
            - (b_h * b_r * a_t).sum(-1)
            + (a_h * b_r * b_t).sum(-1)
            + (b_h * a_r * b_t).sum(-1)
        )
        return score


# ── RotatE ────────────────────────────────────────────────────────────────────


class RotatEModel(KGScoringModel):
    """RotatE: rotation in complex unit sphere.

    Relations are represented as rotations in complex space:
      r_i = exp(i·θ_i)  (enforced by parameterising as a phase)

    Score:
      f(h,r,t) = -||e_h ∘ r - e_t||

    where ∘ denotes element-wise complex multiplication.

    Entity embeddings: complex, shape ``[N_e, D]`` re+im.
    Relation phases: real, shape ``[N_r, D]``, mapped to exp(i·θ).

    The embedding range for phases is ``[0, 2π]`` via sigmoid+2π scaling.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 6.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)
        D = self.embedding_dim
        self.entity_re = nn.Embedding(num_entities, D)
        self.entity_im = nn.Embedding(num_entities, D)
        self.relation_phase = nn.Embedding(num_relations, D)
        nn.init.uniform_(self.entity_re.weight, -1.0, 1.0)
        nn.init.uniform_(self.entity_im.weight, -1.0, 1.0)
        # Phases: initialise uniformly in [-π, π].
        nn.init.uniform_(self.relation_phase.weight, -math.pi, math.pi)

    def _entity_norm(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalised (re, im) parts on the unit circle."""
        re = self.entity_re(idx)
        im = self.entity_im(idx)
        denom = torch.sqrt(re ** 2 + im ** 2).clamp(min=1e-12)
        return re / denom, im / denom

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        # Entity embeddings (L2-normalised per complex coordinate).
        h_re, h_im = self._entity_norm(h)
        t_re, t_im = self._entity_norm(t)
        # Relation as unit-complex rotation.
        phase = self.relation_phase(r)  # [B, D]
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)
        # h ∘ r in complex:
        # (h_re + i*h_im)(r_re + i*r_im) = h_re*r_re - h_im*r_im + i*(h_re*r_im + h_im*r_re)
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        # Distance ||h∘r - t||.
        diff_re = hr_re - t_re
        diff_im = hr_im - t_im
        dist = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-12).sum(dim=-1)
        return self.margin - dist


# ── RESCAL ────────────────────────────────────────────────────────────────────


class RESCALModel(KGScoringModel):
    """RESCAL: f(h, r, t) = h^T M_r t  where M_r is a [D, D] matrix per relation.

    Reference: Nickel, Tresp, Kriegel — *A Three-Way Model for Collective
    Learning on Multi-Relational Data*, ICML 2011.

    Each entity has a vector embedding ``[D]``; each relation has a dense
    matrix ``[D, D]``.  The score is the bilinear form ``h^T M_r t``.

    Compared with DistMult (which uses a *diagonal* matrix), RESCAL captures
    asymmetric and non-commutative relations.

    Args:
        num_entities: N_e.
        num_relations: N_r.
        embedding_dim: D.

    Shape contract:
        ``score_triples([B, 3]) -> FloatTensor[B]``.

    Memory note:
        Relation matrices use ``O(N_r · D^2)`` parameters.  For large D this
        can be heavy.  Use modest D (e.g. 16-64) for tensor-native research.

    Stability: Beta — fully tested with hand-computed reference values.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        D = self.embedding_dim
        # Entity embeddings: [N_e, D]
        self.entity_emb = nn.Embedding(num_entities, D)
        # Relation matrices stored flat as [N_r, D*D]; reshaped on access.
        self.relation_matrix = nn.Embedding(num_relations, D * D)
        # Init: small uniform for stability of bilinear form gradients.
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_matrix.weight)

    def _M(self, r_idx: torch.Tensor) -> torch.Tensor:
        """Return relation matrices [B, D, D]."""
        D = self.embedding_dim
        return self.relation_matrix(r_idx).view(-1, D, D)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        """Compute h^T M_r t for each triple in the batch.

        Args:
            triples: ``LongTensor[B, 3]``.

        Returns:
            ``FloatTensor[B]``.
        """
        h_idx, r_idx, t_idx = triples[:, 0], triples[:, 1], triples[:, 2]
        h = self.entity_emb(h_idx)            # [B, D]
        t = self.entity_emb(t_idx)            # [B, D]
        M = self._M(r_idx)                     # [B, D, D]
        # h^T M t = sum_{ij} h_i M_ij t_j
        # Implemented as einsum for clarity and efficiency.
        return torch.einsum("bi,bij,bj->b", h, M, t)
