"""Multimodal entity/relation projectors for tensor-aware KG learning.

Each projector maps raw tensor features of a specific modality into a
common ``out_dim``-dimensional embedding space.  All projectors are:
  - Differentiable (gradients flow through to input features and parameters).
  - Explicit about their input shape contract (no silent flattening).
  - Optional: the caller decides which modalities are present.

Available projectors:
  VectorEntityProjector   — ``[N, F]`` → ``[N, out_dim]``
  ImageEntityProjector    — ``[N, C, H, W]`` → ``[N, out_dim]``  via avgpool+linear
  TextEntityProjector     — ``[N, D]`` → ``[N, out_dim]``  (pre-computed embeddings)
  RelationFeatureProjector— ``[N_r, F]`` → ``[N_r, out_dim]``
  TripleFeatureProjector  — ``[N_t, F]`` → ``[N_t, out_dim]``

  MultimodalEntityFusion  — fuses multiple modality projections into one
                             entity representation, with a learned fallback
                             embedding for entities without a given modality.

Fusion modes:
  ``"add"``            — sum projected modalities
  ``"concat_project"`` — concatenate, then project down
  ``"gated"``          — sigmoid gate per modality

Stability: Experimental (v0.6.0+).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "VectorEntityProjector",
    "ImageEntityProjector",
    "TextEntityProjector",
    "RelationFeatureProjector",
    "TripleFeatureProjector",
    "MultimodalEntityFusion",
]


# ── Base ──────────────────────────────────────────────────────────────────────


class _BaseProjector(nn.Module):
    """Common interface: ``forward(feat) -> FloatTensor[N, out_dim]``."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ── Vector projector ─────────────────────────────────────────────────────────


class VectorEntityProjector(_BaseProjector):
    """Project ``[N, in_dim]`` vector features → ``[N, out_dim]``.

    Uses a 1-layer MLP with optional ReLU.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output embedding dimension.
        activation: If ``True`` (default), applies ReLU.

    Stability: Experimental.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
        super().__init__(out_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self._act = activation
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 2:
            raise ValueError(
                f"VectorEntityProjector expects 2-D input [N, F]; got {tuple(feat.shape)}"
            )
        out = self.proj(feat.float())
        return F.relu(out) if self._act else out


# Alias.
TextEntityProjector = VectorEntityProjector


# ── Image projector ───────────────────────────────────────────────────────────


class ImageEntityProjector(_BaseProjector):
    """Project image-like entity features ``[N, C, H, W]`` → ``[N, out_dim]``.

    Uses global average pooling across spatial dims, then a linear projection.
    This is lightweight and does NOT require a full CNN — it is appropriate
    when the spatial structure carries global statistics.

    For richer spatial representations, use a full CNN encoder as a
    pre-processing step before passing features to this projector.

    Args:
        in_channels: ``C`` from input shape ``[N, C, H, W]``.
        out_dim: Output dimension.
        activation: If ``True`` (default), applies ReLU after projection.

    Stability: Experimental.
    """

    def __init__(self, in_channels: int, out_dim: int, activation: bool = True) -> None:
        super().__init__(out_dim)
        self.proj = nn.Linear(int(in_channels), out_dim)
        self._act = activation
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError(
                f"ImageEntityProjector expects 4-D input [N, C, H, W]; "
                f"got {tuple(feat.shape)}.  Pass image features directly — "
                f"they are NOT silently flattened."
            )
        # Global average pool: [N, C, H, W] → [N, C]
        pooled = feat.float().mean(dim=[2, 3])
        out = self.proj(pooled)
        return F.relu(out) if self._act else out


# ── User (alias) projector ────────────────────────────────────────────────────

# User/profile entities are typically vector features; reuse VectorEntityProjector.
# A distinct class makes the intent explicit.


class UserEntityProjector(VectorEntityProjector):
    """Alias of VectorEntityProjector for user/profile entity features.

    Accepts ``[N, F]`` user feature vectors and projects to ``[N, out_dim]``.
    """
    pass


# ── Relation feature projector ────────────────────────────────────────────────


class RelationFeatureProjector(_BaseProjector):
    """Project ``[N_r, F]`` relation features → ``[N_r, out_dim]``.

    Stability: Experimental.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
        super().__init__(out_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self._act = activation
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 2:
            raise ValueError(
                f"RelationFeatureProjector expects 2-D input [N_r, F]; got {tuple(feat.shape)}"
            )
        out = self.proj(feat.float())
        return F.relu(out) if self._act else out


# ── Triple feature projector ──────────────────────────────────────────────────


class TripleFeatureProjector(_BaseProjector):
    """Project ``[N_t, F]`` triple/edge features → ``[N_t, out_dim]``.

    Stability: Experimental.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
        super().__init__(out_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self._act = activation
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 2:
            raise ValueError(
                f"TripleFeatureProjector expects 2-D input [N_t, F]; got {tuple(feat.shape)}"
            )
        out = self.proj(feat.float())
        return F.relu(out) if self._act else out


# ── Multimodal entity fusion ──────────────────────────────────────────────────


class MultimodalEntityFusion(nn.Module):
    """Fuse multiple modality projections into a single entity embedding.

    For each entity:
      - Available modalities are projected via their respective projectors.
      - Unavailable modalities are replaced by a learned fallback embedding.
      - Projected representations are combined via ``fusion_mode``.

    All outputs are ``FloatTensor[N_e, out_dim]``.

    Args:
        projectors: ``{modality_name: _BaseProjector}`` mapping.
        out_dim: Common output dimension (all projectors must produce this dim).
        num_entities: Number of entities (for fallback embeddings).
        fusion_mode: ``"add"`` | ``"concat_project"`` | ``"gated"``.
        add_learnable_bias: If True, adds a learnable entity embedding to every output.

    Contract:
        - ``forward(features, masks)`` where ``features`` is the ``entity_features``
          dict from :class:`KnowledgeGraph` and ``masks`` is ``entity_feature_masks``.
        - For modalities not listed in ``masks``, all entities are assumed to
          have the feature (full coverage).
        - Tensors are NOT silently flattened — image features must be 4-D.

    Stability: Experimental.
    """

    def __init__(
        self,
        projectors: Dict[str, _BaseProjector],
        out_dim: int,
        num_entities: int,
        fusion_mode: str = "add",
        add_learnable_bias: bool = True,
    ) -> None:
        super().__init__()
        valid_modes = ("add", "concat_project", "gated")
        if fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {valid_modes}; got {fusion_mode!r}")
        self.out_dim = int(out_dim)
        self.num_entities = int(num_entities)
        self.fusion_mode = fusion_mode
        self.projectors = nn.ModuleDict({k: v for k, v in projectors.items()})
        # Fallback learnable embedding (used for entities without any feature).
        if add_learnable_bias:
            self.fallback_emb = nn.Embedding(num_entities, out_dim)
            nn.init.xavier_uniform_(self.fallback_emb.weight)
        else:
            self.fallback_emb = None
        # Gated fusion: per-modality gate.
        if fusion_mode == "gated":
            self.gates = nn.ParameterDict({
                k: nn.Parameter(torch.ones(1)) for k in projectors
            })
        else:
            self.gates = None
        # Concat-project: downproject concatenated representations.
        if fusion_mode == "concat_project":
            n_mods = len(projectors)
            self.concat_proj = nn.Linear(out_dim * n_mods, out_dim)
            nn.init.xavier_uniform_(self.concat_proj.weight)
            nn.init.zeros_(self.concat_proj.bias)
        else:
            self.concat_proj = None

    def forward(
        self,
        entity_features: Dict[str, torch.Tensor],
        entity_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
        entity_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute entity embeddings from multimodal features.

        Args:
            entity_features: ``{name: Tensor[N_e, *]}`` feature dict from KG.
            entity_feature_masks: ``{name: BoolTensor[N_e]}`` coverage masks.
                When a modality has no mask, full coverage is assumed.
            entity_ids: Optional ``LongTensor[N_e]`` for fallback embedding lookup.
                When ``None``, uses ``arange(N_e)``.

        Returns:
            ``FloatTensor[N_e, out_dim]``.
        """
        device = next(iter(entity_features.values())).device if entity_features else (
            self.fallback_emb.weight.device if self.fallback_emb is not None else torch.device("cpu")
        )
        N = self.num_entities
        masks = entity_feature_masks or {}

        projected: List[torch.Tensor] = []
        for name, projector in self.projectors.items():
            if name not in entity_features:
                # No feature for this modality: use zero projection.
                proj_out = torch.zeros(N, self.out_dim, dtype=torch.float, device=device)
            else:
                feat = entity_features[name].to(device)
                proj_out = projector(feat)  # [N_e, out_dim]
            # Apply mask: for entities without this modality, zero out their contribution.
            if name in masks:
                mask = masks[name].to(device).float().view(-1, 1)  # [N_e, 1]
                proj_out = proj_out * mask
            projected.append(proj_out)

        if not projected:
            # No projectors configured; return fallback only.
            if self.fallback_emb is not None:
                ids = entity_ids if entity_ids is not None else torch.arange(N, device=device)
                return self.fallback_emb(ids)
            return torch.zeros(N, self.out_dim, device=device)

        if self.fusion_mode == "add":
            out = sum(projected)
        elif self.fusion_mode == "gated":
            out = torch.zeros(N, self.out_dim, dtype=torch.float, device=device)
            for name, proj_out in zip(self.projectors.keys(), projected):
                gate = torch.sigmoid(self.gates[name])  # scalar
                out = out + gate * proj_out
        else:  # concat_project
            out = self.concat_proj(torch.cat(projected, dim=-1))

        # Add fallback bias.
        if self.fallback_emb is not None:
            ids = entity_ids if entity_ids is not None else torch.arange(N, device=device)
            out = out + self.fallback_emb(ids)

        return out
