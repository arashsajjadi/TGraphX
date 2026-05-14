"""Attention-based TGraphX candidate node selector ablations.

Three variants on top of the baseline TGraphXCandidateNodeSelector:

tgx_edge_attention:
    Edge-feature-conditioned attention over candidate neighbor nodes.
    Each node queries its neighbors using pairwise edge features as keys.
    Preserves crop tensor dimensions until final pool.

tgx_spatial_attention:
    Lightweight spatial attention over the [C, H, W] crop feature maps
    before pooling. Channel + spatial squeeze-excite gate.

tgx_hybrid_attention:
    CropCNN → spatial attention gate → ConvMP → edge attention →
    pool → metadata branch → concat → selection head.

All variants follow TGraphX identity:
- node_features is [N, C, H, W] through message passing
- Pool only after graph reasoning
- Edge features carry: IoU, center distance, area ratio, same_class,
  same_detector, score_diff, edge_type one-hot
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph

from .candidate_node_selector import CandidateSelectorConfig


# ── Spatial attention gate (channel + spatial SE-style) ──────────────────

class SpatialAttentionGate(nn.Module):
    """SE-style channel + spatial attention for [N, C, H, W] feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        r = max(1, channels // reduction)
        # Channel: global average pool → squeeze → excite
        self.channel_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, r), nn.ReLU(inplace=True),
            nn.Linear(r, channels), nn.Sigmoid(),
        )
        # Spatial: 1×1 conv over mean+max across channels
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        ch = self.channel_se(x).view(x.shape[0], x.shape[1], 1, 1)
        x = x * ch
        sp_mean = x.mean(dim=1, keepdim=True)
        sp_max  = x.max(dim=1, keepdim=True)[0]
        sp_gate = self.spatial(torch.cat([sp_mean, sp_max], dim=1))
        return x * sp_gate


# ── Edge-attention (flat vector, after crop encoding) ─────────────────────

class EdgeAttentionLayer(nn.Module):
    """Edge-feature-conditioned attention over neighbor embeddings.

    For each node i, computes attention weights over its neighbors using
    the pairwise edge feature vector as a key. No cross-cluster leakage.
    """

    def __init__(self, embed_dim: int, edge_feat_dim: int, num_heads: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        # Edge-conditioned attention score
        self.edge_proj = nn.Linear(edge_feat_dim, num_heads)
        # Value transform
        self.val_proj  = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """x: [N, D], edge_index: [2, E], edge_attr: [E, D_e] → [N, D]"""
        if edge_index is None or edge_index.numel() == 0:
            return x
        N, D = x.shape
        src, dst = edge_index[0], edge_index[1]  # src → dst

        # Edge attention scores [E, num_heads]
        raw_scores = self.edge_proj(edge_attr.float())     # [E, H]

        # Per-DESTINATION-NODE softmax (correct: each node normalises
        # over its incoming edges, not globally over all edges).
        attn_scores = torch.zeros_like(raw_scores)
        for node_dst in range(N):
            inc_mask = (dst == node_dst)
            if not inc_mask.any():
                continue
            attn_scores[inc_mask] = F.softmax(raw_scores[inc_mask], dim=0)

        # Value: neighbour embedding
        v = self.val_proj(x)                               # [N, D]

        # Aggregate: for each dst node, weighted sum of src values
        agg = torch.zeros_like(x)
        for h in range(self.num_heads):
            a_h  = attn_scores[:, h:h+1]                  # [E, 1]
            weighted = v[src] * a_h                        # [E, D]
            agg.index_add_(0, dst, weighted)

        out = F.relu(self.out_proj(agg))
        return self.norm(x + out)


# ── tgx_edge_attention model ──────────────────────────────────────────────

class TGXEdgeAttentionSelector(nn.Module):
    """TGraphX candidate selector with edge-feature-conditioned attention.

    Pipeline:
      CropCNN → ConvMP (preserves [C,H,W]) → SpatialPool
        → EdgeAttentionLayer (on flat embeddings) → heads
    """

    def __init__(self, cfg: CandidateSelectorConfig):
        super().__init__()
        self.cfg = cfg
        from .models import CropCNN
        from .source_router import EdgeConditionedMP
        self.crop_enc = CropCNN(3, cfg.crop_channels, cfg.crop_size)
        sp = self.crop_enc.out_spatial
        mp_layers = cfg.num_message_passing if cfg.use_message_passing else 0
        self.ec_layers = nn.ModuleList([
            EdgeConditionedMP(
                in_shape=(cfg.crop_channels, sp, sp),
                out_shape=(cfg.crop_channels, sp, sp),
                edge_feat_dim=cfg.edge_feat_dim,
            ) for _ in range(mp_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        md = cfg.metadata_dim if cfg.metadata_dim is not None else (8 + cfg.num_detectors + cfg.num_classes)
        self.meta_mlp = nn.Sequential(
            nn.Linear(md, cfg.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(cfg.crop_channels + cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True),
        )
        # Edge attention
        self.edge_attn = EdgeAttentionLayer(
            embed_dim=cfg.hidden_dim, edge_feat_dim=cfg.edge_feat_dim, num_heads=2)
        # Heads
        H = cfg.hidden_dim
        self.selection_head    = nn.Linear(H, 1)
        self.tp50_head         = nn.Linear(H, 1)
        self.tp75_head         = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)

    def forward(self, graph: Graph, detector_names: List[str]) -> Dict[str, Any]:
        x   = graph.node_features.float()
        ei  = graph.edge_index
        ea  = graph.edge_features
        md  = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        dev = x.device

        # ConvMP over crop tensors
        h = self.crop_enc(x)
        if ea is not None:
            ea_d = ea.to(dev)
        else:
            ea_d = None
        for layer in self.ec_layers:
            h = F.relu(layer(h, ei, ea_d)) + h

        v = self.pool(h).squeeze(-1).squeeze(-1)       # [N, C]
        if md is None:
            md = torch.zeros(x.shape[0], self.meta_mlp[0].in_features, device=dev)
        m   = self.meta_mlp(md.to(dev).float())
        emb = F.relu(self.node_proj(torch.cat([v, m], dim=1)))  # [N, H]

        # Edge attention on flat embeddings
        if ei is not None and ei.numel() > 0 and ea_d is not None and ea_d.numel() > 0:
            emb = self.edge_attn(emb, ei, ea_d)

        sel  = self.selection_head(emb).squeeze(-1)
        tp50 = self.tp50_head(emb).squeeze(-1)
        tp75 = self.tp75_head(emb).squeeze(-1)
        eiou = self.expected_iou_head(emb).squeeze(-1)
        return {"selection_logit": sel, "tp50_logit": tp50,
                "tp75_logit": tp75, "expected_iou_logit": eiou, "node_emb": emb}


# ── tgx_spatial_attention model ───────────────────────────────────────────

class TGXSpatialAttentionSelector(nn.Module):
    """TGraphX selector with spatial attention gate over crop feature maps.

    Spatial SE-style attention is applied BEFORE pooling, preserving
    TGraphX's [C,H,W] tensor identity during message passing.
    """

    def __init__(self, cfg: CandidateSelectorConfig):
        super().__init__()
        self.cfg = cfg
        from .models import CropCNN
        from .source_router import EdgeConditionedMP
        self.crop_enc  = CropCNN(3, cfg.crop_channels, cfg.crop_size)
        sp = self.crop_enc.out_spatial
        self.spatial_gate = SpatialAttentionGate(cfg.crop_channels, reduction=4)
        mp_layers = cfg.num_message_passing if cfg.use_message_passing else 0
        self.ec_layers = nn.ModuleList([
            EdgeConditionedMP(
                in_shape=(cfg.crop_channels, sp, sp),
                out_shape=(cfg.crop_channels, sp, sp),
                edge_feat_dim=cfg.edge_feat_dim,
            ) for _ in range(mp_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        md = cfg.metadata_dim if cfg.metadata_dim is not None else (8 + cfg.num_detectors + cfg.num_classes)
        self.meta_mlp = nn.Sequential(
            nn.Linear(md, cfg.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(cfg.crop_channels + cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True),
        )
        H = cfg.hidden_dim
        self.selection_head    = nn.Linear(H, 1)
        self.tp50_head         = nn.Linear(H, 1)
        self.tp75_head         = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)

    def forward(self, graph: Graph, detector_names: List[str]) -> Dict[str, Any]:
        x   = graph.node_features.float()
        ei  = graph.edge_index
        ea  = graph.edge_features
        md  = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        dev = x.device

        h = self.crop_enc(x)
        h = self.spatial_gate(h)        # spatial attention BEFORE convMP

        ea_d = ea.to(dev) if ea is not None else None
        for layer in self.ec_layers:
            h = F.relu(layer(h, ei, ea_d)) + h

        v = self.pool(h).squeeze(-1).squeeze(-1)
        if md is None:
            md = torch.zeros(x.shape[0], self.meta_mlp[0].in_features, device=dev)
        m   = self.meta_mlp(md.to(dev).float())
        emb = F.relu(self.node_proj(torch.cat([v, m], dim=1)))

        return {
            "selection_logit":    self.selection_head(emb).squeeze(-1),
            "tp50_logit":         self.tp50_head(emb).squeeze(-1),
            "tp75_logit":         self.tp75_head(emb).squeeze(-1),
            "expected_iou_logit": self.expected_iou_head(emb).squeeze(-1),
            "node_emb":           emb,
        }


# ── tgx_hybrid_attention model ────────────────────────────────────────────

class TGXHybridAttentionSelector(nn.Module):
    """Full hybrid: spatial attention + ConvMP + edge attention + metadata."""

    def __init__(self, cfg: CandidateSelectorConfig):
        super().__init__()
        self.cfg = cfg
        from .models import CropCNN
        from .source_router import EdgeConditionedMP
        self.crop_enc     = CropCNN(3, cfg.crop_channels, cfg.crop_size)
        sp = self.crop_enc.out_spatial
        self.spatial_gate = SpatialAttentionGate(cfg.crop_channels, reduction=4)
        mp_layers = cfg.num_message_passing if cfg.use_message_passing else 0
        self.ec_layers    = nn.ModuleList([
            EdgeConditionedMP(
                in_shape=(cfg.crop_channels, sp, sp),
                out_shape=(cfg.crop_channels, sp, sp),
                edge_feat_dim=cfg.edge_feat_dim,
            ) for _ in range(mp_layers)
        ])
        self.pool         = nn.AdaptiveAvgPool2d(1)
        md = cfg.metadata_dim if cfg.metadata_dim is not None else (8 + cfg.num_detectors + cfg.num_classes)
        self.meta_mlp     = nn.Sequential(
            nn.Linear(md, cfg.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.node_proj    = nn.Sequential(
            nn.Linear(cfg.crop_channels + cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True),
        )
        self.edge_attn    = EdgeAttentionLayer(
            embed_dim=cfg.hidden_dim, edge_feat_dim=cfg.edge_feat_dim, num_heads=2)
        H = cfg.hidden_dim
        self.dropout       = nn.Dropout(p=0.10)
        self.selection_head    = nn.Linear(H, 1)
        self.tp50_head         = nn.Linear(H, 1)
        self.tp75_head         = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)

    def forward(self, graph: Graph, detector_names: List[str]) -> Dict[str, Any]:
        x   = graph.node_features.float()
        ei  = graph.edge_index
        ea  = graph.edge_features
        md  = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        dev = x.device

        h = self.crop_enc(x)
        h = self.spatial_gate(h)               # spatial attention
        ea_d = ea.to(dev) if ea is not None else None
        for layer in self.ec_layers:
            h = F.relu(layer(h, ei, ea_d)) + h  # ConvMP

        v = self.pool(h).squeeze(-1).squeeze(-1)
        if md is None:
            md = torch.zeros(x.shape[0], self.meta_mlp[0].in_features, device=dev)
        m   = self.meta_mlp(md.to(dev).float())
        emb = F.relu(self.node_proj(torch.cat([v, m], dim=1)))

        if ei is not None and ei.numel() > 0 and ea_d is not None and ea_d.numel() > 0:
            emb = self.edge_attn(emb, ei, ea_d)  # edge attention (flat)

        emb = self.dropout(emb)

        return {
            "selection_logit":    self.selection_head(emb).squeeze(-1),
            "tp50_logit":         self.tp50_head(emb).squeeze(-1),
            "tp75_logit":         self.tp75_head(emb).squeeze(-1),
            "expected_iou_logit": self.expected_iou_head(emb).squeeze(-1),
            "node_emb":           emb,
        }


# ── Factory ───────────────────────────────────────────────────────────────

def build_selector(cfg: CandidateSelectorConfig, variant: str) -> nn.Module:
    """Build a candidate-node selector by variant name.

    Variants:
      crop_metadata_mp    — TGraphXCandidateNodeSelector (full V3 encoder)
      flat_crop_mp        — flatten first, then flat-vector GNN
      crop_no_mp          — CNN + pool, no message passing
      metadata_only       — no crops
      tgx_convmp_small    — crop_metadata_mp with 1 MP layer, 8 channels
      tgx_convmp_full     — crop_metadata_mp with 2 MP layers, 16 channels (default)
      tgx_edge_attention  — ConvMP + edge attention
      tgx_spatial_attention — spatial attention gate + ConvMP
      tgx_hybrid_attention  — spatial + ConvMP + edge attention
    """
    if variant in ("crop_metadata_mp", "flat_crop_mp", "crop_no_mp", "metadata_only"):
        from .candidate_node_selector import TGraphXCandidateNodeSelector
        from dataclasses import replace
        cfg2 = CandidateSelectorConfig(
            num_classes=cfg.num_classes, num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size, crop_channels=cfg.crop_channels,
            hidden_dim=cfg.hidden_dim, metadata_dim=cfg.metadata_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            num_message_passing=cfg.num_message_passing,
            use_message_passing=cfg.use_message_passing,
            use_metadata=cfg.use_metadata,
            feature_mode=variant,
        )
        return TGraphXCandidateNodeSelector(cfg2)
    elif variant == "tgx_convmp_small":
        from .candidate_node_selector import TGraphXCandidateNodeSelector
        small_cfg = CandidateSelectorConfig(
            num_classes=cfg.num_classes, num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size, crop_channels=8, hidden_dim=48,
            metadata_dim=cfg.metadata_dim, edge_feat_dim=cfg.edge_feat_dim,
            num_message_passing=1,
            feature_mode="crop_metadata_mp",
        )
        return TGraphXCandidateNodeSelector(small_cfg)
    elif variant == "tgx_convmp_full":
        from .candidate_node_selector import TGraphXCandidateNodeSelector
        cfg2 = CandidateSelectorConfig(
            num_classes=cfg.num_classes, num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size, crop_channels=cfg.crop_channels,
            hidden_dim=cfg.hidden_dim, metadata_dim=cfg.metadata_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            num_message_passing=cfg.num_message_passing,
            feature_mode="crop_metadata_mp",
        )
        return TGraphXCandidateNodeSelector(cfg2)
    elif variant == "tgx_edge_attention":
        return TGXEdgeAttentionSelector(cfg)
    elif variant == "tgx_spatial_attention":
        return TGXSpatialAttentionSelector(cfg)
    elif variant == "tgx_hybrid_attention":
        return TGXHybridAttentionSelector(cfg)
    else:
        raise ValueError(f"Unknown selector variant: {variant!r}")
