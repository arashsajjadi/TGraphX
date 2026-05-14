"""TGXPointerSelector — regularized cross-attention candidate node selector.

The fundamental problem: given N candidate boxes for ONE object cluster,
select the best candidate. This is a SET SELECTION problem.

The correct inductive bias: CROSS-ATTENTION (self-attention over the N nodes).
Each candidate can attend to all other candidates, learning "which neighbors
agree with me, and which ones score higher / have better boxes?"

Architecture:
    per-node encoding:
        small CropCNN(3→crop_ch, crop_size) → AdaptiveAvgPool → [crop_ch]
        concat with metadata_mlp(metadata) → [hidden]
        LayerNorm(Linear(crop_ch + md_out, hidden)) → token [hidden]

    K layers of multi-head self-attention over N tokens:
        attn_out = MHA(Q=X, K=X, V=X, attn_dropout)
        X = LayerNorm(X + dropout(attn_out))
        X = LayerNorm(X + dropout(FFN(X, dropout)))

    per-node heads:
        selection_logit = Linear(hidden, 1)
        tp75_logit      = Linear(hidden, 1)
        eiou_logit      = Linear(hidden, 1)

Key design decisions:
  - crop_size=32 (not 128): fewer parameters, less overfitting
  - hidden=32 (not 64): compact, strong regularization
  - 1-2 attention layers: enough depth for 7-12 nodes
  - dropout=0.15 everywhere
  - LayerNorm: training stability
  - No ConvMP: self-attention IS message passing for small sets
  - TGraphX-distinctive: uses [C,H,W] crop tensors + graph structure

This model should be compared against:
  flat_crop_mp:   pool-first + mean-aggregation (weaker inductive bias)
  metadata_only:  no visual information (no crop encoding)
  crop_metadata_mp: full V3 spatial ConvMP (overfits)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph

from .candidate_node_selector import select_per_cluster
from .candidate_mask import candidate_node_mask
from .graph_builder import NODE_TYPES


# ── Crop augmentation (training only) ────────────────────────────────────────

def augment_crops(
    crops: torch.Tensor,
    *,
    rng: Optional[torch.Generator] = None,
    flip_prob: float = 0.5,
    brightness_range: float = 0.20,
    noise_std: float = 0.015,
) -> torch.Tensor:
    """Random augmentation for a batch of crop tensors [N, 3, H, W].

    Applies horizontal flip, brightness jitter, and mild Gaussian noise.
    All operations are seedable via `rng`.
    """
    N = crops.shape[0]
    out = crops.clone()

    # Per-image horizontal flip
    flip_mask = torch.rand(N, generator=rng) < flip_prob
    if flip_mask.any():
        out[flip_mask] = out[flip_mask].flip(-1)

    # Per-image brightness jitter (applied per channel)
    scale = 1.0 + (torch.rand(N, 3, 1, 1, generator=rng) * 2 - 1) * brightness_range
    out = (out * scale).clamp(0.0, 1.0)

    # Mild additive Gaussian noise (use rng for determinism)
    if noise_std > 0:
        noise = torch.zeros_like(out).normal_(0, noise_std, generator=rng)
        out = (out + noise).clamp(0.0, 1.0)

    return out


# ── Source-type embedding ─────────────────────────────────────────────────────

_SOURCE_TYPE_MAP = {
    "proposal":                0,
    "cluster":                 1,
    "consensus":               2,
    "nms_candidate":           3,
    "soft_nms_candidate":      4,
    "best_proposal_candidate": 5,
}
NUM_SOURCE_TYPES = 6


class SourceTypeEmbedding(nn.Module):
    """Learnable embedding for the node type (detector vs fusion type)."""

    def __init__(self, num_types: int = NUM_SOURCE_TYPES, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_types, embed_dim, padding_idx=None)
        self.embed_dim = embed_dim

    def forward(self, node_types: torch.Tensor) -> torch.Tensor:
        """Map node type IDs (from NODE_TYPES) → embedding [N, embed_dim]."""
        # Map from graph NODE_TYPES values to local source type index
        src_idx = torch.zeros(node_types.shape[0], dtype=torch.long, device=node_types.device)
        for name, local_idx in _SOURCE_TYPE_MAP.items():
            if name in NODE_TYPES:
                mask = node_types == NODE_TYPES[name]
                src_idx[mask] = local_idx
        return self.embed(src_idx)


# ── Transformer feed-forward block ───────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── TGXPointerSelector ────────────────────────────────────────────────────────

@dataclass
class PointerSelectorConfig:
    num_classes: int
    num_detectors: int
    crop_size: int = 32           # smaller crop → fewer params, less overfitting
    crop_channels: int = 8        # compact visual encoding
    hidden_dim: int = 32          # small model, generalizes better
    metadata_dim: Optional[int] = None
    num_attn_layers: int = 2      # 2 self-attention layers over N=7-12 nodes
    num_heads: int = 2            # multi-head attention
    ffn_expansion: int = 2        # FFN inner dim = hidden_dim * expansion
    dropout: float = 0.15         # attention + FFN dropout
    use_crops: bool = True        # set to False for metadata_only ablation
    source_type_embed_dim: int = 8  # learnable source-type embedding


class TGXPointerSelector(nn.Module):
    """Self-attention-based TGraphX candidate node selector.

    Treats the N candidate nodes for one object as a sequence and applies
    multi-head self-attention (cross-candidate comparison). This is the
    correct inductive bias for the node-selection problem:
    each candidate learns "which other candidates agree / disagree with me".

    This is TGraphX-distinctive because:
    - node_features are [N, 3, crop_size, crop_size] — tensor-valued
    - graph structure (N nodes, all candidates for ONE object) is preserved
    - self-attention = learnable graph message passing for small sets
    - final selection: argmax(selection_logit) → selected node → selected box
    """

    def __init__(self, cfg: PointerSelectorConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        # ── Node encoder ─────────────────────────────────────────────────
        if cfg.use_crops:
            from .models import CropCNN
            self.crop_enc = CropCNN(3, cfg.crop_channels, cfg.crop_size)
            crop_flat = cfg.crop_channels
        else:
            self.crop_enc = None
            crop_flat = 0
        self.pool = nn.AdaptiveAvgPool2d(1)

        md_dim = (cfg.metadata_dim if cfg.metadata_dim is not None
                  else 8 + cfg.num_detectors + cfg.num_classes)
        md_out = H
        self.meta_mlp = nn.Sequential(
            nn.Linear(md_dim, H), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(H, md_out),
        )

        # Source-type learnable embedding (detector ID / fusion type)
        self.src_embed = SourceTypeEmbedding(NUM_SOURCE_TYPES, cfg.source_type_embed_dim)

        # Project concat(crop_flat, md_out, src_embed_dim) → H
        in_dim = crop_flat + md_out + cfg.source_type_embed_dim
        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, H), nn.GELU(), nn.LayerNorm(H),
        )

        # ── Self-attention layers ─────────────────────────────────────────
        self.attn_layers = nn.ModuleList()
        self.attn_norms  = nn.ModuleList()
        self.ffn_layers  = nn.ModuleList()
        self.ffn_norms   = nn.ModuleList()
        for _ in range(cfg.num_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(H, cfg.num_heads,
                                      dropout=cfg.dropout, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(H))
            self.ffn_layers.append(FFN(H, cfg.ffn_expansion, cfg.dropout))
            self.ffn_norms.append(nn.LayerNorm(H))

        self.out_drop = nn.Dropout(cfg.dropout)

        # ── Per-node prediction heads ─────────────────────────────────────
        self.selection_head    = nn.Linear(H, 1)
        self.tp50_head         = nn.Linear(H, 1)
        self.tp75_head         = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)

    def encode_nodes(self, graph: Graph) -> torch.Tensor:
        """Encode all nodes → sequence [N, H] for attention."""
        dev = graph.node_features.device
        x  = graph.node_features.float()    # [N, 3, H, W]
        md = (graph.metadata.get("node_metadata")
              if isinstance(graph.metadata, dict) else None)
        nt = (graph.metadata.get("node_types")
              if isinstance(graph.metadata, dict) else None)

        parts = []

        # Visual crop encoding
        if self.crop_enc is not None:
            h  = self.crop_enc(x)                             # [N, C, s, s]
            v  = self.pool(h).squeeze(-1).squeeze(-1)         # [N, C]
            parts.append(v)

        # Metadata encoding
        if md is None:
            md = torch.zeros(x.shape[0], self.meta_mlp[0].in_features,
                             device=dev, dtype=x.dtype)
        m = self.meta_mlp(md.to(dev).float())                 # [N, H]
        parts.append(m)

        # Source-type embedding
        if nt is not None:
            se = self.src_embed(nt.to(dev))                   # [N, src_dim]
        else:
            se = self.src_embed.embed.weight[0].expand(x.shape[0], -1)
        parts.append(se)

        emb = self.node_proj(torch.cat(parts, dim=1))         # [N, H]
        return emb

    def forward(self, graph: Graph, detector_names: List[str]) -> Dict[str, Any]:
        emb = self.encode_nodes(graph)                        # [N, H]

        # Self-attention: treat N candidates as a sequence (batch=1)
        seq = emb.unsqueeze(0)                                # [1, N, H]
        for attn, anorm, ffn, fnorm in zip(
                self.attn_layers, self.attn_norms,
                self.ffn_layers, self.ffn_norms):
            # Pre-norm attention
            attn_out, _ = attn(seq, seq, seq)
            seq = anorm(seq + self.out_drop(attn_out))
            # Pre-norm FFN
            seq = fnorm(seq + self.out_drop(ffn(seq)))

        emb = seq.squeeze(0)                                  # [N, H]

        sel  = self.selection_head(emb).squeeze(-1)           # [N]
        tp50 = self.tp50_head(emb).squeeze(-1)
        tp75 = self.tp75_head(emb).squeeze(-1)
        eiou = self.expected_iou_head(emb).squeeze(-1)

        return {
            "selection_logit":    sel,
            "tp50_logit":         tp50,
            "tp75_logit":         tp75,
            "expected_iou_logit": eiou,
            "node_emb":           emb,
        }


# ── Metadata-only variant (same interface) ───────────────────────────────────

class TGXMetaOnlyPointer(TGXPointerSelector):
    """TGXPointerSelector with use_crops=False (metadata + source embedding only)."""

    def __init__(self, cfg: PointerSelectorConfig):
        cfg_no_crop = PointerSelectorConfig(
            num_classes=cfg.num_classes, num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size, crop_channels=cfg.crop_channels,
            hidden_dim=cfg.hidden_dim, metadata_dim=cfg.metadata_dim,
            num_attn_layers=cfg.num_attn_layers, num_heads=cfg.num_heads,
            ffn_expansion=cfg.ffn_expansion, dropout=cfg.dropout,
            use_crops=False,
            source_type_embed_dim=cfg.source_type_embed_dim,
        )
        super().__init__(cfg_no_crop)


# ── Training loop helpers ─────────────────────────────────────────────────────

def pointer_loss(
    out: Dict[str, torch.Tensor],
    *,
    best_node: torch.Tensor,              # scalar long: index of best candidate
    node_iou:  torch.Tensor,              # [N] float IoU with GT
    cls_ok:    torch.Tensor,              # [N] bool class correct
    cand_mask: torch.Tensor,              # [N] bool candidate eligibility
    w_sel: float = 1.0,
    w_tp75: float = 2.0,
    w_tp50: float = 0.5,
    w_iou:  float = 0.5,
    w_rank: float = 0.5,
    label_smooth: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Loss for one object graph (one cluster).

    Simpler than candidate_selector_loss: no cluster loop needed because
    each graph IS one cluster.
    """
    dev  = out["selection_logit"].device
    sel  = out["selection_logit"]    # [N]
    tp50 = out["tp50_logit"]
    tp75 = out["tp75_logit"]
    eiou = out["expected_iou_logit"]
    N    = sel.shape[0]

    if N == 0 or int(best_node.item()) < 0:
        z = torch.tensor(0.0, device=dev, requires_grad=True)
        return {"total": z, "sel": z, "tp50": z, "tp75": z, "iou": z, "rank": z}

    # ── Selection CE (with label smoothing) ──────────────────────────────
    # Only over candidate nodes
    cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)
    sel_cand = sel[cand_idx]
    # Find local index of best_node within cand_idx
    local_best = (cand_idx == int(best_node.item())).nonzero(as_tuple=False)
    if local_best.numel() == 0:
        z = torch.tensor(0.0, device=dev, requires_grad=True)
        return {"total": z, "sel": z, "tp50": z, "tp75": z, "iou": z, "rank": z}
    local_best_idx = int(local_best[0].item())
    K = sel_cand.shape[0]

    # Smooth label: target = (1 - smooth) at best, smooth/(K-1) elsewhere
    if K > 1 and label_smooth > 0:
        target_soft = torch.full((K,), label_smooth / (K - 1), device=dev)
        target_soft[local_best_idx] = 1.0 - label_smooth
        L_sel = F.kl_div(F.log_softmax(sel_cand, dim=0),
                          target_soft, reduction="sum")
    else:
        L_sel = F.cross_entropy(
            sel_cand.unsqueeze(0),
            torch.tensor([local_best_idx], device=dev))

    # ── TP50 / TP75 BCE ───────────────────────────────────────────────────
    iou  = node_iou[cand_idx]
    clsc = cls_ok[cand_idx].float()
    tp50t = ((iou >= 0.50) & (clsc > 0.5)).float()
    tp75t = ((iou >= 0.75) & (clsc > 0.5)).float()
    L_tp50 = F.binary_cross_entropy_with_logits(tp50[cand_idx], tp50t)
    L_tp75 = F.binary_cross_entropy_with_logits(tp75[cand_idx], tp75t)

    # ── IoU regression ───────────────────────────────────────────────────
    L_iou = F.smooth_l1_loss(torch.sigmoid(eiou[cand_idx]), iou)

    # ── Pairwise ranking (best vs. sampled negatives only) ────────────────
    # Sample up to 4 "other" nodes and enforce best > others
    L_rank = torch.tensor(0.0, device=dev)
    if K > 1:
        others = [i for i in range(K) if i != local_best_idx]
        # Sample at most 4 negatives to keep it O(1) not O(K²)
        if len(others) > 4:
            others = others[:4]  # deterministic, no randomness needed
        best_logit = sel_cand[local_best_idx]
        for neg_i in others:
            L_rank = L_rank + F.softplus(sel_cand[neg_i] - best_logit)
        L_rank = L_rank / max(1, len(others))

    total = (w_sel * L_sel + w_tp50 * L_tp50 + w_tp75 * L_tp75
             + w_iou * L_iou + w_rank * L_rank)
    return {"total": total, "sel": L_sel, "tp50": L_tp50,
            "tp75": L_tp75, "iou": L_iou, "rank": L_rank}
