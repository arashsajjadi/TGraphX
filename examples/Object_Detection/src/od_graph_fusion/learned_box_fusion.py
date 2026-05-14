"""TGraphXLearnedBoxFusion — learned box refinement on top of WBF.

Replaces the source-router head with a regression head that predicts a
bounded residual Δ to the WBF box per cluster. The capped residual oracle
captures +0.26 AP75 / +0.05 mIoU on the real-VOC-car test split with
‖Δ‖∞ ≤ 0.1·diag(WBF), so a model that can learn this residual closes a
material AP75 gap that no source-router can.

Re-uses TGraphXSourceRouterV3's encoder (CropCNN + edge-conditioned
message passing + slot aggregator + slot attention) so we keep all of
the graph-level signal that was built up over previous phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph

from .source_router_v3 import (
    NUM_SOURCES, SOURCE_SLOTS, detector_name_to_slot,
    TGraphXSourceRouterV3,
)
from .pairwise_features import PAIRWISE_FEAT_DIM


# Anchor preference order: WBF → NMS → BestProposal → rt_detr → first available.
ANCHOR_PREFERENCE = [
    SOURCE_SLOTS["wbf"],
    SOURCE_SLOTS["nms_candidate"],
    SOURCE_SLOTS["best_proposal"],
    SOURCE_SLOTS["rt_detr"],
    SOURCE_SLOTS["retinanet"],
    SOURCE_SLOTS["yolo_modern"],
    SOURCE_SLOTS["union"],
]


@dataclass
class LearnedBoxFusionConfig:
    num_classes: int
    num_detectors: int
    crop_size: int
    crop_channels: int = 16
    hidden_dim: int = 64
    metadata_dim: Optional[int] = None
    edge_feat_dim: int = 14
    num_message_passing: int = 2
    num_sources: int = NUM_SOURCES
    fusion_mode: str = "residual"          # one of "residual" | "weighted" | "hybrid"
    delta_cap_frac: float = 0.1            # ‖Δ‖∞ ≤ cap_frac · diag(anchor_box)
    pair_feat_dim: int = PAIRWISE_FEAT_DIM
    source_id_emb_dim: int = 16


class TGraphXLearnedBoxFusion(nn.Module):
    """Learned box refinement over WBF.

    forward returns per cluster:
      final_box_xyxy:      [C, 4] refined box (one per cluster)
      anchor_box_xyxy:     [C, 4] WBF (or fallback) box used as anchor
      delta_box:           [C, 4] predicted Δ (already capped)
      tp50_logit:          [C]
      tp75_logit:          [C]
      expected_iou_logit:  [C]   sigmoid → predicted IoU(final, GT)
      source_weight_logits:[C, S] (only used in fusion_mode in {"weighted","hybrid"})
      anchor_slot:         [C] long — slot used as anchor
      cluster_score:       [C]   — max source confidence per cluster (used to score predictions)
    """

    def __init__(self, cfg: LearnedBoxFusionConfig):
        super().__init__()
        self.cfg = cfg
        self._v3 = TGraphXSourceRouterV3(
            num_classes=cfg.num_classes,
            num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size,
            crop_channels=cfg.crop_channels,
            hidden_dim=cfg.hidden_dim,
            metadata_dim=cfg.metadata_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            num_message_passing=cfg.num_message_passing,
            num_sources=cfg.num_sources,
        )
        H = cfg.hidden_dim
        S = cfg.num_sources
        self.source_id_emb = nn.Embedding(S, cfg.source_id_emb_dim)
        # Per-cluster head input = anchor_slot_emb + mean(slot_embs) + pair_feats_mean
        head_in = H + H + cfg.pair_feat_dim + cfg.source_id_emb_dim
        self.head_trunk = nn.Sequential(
            nn.Linear(head_in, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
        )
        self.delta_head = nn.Linear(H, 4)
        self.tp50_head = nn.Linear(H, 1)
        self.tp75_head = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)
        # Source weight head (per-slot scalar) for weighted-fusion mode.
        # Outputs masked logits over S that we softmax over available slots.
        self.weight_head = nn.Linear(H + cfg.source_id_emb_dim, 1)

    def _pick_anchor_slot_per_cluster(self, slot_mask: torch.Tensor) -> torch.Tensor:
        """For each cluster pick the first preference slot that is available."""
        C, S = slot_mask.shape
        device = slot_mask.device
        out = torch.full((C,), -1, dtype=torch.long, device=device)
        for slot in ANCHOR_PREFERENCE:
            if slot >= S:
                continue
            available = slot_mask[:, slot] & (out < 0)
            out[available] = slot
        # Anything still -1 → first available slot
        for c in range(C):
            if int(out[c].item()) < 0:
                avail = slot_mask[c].nonzero(as_tuple=False).squeeze(-1)
                if avail.numel() > 0:
                    out[c] = int(avail[0].item())
        return out

    def forward(
        self,
        graph: Graph,
        detector_names: List[str],
        *,
        pairwise_feats: Optional[torch.Tensor] = None,    # [C, S, PAIRWISE_FEAT_DIM]
    ) -> Dict[str, Any]:
        device = graph.node_features.device
        v3_out = self._v3(graph, detector_names=detector_names)
        node_emb = v3_out["node_emb"]
        slot_assignments = v3_out["slot_assignments"]
        cluster_of = graph.metadata.get("cluster_of_raw") if isinstance(graph.metadata, dict) else None
        if cluster_of is None or cluster_of.numel() == 0:
            empty = {
                "final_box_xyxy": torch.zeros(0, 4, device=device),
                "anchor_box_xyxy": torch.zeros(0, 4, device=device),
                "delta_box": torch.zeros(0, 4, device=device),
                "tp50_logit": torch.zeros(0, device=device),
                "tp75_logit": torch.zeros(0, device=device),
                "expected_iou_logit": torch.zeros(0, device=device),
                "source_weight_logits": torch.zeros(0, self.cfg.num_sources, device=device),
                "anchor_slot": torch.zeros(0, dtype=torch.long, device=device),
                "cluster_score": torch.zeros(0, device=device),
                "slot_mask": torch.zeros(0, self.cfg.num_sources, dtype=torch.bool, device=device),
                "slot_node_idx": torch.full((0, self.cfg.num_sources), -1, dtype=torch.long, device=device),
            }
            return empty
        cluster_of = cluster_of.to(device)
        n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0

        # Reaggregate slot embeddings + slot mask + slot_node_idx.
        node_score_t = (graph.metadata.get("node_score").to(device)
                         if isinstance(graph.metadata, dict)
                         and graph.metadata.get("node_score") is not None else None)
        slot_emb, slot_mask, slot_node_idx = self._v3.slot_agg(
            node_emb, cluster_of, slot_assignments, n_clusters, node_score_t,
        )
        attn_out, _ = self._v3.slot_attn(slot_emb, slot_emb, slot_emb,
                                          key_padding_mask=~slot_mask)
        slot_emb2 = self._v3.slot_norm(slot_emb + attn_out)  # [C, S, H]

        # Pick anchor slot per cluster.
        anchor_slot = self._pick_anchor_slot_per_cluster(slot_mask)
        C, S, H = slot_emb2.shape

        # Gather anchor slot embedding and anchor box per cluster.
        node_box_global = graph.metadata.get("node_box")
        if node_box_global is None:
            return {
                "final_box_xyxy": torch.zeros(C, 4, device=device),
                "anchor_box_xyxy": torch.zeros(C, 4, device=device),
                "delta_box": torch.zeros(C, 4, device=device),
                "tp50_logit": torch.zeros(C, device=device),
                "tp75_logit": torch.zeros(C, device=device),
                "expected_iou_logit": torch.zeros(C, device=device),
                "source_weight_logits": torch.zeros(C, S, device=device),
                "anchor_slot": anchor_slot, "cluster_score": torch.zeros(C, device=device),
                "slot_mask": slot_mask, "slot_node_idx": slot_node_idx,
            }
        node_box_global = node_box_global.to(device)

        # Build [C, 4] anchor box from slot_node_idx[c, anchor_slot[c]].
        anchor_box = torch.zeros(C, 4, device=device)
        cluster_score = torch.zeros(C, device=device)
        for c in range(C):
            a = int(anchor_slot[c].item())
            if a < 0:
                continue
            ni = int(slot_node_idx[c, a].item())
            if ni < 0:
                continue
            anchor_box[c] = node_box_global[ni]
            # Cluster confidence = max source score available in this cluster.
            avail = slot_mask[c].nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0 and node_score_t is not None:
                node_idxs = slot_node_idx[c, avail]
                node_idxs = node_idxs[node_idxs >= 0]
                if node_idxs.numel() > 0:
                    cluster_score[c] = float(node_score_t[node_idxs].max().item())

        # Anchor slot embedding [C, H] and source-id emb [C, source_id_emb_dim].
        gather_idx = anchor_slot.clamp(min=0).view(C, 1, 1).expand(C, 1, H)
        anchor_slot_emb = slot_emb2.gather(1, gather_idx).squeeze(1)
        anchor_id_emb = self.source_id_emb(anchor_slot.clamp(min=0))

        # Mean of available slot embeddings (cluster context).
        mask_f = slot_mask.float().unsqueeze(-1)        # [C, S, 1]
        denom = mask_f.sum(dim=1).clamp(min=1.0)        # [C, 1]
        mean_slot_emb = (slot_emb2 * mask_f).sum(dim=1) / denom   # [C, H]

        # Pairwise feature mean over non-anchor slots (anchor-vs-self is zeros by construction).
        if pairwise_feats is None:
            pair_mean = torch.zeros(C, self.cfg.pair_feat_dim, device=device)
        else:
            pf = pairwise_feats.to(device)
            mask2 = slot_mask.clone()
            mask2.scatter_(1, anchor_slot.clamp(min=0).unsqueeze(1), False)
            mask2_f = mask2.float().unsqueeze(-1)
            denom2 = mask2_f.sum(dim=1).clamp(min=1.0)
            pair_mean = (pf * mask2_f).sum(dim=1) / denom2

        head_in = torch.cat([anchor_slot_emb, mean_slot_emb, pair_mean, anchor_id_emb], dim=-1)
        h = self.head_trunk(head_in)   # [C, H]

        # Δ box, capped to cap_frac · diag(anchor_box).
        raw_delta = self.delta_head(h)
        # diag per cluster
        aw = (anchor_box[:, 2] - anchor_box[:, 0]).clamp(min=1.0)
        ah = (anchor_box[:, 3] - anchor_box[:, 1]).clamp(min=1.0)
        diag = (aw * aw + ah * ah).sqrt()  # [C]
        cap = self.cfg.delta_cap_frac * diag                # [C]
        delta = torch.tanh(raw_delta) * cap.unsqueeze(-1)   # [C, 4]

        # ── Fusion mode → final_box ──────────────────────────────────
        if self.cfg.fusion_mode == "residual":
            final_box = anchor_box + delta
            source_weight_logits = torch.zeros(C, S, device=device)
        else:
            # Weight head over all S slots; mask absent slots
            slot_ids = torch.arange(S, device=device)
            slot_id_emb = self.source_id_emb(slot_ids).unsqueeze(0).expand(C, S, -1)
            weight_in = torch.cat([slot_emb2, slot_id_emb], dim=-1)
            weight_logits = self.weight_head(weight_in).squeeze(-1)   # [C, S]
            weight_logits = weight_logits.masked_fill(~slot_mask, -1e9)
            w = torch.softmax(weight_logits, dim=-1)                   # [C, S]
            # Slot boxes (use anchor_box for missing slots)
            slot_boxes = anchor_box.unsqueeze(1).expand(C, S, 4).contiguous()
            for c in range(C):
                for s in range(S):
                    if bool(slot_mask[c, s].item()):
                        ni = int(slot_node_idx[c, s].item())
                        if ni >= 0:
                            slot_boxes[c, s] = node_box_global[ni]
            weighted_box = (w.unsqueeze(-1) * slot_boxes).sum(dim=1)
            if self.cfg.fusion_mode == "weighted":
                final_box = weighted_box
            else:   # hybrid
                final_box = weighted_box + delta
            source_weight_logits = weight_logits

        # Image-bound clamping is done by the caller (we need image size).
        tp50 = self.tp50_head(h).squeeze(-1)
        tp75 = self.tp75_head(h).squeeze(-1)
        e_iou = self.expected_iou_head(h).squeeze(-1)

        return {
            "final_box_xyxy": final_box,
            "anchor_box_xyxy": anchor_box,
            "delta_box": delta,
            "tp50_logit": tp50,
            "tp75_logit": tp75,
            "expected_iou_logit": e_iou,
            "source_weight_logits": source_weight_logits,
            "anchor_slot": anchor_slot,
            "cluster_score": cluster_score,
            "slot_mask": slot_mask,
            "slot_node_idx": slot_node_idx,
        }


# ── Losses ──────────────────────────────────────────────────────────


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss between two [N, 4] xyxy box tensors.

    Returns the mean (1 - GIoU). Matches torchvision's formula but is
    self-contained so we don't depend on a non-stable export.
    """
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-9
    iou = inter / union
    # enclosing box
    ex1 = torch.min(pred[:, 0], target[:, 0])
    ey1 = torch.min(pred[:, 1], target[:, 1])
    ex2 = torch.max(pred[:, 2], target[:, 2])
    ey2 = torch.max(pred[:, 3], target[:, 3])
    enclose = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0) + 1e-9
    giou = iou - (enclose - union) / enclose
    return (1.0 - giou).mean()


@dataclass
class FusionLossWeights:
    box: float = 1.0
    giou: float = 1.0
    tp50: float = 1.0
    tp75: float = 2.0
    iou: float = 0.5
    delta_reg: float = 0.05


def learned_fusion_loss(
    out: Dict[str, Any],
    *,
    gt_box: torch.Tensor,           # [C, 4] matched GT (zeros for clusters with no match)
    has_gt: torch.Tensor,            # [C] bool, True if cluster has a matched GT
    iou_at_final: torch.Tensor,      # [C] true IoU(final_box, gt_box) — recomputed each step
    weights: Optional[FusionLossWeights] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or FusionLossWeights()
    device = out["final_box_xyxy"].device
    if not has_gt.any():
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"box": z, "giou": z, "tp50": z, "tp75": z, "iou": z, "delta_reg": z, "total": z}
    has = has_gt.to(device).bool()
    fb = out["final_box_xyxy"][has]
    tg = gt_box[has].to(device)
    L_box = F.smooth_l1_loss(fb, tg)
    L_giou = giou_loss(fb, tg)
    iou = iou_at_final[has].to(device)
    tp50_t = (iou >= 0.5).float()
    tp75_t = (iou >= 0.75).float()
    L_tp50 = F.binary_cross_entropy_with_logits(out["tp50_logit"][has], tp50_t)
    L_tp75 = F.binary_cross_entropy_with_logits(out["tp75_logit"][has], tp75_t)
    L_iou = F.smooth_l1_loss(torch.sigmoid(out["expected_iou_logit"][has]), iou)
    # Δ regularization — discourage huge moves when not needed (using raw cap-scaled Δ).
    L_delta = (out["delta_box"][has].pow(2).sum(dim=-1)).mean()
    total = (weights.box * L_box + weights.giou * L_giou
              + weights.tp50 * L_tp50 + weights.tp75 * L_tp75
              + weights.iou * L_iou + weights.delta_reg * L_delta)
    return {"box": L_box, "giou": L_giou, "tp50": L_tp50, "tp75": L_tp75,
            "iou": L_iou, "delta_reg": L_delta, "total": total}
