"""TGraphXCandidateNodeSelector — paper-faithful TGraphX for detection.

Each candidate detection box is a graph node carrying:
  - its image crop (a tensor [3, H, W], the original TGraphX node feature)
  - its source id (which detector or fusion method produced it)
  - its box, score, and label.

Tensor-aware graph message passing (the V3 encoder) operates over the
crop tensors before any flattening. Per-node heads then predict:

  - selection_logit: cluster-wise softmax over candidate nodes
  - tp50_logit, tp75_logit: calibrated TP probabilities
  - expected_iou: regression toward IoU(box, matched GT)

At inference: for each cluster, pick `argmax(selection_logit)` to get
the selected node, and use its precomputed box + a calibrated score
(tp50 or tp75 head, configurable on validation).

This is NOT box regression and NOT source routing — it is node
selection over visual tensor candidates, the original TGraphX paper
formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph

from .candidate_mask import candidate_node_mask
from .graph_builder import NODE_TYPES
from .source_router_v3 import TGraphXSourceRouterV3


@dataclass
class CandidateSelectorConfig:
    num_classes: int
    num_detectors: int
    crop_size: int
    crop_channels: int = 16
    hidden_dim: int = 64
    metadata_dim: Optional[int] = None
    edge_feat_dim: int = 14
    num_message_passing: int = 2
    use_message_passing: bool = True
    use_metadata: bool = True
    score_head: str = "p_tp50"   # one of "p_tp50" | "p_tp75" | "selection"
    # Ablation control — which feature path to use:
    #   "crop_metadata_mp"   — full paper-faithful: spatial crop tensors
    #                          through tensor-aware ConvMP, then pool + metadata
    #   "flat_crop_mp"       — flatten crop after CNN (spatial pool first),
    #                          then standard flat-vector MP; proves spatial MP matters
    #   "crop_no_mp"         — CropCNN + pool → node embedding, no message passing
    #   "metadata_only"      — skip crop encoder entirely, metadata MLP only
    feature_mode: str = "crop_metadata_mp"


class TGraphXCandidateNodeSelector(nn.Module):
    """Per-node candidate selector — four ablation modes.

    feature_mode="crop_metadata_mp"  (paper-faithful):
        CropCNN → tensor-aware ConvMP → spatial pool → + metadata MLP → head
    feature_mode="flat_crop_mp":
        CropCNN → spatial pool immediately → flat-vector LinMP → + metadata → head
    feature_mode="crop_no_mp":
        CropCNN → spatial pool → + metadata MLP → head  (no MP at all)
    feature_mode="metadata_only":
        skip CropCNN entirely → metadata MLP → head

    The first mode proves spatial tensor MP improves over the others.
    """

    def __init__(self, cfg: CandidateSelectorConfig):
        super().__init__()
        self.cfg = cfg
        mp_layers = cfg.num_message_passing if cfg.use_message_passing else 0

        if cfg.feature_mode == "metadata_only":
            # No crop encoder. Build a minimal metadata-only branch.
            md = cfg.metadata_dim if cfg.metadata_dim is not None else (8 + cfg.num_detectors + cfg.num_classes)
            self._meta_only = nn.Sequential(
                nn.Linear(md, cfg.hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self._v3 = None
        elif cfg.feature_mode == "flat_crop_mp":
            # Pool BEFORE MP; then a flat-vector linear message-passing layer
            # (no spatial dimensions preserved). Simulates standard GNN.
            from .models import CropCNN
            self._crop_enc = CropCNN(3, cfg.crop_channels, cfg.crop_size)
            # Pool is applied BEFORE projection: input to _proj is [N, crop_channels + hidden_dim]
            # (not the full spatial flat_dim = crop_channels * sp * sp)
            md = cfg.metadata_dim if cfg.metadata_dim is not None else (8 + cfg.num_detectors + cfg.num_classes)
            self._meta_mlp = nn.Sequential(
                nn.Linear(md, cfg.hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self._proj = nn.Linear(cfg.crop_channels + cfg.hidden_dim, cfg.hidden_dim)
            # Flat-vector "message passing" — standard Linear over neighbour-aggregated
            # features. This deliberately discards spatial structure.
            self._flat_mp = nn.ModuleList([
                nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(inplace=True))
                for _ in range(max(1, mp_layers))
            ] if mp_layers > 0 else [])
            self._pool = nn.AdaptiveAvgPool2d(1)
            self._v3 = None
        else:
            # "crop_metadata_mp" (default) or "crop_no_mp":
            # Use the full V3 encoder; num_message_passing=0 gives "no MP" variant.
            self._v3 = TGraphXSourceRouterV3(
                num_classes=cfg.num_classes,
                num_detectors=cfg.num_detectors,
                crop_size=cfg.crop_size,
                crop_channels=cfg.crop_channels,
                hidden_dim=cfg.hidden_dim,
                metadata_dim=cfg.metadata_dim,
                edge_feat_dim=cfg.edge_feat_dim,
                num_message_passing=0 if cfg.feature_mode == "crop_no_mp" else mp_layers,
            )

        H = cfg.hidden_dim
        self.selection_head = nn.Linear(H, 1)
        self.tp50_head = nn.Linear(H, 1)
        self.tp75_head = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)

    def _encode_metadata_only(self, graph: Graph) -> torch.Tensor:
        device = graph.node_features.device
        md = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        if md is None:
            in_dim = self._meta_only[0].in_features
            md = torch.zeros(graph.node_features.shape[0], in_dim, device=device)
        return self._meta_only(md.to(device).float())

    def _encode_flat_crop_mp(self, graph: Graph) -> torch.Tensor:
        device = graph.node_features.device
        x = graph.node_features.float().to(device)  # [N, 3, H, W]
        h = self._crop_enc(x)                         # [N, C, S, S]
        v = self._pool(h).squeeze(-1).squeeze(-1)     # [N, C*1*1 flat]
        md = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        if md is None:
            md = torch.zeros(x.shape[0], self._meta_mlp[0].in_features, device=device)
        m = self._meta_mlp(md.to(device).float())
        emb = F.relu(self._proj(torch.cat([v, m], dim=1)))  # [N, H]
        # Flat-vector MP: aggregate neighbours by mean (no edge features, no spatial).
        if self._flat_mp:
            ei = graph.edge_index
            if ei is not None and ei.numel() > 0:
                N = emb.shape[0]
                for layer in self._flat_mp:
                    # mean-aggregate neighbours
                    agg = torch.zeros_like(emb)
                    cnt = torch.zeros(N, device=device)
                    src, dst = ei[0], ei[1]
                    agg.index_add_(0, dst, emb[src])
                    cnt.index_add_(0, dst, torch.ones(src.shape[0], device=device))
                    agg = agg / cnt.clamp(min=1.0).unsqueeze(-1)
                    emb = F.relu(emb + layer(agg))
        return emb

    def forward(self, graph: Graph, detector_names: List[str]) -> Dict[str, Any]:
        mode = self.cfg.feature_mode
        if mode == "metadata_only":
            node_emb = self._encode_metadata_only(graph)
        elif mode == "flat_crop_mp":
            node_emb = self._encode_flat_crop_mp(graph)
        else:
            v3 = self._v3(graph, detector_names=detector_names)
            node_emb = v3["node_emb"]
        sel = self.selection_head(node_emb).squeeze(-1)
        tp50 = self.tp50_head(node_emb).squeeze(-1)
        tp75 = self.tp75_head(node_emb).squeeze(-1)
        eiou = self.expected_iou_head(node_emb).squeeze(-1)
        return {
            "selection_logit": sel,        # [N]
            "tp50_logit": tp50,             # [N]
            "tp75_logit": tp75,             # [N]
            "expected_iou_logit": eiou,     # [N]
            "node_emb": node_emb,
        }


# ── Loss ────────────────────────────────────────────────────────────


@dataclass
class CandidateLossWeights:
    selection_ce: float = 1.0
    tp50_bce: float = 1.0
    tp75_bce: float = 2.0
    iou_reg: float = 0.5
    pairwise_rank: float = 0.5


def candidate_selector_loss(
    out: Dict[str, torch.Tensor],
    *,
    cluster_of: torch.Tensor,      # [N] long, -1 if node not in a cluster
    cand_mask: torch.Tensor,        # [N] bool
    best_node_per_cluster: torch.Tensor,   # [C] long, -1 if none
    node_iou_with_gt: torch.Tensor,        # [N] float in [0, 1]
    node_class_correct: torch.Tensor,      # [N] bool
    weights: Optional[CandidateLossWeights] = None,
) -> Dict[str, torch.Tensor]:
    weights = weights or CandidateLossWeights()
    device = out["selection_logit"].device
    sel = out["selection_logit"]
    tp50 = out["tp50_logit"]
    tp75 = out["tp75_logit"]
    eiou = out["expected_iou_logit"]
    N = sel.shape[0]
    if N == 0 or best_node_per_cluster.shape[0] == 0:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"selection": z, "tp50": z, "tp75": z, "iou": z, "rank": z, "total": z, "n_clusters": 0}

    # Cluster-wise CE over selection logits.
    n_clusters = int(best_node_per_cluster.shape[0])
    L_sel = torch.tensor(0.0, device=device)
    L_rank = torch.tensor(0.0, device=device)
    n_valid_clusters = 0
    for c in range(n_clusters):
        if int(best_node_per_cluster[c].item()) < 0:
            continue
        in_c = (cluster_of == c) & cand_mask
        if not in_c.any():
            continue
        idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
        if idx_c.numel() < 1:
            continue
        target_node = int(best_node_per_cluster[c].item())
        local_target_pos = (idx_c == target_node).nonzero(as_tuple=False)
        if local_target_pos.numel() == 0:
            continue
        local_target = int(local_target_pos[0].item())
        log = sel[idx_c]
        L_sel = L_sel + F.cross_entropy(log.unsqueeze(0),
                                          torch.tensor([local_target], device=device))
        n_valid_clusters += 1
        # Pairwise ranking by IoU
        if idx_c.numel() > 1:
            ious_local = node_iou_with_gt[idx_c]
            for i in range(idx_c.numel()):
                for j in range(i + 1, idx_c.numel()):
                    di = float(ious_local[i].item()); dj = float(ious_local[j].item())
                    if abs(di - dj) < 1e-4:
                        continue
                    if di > dj:
                        L_rank = L_rank + F.softplus(log[j] - log[i]) * 0.5
                    else:
                        L_rank = L_rank + F.softplus(log[i] - log[j]) * 0.5
    if n_valid_clusters > 0:
        L_sel = L_sel / n_valid_clusters
        L_rank = L_rank / n_valid_clusters

    # TP50 / TP75 BCE on all candidate nodes.
    cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)
    if cand_idx.numel() > 0:
        iou = node_iou_with_gt[cand_idx]
        cls_ok = node_class_correct[cand_idx].float()
        tp50_t = ((iou >= 0.5) & (cls_ok > 0.5)).float()
        tp75_t = ((iou >= 0.75) & (cls_ok > 0.5)).float()
        L_tp50 = F.binary_cross_entropy_with_logits(tp50[cand_idx], tp50_t)
        L_tp75 = F.binary_cross_entropy_with_logits(tp75[cand_idx], tp75_t)
        L_iou = F.smooth_l1_loss(torch.sigmoid(eiou[cand_idx]), iou)
    else:
        L_tp50 = torch.tensor(0.0, device=device)
        L_tp75 = torch.tensor(0.0, device=device)
        L_iou = torch.tensor(0.0, device=device)

    total = (weights.selection_ce * L_sel
              + weights.tp50_bce * L_tp50
              + weights.tp75_bce * L_tp75
              + weights.iou_reg * L_iou
              + weights.pairwise_rank * L_rank)
    return {"selection": L_sel, "tp50": L_tp50, "tp75": L_tp75,
            "iou": L_iou, "rank": L_rank, "total": total, "n_clusters": n_valid_clusters}


# ── Inference helper: per-cluster pick ──────────────────────────────


def select_per_cluster(
    out: Dict[str, torch.Tensor],
    *,
    cluster_of: torch.Tensor,
    cand_mask: torch.Tensor,
    node_box: torch.Tensor,
    node_label: torch.Tensor,
    score_head: str = "p_tp50",
) -> Dict[str, torch.Tensor]:
    """Pick one node per cluster, return (boxes, scores, labels)."""
    sel = out["selection_logit"]
    tp50 = out["tp50_logit"]
    tp75 = out["tp75_logit"]
    n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
    boxes = []; scores = []; labels = []
    for c in range(n_clusters):
        in_c = (cluster_of == c) & cand_mask
        if not in_c.any():
            continue
        idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
        pick_local = int(sel[idx_c].argmax().item())
        pick = int(idx_c[pick_local].item())
        boxes.append(node_box[pick])
        if score_head == "p_tp50":
            scores.append(torch.sigmoid(tp50[pick]).cpu())
        elif score_head == "p_tp75":
            scores.append(torch.sigmoid(tp75[pick]).cpu())
        else:
            scores.append(torch.sigmoid(sel[pick]).cpu())
        labels.append(node_label[pick] if node_label is not None else torch.tensor(0, dtype=torch.long))
    if not boxes:
        return {"boxes_xyxy": torch.zeros(0, 4), "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long)}
    return {"boxes_xyxy": torch.stack(boxes).cpu(),
            "scores": torch.stack(scores).float(),
            "labels": torch.stack(labels).cpu()}
