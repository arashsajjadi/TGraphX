"""TGraphX best-source router.

This module reformulates object-detection fusion as a source-routing
problem: for each candidate cluster, pick the single best detector output
(YOLO / DETR / open-vocab / RetinaNet / Union / WBF).

Key components:
- `compute_source_utilities`: per-source utility from GT IoU.
- `TGraphXSourceRouter`: edge-conditioned GNN that predicts source logits.
- `source_routing_loss`: CE + utility-KL + pairwise-ranking.
- `evaluate_source_routing`: source accuracy, oracle-gap recovery, etc.
- `COCO_TO_VOC_MAP`: canonical VOC class mapping for COCO-trained detectors.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import ConvMessagePassing, Graph

# ── Canonical VOC label mapping ────────────────────────────────────────────

COCO_TO_VOC_MAP: Dict[str, str] = {
    "airplane":   "aeroplane",
    "motorcycle": "motorbike",
    "couch":      "sofa",
    "tv":         "tvmonitor",
    "potted plant": "pottedplant",
    "dining table": "diningtable",
}


def canonical_label(label: str) -> str:
    """Map COCO label name to canonical VOC label if applicable."""
    return COCO_TO_VOC_MAP.get(label.lower().strip(), label.lower().strip())


def canonical_label_id(label: str, class_names: List[str]) -> int:
    """Map a raw label name to a canonical class index."""
    c = canonical_label(label)
    try:
        return class_names.index(c)
    except ValueError:
        return -1  # unknown class


# ── Stable-seed for synthetic detectors ───────────────────────────────────

def stable_image_seed(image_id: str, extra: int = 0) -> int:
    """Deterministic seed that is stable across Python processes (no hash())."""
    h = hashlib.sha256(f"{image_id}:{extra}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Utility computation ────────────────────────────────────────────────────

def _box_iou_single(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU of one box vs N boxes. Returns [N]."""
    from .box_ops import box_iou
    return box_iou(box1.unsqueeze(0), boxes2)[0]


def compute_source_utilities(
    node_box: torch.Tensor,       # [N, 4] per-node candidate box
    node_label: torch.Tensor,     # [N] canonical class id
    node_score: torch.Tensor,     # [N] detector score
    cluster_of: torch.Tensor,     # [N] cluster assignment (-1 = not a candidate)
    node_types: torch.Tensor,     # [N] NODE_TYPES values
    gt_boxes: torch.Tensor,        # [G, 4]
    gt_labels: torch.Tensor,       # [G]
    *,
    class_agnostic: bool = False,
    iou_match: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-node utility and best-source label per cluster.

    Returns:
        utility:      [N] float32 — IoU (class-weighted) to best GT.
        best_source:  [num_clusters] long — node index of best source per cluster.
        cluster_mask: [N] bool — nodes eligible as candidates.
    """
    from .graph_builder import NODE_TYPES
    N = node_box.shape[0]
    utility = torch.zeros(N, dtype=torch.float32, device=node_box.device)
    cand_mask = (
        (node_types == NODE_TYPES["proposal"])
        | (node_types == NODE_TYPES["cluster"])
        | (node_types == NODE_TYPES["consensus"])
    )

    if gt_boxes.numel() == 0:
        return utility, torch.zeros(0, dtype=torch.long), cand_mask

    cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)
    if cand_idx.numel() == 0:
        return utility, torch.zeros(0, dtype=torch.long), cand_mask

    from .box_ops import box_iou
    ious = box_iou(node_box[cand_idx], gt_boxes)  # [Nc, G]
    best_iou, best_gt = ious.max(dim=1)

    # CONTINUOUS utility — never threshold IoU before ranking.
    # Even if all candidates are below iou_match=0.5, the highest-IoU candidate
    # is still the best available source for routing supervision.
    for ki, ni in enumerate(cand_idx.tolist()):
        iou = float(best_iou[ki].item())
        g_idx = int(best_gt[ki].item())
        if class_agnostic:
            utility[ni] = iou
        else:
            canonical_match = (int(node_label[ni].item()) == int(gt_labels[g_idx].item()))
            utility[ni] = iou if canonical_match else 0.0
        # Note: we always assign a continuous utility value.
        # iou_match is used ONLY for detection metrics, not for routing labels.

    # Best source per cluster = argmax continuous utility
    n_clusters = int(cluster_of.max().item() + 1) if cluster_of.numel() > 0 else 0
    best_src = torch.full((n_clusters,), -1, dtype=torch.long)
    for c in range(n_clusters):
        mask = (cluster_of == c) & cand_mask
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        utils = utility[idx]
        best_local = int(utils.argmax().item())
        best_src[c] = int(idx[best_local].item())
    return utility, best_src, cand_mask


# ── Source-routing losses ──────────────────────────────────────────────────

def source_routing_loss(
    node_scores: torch.Tensor,     # [N] predicted selection score per node
    utility: torch.Tensor,         # [N] float32 utility
    best_src: torch.Tensor,        # [C] long — best-source node idx per cluster
    cluster_of: torch.Tensor,      # [N] long — cluster assignment
    cand_mask: torch.Tensor,       # [N] bool — eligible candidates
    *,
    beta_kl: float = 5.0,
    pairwise_weight: float = 0.5,
    regret_lambda: float = 1.0,
    baseline_scores: Optional[torch.Tensor] = None,  # [N] base scores for regret
) -> Dict[str, torch.Tensor]:
    """Compute source-routing losses.

    Returns dict of loss terms. Total = CE + KL + pairwise, regret-weighted.
    """
    device = node_scores.device
    total = torch.tensor(0.0, device=device)
    losses: Dict[str, Any] = {}
    valid_clusters = (best_src >= 0).nonzero(as_tuple=False).squeeze(-1)
    if valid_clusters.numel() == 0:
        return {"total": total, "ce": total, "kl": total, "pairwise": total}

    ce_terms = []
    kl_terms = []
    pw_terms = []
    regret_weights = []

    for c_idx in valid_clusters.tolist():
        best_node = int(best_src[c_idx].item())
        # Cluster candidate nodes
        mask = (cluster_of == c_idx) & cand_mask
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        logits = node_scores[idx]
        utils = utility[idx]

        # Which local index is best?
        local_best = int((idx == best_node).nonzero(as_tuple=False)
                         .squeeze(-1).item() if (idx == best_node).any() else 0)

        # CE: pick best source
        lbl = torch.tensor([local_best], dtype=torch.long, device=device)
        ce = F.cross_entropy(logits.unsqueeze(0), lbl)
        ce_terms.append(ce)

        # KL: utility soft labels
        soft_target = F.softmax(beta_kl * utils, dim=0)
        log_pred = F.log_softmax(logits, dim=0)
        kl = F.kl_div(log_pred, soft_target, reduction="sum")
        kl_terms.append(kl)

        # Pairwise ranking over candidate pairs
        Nc = idx.numel()
        if Nc > 1:
            pw_sum = torch.tensor(0.0, device=device)
            n_pairs = 0
            for ai in range(Nc):
                for bi in range(ai + 1, Nc):
                    da = utils[ai].item(); db = utils[bi].item()
                    if abs(da - db) < 1e-6:
                        continue
                    sa = logits[ai]; sb = logits[bi]
                    if da > db:
                        pw_sum += F.softplus(sb - sa)
                    else:
                        pw_sum += F.softplus(sa - sb)
                    n_pairs += 1
            if n_pairs > 0:
                pw_terms.append(pw_sum / n_pairs)

        # Regret weight: how badly does the highest-confidence source do?
        if baseline_scores is not None:
            base = baseline_scores[idx]
            chosen_by_base = int(base.argmax().item())
            regret = float(utils[local_best].item() - utils[chosen_by_base].item())
            regret_weights.append(max(0.0, regret))
        else:
            regret_weights.append(1.0)

    # Per-cluster regret weighting (P3.1 fix):
    # Each cluster's loss is scaled by its own regret, not the batch mean.
    # This focuses learning on clusters where baseline routing is wrong.
    all_cluster_losses = []
    for i in range(len(ce_terms)):
        w_c = 1.0 + regret_lambda * regret_weights[i]
        cluster_loss = (ce_terms[i]
                        + 0.5 * kl_terms[i]
                        + (pairwise_weight * pw_terms[i] if i < len(pw_terms) else torch.tensor(0.0, device=device)))
        all_cluster_losses.append(w_c * cluster_loss)

    if not all_cluster_losses:
        z = torch.tensor(0.0, device=device)
        return {"total": z, "ce": z, "kl": z, "pairwise": z}

    total = torch.stack(all_cluster_losses).mean()
    ce_loss = torch.stack(ce_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    pw_loss = (torch.stack(pw_terms).mean() if pw_terms else torch.tensor(0.0, device=device))
    losses = {"total": total, "ce": ce_loss, "kl": kl_loss, "pairwise": pw_loss}
    return losses


# ── Source-routing metrics ─────────────────────────────────────────────────

def source_selection_accuracy(
    predicted_node: int,
    best_source_node: int,
) -> bool:
    return predicted_node == best_source_node


def oracle_gap_recovery(
    metric_tgx: float,
    metric_baseline: float,
    metric_oracle: float,
    eps: float = 1e-9,
) -> float:
    """Fraction of the oracle-baseline gap closed by TGraphX.

    Returns 0 if TGraphX == baseline, 1 if TGraphX == oracle, <0 if worse.
    """
    denom = max(eps, abs(metric_oracle - metric_baseline))
    return (metric_tgx - metric_baseline) / denom


def evaluate_source_routing(
    all_selected: List[int],
    all_best_src: List[int],
    all_utility_selected: List[float],
    all_utility_oracle: List[float],
    all_utility_baseline: List[float],
    all_iou_selected: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute source-routing evaluation metrics."""
    n = len(all_selected)
    if n == 0:
        return {"source_acc": 0.0, "top2_acc": 0.0, "mean_iou": 0.0,
                "mean_regret": 0.0, "gap_recovery_utility": 0.0,
                "gap_recovery_iou": 0.0}

    correct = sum(s == b for s, b in zip(all_selected, all_best_src))
    mean_util_sel = sum(all_utility_selected) / n
    mean_util_oracle = sum(all_utility_oracle) / n
    mean_util_base = sum(all_utility_baseline) / n
    mean_regret = mean_util_oracle - mean_util_sel

    return {
        "source_acc": correct / n,
        "mean_utility_selected": mean_util_sel,
        "mean_utility_oracle": mean_util_oracle,
        "mean_utility_baseline": mean_util_base,
        "mean_regret": mean_regret,
        "gap_recovery_utility": oracle_gap_recovery(
            mean_util_sel, mean_util_base, mean_util_oracle),
        "mean_iou_selected": (sum(all_iou_selected) / n) if all_iou_selected else 0.0,
    }


# ── Edge-conditioned message module ───────────────────────────────────────

class EdgeConditionedMP(nn.Module):
    """ConvMessagePassing extended with edge-feature gating.

    For each edge (i → j), an edge MLP computes a gate that scales the
    message from node i to node j. The spatial crop features flow through
    ConvMessagePassing; the gate is broadcast over the spatial dims.
    """

    def __init__(self, in_shape: tuple, out_shape: tuple, edge_feat_dim: int):
        super().__init__()
        self.mp = ConvMessagePassing(in_shape=in_shape, out_shape=out_shape)
        C_out = out_shape[0]
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_feat_dim, C_out),
            nn.Sigmoid(),
        )
        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(
        self,
        x: torch.Tensor,          # [N, C, H, W]
        ei: torch.Tensor,          # [2, E]
        edge_attr: Optional[torch.Tensor] = None,  # [E, edge_feat_dim]
    ) -> torch.Tensor:
        """Vectorized edge-conditioned tensor message passing.

        Uses index_add for scatter aggregation (no Python loops over edges).
        Normalizes by destination degree to prevent activation blow-up.
        """
        h_base = self.mp(x, ei)               # [N, C_out, H_out, W_out]
        if edge_attr is None or edge_attr.numel() == 0 or ei.shape[1] == 0:
            return h_base

        N, C_out, H_out, W_out = h_base.shape
        src_nodes, dst_nodes = ei[0], ei[1]

        # Edge gates: [E, C_out]
        gates = self.edge_gate(edge_attr)      # [E, C_out]

        # Gated source features: h_base[src] scaled by gate per edge
        # h_base[src_nodes]: [E, C_out, H_out, W_out]
        src_feats = h_base[src_nodes]          # [E, C_out, H_out, W_out]
        # broadcast gates [E, C_out] → [E, C_out, 1, 1]
        gated = src_feats * gates[:, :, None, None]  # [E, C_out, H_out, W_out]

        # Scatter-add to destination nodes
        h_extra = torch.zeros_like(h_base)
        h_extra.index_add_(0, dst_nodes, gated)

        # Degree normalization: count in-edges per node
        degree = torch.zeros(N, device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst_nodes, torch.ones(ei.shape[1], device=x.device))
        degree = degree.clamp(min=1)
        # Broadcast degree over spatial dims: [N] → [N, 1, 1, 1]
        h_extra = h_extra / degree[:, None, None, None]

        return h_base + 0.1 * h_extra  # residual with small weight


# ── TGraphX Source Router ─────────────────────────────────────────────────

class TGraphXSourceRouter(nn.Module):
    """Graph-based best-source router.

    Predicts a per-node selection score. For each candidate cluster, the node
    with the highest score is selected and its box is returned verbatim.
    """

    def __init__(
        self,
        num_classes: int,
        num_detectors: int,
        crop_size: int,
        crop_channels: int = 16,
        hidden_dim: int = 48,
        metadata_dim: Optional[int] = None,
        edge_feat_dim: int = 14,
        num_message_passing: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_detectors = num_detectors
        self.crop_size = crop_size

        from .models import CropCNN
        self.crop_encoder = CropCNN(in_channels=3, out_channels=crop_channels,
                                     crop_size=crop_size)
        crop_spatial = self.crop_encoder.out_spatial

        # Edge-conditioned message passing layers
        self.ec_layers = nn.ModuleList()
        for _ in range(num_message_passing):
            self.ec_layers.append(EdgeConditionedMP(
                in_shape=(crop_channels, crop_spatial, crop_spatial),
                out_shape=(crop_channels, crop_spatial, crop_spatial),
                edge_feat_dim=edge_feat_dim,
            ))

        md_dim = metadata_dim if metadata_dim is not None else 8 + num_detectors + num_classes
        self.metadata_mlp = nn.Sequential(
            nn.Linear(md_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        fused_dim = crop_channels + hidden_dim
        self.fuse_head = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.ReLU(inplace=True))
        # Source routing head: per-node quality score (higher = better source)
        self.quality_head = nn.Linear(hidden_dim, 1)
        # Calibration head: optional score residual
        self.calib_head = nn.Linear(hidden_dim, 1)

    def forward(self, graph: Graph) -> Dict[str, torch.Tensor]:
        x = graph.node_features                 # [N, 3, H, W]
        ei = graph.edge_index
        ea = graph.edge_features                # [E, edge_feat_dim] or None
        md = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        device = x.device

        h = self.crop_encoder(x)               # [N, C_e, H_e, W_e]
        for ec in self.ec_layers:
            ea_dev = ea.to(device) if ea is not None else None
            h = F.relu(ec(h, ei, ea_dev)) + h

        v = self.spatial_pool(h).squeeze(-1).squeeze(-1)  # [N, C_e]

        if md is None:
            md = torch.zeros(x.shape[0], self.metadata_mlp[0].in_features,
                              device=device, dtype=v.dtype)
        if md.device != device:
            md = md.to(device)
        m = self.metadata_mlp(md)

        fused = torch.cat([v, m], dim=1)
        z = self.fuse_head(fused)

        quality = self.quality_head(z).squeeze(-1)   # [N] quality/routing score
        calib = self.calib_head(z).squeeze(-1)       # [N] calibration residual

        return {
            "quality_logits": quality,
            "calib_residual": calib,
            "node_embedding": z,
            # Back-compat aliases
            "objectness_logits": quality,
            "class_logits": torch.zeros(x.shape[0], self.num_classes, device=device),
            "box_reg": torch.zeros(x.shape[0], 4, device=device),
        }
