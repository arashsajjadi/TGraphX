"""Pairwise (source-vs-anchor) and source-specific features for the anchor router.

The current source-router slot embeddings collapse into a two-source majority
policy because they have no privileged information about *why* one slot
should beat another. The anchor router needs per-pair signals like
"union_box / anchor_box IoU", "score-rank disagreement", "candidate is
aggregate vs raw detector", etc.

Everything here is computed at inference from `graph.metadata` and the
`slot_node_idx` table produced by SourceSlotAggregator. No GT is read.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .box_ops import box_iou
from .source_router_v3 import NUM_SOURCES, SOURCE_SLOTS


# Generic per-(slot, anchor) feature dimension. Keep this fixed so the
# linear projection in AnchorRouter can stay static.
PAIRWISE_FEAT_DIM = 16

# Source-specific (union / yolo / rt_detr / retinanet) extras. Each
# specialist head consumes the generic pairwise features PLUS a few
# specialist scalars (e.g. union: merged-box count). Total dim per
# specialist = PAIRWISE_FEAT_DIM + SPECIALIST_EXTRA_DIM.
SPECIALIST_EXTRA_DIM = 8


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 1e-9 else 0.0


def _to_xywh(box: torch.Tensor) -> Tuple[float, float, float, float]:
    if box.numel() < 4:
        return 0.0, 0.0, 0.0, 0.0
    x1, y1, x2, y2 = (float(box[i].item()) for i in range(4))
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, max(0.0, x2 - x1), max(0.0, y2 - y1)


def pairwise_features_for_cluster(
    cluster_id: int,
    slot_node_idx: torch.Tensor,    # [C, S] long, -1 if slot missing
    slot_avail: torch.Tensor,        # [C, S] bool
    anchor_slot: int,
    node_box: torch.Tensor,          # [N, 4]
    node_score: torch.Tensor,        # [N]
    node_label: torch.Tensor,        # [N] long
    n_proposals_in_cluster: int,
    detector_agreement_entropy: float,
    score_entropy: float,
    box_variance: float,
    proposal_max_iou: float,
) -> torch.Tensor:
    """Compute PAIRWISE_FEAT_DIM features for each slot s vs the anchor.

    Returns [S, PAIRWISE_FEAT_DIM]. Slot=anchor row is zeros. Absent slots
    are zeros.
    """
    C, S = slot_node_idx.shape
    out = torch.zeros(S, PAIRWISE_FEAT_DIM, dtype=torch.float32)
    if cluster_id < 0 or cluster_id >= C:
        return out
    anc_node = int(slot_node_idx[cluster_id, anchor_slot].item()) if 0 <= anchor_slot < S else -1
    anc_ok = anc_node >= 0 and bool(slot_avail[cluster_id, anchor_slot].item())
    if not anc_ok:
        return out
    anc_box = node_box[anc_node]
    anc_score = float(node_score[anc_node].item())
    anc_label = int(node_label[anc_node].item())
    anc_cx, anc_cy, anc_w, anc_h = _to_xywh(anc_box)
    anc_area = anc_w * anc_h + 1e-9
    anc_diag = (anc_w * anc_w + anc_h * anc_h) ** 0.5 + 1e-9

    for s in range(S):
        if s == anchor_slot:
            continue
        if not bool(slot_avail[cluster_id, s].item()):
            continue
        n = int(slot_node_idx[cluster_id, s].item())
        if n < 0:
            continue
        cbox = node_box[n]
        cscore = float(node_score[n].item())
        clabel = int(node_label[n].item())
        iou = float(box_iou(anc_box.unsqueeze(0), cbox.unsqueeze(0))[0, 0].item())
        ccx, ccy, cw, ch = _to_xywh(cbox)
        carea = cw * ch + 1e-9
        center_dx = (ccx - anc_cx) / anc_diag
        center_dy = (ccy - anc_cy) / anc_diag
        w_ratio = _safe_div(cw, anc_w + 1e-9)
        h_ratio = _safe_div(ch, anc_h + 1e-9)
        area_ratio = _safe_div(carea, anc_area)
        score_diff = cscore - anc_score
        score_rank_diff = (1.0 if cscore > anc_score else (-1.0 if cscore < anc_score else 0.0))
        class_agreement = 1.0 if (anc_label == clabel) else 0.0

        # Aggregate-vs-raw indicator: slots 4 (union), 5 (wbf), 6 (nms),
        # 7 (soft_nms), 8 (best_proposal), 9 (calibrated_consensus) are aggregates.
        is_aggregate = 1.0 if s >= SOURCE_SLOTS.get("union", 4) else 0.0
        is_raw_detector = 1.0 - is_aggregate

        feats = [
            iou,
            center_dx,
            center_dy,
            w_ratio - 1.0,
            h_ratio - 1.0,
            area_ratio - 1.0,
            score_diff,
            score_rank_diff,
            class_agreement,
            is_aggregate,
            is_raw_detector,
            float(n_proposals_in_cluster),
            detector_agreement_entropy,
            score_entropy,
            box_variance,
            proposal_max_iou,
        ]
        out[s, :len(feats)] = torch.tensor(feats[:PAIRWISE_FEAT_DIM], dtype=torch.float32)
    return out


def union_specialist_features(
    cluster_id: int,
    slot_node_idx: torch.Tensor,
    slot_avail: torch.Tensor,
    node_box: torch.Tensor,
    node_score: torch.Tensor,
    anchor_slot: int,
    n_proposals_in_cluster: int,
    proposal_mean_pairwise_iou: float,
    proposal_max_iou_to_union: float,
) -> torch.Tensor:
    """Specialist features for union (slot 4). [SPECIALIST_EXTRA_DIM]."""
    S = slot_node_idx.shape[1]
    union_slot = SOURCE_SLOTS.get("union", 4)
    out = torch.zeros(SPECIALIST_EXTRA_DIM, dtype=torch.float32)
    if not bool(slot_avail[cluster_id, union_slot].item()):
        return out
    un = int(slot_node_idx[cluster_id, union_slot].item())
    if un < 0:
        return out
    ubox = node_box[un]
    _, _, uw, uh = _to_xywh(ubox)
    uarea = uw * uh + 1e-9
    if anchor_slot >= 0:
        anc = int(slot_node_idx[cluster_id, anchor_slot].item())
        if anc >= 0:
            _, _, aw, ah = _to_xywh(node_box[anc])
            aa = aw * ah + 1e-9
            anc_to_union = _safe_div(uarea, aa) - 1.0
        else:
            anc_to_union = 0.0
    else:
        anc_to_union = 0.0
    proposal_areas = []
    for s in range(S):
        if s == union_slot:
            continue
        if not bool(slot_avail[cluster_id, s].item()):
            continue
        n = int(slot_node_idx[cluster_id, s].item())
        if n < 0:
            continue
        _, _, w, h = _to_xywh(node_box[n])
        proposal_areas.append(w * h)
    mean_prop = (sum(proposal_areas) / len(proposal_areas)) if proposal_areas else 1.0
    out[0] = float(n_proposals_in_cluster)
    out[1] = proposal_mean_pairwise_iou
    out[2] = _safe_div(uarea, mean_prop) - 1.0
    out[3] = anc_to_union
    out[4] = float(len(proposal_areas))
    out[5] = proposal_max_iou_to_union
    out[6] = 1.0 if uarea > mean_prop else 0.0   # union *expands*
    out[7] = 1.0 if uarea < mean_prop else 0.0   # union *shrinks*
    return out


def yolo_specialist_features(
    cluster_id: int,
    slot_node_idx: torch.Tensor,
    slot_avail: torch.Tensor,
    node_box: torch.Tensor,
    node_score: torch.Tensor,
    anchor_slot: int,
    yolo_score_percentile: float = 0.5,
) -> torch.Tensor:
    """Specialist features for yolo_modern (slot 0). [SPECIALIST_EXTRA_DIM]."""
    yolo_slot = SOURCE_SLOTS.get("yolo_modern", 0)
    rtdetr_slot = SOURCE_SLOTS.get("rt_detr", 2)
    out = torch.zeros(SPECIALIST_EXTRA_DIM, dtype=torch.float32)
    if not bool(slot_avail[cluster_id, yolo_slot].item()):
        return out
    yn = int(slot_node_idx[cluster_id, yolo_slot].item())
    if yn < 0:
        return out
    ybox = node_box[yn]
    ys = float(node_score[yn].item())
    out[0] = yolo_score_percentile
    out[1] = ys
    if anchor_slot >= 0:
        an = int(slot_node_idx[cluster_id, anchor_slot].item())
        if an >= 0:
            iou_ya = float(box_iou(ybox.unsqueeze(0), node_box[an].unsqueeze(0))[0, 0].item())
            out[2] = iou_ya
            out[3] = ys - float(node_score[an].item())
    if bool(slot_avail[cluster_id, rtdetr_slot].item()):
        rn = int(slot_node_idx[cluster_id, rtdetr_slot].item())
        if rn >= 0:
            iou_yr = float(box_iou(ybox.unsqueeze(0), node_box[rn].unsqueeze(0))[0, 0].item())
            out[4] = iou_yr
            out[5] = 1.0 - iou_yr  # localization disagreement with rt_detr
            out[6] = ys - float(node_score[rn].item())
    out[7] = 1.0 if ys > 0.7 else 0.0
    return out


def cluster_box_variance(
    cluster_id: int,
    slot_node_idx: torch.Tensor,
    slot_avail: torch.Tensor,
    node_box: torch.Tensor,
) -> float:
    """Mean pairwise center-distance variance across all available slots."""
    S = slot_node_idx.shape[1]
    centers = []
    for s in range(S):
        if not bool(slot_avail[cluster_id, s].item()):
            continue
        n = int(slot_node_idx[cluster_id, s].item())
        if n < 0:
            continue
        cx, cy, _, _ = _to_xywh(node_box[n])
        centers.append((cx, cy))
    if len(centers) < 2:
        return 0.0
    xs = torch.tensor([c[0] for c in centers])
    ys = torch.tensor([c[1] for c in centers])
    return float((xs.var(unbiased=False) + ys.var(unbiased=False)).item())


def cluster_score_entropy(
    cluster_id: int,
    slot_node_idx: torch.Tensor,
    slot_avail: torch.Tensor,
    node_score: torch.Tensor,
) -> float:
    """Shannon entropy of (normalized) per-slot scores in this cluster."""
    S = slot_node_idx.shape[1]
    scores = []
    for s in range(S):
        if not bool(slot_avail[cluster_id, s].item()):
            continue
        n = int(slot_node_idx[cluster_id, s].item())
        if n < 0:
            continue
        scores.append(max(1e-6, float(node_score[n].item())))
    if not scores:
        return 0.0
    t = torch.tensor(scores)
    p = t / t.sum()
    return float(-(p * (p + 1e-9).log()).sum().item())
