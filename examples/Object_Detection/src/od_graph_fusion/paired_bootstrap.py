"""Paired bootstrap for AP comparisons.

Given two methods evaluated on the same images, draws B bootstrap samples
of images (with replacement) and computes P(A > B) = fraction of resamples
where mean AP_A > mean AP_B. The image-level pairing is preserved on every
resample, which is critical for low-sample (n<=50) test splits.

Used by Step 06 verdict logic. Never used at training time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _ap_for_image(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> float:
    """Per-image AP at a given IoU threshold.

    Used to build the paired vector of per-image APs that the bootstrap
    resamples. Greedy GT matching by descending score is the standard
    VOC-style decision rule.
    """
    from .box_ops import box_iou
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 1.0
    if pred_boxes.numel() == 0:
        return 0.0
    if gt_boxes.numel() == 0:
        return 0.0

    order = pred_scores.argsort(descending=True)
    pb = pred_boxes[order]
    pl = pred_labels[order]
    matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp = torch.zeros(pb.shape[0], dtype=torch.float32)
    if pb.shape[0] > 0:
        ious = box_iou(pb, gt_boxes)
        for i in range(pb.shape[0]):
            best_gt = -1; best_iou = iou_threshold
            for g in range(gt_boxes.shape[0]):
                if matched[g]:
                    continue
                if not class_agnostic and int(pl[i].item()) != int(gt_labels[g].item()):
                    continue
                v = float(ious[i, g].item())
                if v >= best_iou:
                    best_iou = v
                    best_gt = g
            if best_gt >= 0:
                matched[best_gt] = True
                tp[i] = 1.0

    n_gt = int(gt_boxes.shape[0])
    # 11-point interpolated AP — small-sample stable, matches VOC07.
    recalls = tp.cumsum(0) / max(1, n_gt)
    precisions = tp.cumsum(0) / torch.arange(1, tp.shape[0] + 1, dtype=torch.float32)
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        mask = recalls >= t
        p = float(precisions[mask].max().item()) if mask.any() else 0.0
        ap += p / 11.0
    return ap


def per_image_aps(
    preds: Sequence,
    gts: Sequence,
    iou_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> Tuple[List[str], torch.Tensor]:
    """Return (image_ids in order, per-image AP vector) for paired comparisons.

    preds: sequence of DetectionPrediction-like objects with .image_id,
           .boxes_xyxy, .scores, .labels.
    gts:   sequence of GroundTruth-like objects with .image_id, .boxes_xyxy,
           .labels.
    """
    gt_by_id = {g.image_id: g for g in gts}
    ids: List[str] = []
    aps: List[float] = []
    for p in preds:
        g = gt_by_id.get(p.image_id)
        if g is None:
            continue
        ap = _ap_for_image(
            p.boxes_xyxy, p.scores, p.labels,
            g.boxes_xyxy, g.labels,
            iou_threshold=iou_threshold, class_agnostic=class_agnostic,
        )
        ids.append(p.image_id)
        aps.append(ap)
    return ids, torch.tensor(aps, dtype=torch.float32)


def paired_bootstrap(
    ap_a: torch.Tensor,
    ap_b: torch.Tensor,
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap of mean AP difference.

    ap_a, ap_b: 1-D tensors of per-image APs over the *same* images, same order.

    Returns dict with:
      p_a_gt_b: fraction of resamples where mean(A) > mean(B)
      mean_diff: mean(A) - mean(B)
      ci95_low / ci95_high: 2.5/97.5 percentile of resampled (mean_a - mean_b)
      n_images: paired sample size
    """
    if ap_a.shape != ap_b.shape:
        raise ValueError(f"paired vectors must have same shape, got {ap_a.shape} vs {ap_b.shape}")
    n = int(ap_a.shape[0])
    if n == 0:
        return {"p_a_gt_b": 0.5, "mean_diff": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n_images": 0}

    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_resamples, n), generator=g)
    a_means = ap_a[idx].mean(dim=1)
    b_means = ap_b[idx].mean(dim=1)
    diffs = a_means - b_means
    p_gt = float((diffs > 0).float().mean().item())
    qs = torch.tensor([0.025, 0.975])
    lo, hi = torch.quantile(diffs, qs).tolist()
    return {
        "p_a_gt_b": p_gt,
        "mean_diff": float(diffs.mean().item()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_images": n,
    }


def verdict_from_bootstrap(
    boot: Dict[str, float],
    *,
    win_threshold: float = 0.95,
    tie_threshold: float = 0.85,
) -> str:
    """Interpret paired bootstrap output as a 3-way verdict label.

    "WIN":           p_a_gt_b >= win_threshold AND mean_diff > 0
    "TIE":           tie_threshold <= p_a_gt_b < win_threshold OR
                     (mean_diff close to 0 and CI straddles 0)
    "NOT_YET_WIN":   p_a_gt_b < tie_threshold
    """
    p = boot.get("p_a_gt_b", 0.5)
    md = boot.get("mean_diff", 0.0)
    lo = boot.get("ci95_low", 0.0)
    hi = boot.get("ci95_high", 0.0)
    if p >= win_threshold and md > 0.0:
        return "WIN"
    if (lo <= 0.0 <= hi) or (tie_threshold <= p < win_threshold):
        return "TIE"
    return "NOT_YET_WIN"
