"""Detection evaluation metrics.

Implements a simple but honest AP / precision / recall / F1 evaluator,
plus AP@multiple IoU thresholds. Not COCO-API exact, but well-defined
and reproducible.

This module is intentionally CPU-side and does not require pycocotools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from .box_ops import box_iou


@dataclass
class DetectionPrediction:
    """Final detection output for one image (after fusion or single detector)."""
    image_id: str
    boxes_xyxy: torch.Tensor      # [N, 4]
    scores: torch.Tensor          # [N]
    labels: torch.Tensor          # [N] long


@dataclass
class GroundTruth:
    image_id: str
    boxes_xyxy: torch.Tensor
    labels: torch.Tensor


def _match_predictions(
    pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
    iou_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match predictions to GT greedily (highest score first).

    Returns:
        tp:  [N_pred] bool. True if matched & class agrees.
        fn:  scalar (number of unmatched GT).
    """
    N = pred_boxes.shape[0]
    G = gt_boxes.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.bool), torch.tensor(G)
    if G == 0:
        return torch.zeros(N, dtype=torch.bool), torch.tensor(0)

    order = pred_scores.argsort(descending=True)
    matched_gt = torch.zeros(G, dtype=torch.bool)
    tp = torch.zeros(N, dtype=torch.bool)
    ious = box_iou(pred_boxes, gt_boxes)

    for idx in order.tolist():
        # Restrict to GT of matching class
        cls_mask = gt_labels == pred_labels[idx]
        cls_mask = cls_mask & (~matched_gt)
        if cls_mask.sum() == 0:
            continue
        row = ious[idx].clone()
        row[~cls_mask] = -1.0
        best_iou, best_g = row.max(dim=0)
        if best_iou.item() >= iou_threshold:
            tp[idx] = True
            matched_gt[best_g] = True

    fn = (~matched_gt).sum()
    return tp, fn


def evaluate_predictions(
    predictions: List[DetectionPrediction],
    ground_truths: List[GroundTruth],
    iou_threshold: float = 0.5,
    num_classes: int = 20,
) -> Dict[str, Any]:
    """Compute AP@iou_threshold, precision, recall, F1.

    Pools per-class AP using all images.
    """
    gt_by_id = {gt.image_id: gt for gt in ground_truths}
    # Per-class TP/FP/FN aggregation
    per_class_scores: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    per_class_tp: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    per_class_total_gt: Dict[int, int] = {c: 0 for c in range(num_classes)}

    n_pred = 0; n_tp = 0; n_fp = 0
    total_gt = 0

    for pred in predictions:
        gt = gt_by_id.get(pred.image_id)
        if gt is None:
            continue
        total_gt += int(gt.boxes_xyxy.shape[0])
        for c in gt.labels.tolist():
            per_class_total_gt[int(c)] = per_class_total_gt.get(int(c), 0) + 1

        tp, _fn = _match_predictions(
            pred.boxes_xyxy, pred.scores, pred.labels,
            gt.boxes_xyxy, gt.labels, iou_threshold,
        )
        n_pred += pred.boxes_xyxy.shape[0]
        n_tp += int(tp.sum().item())
        n_fp += int((~tp).sum().item())
        for i in range(pred.boxes_xyxy.shape[0]):
            c = int(pred.labels[i].item())
            per_class_scores.setdefault(c, []).append(float(pred.scores[i].item()))
            per_class_tp.setdefault(c, []).append(int(tp[i].item()))

    precision = n_tp / max(1, n_pred)
    recall = n_tp / max(1, total_gt)
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))

    # Per-class AP (11-point or all-point; use all-point precision-recall AUC)
    per_class_ap: Dict[int, float] = {}
    for c in per_class_scores:
        scores = torch.tensor(per_class_scores[c], dtype=torch.float32)
        tps = torch.tensor(per_class_tp[c], dtype=torch.float32)
        total = per_class_total_gt.get(c, 0)
        if scores.numel() == 0 or total == 0:
            per_class_ap[c] = 0.0
            continue
        order = scores.argsort(descending=True)
        tps_ord = tps[order]
        fps_ord = 1 - tps_ord
        cum_tp = tps_ord.cumsum(0)
        cum_fp = fps_ord.cumsum(0)
        prec = cum_tp / (cum_tp + cum_fp).clamp(min=1)
        rec = cum_tp / max(1, total)
        # Make precision monotonically decreasing (PASCAL VOC style)
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i].item(), prec[i + 1].item())
        # AP = area under PR curve
        ap = 0.0
        last_r = 0.0
        for i in range(len(prec)):
            ap += (rec[i].item() - last_r) * prec[i].item()
            last_r = rec[i].item()
        per_class_ap[c] = float(ap)

    mAP = sum(per_class_ap.values()) / max(1, len(per_class_ap))
    return {
        "iou_threshold": iou_threshold,
        "num_predictions": n_pred,
        "num_tp": n_tp, "num_fp": n_fp,
        "num_gt": total_gt,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class_ap": per_class_ap,
        "mAP": float(mAP),
        "AP": float(mAP),  # alias
    }


def evaluate_at_multiple_ious(
    predictions: List[DetectionPrediction],
    ground_truths: List[GroundTruth],
    iou_thresholds: List[float] = (0.5, 0.75),
    num_classes: int = 20,
) -> Dict[str, Any]:
    out = {}
    for t in iou_thresholds:
        r = evaluate_predictions(predictions, ground_truths,
                                  iou_threshold=t, num_classes=num_classes)
        out[f"AP@{t:.2f}"] = r["AP"]
        out[f"precision@{t:.2f}"] = r["precision"]
        out[f"recall@{t:.2f}"] = r["recall"]
        out[f"f1@{t:.2f}"] = r["f1"]
    return out
