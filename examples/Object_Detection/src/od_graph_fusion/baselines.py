"""Classical detection fusion baselines (NMS, Soft-NMS, WBF)."""
from __future__ import annotations

from typing import List, Tuple

import torch

from .box_ops import box_iou, weighted_box_average


def pool_detector_results(detector_results: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate boxes/scores/labels/detector-ids from a list of DetectionResults."""
    boxes_list, scores_list, labels_list, det_list = [], [], [], []
    for i, r in enumerate(detector_results):
        if r is None or r.num_detections() == 0:
            continue
        boxes_list.append(r.boxes_xyxy)
        scores_list.append(r.scores)
        labels_list.append(r.label_ids)
        det_list.append(torch.full((r.boxes_xyxy.shape[0],), i, dtype=torch.long))
    if not boxes_list:
        return (torch.zeros(0, 4), torch.zeros(0),
                torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
    return (torch.cat(boxes_list), torch.cat(scores_list),
            torch.cat(labels_list), torch.cat(det_list))


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """Standard NMS. Returns kept indices (long)."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i:i+1], boxes[rest])[0]
        order = rest[ious < iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def soft_nms(
    boxes: torch.Tensor, scores: torch.Tensor,
    sigma: float = 0.5, score_threshold: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gaussian Soft-NMS. Returns (kept indices, decayed scores)."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0)
    decayed = scores.clone().float()
    N = boxes.shape[0]
    kept = []
    indices = torch.arange(N)
    while decayed.numel() > 0:
        i = int(decayed.argmax().item())
        if decayed[i].item() < score_threshold:
            break
        global_i = int(indices[i].item())
        kept.append(global_i)
        # Decay
        ious = box_iou(boxes[global_i:global_i+1], boxes[indices])[0]
        decay = torch.exp(-(ious ** 2) / sigma)
        decayed = decayed * decay
        # Remove the chosen one
        mask = torch.ones_like(decayed, dtype=torch.bool)
        mask[i] = False
        decayed = decayed[mask]
        indices = indices[mask]
    return torch.tensor(kept, dtype=torch.long), scores[kept] if kept else torch.zeros(0)


def weighted_boxes_fusion(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
    iou_threshold: float = 0.55,
    skip_box_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation of Weighted Boxes Fusion (Solovyev et al., 2021).

    Groups boxes by IoU + class agreement, then averages each group.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels
    # Keep only those above threshold
    keep = scores >= skip_box_threshold
    boxes = boxes[keep]; scores = scores[keep]; labels = labels[keep]
    if boxes.numel() == 0:
        return boxes, scores, labels

    order = scores.argsort(descending=True)
    boxes = boxes[order]; scores = scores[order]; labels = labels[order]

    fused_boxes = []
    fused_scores = []
    fused_labels = []
    used = torch.zeros(boxes.shape[0], dtype=torch.bool)

    for i in range(boxes.shape[0]):
        if used[i]:
            continue
        ious = box_iou(boxes[i:i+1], boxes)[0]
        # cluster = same class & IoU >= threshold
        cluster_mask = (ious >= iou_threshold) & (labels == labels[i]) & (~used)
        cluster_idx = cluster_mask.nonzero(as_tuple=True)[0]
        if cluster_idx.numel() == 0:
            continue
        group_boxes = boxes[cluster_idx]
        group_scores = scores[cluster_idx]
        fb = weighted_box_average(group_boxes, group_scores)
        fs = group_scores.mean() * min(1.0, group_scores.numel() / 3.0)  # mild support boost
        fused_boxes.append(fb)
        fused_scores.append(fs)
        fused_labels.append(int(labels[i].item()))
        used[cluster_idx] = True

    if not fused_boxes:
        return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return (torch.stack(fused_boxes, dim=0),
            torch.stack(fused_scores, dim=0) if fused_scores else torch.zeros(0),
            torch.tensor(fused_labels, dtype=torch.long))


def best_single_detector_baseline(detector_results: List, conf_threshold: float = 0.25):
    """Pick the highest-confidence detection per cluster from a single detector."""
    return pool_detector_results(detector_results)
