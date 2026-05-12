"""Matching proposals to candidate clusters and ground-truth boxes."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .box_ops import box_iou


def cluster_proposals(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    detector_ids: torch.Tensor,
    iou_threshold: float = 0.5,
    require_same_class: bool = True,
) -> torch.Tensor:
    """Greedy clustering of proposals across detectors.

    Returns a ``[N_proposals]`` long tensor of cluster IDs (0..C-1).

    Two proposals belong to the same cluster if:
      - IoU >= ``iou_threshold``;
      - and ``require_same_class`` is False OR they share a class label;
      - and they come from different detectors (same-detector suppression is
        modelled as a separate edge, not as clustering).
    """
    N = boxes.shape[0]
    if N == 0:
        return torch.zeros(0, dtype=torch.long)

    ious = box_iou(boxes, boxes)
    # Suppress self-pairs by setting diagonal to 0
    ious.fill_diagonal_(0.0)

    cluster_id = torch.full((N,), -1, dtype=torch.long)
    next_id = 0
    # Process in confidence order if scores embedded? We don't have them here;
    # process by index. Stability is fine for FAST_SMOKE.
    for i in range(N):
        if cluster_id[i] != -1:
            continue
        cluster_id[i] = next_id
        for j in range(i + 1, N):
            if cluster_id[j] != -1:
                continue
            if ious[i, j].item() < iou_threshold:
                continue
            if detector_ids[i].item() == detector_ids[j].item():
                # Same detector — represented via same_detector_suppression edge,
                # not via cluster merging.
                continue
            if require_same_class and labels[i].item() != labels[j].item():
                continue
            cluster_id[j] = next_id
        next_id += 1
    return cluster_id


def match_to_gt(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each proposal, find best-matching GT.

    Returns:
        matched_gt_idx: [N] long. -1 if unmatched.
        matched_iou:    [N] float.
        is_correct:     [N] bool. True if matched IoU >= threshold and labels agree.
    """
    N = boxes.shape[0]
    G = gt_boxes.shape[0]
    if N == 0 or G == 0:
        return (torch.full((N,), -1, dtype=torch.long),
                torch.zeros(N, dtype=torch.float32),
                torch.zeros(N, dtype=torch.bool))

    ious = box_iou(boxes, gt_boxes)
    best_iou, best_idx = ious.max(dim=1)
    same_class = (labels[:, None] == gt_labels[None, :]).gather(1, best_idx.unsqueeze(1)).squeeze(1)
    is_correct = (best_iou >= iou_threshold) & same_class
    matched_gt_idx = torch.where(best_iou >= iou_threshold, best_idx, torch.full_like(best_idx, -1))
    return matched_gt_idx, best_iou, is_correct
