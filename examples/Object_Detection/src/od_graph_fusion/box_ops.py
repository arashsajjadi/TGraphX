"""Bounding-box geometry utilities (xyxy format, normalized to image size).

All functions are vectorized PyTorch. Boxes are ``[N, 4]`` ``(x1, y1, x2, y2)``.
"""
from __future__ import annotations

from typing import Tuple

import torch


def clip_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Clip xyxy boxes to ``(H, W)``."""
    H, W = image_size
    out = boxes.clone()
    out[:, 0] = out[:, 0].clamp(0, W - 1)
    out[:, 1] = out[:, 1].clamp(0, H - 1)
    out[:, 2] = out[:, 2].clamp(0, W - 1)
    out[:, 3] = out[:, 3].clamp(0, H - 1)
    return out


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Area of xyxy boxes ``[N]``."""
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return w * h


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU. Returns ``[N1, N2]``.

    Both inputs are xyxy. Empty inputs return a correctly-shaped zero matrix.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]),
                           device=boxes1.device, dtype=boxes1.dtype)
    a1 = box_area(boxes1)
    a2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU ``[N1, N2]``."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]),
                           device=boxes1.device, dtype=boxes1.dtype)
    a1 = box_area(boxes1)
    a2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter
    iou = inter / union.clamp(min=1e-9)
    # enclosing box
    lt_e = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_e = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    enc = wh_e[..., 0] * wh_e[..., 1]
    return iou - (enc - union) / enc.clamp(min=1e-9)


def box_center(boxes: torch.Tensor) -> torch.Tensor:
    """Return ``[N, 2]`` (cx, cy)."""
    return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], dim=1)


def box_wh(boxes: torch.Tensor) -> torch.Tensor:
    """Return ``[N, 2]`` (w, h)."""
    return torch.stack([(boxes[:, 2] - boxes[:, 0]).clamp(min=0),
                        (boxes[:, 3] - boxes[:, 1]).clamp(min=0)], dim=1)


def center_distance(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean center distance ``[N1, N2]``."""
    c1 = box_center(boxes1)
    c2 = box_center(boxes2)
    return torch.cdist(c1, c2, p=2)


def weighted_box_average(
    boxes: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted average box. ``boxes [N, 4]``, ``weights [N]``."""
    if boxes.numel() == 0:
        return torch.zeros(4, device=boxes.device, dtype=boxes.dtype)
    w = weights.clamp(min=1e-9)
    return (boxes * w[:, None]).sum(0) / w.sum()


def union_box(boxes: torch.Tensor) -> torch.Tensor:
    """Smallest enclosing box of N input boxes. ``[4]``."""
    if boxes.numel() == 0:
        return torch.zeros(4, device=boxes.device, dtype=boxes.dtype)
    x1 = boxes[:, 0].min()
    y1 = boxes[:, 1].min()
    x2 = boxes[:, 2].max()
    y2 = boxes[:, 3].max()
    return torch.stack([x1, y1, x2, y2])


def intersection_box(boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise intersection box of N input boxes. Returns zeros if no overlap."""
    if boxes.numel() == 0:
        return torch.zeros(4, device=boxes.device, dtype=boxes.dtype)
    x1 = boxes[:, 0].max()
    y1 = boxes[:, 1].max()
    x2 = boxes[:, 2].min()
    y2 = boxes[:, 3].min()
    if x2 < x1 or y2 < y1:
        return torch.zeros(4, device=boxes.device, dtype=boxes.dtype)
    return torch.stack([x1, y1, x2, y2])


def normalize_boxes(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """Normalize xyxy to [0, 1] using image (H, W)."""
    H, W = image_size
    if boxes.numel() == 0:
        return boxes.clone()
    out = boxes.clone().float()
    out[:, 0] /= W
    out[:, 1] /= H
    out[:, 2] /= W
    out[:, 3] /= H
    return out
