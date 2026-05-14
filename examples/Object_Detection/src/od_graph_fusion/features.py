"""Node and edge feature extraction for detection graphs."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .box_ops import box_area, box_center, box_iou


def crop_tensor_from_image(
    image: torch.Tensor,
    box_xyxy: torch.Tensor,
    crop_size: int = 64,
) -> torch.Tensor:
    """Crop a tensor region from ``image [C, H, W]`` and resize to ``[C, crop_size, crop_size]``.

    Preserves tensor-native semantics: returns a 3-D tensor with the original
    channel dim. Tiny / negative boxes are returned as zeros of the right shape.
    """
    C, H, W = image.shape
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(W, int(round(x2)))
    y2 = min(H, int(round(y2)))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return torch.zeros(C, crop_size, crop_size, dtype=image.dtype, device=image.device)
    crop = image[:, y1:y2, x1:x2]
    # Resize via bilinear; uses 4-D input
    crop = F.interpolate(crop.unsqueeze(0), size=(crop_size, crop_size),
                          mode="bilinear", align_corners=False)[0]
    return crop


def proposal_metadata(
    box: torch.Tensor,
    score: float,
    label_id: int,
    detector_id: int,
    num_detectors: int,
    num_classes: int,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Build a metadata vector for a single proposal node.

    Layout (length: 8 + num_detectors + num_classes):
        - normalized cx, cy, w, h         (4)
        - score                            (1)
        - log area                         (1)
        - aspect ratio                     (1)
        - is_proposal_flag                 (1)
        - detector one-hot                 (num_detectors)
        - class one-hot                    (num_classes)
    """
    H, W = image_size
    x1, y1, x2, y2 = box.tolist()
    cx = ((x1 + x2) / 2) / max(W, 1)
    cy = ((y1 + y2) / 2) / max(H, 1)
    w = (x2 - x1) / max(W, 1)
    h = (y2 - y1) / max(H, 1)
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    log_area = torch.log(torch.tensor(area)).item() / 10.0  # rough normalization
    ar = (x2 - x1 + 1e-6) / (y2 - y1 + 1e-6)
    ar = max(0.05, min(20.0, ar))

    det_onehot = torch.zeros(num_detectors)
    if 0 <= detector_id < num_detectors:
        det_onehot[detector_id] = 1.0
    cls_onehot = torch.zeros(num_classes)
    if 0 <= label_id < num_classes:
        cls_onehot[label_id] = 1.0

    scalar = torch.tensor([cx, cy, w, h, score, log_area, ar, 1.0],
                          dtype=torch.float32)
    return torch.cat([scalar, det_onehot, cls_onehot], dim=0)


def cluster_metadata(
    cluster_box: torch.Tensor,
    mean_score: float,
    max_score: float,
    num_supporting: int,
    detector_diversity: float,
    num_detectors: int,
    num_classes: int,
    label_id: int,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """Metadata vector for a candidate-cluster node. Same length as proposal_metadata."""
    H, W = image_size
    x1, y1, x2, y2 = cluster_box.tolist()
    cx = ((x1 + x2) / 2) / max(W, 1)
    cy = ((y1 + y2) / 2) / max(H, 1)
    w = (x2 - x1) / max(W, 1)
    h = (y2 - y1) / max(H, 1)
    log_n = min(1.0, num_supporting / max(num_detectors, 1))
    diversity = max(0.0, min(1.0, detector_diversity))

    det_onehot = torch.zeros(num_detectors)
    cls_onehot = torch.zeros(num_classes)
    if 0 <= label_id < num_classes:
        cls_onehot[label_id] = 1.0

    # detector_diversity stored in a dedicated scalar (index 7); det_onehot is
    # all-zeros for fusion nodes (no single source detector). Previously this
    # was incorrectly set to det_onehot * diversity, which broke the one-hot
    # semantics. The diversity scalar is encoded separately in the log_n slot.
    scalar = torch.tensor([cx, cy, w, h, max_score, mean_score, log_n, 0.0],
                          dtype=torch.float32)
    return torch.cat([scalar, det_onehot, cls_onehot], dim=0)


def edge_feature_vector(
    box_a: torch.Tensor,
    box_b: torch.Tensor,
    score_a: float,
    score_b: float,
    label_a: int,
    label_b: int,
    detector_a: int,
    detector_b: int,
    edge_type_id: int,
    num_edge_types: int,
) -> torch.Tensor:
    """Build an edge feature vector.

    Layout:
        - iou
        - center_distance (image-normalized would require image_size; we use raw pixels / 100)
        - area ratio (small / large)
        - same_class
        - same_detector
        - score difference
        - edge type one-hot   (num_edge_types)
    """
    iou_v = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0))[0, 0].item()
    ca = box_center(box_a.unsqueeze(0))[0]
    cb = box_center(box_b.unsqueeze(0))[0]
    cd = (ca - cb).norm().item() / 100.0
    aa = max(box_area(box_a.unsqueeze(0))[0].item(), 1e-6)
    ab = max(box_area(box_b.unsqueeze(0))[0].item(), 1e-6)
    ar_ratio = min(aa, ab) / max(aa, ab)
    same_class = 1.0 if label_a == label_b else 0.0
    same_det = 1.0 if detector_a == detector_b else 0.0
    score_diff = abs(float(score_a) - float(score_b))

    edge_onehot = torch.zeros(num_edge_types)
    if 0 <= edge_type_id < num_edge_types:
        edge_onehot[edge_type_id] = 1.0

    scalar = torch.tensor([iou_v, cd, ar_ratio, same_class, same_det, score_diff],
                          dtype=torch.float32)
    return torch.cat([scalar, edge_onehot], dim=0)
