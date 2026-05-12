"""torchvision RetinaNet adapter."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult


# COCO 80-class names that torchvision detectors return (1-indexed).
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


class RetinaNetAdapter(BaseDetector):
    """RetinaNet (torchvision) — always available with COCO-pretrained weights."""

    name = "retinanet"
    family = "anchor_based_cnn"

    def _check_available(self) -> None:
        from torchvision.models.detection import retinanet_resnet50_fpn  # noqa: F401
        self._model_identifier = "retinanet_resnet50_fpn_v2"

    def load(self, device: Optional[str] = None) -> None:
        from torchvision.models.detection import (
            retinanet_resnet50_fpn_v2, retinanet_resnet50_fpn,
        )
        if device is not None:
            self.device = device
        # Try v2 first
        try:
            self._model = retinanet_resnet50_fpn_v2(weights="DEFAULT", box_score_thresh=self.conf)
            self._model_identifier = "retinanet_resnet50_fpn_v2"
        except Exception:
            self._model = retinanet_resnet50_fpn(weights="DEFAULT", box_score_thresh=self.conf)
            self._model_identifier = "retinanet_resnet50_fpn"
        self._model.eval().to(self.device)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        # image is [C, H, W] in [0, 1]
        if image.dim() != 3:
            raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
        H, W = int(image.shape[1]), int(image.shape[2])
        x = image.unsqueeze(0).to(self.device)
        t0 = time.time()
        outputs = self._model(x)[0]
        runtime = (time.time() - t0) * 1000

        boxes = outputs["boxes"].detach().cpu()
        scores = outputs["scores"].detach().cpu()
        labels_idx = outputs["labels"].detach().cpu().tolist()

        keep = scores >= self.conf
        boxes = boxes[keep]
        scores = scores[keep]
        labels_idx = [l for l, k in zip(labels_idx, keep.tolist()) if k]
        labels = [COCO_INSTANCE_CATEGORY_NAMES[i] if 0 <= i < len(COCO_INSTANCE_CATEGORY_NAMES) else "unknown"
                  for i in labels_idx]

        if class_filter is not None:
            mask = torch.tensor([l in class_filter for l in labels], dtype=torch.bool)
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [l for l, m in zip(labels, mask.tolist()) if m]
            labels_idx = [i for i, m in zip(labels_idx, mask.tolist()) if m]

        return DetectionResult(
            image_id=image_id, model_name=self.name,
            boxes_xyxy=boxes, scores=scores,
            label_ids=torch.tensor(labels_idx, dtype=torch.long),
            labels=labels, image_size=(H, W),
            device=self.device, runtime_ms=runtime,
        )
