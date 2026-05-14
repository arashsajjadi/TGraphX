"""Torchvision Faster R-CNN adapter."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult

# COCO class names (80 classes, 1-indexed in torchvision models)
COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class FasterRCNNAdapter(BaseDetector):
    """Torchvision Faster R-CNN ResNet-50 FPN v2 (COCO weights)."""

    name = "faster_rcnn"
    family = "two_stage_cnn"

    def _check_available(self) -> None:
        import torchvision  # noqa: F401
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2  # noqa: F401
        self._model_identifier = "fasterrcnn_resnet50_fpn_v2_coco"

    def load(self, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = device
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        dev = self.device
        self._model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        ).to(dev).eval()
        self._model_identifier = "fasterrcnn_resnet50_fpn_v2_coco"

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        if image.dim() != 3:
            raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
        H, W = int(image.shape[1]), int(image.shape[2])
        dev = self.device
        inp = [image.float().to(dev)]
        t0 = time.time()
        try:
            preds = self._model(inp)
        except Exception as exc:
            return self.empty_result(image_id, (H, W), error=str(exc))
        runtime = (time.time() - t0) * 1000

        p = preds[0]
        boxes = p["boxes"].detach().cpu()
        scores = p["scores"].detach().cpu()
        label_indices = p["labels"].detach().cpu().long()

        # Filter by confidence threshold
        keep = scores >= self.conf
        boxes = boxes[keep]; scores = scores[keep]; label_indices = label_indices[keep]
        if boxes.numel() == 0:
            return self.empty_result(image_id, (H, W))

        # Map torchvision's 1-indexed COCO labels to canonical names
        raw_labels = [COCO_NAMES[int(i)] if int(i) < len(COCO_NAMES) else str(int(i))
                      for i in label_indices.tolist()]
        from ..source_router import canonical_label as _canon
        labels = [_canon(l) for l in raw_labels]

        if class_filter is not None:
            mask = torch.tensor([l in class_filter for l in labels], dtype=torch.bool)
            if not mask.any():
                return self.empty_result(image_id, (H, W))
            boxes = boxes[mask]; scores = scores[mask]
            labels = [l for l, m in zip(labels, mask.tolist()) if m]

        from ..source_router import canonical_label_id as _clid
        dataset_classes = list(class_filter) if class_filter is not None else []
        can_ids = torch.tensor([_clid(l, dataset_classes) for l in labels], dtype=torch.long)
        return DetectionResult(
            image_id=image_id, model_name=self.name,
            boxes_xyxy=boxes, scores=scores,
            label_ids=can_ids, labels=labels,
            image_size=(H, W), device=self.device, runtime_ms=runtime,
        )
