"""Torchvision SSDLite adapter."""
from __future__ import annotations

import time
from typing import Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult
from .faster_rcnn import COCO_NAMES


class SSDAdapter(BaseDetector):
    """Torchvision SSDLite320 MobileNetV3 (COCO weights)."""

    name = "ssd"
    family = "anchor_based_cnn"

    def _check_available(self) -> None:
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large  # noqa
        self._model_identifier = "ssdlite320_mobilenet_v3_large_coco"

    def load(self, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = device
        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )
        self._model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
        ).to(self.device).eval()
        self._model_identifier = "ssdlite320_mobilenet_v3_large_coco"

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        H, W = int(image.shape[1]), int(image.shape[2])
        t0 = time.time()
        try:
            preds = self._model([image.float().to(self.device)])
        except Exception as exc:
            return self.empty_result(image_id, (H, W), error=str(exc))
        runtime = (time.time() - t0) * 1000
        p = preds[0]
        boxes = p["boxes"].detach().cpu()
        scores = p["scores"].detach().cpu()
        label_indices = p["labels"].detach().cpu().long()
        keep = scores >= self.conf
        boxes = boxes[keep]; scores = scores[keep]; label_indices = label_indices[keep]
        if boxes.numel() == 0:
            return self.empty_result(image_id, (H, W))
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
