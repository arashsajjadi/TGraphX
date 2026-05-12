"""Ultralytics YOLO adapter (modern YOLO family)."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult


class YOLOAdapter(BaseDetector):
    """Modern Ultralytics YOLO (v8/v10/v11)."""

    name = "yolo_modern"
    family = "anchor_free_cnn"

    def __init__(self, model_path: str = "yolo11n.pt", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

    def _check_available(self) -> None:
        import ultralytics  # noqa: F401
        self._model_identifier = self.model_path

    def load(self, device: Optional[str] = None) -> None:
        from ultralytics import YOLO
        if device is not None:
            self.device = device
        # Try the requested model; ultralytics will download if cached
        try:
            self._model = YOLO(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"YOLO model '{self.model_path}' failed to load: {exc}. "
                "Try a smaller variant such as yolo11n.pt or pre-download weights."
            ) from exc
        self._model_identifier = self.model_path

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        if image.dim() != 3:
            raise ValueError(f"image must be [C, H, W], got {tuple(image.shape)}")
        H, W = int(image.shape[1]), int(image.shape[2])
        # Ultralytics accepts PIL or numpy or torch (HWC uint8). Convert from CHW float.
        import numpy as np
        np_img = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        t0 = time.time()
        try:
            results = self._model.predict(
                np_img, conf=self.conf, iou=self.iou,
                device=self.device if self.device != "cpu" else "cpu",
                verbose=False,
            )
        except Exception as exc:
            return self.empty_result(image_id, (H, W), error=str(exc))
        runtime = (time.time() - t0) * 1000

        if not results:
            return self.empty_result(image_id, (H, W))
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return self.empty_result(image_id, (H, W))
        boxes = r0.boxes.xyxy.detach().cpu()
        scores = r0.boxes.conf.detach().cpu()
        labels_idx = r0.boxes.cls.detach().cpu().long().tolist()
        names = r0.names  # dict idx → name
        from ..source_router import canonical_label as _canon
        raw_labels = [names.get(int(i), str(i)) for i in labels_idx]
        labels = [_canon(l) for l in raw_labels]

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
