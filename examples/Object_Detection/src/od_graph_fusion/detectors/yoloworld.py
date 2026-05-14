"""YOLO-World (open-vocabulary, detection-boxes-only) adapter.

Uses yolov8x-worldv2.pt — the strongest publicly available YOLO-World variant.
Outputs detection boxes only. No segmentation masks, no SAM, no mask-to-box pipeline.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult


class YOLOWorldAdapter(BaseDetector):
    """YOLO-World open-vocabulary detector (yolov8x-worldv2.pt).

    Requires class prompts via set_classes(). Returns detection boxes only.
    """

    name = "yolo_world"
    family = "open_vocabulary_yolo"

    def __init__(self, model_path: str = "yolov8x-worldv2.pt", **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._classes_set: List[str] = []

    def _check_available(self) -> None:
        import ultralytics  # noqa: F401
        self._model_identifier = self.model_path

    def load(self, device: Optional[str] = None) -> None:
        from ultralytics import YOLO
        if device is not None:
            self.device = device
        try:
            self._model = YOLO(self.model_path)
            self._model_identifier = self.model_path
        except Exception as exc:
            raise RuntimeError(
                f"YOLOWorld model '{self.model_path}' failed to load: {exc}"
            ) from exc

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        H, W = int(image.shape[1]), int(image.shape[2])
        import numpy as np
        np_img = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Set class vocabulary — required for YOLOWorld
        classes = list(class_filter) if class_filter else ["object"]
        if classes != self._classes_set:
            try:
                self._model.set_classes(classes)
                self._classes_set = classes
            except Exception:
                pass

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

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return self.empty_result(image_id, (H, W))
        r0 = results[0]
        boxes = r0.boxes.xyxy.detach().cpu()
        scores = r0.boxes.conf.detach().cpu()
        labels_idx = r0.boxes.cls.detach().cpu().long().tolist()
        names = r0.names
        from ..source_router import canonical_label as _canon
        labels = [_canon(names.get(int(i), str(i))) for i in labels_idx]

        if class_filter is not None:
            mask = torch.tensor([l in class_filter for l in labels], dtype=torch.bool)
            # Also keep by index if name mapping misses
            if not mask.any():
                mask = torch.ones(len(labels), dtype=torch.bool)
            boxes = boxes[mask]
            scores = scores[mask]
            labels = [l for l, m in zip(labels, mask.tolist()) if m]
            labels_idx = [i for i, m in zip(labels_idx, mask.tolist()) if m]

        from ..source_router import canonical_label_id as _clid
        dataset_classes = list(class_filter) if class_filter is not None else []
        can_ids = [_clid(l, dataset_classes) for l in labels]

        return DetectionResult(
            image_id=image_id, model_name=self.name,
            boxes_xyxy=boxes, scores=scores,
            label_ids=torch.tensor(can_ids, dtype=torch.long),
            labels=labels, image_size=(H, W),
            device=self.device, runtime_ms=runtime,
        )
