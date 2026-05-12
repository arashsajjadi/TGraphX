"""Open-vocabulary YOLO adapter (YOLOE / YOLO-World)."""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult


class YOLOEAdapter(BaseDetector):
    """Open-vocabulary YOLO via Ultralytics (YOLOE or YOLO-World).

    Tries the configured model first, then falls back through known candidates.
    """

    name = "yolo_open_vocab"
    family = "open_vocabulary_yolo"

    _candidates = [
        "yoloe-11s-seg.pt", "yoloe-11m-seg.pt",
        "yolov8s-world.pt", "yolov8m-world.pt",
    ]

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path

    def _check_available(self) -> None:
        import ultralytics  # noqa: F401
        # We accept "available" if ultralytics imports; concrete model load happens in load().

    def load(self, device: Optional[str] = None) -> None:
        from ultralytics import YOLO
        if device is not None:
            self.device = device
        candidates = ([self.model_path] if self.model_path else []) + self._candidates
        last_err = None
        for cand in candidates:
            if not cand:
                continue
            try:
                self._model = YOLO(cand)
                self._model_identifier = cand
                return
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"No open-vocabulary YOLO model could be loaded. "
            f"Tried: {candidates}. Last error: {last_err}"
        )

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        H, W = int(image.shape[1]), int(image.shape[2])
        import numpy as np
        np_img = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

        # Set open-vocabulary prompts if class_filter provided and model supports it
        if class_filter is not None:
            try:
                if hasattr(self._model, "set_classes"):
                    self._model.set_classes(list(class_filter))
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
