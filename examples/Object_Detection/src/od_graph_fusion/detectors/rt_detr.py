"""RT-DETR adapter (transformer-family detector).

Tries Ultralytics' RT-DETR first; falls back to HuggingFace transformers'
``RTDetrForObjectDetection`` if available.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult


class RTDETRAdapter(BaseDetector):
    name = "rt_detr"
    family = "transformer_detector"

    def __init__(
        self,
        source: str = "ultralytics",
        ultralytics_model: str = "rtdetr-l.pt",
        hf_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source = source
        self.ultralytics_model = ultralytics_model
        self.hf_model = hf_model
        self._backend: Optional[str] = None
        self._processor = None  # for HF path

    def _check_available(self) -> None:
        if self.source == "ultralytics":
            import ultralytics  # noqa: F401
        else:
            import transformers  # noqa: F401

    def load(self, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = device
        last_err = None
        # Try ultralytics first
        if self.source in ("ultralytics", "auto"):
            try:
                from ultralytics import RTDETR
                self._model = RTDETR(self.ultralytics_model)
                self._model_identifier = f"ultralytics:{self.ultralytics_model}"
                self._backend = "ultralytics"
                return
            except Exception as exc:
                last_err = exc

        # Fall back to HF transformers
        try:
            from transformers import AutoImageProcessor
            try:
                from transformers import RTDetrForObjectDetection as _RTDetr
            except ImportError:
                from transformers import RTDetrV2ForObjectDetection as _RTDetr  # type: ignore
            self._processor = AutoImageProcessor.from_pretrained(self.hf_model)
            self._model = _RTDetr.from_pretrained(self.hf_model)
            self._model.eval().to(self.device)
            self._model_identifier = f"huggingface:{self.hf_model}"
            self._backend = "huggingface"
            return
        except Exception as exc:
            last_err = exc

        raise RuntimeError(
            f"RT-DETR could not be loaded via any backend. Last error: {last_err}"
        )

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        if self._model is None:
            self.load()
        H, W = int(image.shape[1]), int(image.shape[2])

        if self._backend == "ultralytics":
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
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return self.empty_result(image_id, (H, W))
            r0 = results[0]
            boxes = r0.boxes.xyxy.detach().cpu()
            scores = r0.boxes.conf.detach().cpu()
            labels_idx = r0.boxes.cls.detach().cpu().long().tolist()
            names = r0.names
            labels = [names.get(int(i), str(i)) for i in labels_idx]
        else:  # huggingface
            from PIL import Image as PILImage
            import numpy as np
            np_img = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = PILImage.fromarray(np_img)
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            t0 = time.time()
            try:
                outputs = self._model(**inputs)
            except Exception as exc:
                return self.empty_result(image_id, (H, W), error=str(exc))
            target_sizes = torch.tensor([[H, W]], device=self.device)
            try:
                processed = self._processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=self.conf
                )
            except Exception as exc:
                return self.empty_result(image_id, (H, W), error=str(exc))
            runtime = (time.time() - t0) * 1000
            r0 = processed[0]
            boxes = r0["boxes"].detach().cpu()
            scores = r0["scores"].detach().cpu()
            labels_idx = r0["labels"].detach().cpu().long().tolist()
            id2label = getattr(self._model.config, "id2label", {})
            labels = [id2label.get(int(i), str(i)) for i in labels_idx]

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
