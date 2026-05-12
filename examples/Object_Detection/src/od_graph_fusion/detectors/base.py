"""Unified detector interface."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class DetectionResult:
    """Standard detector output for one image."""
    image_id: str
    model_name: str
    boxes_xyxy: torch.Tensor       # [N, 4]
    scores: torch.Tensor           # [N]
    label_ids: torch.Tensor        # [N] long, dataset-class space
    labels: List[str]              # [N]
    image_size: Tuple[int, int]    # (H, W)
    device: str = "cpu"
    runtime_ms: float = 0.0
    raw: Optional[Any] = None
    error: Optional[str] = None

    def num_detections(self) -> int:
        return int(self.boxes_xyxy.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "model_name": self.model_name,
            "num_detections": self.num_detections(),
            "boxes_xyxy": self.boxes_xyxy.detach().cpu().tolist(),
            "scores": self.scores.detach().cpu().tolist(),
            "label_ids": self.label_ids.detach().cpu().tolist(),
            "labels": self.labels,
            "image_size": list(self.image_size),
            "device": self.device,
            "runtime_ms": self.runtime_ms,
            "error": self.error,
        }


class BaseDetector:
    """Base detector interface."""

    name: str = "base"
    family: str = "base"

    def __init__(self, device: str = "cpu", conf: float = 0.25,
                 iou: float = 0.45, **kwargs):
        self.device = device
        self.conf = conf
        self.iou = iou
        self._available: Optional[bool] = None
        self._unavailable_reason: Optional[str] = None
        self._model = None
        self._model_identifier: Optional[str] = None

    def available(self) -> bool:
        if self._available is None:
            try:
                self._check_available()
                self._available = True
            except Exception as exc:
                self._available = False
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
        return self._available

    def unavailable_reason(self) -> str:
        if self.available():
            return ""
        return self._unavailable_reason or "unknown"

    def model_identifier(self) -> str:
        return self._model_identifier or "unknown"

    def _check_available(self) -> None:
        raise NotImplementedError

    def load(self, device: Optional[str] = None) -> None:
        raise NotImplementedError

    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None) -> DetectionResult:
        raise NotImplementedError

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def empty_result(self, image_id: str, image_size: Tuple[int, int],
                      error: Optional[str] = None) -> DetectionResult:
        return DetectionResult(
            image_id=image_id, model_name=self.name,
            boxes_xyxy=torch.zeros(0, 4), scores=torch.zeros(0),
            label_ids=torch.zeros(0, dtype=torch.long), labels=[],
            image_size=image_size, device=self.device,
            runtime_ms=0.0, error=error,
        )
