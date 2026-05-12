"""Detector adapters with a unified interface."""
from __future__ import annotations

from .base import BaseDetector, DetectionResult
from .registry import build_detectors, build_synthetic_detector
from .retinanet import RetinaNetAdapter
from .yolo import YOLOAdapter
from .yoloe import YOLOEAdapter
from .rt_detr import RTDETRAdapter

__all__ = [
    "BaseDetector", "DetectionResult",
    "RetinaNetAdapter", "YOLOAdapter", "YOLOEAdapter", "RTDETRAdapter",
    "build_detectors", "build_synthetic_detector",
]
