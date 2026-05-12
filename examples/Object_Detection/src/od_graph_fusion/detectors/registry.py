"""Detector registry / builder.

Builds a list of available detectors from a config dict, with graceful
fallback to a synthetic detector when no real detector loads or when the
config requests synthetic mode.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import torch

from .base import BaseDetector, DetectionResult
from .retinanet import RetinaNetAdapter
from .yolo import YOLOAdapter
from .yoloe import YOLOEAdapter
from .rt_detr import RTDETRAdapter


# ── Synthetic detector ─────────────────────────────────────────────────────


class SyntheticDetector(BaseDetector):
    """Deterministic, no-network detector that emits boxes near ground truth.

    Used for FAST_SMOKE so the pipeline can be exercised without any model
    downloads. Each synthetic detector ID has a slightly different jitter
    profile so the resulting graphs are non-trivial.
    """

    def __init__(self, name: str, family: str, jitter: float = 0.05,
                 conf_min: float = 0.5, conf_max: float = 0.95,
                 drop_rate: float = 0.0, false_positive_rate: float = 0.0,
                 seed: int = 0, class_names: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.family = family
        self.jitter = jitter
        self.conf_min = conf_min
        self.conf_max = conf_max
        self.drop_rate = drop_rate
        self.false_positive_rate = false_positive_rate
        self._seed = seed
        self._class_names = class_names or []
        self._model_identifier = f"synthetic:{name}"
        self._available = True

    def _check_available(self) -> None:
        pass

    def load(self, device: Optional[str] = None) -> None:
        if device is not None:
            self.device = device

    def predict(self, image: torch.Tensor, image_id: str,
                class_filter: Optional[Sequence[str]] = None,
                gt_boxes: Optional[torch.Tensor] = None,
                gt_labels: Optional[torch.Tensor] = None) -> DetectionResult:
        """Synthetic predictor takes GT as a hidden hint (set by registry helper)."""
        H, W = int(image.shape[1]), int(image.shape[2])
        rng = random.Random(hash((image_id, self._seed)) % (2**31))
        gen = torch.Generator().manual_seed(hash((image_id, self._seed, 1)) % (2**31))

        if gt_boxes is None or gt_boxes.numel() == 0:
            return self.empty_result(image_id, (H, W))

        boxes_list, scores_list, label_ids_list, labels_list = [], [], [], []
        for i in range(gt_boxes.shape[0]):
            if rng.random() < self.drop_rate:
                continue
            box = gt_boxes[i].clone().float()
            # Per-box jitter proportional to size
            w = (box[2] - box[0]).clamp(min=1)
            h = (box[3] - box[1]).clamp(min=1)
            jitter_xy = (torch.randn(2, generator=gen) * self.jitter)
            jitter_wh = (torch.randn(2, generator=gen) * self.jitter)
            box[0] += jitter_xy[0] * w
            box[1] += jitter_xy[1] * h
            box[2] += jitter_wh[0] * w
            box[3] += jitter_wh[1] * h
            box[0] = box[0].clamp(0, W - 1)
            box[1] = box[1].clamp(0, H - 1)
            box[2] = box[2].clamp(0, W - 1)
            box[3] = box[3].clamp(0, H - 1)
            if box[2] <= box[0] + 2 or box[3] <= box[1] + 2:
                continue
            score = rng.uniform(self.conf_min, self.conf_max)
            label_id = int(gt_labels[i].item()) if gt_labels is not None else 0
            label_name = (self._class_names[label_id]
                          if 0 <= label_id < len(self._class_names) else str(label_id))
            boxes_list.append(box)
            scores_list.append(score)
            label_ids_list.append(label_id)
            labels_list.append(label_name)

        # False positives
        for _ in range(int(rng.random() < self.false_positive_rate)):
            w = rng.randint(20, max(21, W // 4))
            h = rng.randint(20, max(21, H // 4))
            x1 = rng.randint(0, W - w - 1)
            y1 = rng.randint(0, H - h - 1)
            boxes_list.append(torch.tensor([x1, y1, x1 + w, y1 + h], dtype=torch.float32))
            scores_list.append(rng.uniform(self.conf_min * 0.5, self.conf_min))
            cls = rng.randint(0, max(0, len(self._class_names) - 1))
            label_ids_list.append(cls)
            labels_list.append(self._class_names[cls] if cls < len(self._class_names) else str(cls))

        if not boxes_list:
            return self.empty_result(image_id, (H, W))

        boxes = torch.stack(boxes_list, dim=0)
        scores = torch.tensor(scores_list, dtype=torch.float32)
        label_ids = torch.tensor(label_ids_list, dtype=torch.long)

        if class_filter is not None:
            mask = torch.tensor([l in class_filter for l in labels_list], dtype=torch.bool)
            boxes = boxes[mask]; scores = scores[mask]; label_ids = label_ids[mask]
            labels_list = [l for l, m in zip(labels_list, mask.tolist()) if m]

        return DetectionResult(
            image_id=image_id, model_name=self.name,
            boxes_xyxy=boxes, scores=scores,
            label_ids=label_ids, labels=labels_list,
            image_size=(H, W), device=self.device, runtime_ms=0.1,
        )


def build_synthetic_detector(name: str, family: str, seed: int = 0,
                              jitter: float = 0.05, drop_rate: float = 0.0,
                              class_names: Optional[List[str]] = None) -> SyntheticDetector:
    return SyntheticDetector(name=name, family=family, seed=seed,
                              jitter=jitter, drop_rate=drop_rate,
                              class_names=class_names)


def build_detectors(config: Dict[str, Any], class_names: List[str]) -> Dict[str, BaseDetector]:
    """Return ``{name: detector}`` for all configured/available detectors.

    Honors ``detectors.use_real``. If a real detector cannot load, it is
    replaced by a synthetic detector of the same name with a clear note in
    its availability log.
    """
    dcfg = config.get("detectors", {})
    use_real = bool(dcfg.get("use_real", False))
    device = config.get("device", "auto")
    conf = float(dcfg.get("conf_threshold", 0.25))
    iou = float(dcfg.get("iou_threshold", 0.45))

    requested = {
        "retinanet": dcfg.get("use_retinanet", True),
        "yolo_modern": dcfg.get("use_yolo", True),
        "yolo_open_vocab": dcfg.get("use_yoloe", True),
        "rt_detr": dcfg.get("use_rtdetr", True),
    }

    detectors: Dict[str, BaseDetector] = {}

    def _try_real(adapter_cls, name: str, family: str, **kwargs) -> Optional[BaseDetector]:
        if not use_real:
            return None
        try:
            adapter = adapter_cls(device=device, conf=conf, iou=iou, **kwargs)
            if not adapter.available():
                return None
            adapter.load()
            return adapter
        except Exception as exc:
            print(f"[detectors] {name}: real load failed ({type(exc).__name__}: {exc}); using synthetic.")
            return None

    # RetinaNet — usually works
    if requested["retinanet"]:
        d = _try_real(RetinaNetAdapter, "retinanet", "anchor_based_cnn")
        if d is None:
            d = build_synthetic_detector("retinanet", "anchor_based_cnn",
                                          seed=1, jitter=0.05, drop_rate=0.1,
                                          class_names=class_names)
        detectors[d.name] = d

    if requested["yolo_modern"]:
        d = _try_real(YOLOAdapter, "yolo_modern", "anchor_free_cnn",
                       model_path=dcfg.get("yolo_model", "yolo11n.pt"))
        if d is None:
            d = build_synthetic_detector("yolo_modern", "anchor_free_cnn",
                                          seed=2, jitter=0.08, drop_rate=0.05,
                                          class_names=class_names)
        detectors[d.name] = d

    if requested["yolo_open_vocab"]:
        d = _try_real(YOLOEAdapter, "yolo_open_vocab", "open_vocabulary_yolo",
                       model_path=dcfg.get("yoloe_model"))
        if d is None:
            d = build_synthetic_detector("yolo_open_vocab", "open_vocabulary_yolo",
                                          seed=3, jitter=0.10, drop_rate=0.15,
                                          class_names=class_names)
        detectors[d.name] = d

    if requested["rt_detr"]:
        d = _try_real(
            RTDETRAdapter, "rt_detr", "transformer_detector",
            source=dcfg.get("rtdetr_source", "ultralytics"),
            ultralytics_model=dcfg.get("rtdetr_model", "rtdetr-l.pt"),
            hf_model=dcfg.get("rtdetr_hf_model", "PekingU/rtdetr_r50vd_coco_o365"),
        )
        if d is None:
            d = build_synthetic_detector("rt_detr", "transformer_detector",
                                          seed=4, jitter=0.06, drop_rate=0.08,
                                          class_names=class_names)
        detectors[d.name] = d

    return detectors


def detector_availability_report(detectors: Dict[str, BaseDetector]) -> Dict[str, Any]:
    report = {}
    for name, det in detectors.items():
        is_synth = "synthetic" in det.model_identifier()
        report[name] = {
            "family": det.family,
            "available": det.available(),
            "model_identifier": det.model_identifier(),
            "is_synthetic": is_synth,
            "device": det.device,
            "unavailable_reason": det.unavailable_reason() if not det.available() else None,
        }
    return report
