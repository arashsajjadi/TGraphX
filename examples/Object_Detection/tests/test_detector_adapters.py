"""Detector adapter sanity tests (no real model downloads required)."""
import torch
import pytest

from od_graph_fusion.detectors.base import DetectionResult
from od_graph_fusion.detectors.registry import (
    build_detectors, build_synthetic_detector, detector_availability_report,
)


def test_synthetic_detector_returns_proper_result():
    det = build_synthetic_detector("test_det", "synthetic", seed=1,
                                    class_names=["a", "b"])
    image = torch.rand(3, 64, 64)
    gt = torch.tensor([[5., 5., 25., 25.]])
    gtl = torch.tensor([0])
    res = det.predict(image, "img", gt_boxes=gt, gt_labels=gtl)
    assert isinstance(res, DetectionResult)
    assert res.num_detections() >= 0


def test_synthetic_detector_empty_gt_returns_empty():
    det = build_synthetic_detector("t", "synthetic", seed=1, class_names=["a"])
    image = torch.rand(3, 64, 64)
    res = det.predict(image, "img", gt_boxes=None, gt_labels=None)
    assert res.num_detections() == 0


def test_availability_report_fields():
    det = build_synthetic_detector("t", "synthetic", seed=1, class_names=["a"])
    rep = detector_availability_report({"t": det})
    assert "t" in rep
    assert "available" in rep["t"]
    assert "is_synthetic" in rep["t"]


def test_build_detectors_synthetic_mode():
    cfg = {
        "device": "cpu",
        "detectors": {"use_real": False, "use_yolo": True, "use_yoloe": True,
                      "use_rtdetr": True, "use_retinanet": True},
    }
    dets = build_detectors(cfg, class_names=["car", "person"])
    assert "retinanet" in dets
    # Detector names updated: yolo_modern → yolo26x, rt_detr → rtdetr_x, yolo_open_vocab → yolo_world
    assert any(k in dets for k in ("yolo26x", "yolo_modern"))
    assert any(k in dets for k in ("rtdetr_x", "rt_detr"))
    # yolo_world replaces yolo_open_vocab; use_yoloe legacy flag maps to yolo_world slot
    assert any(k in dets for k in ("yolo_world", "yolo_open_vocab"))


def test_detection_result_to_dict():
    r = DetectionResult(
        image_id="img", model_name="m",
        boxes_xyxy=torch.tensor([[0., 0., 5., 5.]]),
        scores=torch.tensor([0.9]),
        label_ids=torch.tensor([0]),
        labels=["car"], image_size=(64, 64),
    )
    d = r.to_dict()
    import json
    json.dumps(d)
    assert d["num_detections"] == 1
