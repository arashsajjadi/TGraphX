import torch

from od_graph_fusion.baselines import (
    nms, soft_nms, weighted_boxes_fusion, pool_detector_results,
)
from od_graph_fusion.detectors.base import DetectionResult


def test_nms_keeps_highest_score():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    scores = torch.tensor([0.9, 0.5])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0]


def test_nms_keeps_disjoint():
    boxes = torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.5])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_wbf_merges_overlapping_same_class():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    scores = torch.tensor([0.8, 0.7])
    labels = torch.tensor([0, 0])
    fb, fs, fl = weighted_boxes_fusion(boxes, scores, labels,
                                        iou_threshold=0.4)
    assert fb.shape[0] == 1
    assert fl[0].item() == 0


def test_wbf_keeps_different_classes_separate():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    scores = torch.tensor([0.8, 0.7])
    labels = torch.tensor([0, 1])
    fb, fs, fl = weighted_boxes_fusion(boxes, scores, labels, iou_threshold=0.4)
    assert fb.shape[0] == 2


def test_pool_detector_results():
    r0 = DetectionResult("img", "d0",
                          torch.tensor([[0., 0., 10., 10.]]),
                          torch.tensor([0.9]),
                          torch.tensor([0]), ["car"], (100, 100))
    r1 = DetectionResult("img", "d1",
                          torch.tensor([[5., 5., 15., 15.]]),
                          torch.tensor([0.8]),
                          torch.tensor([0]), ["car"], (100, 100))
    b, s, l, d = pool_detector_results([r0, r1])
    assert b.shape == (2, 4)
    assert d.tolist() == [0, 1]


def test_soft_nms_runs():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    scores = torch.tensor([0.9, 0.8])
    keep, decayed = soft_nms(boxes, scores, sigma=0.5)
    assert keep.numel() > 0
