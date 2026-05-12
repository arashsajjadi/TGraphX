"""Tests for box_ops."""
import torch

from od_graph_fusion.box_ops import (
    box_area, box_iou, box_giou, box_center, box_wh,
    weighted_box_average, union_box, intersection_box, normalize_boxes,
    clip_boxes,
)


def test_box_area_simple():
    b = torch.tensor([[0., 0., 10., 10.], [0., 0., 0., 0.]])
    a = box_area(b)
    assert a[0].item() == 100
    assert a[1].item() == 0


def test_box_iou_identical_is_one():
    b = torch.tensor([[0., 0., 10., 10.]])
    iou = box_iou(b, b)
    assert torch.allclose(iou, torch.tensor([[1.0]]))


def test_box_iou_no_overlap_is_zero():
    a = torch.tensor([[0., 0., 5., 5.]])
    b = torch.tensor([[10., 10., 15., 15.]])
    assert box_iou(a, b).item() == 0


def test_box_iou_half_overlap():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[5., 0., 15., 10.]])
    # Intersection 5*10=50; Union = 100+100-50 = 150; IoU = 50/150
    assert abs(box_iou(a, b).item() - 50/150) < 1e-6


def test_box_iou_empty():
    a = torch.zeros(0, 4)
    b = torch.tensor([[0., 0., 10., 10.]])
    assert box_iou(a, b).shape == (0, 1)


def test_giou_less_than_iou():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[20., 20., 30., 30.]])
    iou = box_iou(a, b).item()
    g = box_giou(a, b).item()
    # GIoU penalises distant boxes, so g <= iou
    assert g <= iou


def test_union_box():
    boxes = torch.tensor([[0., 0., 5., 5.], [3., 3., 10., 10.]])
    u = union_box(boxes)
    assert u.tolist() == [0., 0., 10., 10.]


def test_intersection_box_overlap():
    boxes = torch.tensor([[0., 0., 5., 5.], [3., 3., 10., 10.]])
    inter = intersection_box(boxes)
    # max x1=3, max y1=3, min x2=5, min y2=5
    assert inter.tolist() == [3., 3., 5., 5.]


def test_intersection_box_no_overlap_returns_zero():
    boxes = torch.tensor([[0., 0., 5., 5.], [10., 10., 15., 15.]])
    inter = intersection_box(boxes)
    assert inter.sum().item() == 0


def test_normalize_boxes():
    b = torch.tensor([[0., 0., 50., 100.]])
    n = normalize_boxes(b, (100, 50))
    assert n.tolist() == [[0., 0., 1., 1.]]


def test_clip_boxes():
    b = torch.tensor([[-5., -5., 200., 200.]])
    c = clip_boxes(b, (100, 100))
    assert c.tolist() == [[0., 0., 99., 99.]]


def test_weighted_box_average():
    boxes = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.]])
    scores = torch.tensor([1.0, 1.0])
    avg = weighted_box_average(boxes, scores)
    assert torch.allclose(avg, torch.tensor([2.5, 2.5, 12.5, 12.5]))
