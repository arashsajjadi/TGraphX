import torch

from od_graph_fusion.matching import cluster_proposals, match_to_gt


def test_cluster_proposals_empty():
    cid = cluster_proposals(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
                             torch.zeros(0, dtype=torch.long))
    assert cid.numel() == 0


def test_cluster_two_overlapping_different_detectors():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    labels = torch.tensor([0, 0])
    det_ids = torch.tensor([0, 1])
    cid = cluster_proposals(boxes, labels, det_ids, iou_threshold=0.5,
                             require_same_class=True)
    assert cid[0].item() == cid[1].item()


def test_cluster_two_overlapping_same_detector_kept_separate():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    labels = torch.tensor([0, 0])
    det_ids = torch.tensor([0, 0])
    cid = cluster_proposals(boxes, labels, det_ids, iou_threshold=0.5)
    assert cid[0].item() != cid[1].item()


def test_cluster_different_classes_not_merged():
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]])
    labels = torch.tensor([0, 1])
    det_ids = torch.tensor([0, 1])
    cid = cluster_proposals(boxes, labels, det_ids, iou_threshold=0.5,
                             require_same_class=True)
    assert cid[0].item() != cid[1].item()


def test_match_to_gt_correct():
    pred_boxes = torch.tensor([[0., 0., 10., 10.]])
    pred_labels = torch.tensor([0])
    gt_boxes = torch.tensor([[1., 1., 11., 11.]])
    gt_labels = torch.tensor([0])
    matched, iou, correct = match_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels)
    assert matched[0].item() == 0
    assert iou[0].item() > 0.5
    assert correct[0].item()


def test_match_to_gt_wrong_class():
    pred_boxes = torch.tensor([[0., 0., 10., 10.]])
    pred_labels = torch.tensor([1])
    gt_boxes = torch.tensor([[0., 0., 10., 10.]])
    gt_labels = torch.tensor([0])
    matched, iou, correct = match_to_gt(pred_boxes, pred_labels, gt_boxes, gt_labels)
    assert not correct[0].item()
