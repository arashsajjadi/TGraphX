import torch

from od_graph_fusion.evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)


def test_perfect_predictions_give_perfect_recall():
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
    )
    gt = GroundTruth(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        labels=torch.tensor([0]),
    )
    r = evaluate_predictions([pred], [gt], iou_threshold=0.5, num_classes=1)
    assert r["recall"] == 1.0
    assert r["precision"] == 1.0


def test_wrong_class_gives_zero_recall():
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([1]),
    )
    gt = GroundTruth(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        labels=torch.tensor([0]),
    )
    r = evaluate_predictions([pred], [gt], iou_threshold=0.5, num_classes=2)
    assert r["recall"] == 0


def test_no_predictions_gives_zero_recall():
    gt = GroundTruth(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        labels=torch.tensor([0]),
    )
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.zeros(0, 4),
        scores=torch.zeros(0),
        labels=torch.zeros(0, dtype=torch.long),
    )
    r = evaluate_predictions([pred], [gt], iou_threshold=0.5, num_classes=1)
    assert r["recall"] == 0
