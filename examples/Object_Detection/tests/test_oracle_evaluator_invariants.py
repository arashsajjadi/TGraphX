"""Oracle / evaluator invariant tests.

These tests prevent the failure mode of v2 where the reported oracle AP
was below TGraphX AP, which is logically impossible when oracle and
TGraphX share the same candidate pool.
"""
import torch

from od_graph_fusion.evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)


def _gt(image_id="img", n=2):
    return GroundTruth(
        image_id=image_id,
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.], [50., 50., 70., 70.]])[:n],
        labels=torch.tensor([0, 1])[:n],
    )


def test_perfect_oracle_class_agnostic_is_one():
    """Class-agnostic AP of perfect predictions on simple GT must be 1.0."""
    gt = _gt(n=2)
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=gt.boxes_xyxy.clone(),
        scores=torch.tensor([0.9, 0.8]),
        labels=gt.labels.clone(),
    )
    r = evaluate_predictions([pred], [gt], iou_threshold=0.5,
                              num_classes=2, class_agnostic=True)
    assert r["AP"] == 1.0, f"Perfect predictions should give AP=1.0, got {r['AP']}"


def test_perfect_oracle_class_aware_is_one():
    gt = _gt(n=2)
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=gt.boxes_xyxy.clone(),
        scores=torch.tensor([0.9, 0.8]),
        labels=gt.labels.clone(),
    )
    r = evaluate_predictions([pred], [gt], iou_threshold=0.5,
                              num_classes=2, class_agnostic=False)
    assert r["AP"] == 1.0


def test_oracle_geq_subset_invariant_class_agnostic():
    """Oracle (best-IoU proposal per GT) must dominate any subset selector
    in class-agnostic mode."""
    gt = _gt(n=2)
    # Candidate pool: 3 proposals — 2 perfect ones and 1 bad one
    cand_boxes = torch.tensor([
        [0., 0., 10., 10.],     # perfect for GT[0]
        [50., 50., 70., 70.],   # perfect for GT[1]
        [100., 100., 110., 110.],  # bad: far from both
    ])
    cand_scores = torch.tensor([0.8, 0.7, 0.95])
    cand_labels = torch.tensor([0, 1, 0])

    # "Selector" picks the two perfect ones
    selector = DetectionPrediction(
        image_id="img",
        boxes_xyxy=cand_boxes[:2], scores=cand_scores[:2], labels=cand_labels[:2],
    )
    r_sel = evaluate_predictions([selector], [gt], iou_threshold=0.5,
                                   num_classes=2, class_agnostic=True)

    # "Oracle" picks the best per GT (also the two perfect ones)
    oracle = DetectionPrediction(
        image_id="img",
        boxes_xyxy=cand_boxes[:2], scores=cand_scores[:2], labels=cand_labels[:2],
    )
    r_or = evaluate_predictions([oracle], [gt], iou_threshold=0.5,
                                  num_classes=2, class_agnostic=True)

    # When oracle and selector are identical predictions, APs must match
    assert abs(r_or["AP"] - r_sel["AP"]) < 1e-6


def test_oracle_dominates_noisy_selector():
    """A selector that includes spurious low-IoU boxes must not exceed an
    oracle that picks the best proposals."""
    gt = _gt(n=2)
    cand_boxes = torch.tensor([
        [0., 0., 10., 10.],
        [50., 50., 70., 70.],
        [100., 100., 110., 110.],
    ])
    cand_scores = torch.tensor([0.8, 0.7, 0.95])
    cand_labels = torch.tensor([0, 1, 0])

    # Noisy selector keeps all 3
    noisy = DetectionPrediction(
        image_id="img",
        boxes_xyxy=cand_boxes, scores=cand_scores, labels=cand_labels,
    )
    r_noisy = evaluate_predictions([noisy], [gt], iou_threshold=0.5,
                                     num_classes=2, class_agnostic=True)

    # Oracle keeps only the two perfect ones
    oracle = DetectionPrediction(
        image_id="img",
        boxes_xyxy=cand_boxes[:2], scores=cand_scores[:2], labels=cand_labels[:2],
    )
    r_or = evaluate_predictions([oracle], [gt], iou_threshold=0.5,
                                  num_classes=2, class_agnostic=True)

    # Oracle AP must be at least as high as noisy AP
    assert r_or["AP"] >= r_noisy["AP"] - 1e-6, (
        f"INVARIANT VIOLATED: oracle AP ({r_or['AP']:.4f}) < "
        f"noisy selector AP ({r_noisy['AP']:.4f})"
    )


def test_oracle_geq_individual_detectors_class_agnostic():
    """When oracle picks the best proposal per GT from a pool that includes
    all detector proposals, oracle AP >= any individual detector AP."""
    gt = _gt(n=2)
    # Detector A: hits GT0 well, misses GT1
    det_a = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[1., 1., 11., 11.], [200., 200., 210., 210.]]),
        scores=torch.tensor([0.9, 0.4]),
        labels=torch.tensor([0, 1]),
    )
    # Detector B: hits GT1 well, misses GT0
    det_b = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[200., 200., 210., 210.], [51., 51., 71., 71.]]),
        scores=torch.tensor([0.3, 0.8]),
        labels=torch.tensor([0, 1]),
    )

    # Oracle: best per GT from pooled candidates
    pool_boxes = torch.cat([det_a.boxes_xyxy, det_b.boxes_xyxy])
    pool_scores = torch.cat([det_a.scores, det_b.scores])
    pool_labels = torch.cat([det_a.labels, det_b.labels])
    # Best for GT0 = idx 0 (det_a's first), best for GT1 = idx 3 (det_b's second)
    oracle = DetectionPrediction(
        image_id="img",
        boxes_xyxy=pool_boxes[[0, 3]],
        scores=pool_scores[[0, 3]],
        labels=pool_labels[[0, 3]],
    )
    r_a = evaluate_predictions([det_a], [gt], iou_threshold=0.5,
                                  num_classes=2, class_agnostic=True)
    r_b = evaluate_predictions([det_b], [gt], iou_threshold=0.5,
                                  num_classes=2, class_agnostic=True)
    r_or = evaluate_predictions([oracle], [gt], iou_threshold=0.5,
                                  num_classes=2, class_agnostic=True)

    assert r_or["AP"] + 1e-6 >= r_a["AP"], (
        f"Oracle ({r_or['AP']:.4f}) < detector A ({r_a['AP']:.4f})"
    )
    assert r_or["AP"] + 1e-6 >= r_b["AP"], (
        f"Oracle ({r_or['AP']:.4f}) < detector B ({r_b['AP']:.4f})"
    )


def test_class_agnostic_returns_mode_field():
    gt = _gt(n=1)
    pred = DetectionPrediction(
        image_id="img",
        boxes_xyxy=gt.boxes_xyxy.clone(),
        scores=torch.tensor([0.9]),
        labels=gt.labels.clone(),
    )
    r_ca = evaluate_predictions([pred], [gt], iou_threshold=0.5,
                                  num_classes=1, class_agnostic=True)
    r_aw = evaluate_predictions([pred], [gt], iou_threshold=0.5,
                                  num_classes=1, class_agnostic=False)
    assert r_ca.get("mode") == "class_agnostic"
    assert r_aw.get("mode") == "class_aware"


def test_class_aware_excludes_classes_with_no_gt():
    """Classes that exist only in predictions (not in GT) must not drag mAP
    down to zero (the old bug)."""
    gt = GroundTruth(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        labels=torch.tensor([0]),
    )
    # Prediction in a totally different class (e.g. COCO id 2 vs VOC id 0).
    # In class-aware mode that prediction is an FP at class 2 with no GT.
    # mAP must NOT include the spurious class-2 bucket.
    pred_wrong_class = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([2]),
    )
    pred_correct = DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor([[0., 0., 10., 10.]]),
        scores=torch.tensor([0.9]),
        labels=torch.tensor([0]),
    )
    r_wrong = evaluate_predictions([pred_wrong_class], [gt], iou_threshold=0.5,
                                     num_classes=10, class_agnostic=False)
    r_correct = evaluate_predictions([pred_correct], [gt], iou_threshold=0.5,
                                        num_classes=10, class_agnostic=False)
    # Wrong-class prediction must have AP=0 (no class match), but the
    # average should not include phantom 0s from unused class buckets.
    assert r_correct["AP"] == 1.0
    assert r_wrong["AP"] == 0.0
