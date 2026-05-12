"""Class-aware oracle invariant tests.

Defines three distinct oracle types and validates their ordering:

1. localization_oracle: ignores class labels, picks best-IoU proposal per GT.
   - In class-agnostic mode, must be >= every other method.
   - In class-aware mode with wrong labels, can be LOWER than TGraphX.
   - Correct name: localization_oracle.

2. class_aware_candidate_oracle: picks the best correctly-labeled proposal per GT.
   - In class-aware mode, must be >= TGraphX and >= any classical fusion.
   - If no correctly-labeled proposal exists for a GT, that GT is uncovered.

3. true_upper_bound: GT labels assigned to best-IoU proposal, AP=1.0 if detectors
   cover every GT spatially. Only reachable with GT-label correction — not deployable.
"""
import torch
import pytest

from od_graph_fusion.evaluation import (
    DetectionPrediction, GroundTruth, evaluate_predictions,
)


def _pred(boxes, scores, labels):
    return DetectionPrediction(
        image_id="img",
        boxes_xyxy=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def _gt(boxes, labels):
    return GroundTruth(
        image_id="img",
        boxes_xyxy=torch.tensor(boxes, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.long),
    )


def test_localization_oracle_class_agnostic_is_upper_bound():
    """Localization oracle must dominate in class-agnostic mode."""
    gt = _gt([[0., 0., 10., 10.]], [5])
    # Localization oracle: best IoU, wrong label
    loc_oracle = _pred([[0., 0., 10., 10.]], [0.9], [2])  # COCO label
    # TGraphX: correct label
    tgx = _pred([[1., 1., 9., 9.]], [0.8], [5])           # correct VOC label

    r_or = evaluate_predictions([loc_oracle], [gt], iou_threshold=0.5,
                                 num_classes=20, class_agnostic=True)
    r_tgx = evaluate_predictions([tgx], [gt], iou_threshold=0.5,
                                   num_classes=20, class_agnostic=True)
    # Class-agnostic: oracle must win (better box)
    assert r_or["AP"] >= r_tgx["AP"] - 1e-6


def test_class_aware_candidate_oracle_dominates():
    """A class-aware candidate oracle (best correctly-labeled proposal per GT)
    must dominate TGraphX and other methods in class-aware mode."""
    gt = _gt([[0., 0., 10., 10.]], [5])
    # Correctly-labeled proposal (VOC label=5 matches GT label=5)
    ca_oracle = _pred([[0., 0., 10., 10.]], [0.9], [5])
    # TGraphX might pick a slightly worse box with correct label
    tgx = _pred([[1., 1., 9., 9.]], [0.8], [5])

    r_ca_or = evaluate_predictions([ca_oracle], [gt], iou_threshold=0.5,
                                     num_classes=20, class_agnostic=False)
    r_tgx = evaluate_predictions([tgx], [gt], iou_threshold=0.5,
                                   num_classes=20, class_agnostic=False)
    assert r_ca_or["AP"] >= r_tgx["AP"] - 1e-6


def test_localization_oracle_can_be_below_tgraphx_in_class_aware():
    """The localization oracle (best-IoU but potentially wrong label) can legally
    score below TGraphX in class-aware mode when detector labels are wrong.
    This is NOT a bug — it's why we need three oracle types."""
    gt = _gt([[0., 0., 10., 10.]], [5])
    loc_oracle = _pred([[0., 0., 10., 10.]], [0.9], [2])  # best IoU, wrong label
    tgx = _pred([[0., 0., 10., 10.]], [0.8], [5])          # same box, correct label

    r_or = evaluate_predictions([loc_oracle], [gt], iou_threshold=0.5,
                                 num_classes=20, class_agnostic=False)
    r_tgx = evaluate_predictions([tgx], [gt], iou_threshold=0.5,
                                   num_classes=20, class_agnostic=False)
    # Class-aware: TGraphX wins because it has the correct label
    assert r_tgx["AP"] > r_or["AP"]
    # This is expected and valid — NOT an oracle violation


def test_true_upper_bound_with_gt_corrected_labels():
    """A true upper bound uses GT labels on the best-IoU proposal. AP near 1.0."""
    gt = _gt([[0., 0., 10., 10.]], [5])
    # True upper bound: best IoU proposal with GT label assigned
    true_ub = _pred([[0., 0., 10., 10.]], [0.9], [5])  # GT label
    r = evaluate_predictions([true_ub], [gt], iou_threshold=0.5,
                               num_classes=20, class_agnostic=False)
    assert r["AP"] == 1.0, f"True upper bound should be 1.0, got {r['AP']}"


def test_class_aware_candidate_oracle_zero_when_no_correct_label():
    """If no proposal has the correct class label, class-aware candidate oracle = 0."""
    gt = _gt([[0., 0., 10., 10.]], [5])
    # All proposals have wrong labels
    ca_oracle = _pred([[0., 0., 10., 10.], [2., 2., 8., 8.]], [0.9, 0.8], [2, 3])
    r = evaluate_predictions([ca_oracle], [gt], iou_threshold=0.5,
                               num_classes=20, class_agnostic=False)
    assert r["AP"] == 0.0


def test_oracle_naming_consistency():
    """Document that class-aware AP can EXCEED class-agnostic AP when
    wrong-label predictions pollute the class-agnostic global PR curve.

    Explanation: in class-agnostic mode ALL predictions compete for GT slots,
    including wrong-label FPs that lower global precision. In class-aware mode,
    wrong-label predictions are isolated in buckets with no GT, so they do NOT
    drag down AP for the correctly-labeled classes.

    This is an expected and valid property — it is NOT a bug.
    The implication for our experiment: when COCO-labeled detectors generate
    wrong-label predictions, TGraphX (which selects correctly-labeled YOLOE
    candidates) can have class-aware AP > class-agnostic AP, while the
    localization oracle (best-IoU but potentially wrong-label) will have
    class-aware AP = 0.
    """
    gt = _gt([[0., 0., 10., 10.], [50., 50., 60., 60.]], [5, 6])
    # Pool: two correct-label proposals and two wrong-label proposals
    pool = _pred(
        [[0., 0., 10., 10.], [50., 50., 60., 60.],
         [0., 0., 10., 10.], [50., 50., 60., 60.]],
        [0.9, 0.8, 0.85, 0.75],
        [5, 6, 2, 3],   # first two correct, last two wrong (COCO ids)
    )

    r_ca = evaluate_predictions([pool], [gt], iou_threshold=0.5,
                                  num_classes=20, class_agnostic=False)
    r_loc = evaluate_predictions([pool], [gt], iou_threshold=0.5,
                                   num_classes=20, class_agnostic=True)
    # In class-aware mode, wrong-label FPs don't enter the correct-class PR curve
    # so class-aware AP=1.0 > class-agnostic AP=0.83
    assert r_ca["AP"] > r_loc["AP"] - 1e-6, (
        "With wrong-label FPs in pool, class-aware should not be dramatically lower"
    )
    # Class-agnostic should be well below 1.0 because of the FPs in the global curve
    assert r_loc["AP"] < 1.0 - 1e-6
    # Class-aware mAP should be 1.0 because both correct classes have AP=1.0
    assert abs(r_ca["AP"] - 1.0) < 1e-6
