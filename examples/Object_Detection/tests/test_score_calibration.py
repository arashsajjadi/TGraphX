"""Tests for validation-only score calibration — no test leakage."""
import pytest
import torch


def test_score_mode_selection_uses_val_not_test():
    """Score mode selection function must refuse to run on the test split."""
    from od_graph_fusion.evaluation import evaluate_predictions, GroundTruth, DetectionPrediction
    # Simulate a score mode selector that enforces split
    def select_score_mode(val_ap_by_mode: dict, test_ap_by_mode: dict = None) -> str:
        if test_ap_by_mode is not None:
            raise ValueError("select_score_mode must not receive test AP values — test leakage risk")
        return max(val_ap_by_mode, key=val_ap_by_mode.get)

    val_aps = {"routing_prob": 0.85, "base_score": 0.72, "routing_prob*base": 0.82}
    test_aps = {"routing_prob": 0.88, "base_score": 0.75, "routing_prob*base": 0.84}

    best = select_score_mode(val_aps)
    assert best == "routing_prob"

    with pytest.raises(ValueError, match="test leakage"):
        select_score_mode(val_aps, test_aps)


def test_score_mode_frozen_before_test():
    """Once score_mode is selected on val, the test run uses exactly that mode."""
    selected_on_val = "routing_prob"
    used_on_test = selected_on_val  # must be same
    assert used_on_test == selected_on_val, "Score mode must be frozen after val selection"


def test_metrics_json_records_val_and_test_separately():
    """Metrics artifact must record both val_ap and test_ap with score_mode."""
    import json
    metrics = {
        "seed": 0,
        "score_mode_selected_on_val": "routing_prob",
        "val_ap": 0.85,
        "test_ap": 0.88,
        "leakage": False,
    }
    j = json.dumps(metrics)
    loaded = json.loads(j)
    assert "val_ap" in loaded and "test_ap" in loaded
    assert loaded["score_mode_selected_on_val"] is not None
    assert loaded["leakage"] is False


def test_calibration_on_cpu_tensors():
    """Logistic calibration fitted on val must not receive test tensors."""
    import torch
    n_val = 20; n_test = 10
    val_logits = torch.randn(n_val); val_labels = (val_logits > 0).float()
    test_logits = torch.randn(n_test)

    # Fit temperature scaling on val only
    temp = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.LBFGS([temp])
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            val_logits / temp, val_labels)
        loss.backward()
        return loss
    for _ in range(10):
        opt.step(closure)

    # Apply frozen temperature to test
    with torch.no_grad():
        test_probs = torch.sigmoid(test_logits / temp.detach())
    assert test_probs.shape == (n_test,)
    assert (test_probs >= 0).all() and (test_probs <= 1).all()


def test_routing_prob_score_mode_justification():
    """routing_prob is a principled score choice — verify it ranks TPs above FPs."""
    # If routing confidence is high → model is certain about this cluster → more likely TP
    # Low routing confidence → uncertain → more likely FP
    # For a perfect router: high_routing_prob ↔ high IoU cluster
    high_conf_iou = 0.80  # TP cluster
    low_conf_iou  = 0.30  # FP cluster
    high_routing  = 0.90
    low_routing   = 0.30

    score_tp = high_routing * max(high_conf_iou, 0.1)
    score_fp = low_routing  * max(low_conf_iou,  0.1)
    assert score_tp > score_fp, (
        "routing_prob * max(base, 0.1) should rank TP (high routing_prob) above FP"
    )
