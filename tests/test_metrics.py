"""Metric tests with hand-computed values (v0.2.9)."""
from __future__ import annotations

import pytest
import torch

from tgraphx.metrics import (
    accuracy,
    average_precision,
    classification_report,
    confusion_matrix,
    edge_classification_report,
    graph_classification_report,
    graph_regression_report,
    hits_at_k,
    link_prediction_report,
    mae,
    mean_reciprocal_rank,
    mse,
    ndcg_at_k,
    node_classification_report,
    precision_recall_f1,
    r2_score,
    regression_report,
    rmse,
    roc_auc,
    top_k_accuracy,
)


# ── classification ───────────────────────────────────────────────────────────


class TestAccuracy:
    def test_idx_form(self):
        preds = torch.tensor([0, 1, 1, 1])
        labels = torch.tensor([0, 1, 1, 0])
        assert accuracy(preds, labels) == pytest.approx(0.75)

    def test_logits_form(self):
        # All logits favour class 1 except sample 0.
        preds = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, 1, 0])
        assert accuracy(preds, labels) == pytest.approx(0.75)

    def test_empty(self):
        preds = torch.zeros(0, dtype=torch.long)
        labels = torch.zeros(0, dtype=torch.long)
        assert accuracy(preds, labels) == 0.0


class TestTopKAccuracy:
    def test_basic(self):
        # Class with second-highest score is correct for half the rows.
        logits = torch.tensor([
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ])
        labels = torch.tensor([1, 0])
        # top-1: 50% (first row predicted 0); top-2: 100%.
        assert top_k_accuracy(logits, labels, k=1) == pytest.approx(0.5)
        assert top_k_accuracy(logits, labels, k=2) == pytest.approx(1.0)


class TestConfusionMatrix:
    def test_layout(self):
        preds = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 1, 1, 1])
        cm = confusion_matrix(preds, labels, num_classes=2)
        # Row = true, column = predicted.
        # true=0,pred=0: 1 ; true=1,pred=0: 1 ; true=1,pred=1: 2
        assert cm.tolist() == [[1, 0], [1, 2]]


class TestPRF:
    def test_macro_average(self):
        # 2 classes, perfect predictions.
        preds = torch.tensor([0, 1, 0, 1])
        labels = torch.tensor([0, 1, 0, 1])
        out = precision_recall_f1(preds, labels, num_classes=2, average="macro")
        assert out["precision"] == pytest.approx(1.0)
        assert out["recall"] == pytest.approx(1.0)
        assert out["f1"] == pytest.approx(1.0)

    def test_zero_division(self):
        preds = torch.tensor([0, 0, 0, 0])
        labels = torch.tensor([0, 1, 0, 1])
        # Class 1 has no predictions → precision is zero_division.
        out = precision_recall_f1(preds, labels, num_classes=2,
                                  average="macro", zero_division=0.0)
        # Per-class precision: cls 0 = 2/4, cls 1 = 0 (no preds)
        assert out["precision_per_class"][1] == 0.0


# ── regression ───────────────────────────────────────────────────────────────


class TestRegression:
    def test_known_values(self):
        p = torch.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([1.5, 2.5, 2.5])
        assert mae(p, t) == pytest.approx(0.5)
        assert mse(p, t) == pytest.approx(0.25)
        assert rmse(p, t) == pytest.approx(0.5)

    def test_r2_perfect(self):
        p = torch.tensor([1.0, 2.0, 3.0])
        assert r2_score(p, p) == pytest.approx(1.0)

    def test_r2_zero_variance(self):
        # All targets identical → variance 0 → returns 0.0.
        t = torch.tensor([1.0, 1.0, 1.0])
        p = torch.tensor([1.0, 1.0, 0.5])
        assert r2_score(p, t) == 0.0

    def test_regression_report_keys(self):
        out = regression_report(torch.zeros(3), torch.zeros(3))
        for k in ("mae", "mse", "rmse", "r2"):
            assert k in out


# ── ranking ──────────────────────────────────────────────────────────────────


class TestRanking:
    def test_hits_at_1(self):
        # Sample 0: top-1 is index 1 (correct); sample 1: top-1 is 0 (correct).
        scores = torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.5, 0.0]])
        target = torch.tensor([1, 0])
        assert hits_at_k(scores, target, k=1) == pytest.approx(1.0)

    def test_mrr(self):
        # Target index 1; sorted desc indices = [1, 2, 0]. Rank of target=1 → 1/1.
        scores = torch.tensor([[0.0, 1.0, 0.5]])
        assert mean_reciprocal_rank(scores, torch.tensor([1])) == pytest.approx(1.0)
        # Target at second position.
        scores2 = torch.tensor([[0.5, 1.0, 2.0]])
        assert mean_reciprocal_rank(scores2, torch.tensor([1])) == pytest.approx(1 / 2)

    def test_ndcg_perfect(self):
        scores = torch.tensor([[0.0, 1.0, 0.5]])
        assert ndcg_at_k(scores, torch.tensor([1]), k=2) == pytest.approx(1.0)


# ── link prediction ──────────────────────────────────────────────────────────


class TestLinkPrediction:
    def test_perfect_separation(self):
        pos = torch.tensor([0.9, 0.8, 0.7])
        neg = torch.tensor([0.1, 0.2, 0.3])
        assert roc_auc(pos, neg) == pytest.approx(1.0)
        assert average_precision(pos, neg) == pytest.approx(1.0)

    def test_random_scores_close_to_half(self):
        torch.manual_seed(0)
        pos = torch.randn(500)
        neg = torch.randn(500)
        auc = roc_auc(pos, neg)
        # Expect ≈ 0.5 ± reasonable margin.
        assert 0.3 < auc < 0.7

    def test_link_report_shape(self):
        out = link_prediction_report(torch.tensor([0.9]), torch.tensor([0.1]))
        assert out["num_pos"] == 1
        assert out["num_neg"] == 1


# ── reports ──────────────────────────────────────────────────────────────────


class TestReports:
    def test_graph_classification_report(self):
        out = graph_classification_report(
            torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]),
        )
        assert "accuracy" in out and "f1" in out

    def test_node_classification_report_with_mask(self):
        preds = torch.tensor([0, 1, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])
        mask = torch.tensor([True, False, True, True])
        out = node_classification_report(preds, labels, mask=mask)
        # Masked: preds=[0,1,1], labels=[0,1,1] → acc=1.0
        assert out["accuracy"] == pytest.approx(1.0)

    def test_edge_classification_report(self):
        out = edge_classification_report(
            torch.tensor([1, 1, 0]), torch.tensor([1, 0, 0]),
        )
        assert "f1" in out

    def test_graph_regression_report(self):
        out = graph_regression_report(torch.ones(3), torch.zeros(3))
        assert out["mae"] == pytest.approx(1.0)


# ── autograd hygiene ─────────────────────────────────────────────────────────


class TestAutogradHygiene:
    def test_metrics_do_not_retain_graph(self):
        x = torch.randn(8, 4, requires_grad=True)
        labels = torch.randint(0, 4, (8,))
        # accuracy must not retain graph through .item().
        acc = accuracy(x, labels)
        assert isinstance(acc, float)

    def test_regression_returns_python_floats(self):
        p = torch.randn(10, requires_grad=True)
        t = torch.randn(10)
        for name in ("mae", "mse", "rmse"):
            fn = {"mae": mae, "mse": mse, "rmse": rmse}[name]
            v = fn(p, t)
            assert isinstance(v, float)
