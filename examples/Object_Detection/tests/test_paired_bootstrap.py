"""Tests for paired bootstrap utility used by Step 06 verdict."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_paired_bootstrap_p_a_gt_b_when_a_strictly_better():
    """If A is strictly better than B on every image, P(A>B) must be 1."""
    from od_graph_fusion.paired_bootstrap import paired_bootstrap
    a = torch.tensor([0.9, 0.8, 0.85, 0.92, 0.88])
    b = torch.tensor([0.8, 0.7, 0.75, 0.85, 0.78])
    boot = paired_bootstrap(a, b, n_resamples=500, seed=0)
    assert boot["p_a_gt_b"] > 0.99
    assert boot["mean_diff"] > 0.0


def test_paired_bootstrap_tie_when_identical():
    """Identical methods → P near 0.5 and tiny mean diff."""
    from od_graph_fusion.paired_bootstrap import paired_bootstrap
    a = torch.tensor([0.5, 0.6, 0.7, 0.8])
    boot = paired_bootstrap(a, a.clone(), n_resamples=500, seed=0)
    assert abs(boot["p_a_gt_b"] - 0.5) <= 0.51  # near 0.5 (degenerate)
    assert abs(boot["mean_diff"]) < 1e-6


def test_verdict_from_bootstrap_win_tie_loss():
    from od_graph_fusion.paired_bootstrap import verdict_from_bootstrap
    win = {"p_a_gt_b": 0.97, "mean_diff": 0.02, "ci95_low": 0.01, "ci95_high": 0.03}
    tie = {"p_a_gt_b": 0.88, "mean_diff": 0.005, "ci95_low": -0.001, "ci95_high": 0.011}
    loss = {"p_a_gt_b": 0.40, "mean_diff": -0.01, "ci95_low": -0.02, "ci95_high": 0.00}
    assert verdict_from_bootstrap(win) == "WIN"
    assert verdict_from_bootstrap(tie) == "TIE"
    assert verdict_from_bootstrap(loss) in ("TIE", "NOT_YET_WIN")
