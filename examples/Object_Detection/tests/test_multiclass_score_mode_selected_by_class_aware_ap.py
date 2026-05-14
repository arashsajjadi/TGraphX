"""Step 05 must select the score mode using CLASS-AWARE AP for multi-class.

This regression test reads the patched Step 05 evaluator source and asserts:
  1. The branch that picks the val metric calls evaluate_predictions with
     class_agnostic=False at least once when is_multiclass is True.
  2. The "selection_metric" recorded in metrics is "class_aware_AP" when
     is_multiclass is True.

The patched script lives at scripts/05_evaluate.py.
"""
from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py"


def test_step05_passes_class_agnostic_false_for_multiclass():
    src = SCRIPT.read_text()
    assert "selection_class_agnostic = not is_multiclass" in src, (
        "Step 05 must compute selection_class_agnostic = not is_multiclass."
    )
    assert "val_ap_aware" in src, (
        "Step 05 must compute class-aware AP and record val_ap_aware."
    )
    assert "selection_metric" in src and "class_aware_AP" in src, (
        "Step 05 must record selection_metric in val_score_modes."
    )


def test_step05_records_paired_bootstrap_against_baselines():
    src = SCRIPT.read_text()
    # Part 9 deliverable: Step 05 emits baseline_methods AND
    # paired_bootstrap_vs_baselines so Step 06 verdict can use them.
    assert "baseline_methods" in src
    assert "paired_bootstrap_vs_baselines" in src
