"""Step 06 verdict must compare TGraphX to the strongest VALIDATION-selected
baseline using a paired bootstrap, NOT to a hardcoded absolute AP threshold."""
from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_make_report.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _run_step06_in_temp(tmp_path, seeds_metrics):
    (tmp_path / "graph_audit.json").write_text(json.dumps({"avg_nodes": 10, "avg_edges": 20, "detector_names": ["a","b","c"]}))
    (tmp_path / "dataset_inventory.json").write_text(json.dumps({
        "num_records": 200, "class_names": ["car"], "source": "voc2007",
    }))
    (tmp_path / "split_manifest.json").write_text(json.dumps({"num_classes": 1, "config": "voc2007"}))
    for i, m in enumerate(seeds_metrics):
        (tmp_path / f"metrics_seed{i}.json").write_text(json.dumps(m))
    # Run the script
    import runpy
    sys.argv = ["06_make_report.py", "--run-dir", str(tmp_path), "--force"]
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return (tmp_path / "report.md").read_text()


def test_verdict_says_win_when_paired_bootstrap_is_confident(tmp_path):
    # 5 seeds where TGraphX beats NMS by a clear margin and the paired
    # bootstrap is confident.
    seeds = []
    for s in range(5):
        seeds.append({
            "seed": s,
            "test_metrics_selected_mode": {"headline_ap": 0.88},
            "baseline_methods": {
                "fusion::nms": {"headline_ap": 0.85, "test_ap_class_agnostic": 0.85, "test_ap_class_aware": 0.85},
                "fusion::wbf": {"headline_ap": 0.84, "test_ap_class_agnostic": 0.84, "test_ap_class_aware": 0.84},
                "fusion::tgraphx": {"headline_ap": 0.88, "test_ap_class_agnostic": 0.88, "test_ap_class_aware": 0.88},
            },
            "paired_bootstrap_vs_baselines": {
                "fusion::nms": {"p_a_gt_b": 0.97, "mean_diff": 0.03, "ci95_low": 0.01, "ci95_high": 0.05},
                "fusion::wbf": {"p_a_gt_b": 0.99, "mean_diff": 0.04, "ci95_low": 0.02, "ci95_high": 0.06},
            },
            "selected_score_mode": "p_tp50",
            "score_mode_selection_metric": "class_agnostic_AP",
        })
    report = _run_step06_in_temp(tmp_path, seeds)
    assert "REAL_VOC_CAR_WIN" in report


def test_verdict_says_tie_when_close_and_bootstrap_inconclusive(tmp_path):
    seeds = []
    for s in range(5):
        seeds.append({
            "seed": s,
            "test_metrics_selected_mode": {"headline_ap": 0.866},
            "baseline_methods": {
                "fusion::nms": {"headline_ap": 0.864, "test_ap_class_agnostic": 0.864, "test_ap_class_aware": 0.864},
                "fusion::tgraphx": {"headline_ap": 0.866, "test_ap_class_agnostic": 0.866, "test_ap_class_aware": 0.866},
            },
            "paired_bootstrap_vs_baselines": {
                "fusion::nms": {"p_a_gt_b": 0.88, "mean_diff": 0.002, "ci95_low": -0.005, "ci95_high": 0.009},
            },
            "selected_score_mode": "p_tp50",
            "score_mode_selection_metric": "class_agnostic_AP",
        })
    report = _run_step06_in_temp(tmp_path, seeds)
    assert "SAFE_TIE" in report or "NOT_YET_WIN" in report


def test_verdict_does_not_use_fixed_ap_threshold():
    """The literal threshold `> 0.85` MUST NOT appear unguarded in 06_make_report
    (verdict must come from paired-bootstrap fields, not a magic constant)."""
    src = SCRIPT.read_text()
    # The old code used `if mean_ap > 0.85:` for car-only and `> 0.50` for
    # multi-class. Both are gone.
    assert "mean_ap > 0.85" not in src
    assert "mean_ap > 0.50" not in src
    # New code must reference paired_bootstrap_vs_baselines.
    assert "paired_bootstrap_vs_baselines" in src
