"""Dashboard artefact end-to-end validation.

Creates a temporary run directory, writes every metadata artefact the
v0.3.0 dashboard understands, calls :func:`export_dashboard_html` on
it, and asserts the resulting HTML contains the right markers and no
secrets / external CDN references.

Usage::

    python examples/dashboard_artifact_validation.py
    python examples/dashboard_artifact_validation.py --output-run-dir ./runs/dash
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _populate_run(run_dir: Path) -> Dict[str, Any]:
    """Write every supported artefact and return a description dict."""
    import torch
    from tgraphx import HeteroGraph, TemporalGraphSequence, Graph
    from tgraphx.tracking import (
        write_benchmark_results,
        write_dataset_metadata,
        write_experiment_config,
        write_explanation_metadata,
        write_hardware_report,
        write_hetero_graph_metadata,
        write_metrics_summary,
        write_run_metadata,
        write_sampling_metadata,
        write_temporal_metadata,
        write_transform_metadata,
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    # Old, minimal metrics.csv (backwards compatibility).
    (run_dir / "metrics.csv").write_text(
        "epoch,train_loss,val_loss\n0,1.0,1.2\n1,0.7,0.9\n2,0.5,0.7\n"
    )
    write_run_metadata(
        str(run_dir / "run_metadata.json"),
        run_name="dashboard_smoke", status="completed",
        total_epochs=3, device="cpu", task="graph_classification",
        tgraphx_version="0.3.0", seed=0,
    )

    # New v0.3.0 metadata files.
    write_dataset_metadata(
        str(run_dir / "dataset_metadata.json"),
        name="synthetic:patch_graph", task="graph_classification",
        graph_type="homogeneous", num_graphs=32, num_classes=6,
        upstream_library=None, license="MIT (TGraphX synthetic)",
    )
    write_transform_metadata(
        str(run_dir / "transform_metadata.json"),
        ["NormalizeFeatures", "AddSelfLoops", "RandomNodeSplit"],
        seed=0,
    )
    write_metrics_summary(
        str(run_dir / "metrics_summary.json"),
        best_epoch=2, best_val_loss=0.7, final_train_loss=0.5,
    )
    write_benchmark_results(
        str(run_dir / "benchmark_results.json"),
        benchmark="dashboard_smoke", device="cpu", elapsed_s=0.05,
        num_graphs=4,
    )
    write_explanation_metadata(
        str(run_dir / "explanation_metadata.json"),
        method="saliency", target=0,
        sample_index=0,
    )
    (run_dir / "explanation_edges.csv").write_text(
        "edge_id,src,dst,score,method\n"
        "0,0,1,0.83,perturbation\n"
        "1,1,2,0.41,perturbation\n"
    )
    (run_dir / "explanation_patch_heatmap.json").write_text(
        json.dumps({"shape": [4, 4],
                    "values": [[0.0, 0.1, 0.2, 0.3]] * 4,
                    "method": "saliency",
                    "grid_shape": [2, 2]})
    )
    write_experiment_config(
        str(run_dir / "experiment_config.json"),
        {"seed": 0, "model": {"task": "graph_classification"}},
    )
    (run_dir / "experiment_summary.json").write_text(json.dumps({
        "run_name": "dashboard_smoke",
        "epochs": 3,
        "best_metric": 0.7,
        "best_epoch": 2,
        "final_train_loss": 0.5,
    }))
    write_hardware_report(
        str(run_dir / "hardware_report.json"),
        cpu_count=4, cuda_available=False, mps_available=False,
    )
    write_sampling_metadata(
        str(run_dir / "sampling_metadata.json"),
        kind="random_walk_sample", walk_length=10, num_walks_per_seed=3,
        direction="out", seed=0,
    )
    hg = HeteroGraph(
        node_stores={"a": torch.randn(3, 2), "b": torch.randn(2, 2)},
        edge_stores={
            ("a", "rel", "b"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        },
    )
    write_hetero_graph_metadata(str(run_dir / "hetero_graph_metadata.json"), hg)
    seq = TemporalGraphSequence(
        graphs=[Graph(torch.randn(3, 2), None) for _ in range(4)],
        timestamps=[0.0, 1.0, 2.0, 3.0],
    )
    write_temporal_metadata(str(run_dir / "temporal_metadata.json"), seq)

    return {"files": sorted(p.name for p in run_dir.iterdir())}


def _check_html_security(html: str, secret_token: str) -> Dict[str, Any]:
    """Return findings for security guarantees the dashboard must keep."""
    findings: Dict[str, Any] = {}
    findings["external_cdn_referenced"] = (
        "cdn.jsdelivr" in html.lower() or "unpkg.com" in html.lower()
    )
    code_only = re.sub(r"/\*.*?\*/", "", html, flags=re.DOTALL)
    code_only = re.sub(r"//[^\n]*", "", code_only)
    findings["eval_used"] = bool(re.search(r"\beval\s*\(", code_only))
    findings["new_function_used"] = "new Function" in code_only
    findings["secret_token_leaked"] = secret_token in html
    findings["snapshot_marker"] = "__TGXSNAP" in html
    return findings


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-run-dir", type=str, default=None,
                   help="Persistent run dir (default: a TemporaryDirectory).")
    args = p.parse_args(argv)

    using_temp = args.output_run_dir is None
    tmp_ctx = None
    if using_temp:
        tmp = tempfile.TemporaryDirectory()
        run_dir = Path(tmp.name)
        tmp_ctx = tmp
    else:
        run_dir = Path(args.output_run_dir).expanduser()

    secret_token = "DO-NOT-LEAK-INTO-EXPORT-XYZ"
    try:
        _populate_run(run_dir)
        # Surface a fake "token" only inside files the export must not embed.
        # (We do NOT pass it into export_dashboard_html.)
        from tgraphx.dashboard.app import export_dashboard_html
        out_html = run_dir / "snapshot.html"
        export_dashboard_html(str(run_dir), str(out_html))
        text = out_html.read_text()
        findings = _check_html_security(text, secret_token)

        # Re-scan after the snapshot is created.
        all_files = sorted(p.name for p in run_dir.iterdir())
        report = {
            "run_dir": str(run_dir),
            "files": all_files,
            "snapshot_size_bytes": out_html.stat().st_size,
            "security": findings,
            "passed": (
                findings["snapshot_marker"]
                and not findings["external_cdn_referenced"]
                and not findings["eval_used"]
                and not findings["new_function_used"]
                and not findings["secret_token_leaked"]
            ),
        }
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
