"""Tests for v0.3.0 dashboard metadata writers + dashboard backwards compat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tgraphx import (
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


class TestMetadataWriters:
    def test_run_metadata(self, tmp_path):
        path = tmp_path / "run_metadata.json"
        write_run_metadata(str(path), run_name="demo", status="completed",
                           total_epochs=3, device="cpu", task="graph_classification")
        data = json.loads(path.read_text())
        assert data["run_name"] == "demo"
        assert data["status"] == "completed"

    def test_dataset_metadata(self, tmp_path):
        path = tmp_path / "dataset_metadata.json"
        write_dataset_metadata(str(path),
                               name="synthetic:patch_graph",
                               task="graph_classification",
                               num_graphs=32, num_classes=6)
        data = json.loads(path.read_text())
        assert data["name"] == "synthetic:patch_graph"

    def test_transform_metadata_from_compose(self, tmp_path):
        from tgraphx.transforms import AddSelfLoops, Compose, NormalizeFeatures
        pipeline = Compose([NormalizeFeatures(), AddSelfLoops()])
        path = tmp_path / "transform_metadata.json"
        write_transform_metadata(str(path), pipeline, seed=0)
        data = json.loads(path.read_text())
        assert data["pipeline"] == ["NormalizeFeatures", "AddSelfLoops"]

    def test_metrics_summary(self, tmp_path):
        path = tmp_path / "metrics_summary.json"
        write_metrics_summary(str(path), best_epoch=4, best_val_loss=0.21)
        assert "best_val_loss" in json.loads(path.read_text())

    def test_benchmark_results(self, tmp_path):
        path = tmp_path / "benchmark_results.json"
        write_benchmark_results(str(path), benchmark="smoke",
                                elapsed_s=0.05, device="cpu")
        assert json.loads(path.read_text())["benchmark"] == "smoke"

    def test_explanation_metadata(self, tmp_path):
        path = tmp_path / "explanation_metadata.json"
        write_explanation_metadata(str(path), method="saliency", target=2)
        assert json.loads(path.read_text())["method"] == "saliency"

    def test_experiment_config_from_dict(self, tmp_path):
        path = tmp_path / "experiment_config.json"
        write_experiment_config(str(path), {"seed": 0, "model": {"task": "node_classification"}})
        data = json.loads(path.read_text())
        assert data["seed"] == 0

    def test_experiment_config_from_obj(self, tmp_path):
        from tgraphx.experiments import (
            DatasetConfig, ExperimentConfig, ModelConfig,
        )
        cfg = ExperimentConfig(
            seed=0, run_name="t",
            dataset=DatasetConfig(name="synthetic:patch_graph"),
            model=ModelConfig(task="graph_classification"),
        )
        path = tmp_path / "experiment_config.json"
        write_experiment_config(str(path), cfg)
        assert json.loads(path.read_text())["seed"] == 0

    def test_hardware_report(self, tmp_path):
        path = tmp_path / "hardware_report.json"
        write_hardware_report(str(path), cpu_count=4, cuda_available=False)
        assert json.loads(path.read_text())["cuda_available"] is False

    def test_sampling_metadata(self, tmp_path):
        path = tmp_path / "sampling_metadata.json"
        write_sampling_metadata(str(path), kind="random_walk_sample",
                                walk_length=10, seed=0)
        assert json.loads(path.read_text())["kind"] == "random_walk_sample"

    def test_hetero_graph_metadata_from_obj(self, tmp_path):
        from tgraphx import HeteroGraph
        import torch
        hg = HeteroGraph(
            node_stores={"a": torch.randn(3, 2), "b": torch.randn(2, 2)},
            edge_stores={
                ("a", "rel", "b"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            },
        )
        path = tmp_path / "hetero_graph_metadata.json"
        write_hetero_graph_metadata(str(path), hg)
        data = json.loads(path.read_text())
        assert "node_types" in data and set(data["node_types"]) == {"a", "b"}

    def test_temporal_metadata_from_obj(self, tmp_path):
        from tgraphx import Graph, TemporalGraphSequence
        import torch
        seq = TemporalGraphSequence(
            graphs=[Graph(torch.randn(3, 2), None) for _ in range(4)],
            timestamps=[0.0, 1.0, 2.0, 3.0],
        )
        path = tmp_path / "temporal_metadata.json"
        write_temporal_metadata(str(path), seq)
        data = json.loads(path.read_text())
        assert data["num_snapshots"] == 4

    def test_atomic_failure_cleans_tmp(self, tmp_path, monkeypatch):
        path = tmp_path / "x.json"

        import os
        original_replace = os.replace

        def boom(*a, **kw):
            raise OSError("simulated")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            write_run_metadata(str(path), run_name="x")
        # No leftover .tmp files.
        leftovers = list(tmp_path.glob("x.json.*tmp"))
        assert leftovers == []
        monkeypatch.setattr(os, "replace", original_replace)


# ── Dashboard backwards compatibility ────────────────────────────────────────


class TestDashboardBackwardsCompat:
    def test_old_minimal_metrics_csv_export(self, tmp_path):
        """Old-style ``metrics.csv`` + ``run_metadata.json`` still renders."""
        from tgraphx.dashboard.app import export_dashboard_html
        (tmp_path / "metrics.csv").write_text(
            "epoch,train_loss,val_loss\n0,1.0,1.2\n1,0.7,0.9\n"
        )
        (tmp_path / "run_metadata.json").write_text(
            '{"run_name":"old","status":"completed","total_epochs":2}'
        )
        out = tmp_path / "snapshot.html"
        export_dashboard_html(str(tmp_path), str(out))
        assert out.exists()
        text = out.read_text()
        assert "old" in text

    def test_dashboard_handles_new_metadata_files(self, tmp_path):
        """Adding the new v0.3.0 metadata files must not break the dashboard."""
        from tgraphx.dashboard.app import export_dashboard_html
        (tmp_path / "metrics.csv").write_text("epoch,train_loss\n0,0.5\n")
        write_run_metadata(str(tmp_path / "run_metadata.json"),
                           run_name="v030", status="completed", total_epochs=1)
        write_dataset_metadata(str(tmp_path / "dataset_metadata.json"),
                               name="synthetic:patch_graph", task="graph_classification")
        write_explanation_metadata(str(tmp_path / "explanation_metadata.json"),
                                    method="saliency", target=0)
        write_benchmark_results(str(tmp_path / "benchmark_results.json"),
                                 benchmark="smoke", elapsed_s=0.1)
        out = tmp_path / "snapshot.html"
        export_dashboard_html(str(tmp_path), str(out))
        assert out.exists()

    def test_dashboard_cli_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "tgraphx.dashboard", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
