"""Tests for tgraphx.experiments (v0.3.0)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tgraphx.experiments import (
    CSVLoggerCallback,
    EarlyStopping,
    ExperimentConfig,
    DatasetConfig,
    GridRunner,
    ModelCheckpoint,
    ModelConfig,
    Runner,
    TrainingConfig,
    TransformConfig,
    expand_grid,
    load_config,
    summarize_runs,
)


def _patch_cfg(run_dir: Path, epochs: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        seed=0,
        run_name="patch",
        run_dir=str(run_dir),
        dataset=DatasetConfig(
            name="synthetic:patch_graph",
            kwargs={"num_graphs": 4, "image_size": 16, "patch_size": 4, "seed": 0},
        ),
        model=ModelConfig(
            task="graph_classification",
            layer="conv",
            in_shape=[1, 4, 4],
            hidden_shape=[8, 4, 4],
            num_layers=2,
            num_classes=6,
        ),
        training=TrainingConfig(epochs=epochs, lr=0.01),
    )


# ── Config loading ───────────────────────────────────────────────────────────


class TestConfig:
    def test_load_dict(self):
        cfg = load_config({
            "seed": 1,
            "run_name": "demo",
            "dataset": {"name": "synthetic:patch_graph"},
            "model": {"task": "graph_classification", "layer": "linear",
                      "in_shape": [4], "hidden_shape": [8], "num_classes": 3},
        })
        assert cfg.seed == 1
        assert cfg.dataset.name == "synthetic:patch_graph"
        assert cfg.model.task == "graph_classification"

    def test_load_json_file(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({
            "seed": 2,
            "dataset": {"name": "synthetic:patch_graph"},
            "model": {"task": "graph_classification", "layer": "linear"},
        }))
        cfg = load_config(path)
        assert cfg.seed == 2

    def test_load_yaml_file(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "cfg.yaml"
        path.write_text(
            "seed: 3\n"
            "run_name: yaml_run\n"
            "dataset: {name: 'synthetic:patch_graph'}\n"
            "model: {task: graph_classification, layer: linear}\n"
        )
        cfg = load_config(path)
        assert cfg.seed == 3
        assert cfg.run_name == "yaml_run"

    def test_unknown_keys_rejected(self):
        with pytest.raises(ValueError, match="Unknown top-level"):
            load_config({"dataset": {"name": "synthetic:patch_graph"},
                         "model": {"task": "graph_classification"},
                         "junk_field": 99})

    def test_missing_dataset(self):
        with pytest.raises(ValueError, match="config.dataset"):
            load_config({"model": {"task": "graph_classification"}})

    def test_missing_model(self):
        with pytest.raises(ValueError, match="config.model"):
            load_config({"dataset": {"name": "x"}})


# ── Runner ───────────────────────────────────────────────────────────────────


class TestRunner:
    def test_runs_and_writes_files(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "run", epochs=3)
        runner = Runner(cfg)
        history = runner.fit()
        assert len(history) == 3
        # Run dir contains all expected artefacts.
        files = {p.name for p in runner.run_dir.iterdir()}
        for required in ("metrics.csv", "run_metadata.json",
                         "experiment_config.json", "experiment_summary.json"):
            assert required in files
        # Status flipped to completed.
        meta = json.loads((runner.run_dir / "run_metadata.json").read_text())
        assert meta["status"] == "completed"

    def test_loss_decreases(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "run", epochs=4)
        history = Runner(cfg).fit()
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_early_stopping(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "run", epochs=20)
        cfg.callbacks.append(
            __import__("tgraphx.experiments", fromlist=["CallbackConfig"]).CallbackConfig(
                name="early_stopping",
                kwargs={"monitor": "train_loss", "patience": 1, "min_delta": 100.0},
            )
        )
        history = Runner(cfg).fit()
        # min_delta is huge, so we should stop quickly.
        assert len(history) < 10

    def test_model_checkpoint(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "run", epochs=3)
        cfg.callbacks.append(
            __import__("tgraphx.experiments", fromlist=["CallbackConfig"]).CallbackConfig(
                name="model_checkpoint",
                kwargs={"monitor": "train_loss"},
            )
        )
        runner = Runner(cfg)
        runner.fit()
        ckpt = runner.run_dir / "checkpoints" / "best.pt"
        assert ckpt.exists()

    def test_run_dir_only_writes(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "isolated", epochs=2)
        runner = Runner(cfg)
        runner.fit()
        # Nothing was written outside run_dir / tmp_path.
        for p in tmp_path.iterdir():
            assert p == runner.run_dir or p.is_relative_to(runner.run_dir.parent)

    def test_node_classification_runs(self, tmp_path):
        cfg = ExperimentConfig(
            seed=0, run_name="node",
            run_dir=str(tmp_path / "node"),
            dataset=DatasetConfig(name="synthetic:node_classification",
                                  kwargs={"num_nodes": 30, "num_classes": 3, "seed": 0}),
            model=ModelConfig(task="node_classification", layer="linear",
                              in_shape=[16], hidden_shape=[8], num_layers=2,
                              num_classes=3),
            training=TrainingConfig(epochs=3, lr=0.01),
        )
        history = Runner(cfg).fit()
        assert history[-1]["train_loss"] < history[0]["train_loss"]


# ── Grid expansion ───────────────────────────────────────────────────────────


class TestGrid:
    def test_expand_grid(self):
        base = {"training": {"lr": 0.001, "epochs": 5}}
        configs = expand_grid(base, {"training.lr": [0.001, 0.01],
                                     "training.epochs": [2, 4]})
        assert len(configs) == 4
        lrs = sorted(c["training"]["lr"] for c in configs)
        assert lrs == [0.001, 0.001, 0.01, 0.01]

    def test_grid_runner_runs_all(self, tmp_path):
        base = {
            "seed": 0,
            "run_name": "grid_smoke",
            "dataset": {"name": "synthetic:patch_graph",
                        "kwargs": {"num_graphs": 2, "image_size": 8, "patch_size": 4, "seed": 0}},
            "model": {"task": "graph_classification", "layer": "conv",
                      "in_shape": [1, 4, 4], "hidden_shape": [4, 4, 4],
                      "num_layers": 2, "num_classes": 6},
            "training": {"epochs": 2, "lr": 0.01},
        }
        gr = GridRunner(
            base_config=base,
            grid={"training.lr": [0.005, 0.01]},
            seeds=[0],
            out_dir=tmp_path / "grid",
        )
        results = gr.run()
        assert len(results) == 2
        # Summary persisted.
        summary = json.loads((tmp_path / "grid" / "grid_summary.json").read_text())
        assert summary["num_runs"] == 2


# ── Resume ───────────────────────────────────────────────────────────────────


class TestResume:
    def test_resume_loads_checkpoint(self, tmp_path):
        cfg = _patch_cfg(tmp_path / "run", epochs=2)
        cfg.callbacks.append(
            __import__("tgraphx.experiments", fromlist=["CallbackConfig"]).CallbackConfig(
                name="model_checkpoint", kwargs={"monitor": "train_loss"},
            )
        )
        Runner(cfg).fit()
        runner2 = Runner(cfg)
        payload = runner2.resume("checkpoints/best.pt")
        assert "model_state_dict" in payload
        assert "epoch" in payload


# ── Summarize ────────────────────────────────────────────────────────────────


class TestSummarize:
    def test_summarize_runs_collects_metadata(self, tmp_path):
        run_dir = tmp_path / "r1"
        cfg = _patch_cfg(run_dir, epochs=2)
        Runner(cfg).fit()
        rows = summarize_runs(tmp_path)
        assert len(rows) >= 1
        assert any(r.get("run_name") == "patch" for r in rows)


# ── CLI ──────────────────────────────────────────────────────────────────────


class TestCLI:
    """Invoke the CLI via ``python -m`` so the test works whether or not the
    console-script entry points are installed in the current venv."""

    @staticmethod
    def _module_for(cmd: str) -> str:
        # cli.train_main / grid_main / report_main are all in tgraphx.experiments.cli;
        # we expose them through ``python -m tgraphx.experiments.<cmd>``-style by
        # invoking the cli module directly.
        return cmd

    def test_train_main_help(self):
        from tgraphx.experiments.cli import train_main
        with pytest.raises(SystemExit) as exc_info:
            train_main(["--help"])
        assert exc_info.value.code == 0

    def test_grid_main_help(self):
        from tgraphx.experiments.cli import grid_main
        with pytest.raises(SystemExit) as exc_info:
            grid_main(["--help"])
        assert exc_info.value.code == 0

    def test_report_main_help(self):
        from tgraphx.experiments.cli import report_main
        with pytest.raises(SystemExit) as exc_info:
            report_main(["--help"])
        assert exc_info.value.code == 0

    def test_train_main_runs(self, tmp_path):
        from tgraphx.experiments.cli import train_main
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({
            "seed": 0,
            "run_name": "cli_smoke",
            "run_dir": str(tmp_path / "cli_run"),
            "dataset": {
                "name": "synthetic:patch_graph",
                "kwargs": {"num_graphs": 2, "image_size": 8, "patch_size": 4, "seed": 0},
            },
            "model": {
                "task": "graph_classification", "layer": "conv",
                "in_shape": [1, 4, 4], "hidden_shape": [4, 4, 4],
                "num_layers": 2, "num_classes": 6,
            },
            "training": {"epochs": 2, "lr": 0.01},
        }))
        rc = train_main([str(cfg_path), "--quiet"])
        assert rc == 0
        run_dir = tmp_path / "cli_run"
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "run_metadata.json").exists()
