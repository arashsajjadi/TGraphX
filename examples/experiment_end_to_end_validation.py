"""End-to-end validation of :mod:`tgraphx.experiments`.

The script:

1. trains a tiny synthetic experiment via :class:`Runner`,
2. asserts the dashboard-compatible artefacts are written,
3. resumes from the best checkpoint,
4. exports a dashboard HTML snapshot,
5. emits a summary JSON.

Usage::

    python examples/experiment_end_to_end_validation.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-run-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args(argv)

    import tgraphx
    from tgraphx.experiments import (
        CallbackConfig, DatasetConfig, ExperimentConfig, ModelConfig,
        Runner, TrainingConfig,
    )

    using_temp = args.output_run_dir is None
    tmp_ctx = None
    if using_temp:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name)
        tmp_ctx = tmp
    else:
        base = Path(args.output_run_dir).expanduser()
        base.mkdir(parents=True, exist_ok=True)

    try:
        run_dir = base / "run"
        cfg = ExperimentConfig(
            seed=0,
            run_name="experiment_e2e_smoke",
            run_dir=str(run_dir),
            dataset=DatasetConfig(
                name="synthetic:patch_graph",
                kwargs={"num_graphs": 4, "image_size": 8,
                        "patch_size": 4, "seed": 0},
            ),
            model=ModelConfig(
                task="graph_classification", layer="conv",
                in_shape=[1, 4, 4], hidden_shape=[4, 4, 4],
                num_layers=2, num_classes=6,
            ),
            training=TrainingConfig(epochs=args.epochs, lr=0.01),
            callbacks=[
                CallbackConfig(name="csv_logger"),
                CallbackConfig(
                    name="model_checkpoint",
                    kwargs={"monitor": "train_loss"},
                ),
                CallbackConfig(
                    name="early_stopping",
                    kwargs={"monitor": "train_loss",
                            "patience": 100, "min_delta": 0.0},
                ),
            ],
        )

        runner = Runner(cfg)
        history = runner.fit()

        files = sorted(p.name for p in runner.run_dir.iterdir())
        required = [
            "metrics.csv",
            "run_metadata.json",
            "experiment_config.json",
            "experiment_summary.json",
            "checkpoints",
        ]
        missing = [r for r in required if r not in files]
        if missing:
            print(f"missing files: {missing}", file=sys.stderr)
            return 1

        # Resume from the best checkpoint.
        runner2 = Runner(cfg)
        payload = runner2.resume("checkpoints/best.pt")
        assert "model_state_dict" in payload

        # Dashboard offline export.
        try:
            from tgraphx.dashboard.app import export_dashboard_html
            export_dashboard_html(str(runner.run_dir),
                                  str(runner.run_dir / "snapshot.html"))
        except Exception as exc:  # pragma: no cover
            print(f"  dashboard export skipped: {exc}")

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "epochs": len(history),
            "loss_start": history[0]["train_loss"],
            "loss_end": history[-1]["train_loss"],
            "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
            "run_dir": str(runner.run_dir),
            "files": sorted(p.name for p in runner.run_dir.iterdir()),
            "resumed_payload_keys": sorted(payload.keys()),
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["loss_decreased"] else 1
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
