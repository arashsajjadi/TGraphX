"""End-to-end validation of :mod:`tgraphx.explain` after real training.

The script trains a tiny synthetic patch-graph classifier, then runs
saliency / IG / edge-perturbation / patch-heatmap, and verifies the
exported artefacts are dashboard-readable.

Usage::

    python examples/explainability_end_to_end_validation.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-run-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-edges", type=int, default=16)
    args = p.parse_args(argv)

    import tgraphx
    from tgraphx import GraphBatch, build_model, set_seed
    from tgraphx.datasets import SyntheticPatchGraphDataset
    from tgraphx.experiments.callbacks import CSVLoggerCallback, RunState
    from tgraphx.explain import (
        edge_perturbation_attribution,
        export_edge_scores_csv,
        export_explanation_metadata,
        export_patch_heatmap_json,
        integrated_gradients,
        node_feature_saliency,
        patch_saliency_to_image_grid,
    )
    from tgraphx.tracking import (
        write_dataset_metadata, write_metrics_summary, write_run_metadata,
    )

    set_seed(args.seed)

    using_temp = args.output_run_dir is None
    tmp_ctx = None
    if using_temp:
        tmp = tempfile.TemporaryDirectory()
        run_dir = Path(tmp.name)
        tmp_ctx = tmp
    else:
        run_dir = Path(args.output_run_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Train.
        ds = SyntheticPatchGraphDataset(num_graphs=8, image_size=16,
                                        patch_size=4, seed=args.seed)
        items = list(ds)
        batch = GraphBatch(items)
        in_shape = tuple(batch.node_features.shape[1:])
        model = build_model(
            task="graph_classification", layer="conv",
            in_shape=in_shape, hidden_shape=(8, in_shape[1], in_shape[2]),
            num_layers=2, num_classes=ds.metadata.num_classes,
            pooling="mean",
        )
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = nn.functional.cross_entropy

        write_run_metadata(
            str(run_dir / "run_metadata.json"),
            run_name="explainability_e2e_smoke", status="running",
            tgraphx_version=tgraphx.__version__, seed=args.seed,
            task="graph_classification",
        )
        write_dataset_metadata(
            str(run_dir / "dataset_metadata.json"),
            **ds.metadata.to_dict(),
        )

        state = RunState(run_dir=run_dir)
        csv_cb = CSVLoggerCallback()
        history = []
        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
            loss = loss_fn(logits, batch.graph_labels.long())
            loss.backward()
            opt.step()
            metrics = {"train_loss": float(loss.item())}
            csv_cb.on_epoch_end(state, epoch, metrics)
            history.append({"epoch": epoch, **metrics})

        # 2) Explainability on the first sample.
        first = items[0]
        target = int(first.graph_label)
        sal = node_feature_saliency(model, first, target=target)
        ig = integrated_gradients(model, first, target=target, steps=8)

        # Verify shapes / finiteness.
        for name, t in (("saliency", sal), ("integrated_gradients", ig)):
            assert t.shape == first.node_features.shape, (name, t.shape)
            assert torch.isfinite(t).all().item(), name

        edge_scores = edge_perturbation_attribution(
            model, first, target=target, max_edges=args.max_edges,
        )
        assert edge_scores.numel() == args.max_edges

        heatmap = patch_saliency_to_image_grid(
            sal, grid_shape=first.metadata["grid_shape"],
        )

        # 3) Exports.
        export_explanation_metadata(
            str(run_dir / "explanation_metadata.json"),
            method="saliency+integrated_gradients",
            target=target,
            extra={"sample_index": 0,
                   "max_abs_saliency": float(sal.abs().max())},
        )
        export_edge_scores_csv(
            str(run_dir / "explanation_edges.csv"),
            first.edge_index[:, : edge_scores.numel()],
            edge_scores, method="perturbation", top_k=8,
        )
        export_patch_heatmap_json(
            str(run_dir / "explanation_patch_heatmap.json"),
            heatmap, grid_shape=first.metadata["grid_shape"],
            method="saliency",
        )

        write_metrics_summary(
            str(run_dir / "metrics_summary.json"),
            initial_train_loss=history[0]["train_loss"],
            final_train_loss=history[-1]["train_loss"],
            note="Synthetic explainability smoke; not a benchmark.",
        )

        # 4) Dashboard offline export.
        try:
            from tgraphx.dashboard.app import export_dashboard_html
            export_dashboard_html(str(run_dir), str(run_dir / "snapshot.html"))
        except Exception as exc:  # pragma: no cover
            print(f"  dashboard export skipped: {exc}")

        # 5) Verify exports are well-formed.
        meta = json.loads((run_dir / "explanation_metadata.json").read_text())
        assert meta["method"]

        edges_text = (run_dir / "explanation_edges.csv").read_text()
        assert edges_text.startswith("edge_id,src,dst,score,method")

        heatmap_payload = json.loads(
            (run_dir / "explanation_patch_heatmap.json").read_text()
        )
        assert heatmap_payload["shape"] == list(heatmap.shape)

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "epochs": len(history),
            "loss_start": history[0]["train_loss"],
            "loss_end": history[-1]["train_loss"],
            "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
            "saliency_shape": list(sal.shape),
            "integrated_gradients_shape": list(ig.shape),
            "edge_perturbation_shape": list(edge_scores.shape),
            "heatmap_shape": list(heatmap.shape),
            "run_dir": str(run_dir),
            "files": sorted(p.name for p in run_dir.iterdir()),
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["loss_decreased"] else 1
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
