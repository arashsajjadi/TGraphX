"""MNIST → TGraphX patch graph manual validation (real download).

This script **must** be run with ``--download`` before any network
access happens.  It is intentionally not exercised in CI.

Usage::

    python examples/public_datasets/mnist_patch_smoke.py --download \\
        --max-samples 100 --epochs 3
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    make_parser, mark_run_completed, resolve_device, soft_skip,
    write_run_provenance, write_summary_json,
)


def main(argv=None) -> int:
    parser = make_parser(
        "mnist_patch_smoke",
        "Manual MNIST patch-graph validation through the torchvision adapter.",
    )
    args = parser.parse_args(argv)

    try:
        import torchvision  # noqa: F401
    except ImportError:
        return soft_skip("torchvision not installed", args.strict)

    if not args.download:
        print("This script requires --download (TGraphX never downloads silently).")
        print("Re-run with: --download --max-samples 100 --epochs 3")
        return 0

    import torch
    import torch.nn as nn
    import tgraphx
    from tgraphx import GraphBatch, build_model
    from tgraphx.datasets import MNISTPatchGraphDataset
    from tgraphx.experiments.callbacks import CSVLoggerCallback, RunState
    from tgraphx.explain import (
        export_edge_scores_csv, export_explanation_metadata,
        export_patch_heatmap_json, integrated_gradients,
        node_feature_saliency, patch_saliency_to_image_grid,
    )
    from tgraphx.tracking import write_dataset_metadata, write_metrics_summary

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

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
        ds = MNISTPatchGraphDataset(
            root=args.root,
            download=True,
            patch_size=7,
            train=True,
        )
        n = max(2, min(args.max_samples, len(ds)))
        items = [ds[i] for i in range(n)]
        batch = GraphBatch(items).to(device)
        in_shape = tuple(batch.node_features.shape[1:])

        write_run_provenance(
            run_dir,
            run_name="mnist_patch_smoke",
            device=str(device), task="graph_classification",
            seed=args.seed, dataset="torchvision:mnist_patch",
            num_samples=n,
        )
        write_dataset_metadata(
            str(run_dir / "dataset_metadata.json"),
            **{**ds.metadata.to_dict(), "num_graphs": n,
               "validation_note": "manual_run; --max-samples capped"},
        )

        model = build_model(
            task="graph_classification", layer="conv",
            in_shape=in_shape, hidden_shape=(8, in_shape[1], in_shape[2]),
            num_layers=2, num_classes=10, pooling="mean",
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = nn.functional.cross_entropy

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
            print(f"  epoch {epoch}  train_loss={metrics['train_loss']:.4f}")

        # Sanity accuracy on the same capped batch (NOT a benchmark).
        model.eval()
        with torch.no_grad():
            preds = model(batch.node_features, batch.edge_index,
                          batch=batch.batch).argmax(dim=-1)
            acc = float((preds == batch.graph_labels.long()).float().mean())

        # Explainability on the first sample.
        first = items[0].to(device)
        target = int(first.graph_label)
        sal = node_feature_saliency(model, first, target=target)
        ig = integrated_gradients(model, first, target=target, steps=8)
        heatmap = patch_saliency_to_image_grid(
            sal.cpu(), grid_shape=first.metadata["grid_shape"],
        )

        export_explanation_metadata(
            str(run_dir / "explanation_metadata.json"),
            method="saliency+integrated_gradients",
            target=target,
            extra={"sample_index": 0, "max_abs_saliency": float(sal.abs().max())},
        )
        export_patch_heatmap_json(
            str(run_dir / "explanation_patch_heatmap.json"),
            heatmap, grid_shape=first.metadata["grid_shape"],
            method="saliency",
        )
        # Edge scores from a tiny perturbation pass on the original sample.
        from tgraphx.explain import edge_perturbation_attribution
        edge_scores = edge_perturbation_attribution(
            model, first, target=target, max_edges=args.max_edges,
        )
        export_edge_scores_csv(
            str(run_dir / "explanation_edges.csv"),
            first.edge_index[:, : edge_scores.numel()],
            edge_scores, method="perturbation", top_k=10,
        )

        write_metrics_summary(
            str(run_dir / "metrics_summary.json"),
            final_train_loss=history[-1]["train_loss"],
            initial_train_loss=history[0]["train_loss"],
            sanity_accuracy_capped=acc,
            note="Capped manual MNIST validation; not a benchmark / SOTA result.",
        )
        mark_run_completed(run_dir, total_epochs=len(history),
                           final_train_loss=history[-1]["train_loss"])

        try:
            from tgraphx.dashboard.app import export_dashboard_html
            export_dashboard_html(str(run_dir), str(run_dir / "snapshot.html"))
        except Exception as exc:  # pragma: no cover
            print(f"  dashboard export skipped: {exc}")

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "device": str(device),
            "num_samples": n,
            "epochs": len(history),
            "loss_start": history[0]["train_loss"],
            "loss_end": history[-1]["train_loss"],
            "sanity_accuracy_capped": acc,
            "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
            "run_dir": str(run_dir),
            "files": sorted(p.name for p in run_dir.iterdir()),
        }
        write_summary_json(run_dir, summary)
        print("\n" + json.dumps(summary, indent=2))
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
