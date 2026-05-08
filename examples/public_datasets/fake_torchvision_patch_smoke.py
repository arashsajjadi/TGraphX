"""FakeData → TGraphX patch graph smoke validation.

This script is **CI-safe**: ``torchvision.datasets.FakeData`` synthesises
images in memory, so no network is touched.  It exercises the full
torchvision adapter path, the dataset → graph conversion, a tiny
training loop, and dashboard artefact writing.

Usage::

    python examples/public_datasets/fake_torchvision_patch_smoke.py --epochs 2
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Allow running the script directly (without -m).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    make_parser, mark_run_completed, resolve_device, soft_skip,
    write_run_provenance, write_summary_json,
)


def main(argv=None) -> int:
    parser = make_parser(
        "fake_torchvision_patch_smoke",
        "TGraphX validation against torchvision.datasets.FakeData (no network).",
    )
    args = parser.parse_args(argv)

    try:
        import torchvision  # noqa: F401
    except ImportError:
        return soft_skip("torchvision not installed", args.strict)

    import torch
    import torch.nn as nn
    import tgraphx
    from tgraphx import GraphBatch, build_model
    from tgraphx.datasets import FakeDataPatchGraphDataset
    from tgraphx.tracking import (
        write_dataset_metadata, write_metrics_summary,
    )
    from tgraphx.experiments.callbacks import CSVLoggerCallback, RunState

    torch.manual_seed(args.seed)

    using_temp = args.output_run_dir is None
    tmp_ctx: tempfile._TemporaryFileWrapper | None = None
    if using_temp:
        tmp = tempfile.TemporaryDirectory()
        run_dir = Path(tmp.name)
        tmp_ctx = tmp
    else:
        run_dir = Path(args.output_run_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = resolve_device(args.device)
        n = max(2, min(args.max_samples, 32))

        ds = FakeDataPatchGraphDataset(
            root=str(run_dir / "fakedata_cache"),
            upstream_kwargs={
                "size": n,
                "image_size": (3, 16, 16),
                "num_classes": 4,
            },
            patch_size=4,
        )
        items = [ds[i] for i in range(len(ds))]
        batch = GraphBatch(items).to(device)
        in_shape = tuple(batch.node_features.shape[1:])

        write_run_provenance(
            run_dir,
            run_name="fake_torchvision_patch_smoke",
            device=str(device), task="graph_classification",
            seed=args.seed,
        )
        write_dataset_metadata(
            str(run_dir / "dataset_metadata.json"),
            **ds.metadata.to_dict(),
        )

        model = build_model(
            task="graph_classification", layer="conv",
            in_shape=in_shape, hidden_shape=(8, in_shape[1], in_shape[2]),
            num_layers=2, num_classes=4, pooling="mean",
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

        # Sanity accuracy on the same batch (this is a smoke test, NOT a benchmark).
        model.eval()
        with torch.no_grad():
            preds = model(batch.node_features, batch.edge_index,
                          batch=batch.batch).argmax(dim=-1)
            acc = float((preds == batch.graph_labels.long()).float().mean())

        write_metrics_summary(
            str(run_dir / "metrics_summary.json"),
            final_train_loss=history[-1]["train_loss"],
            initial_train_loss=history[0]["train_loss"],
            sanity_accuracy=acc,
            note="Synthetic FakeData; not a benchmark.",
        )
        mark_run_completed(run_dir, total_epochs=len(history),
                           final_train_loss=history[-1]["train_loss"])

        # Dashboard offline export (optional).
        try:
            from tgraphx.dashboard.app import export_dashboard_html
            export_dashboard_html(str(run_dir), str(run_dir / "snapshot.html"))
        except Exception as exc:  # pragma: no cover  (best-effort export)
            print(f"  dashboard export skipped: {exc}")

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "device": str(device),
            "epochs": len(history),
            "loss_start": history[0]["train_loss"],
            "loss_end": history[-1]["train_loss"],
            "sanity_accuracy": acc,
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
