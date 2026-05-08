"""DGL Cora manual validation.

Usage::

    python examples/public_datasets/dgl_cora_smoke.py --download --epochs 3
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
        "dgl_cora_smoke",
        "Manual DGL/Cora validation via tgraphx.datasets.DGLCitationDatasetAdapter.",
    )
    args = parser.parse_args(argv)

    try:
        import dgl  # noqa: F401
    except ImportError:
        return soft_skip("dgl not installed", args.strict)

    if not args.download:
        print("This script requires --download.")
        return 0

    import torch
    import tgraphx
    from tgraphx import build_model
    from tgraphx.datasets import DGLCitationDatasetAdapter
    from tgraphx.experiments.callbacks import CSVLoggerCallback, RunState
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
        ds = DGLCitationDatasetAdapter(name="cora", root=args.root, download=True)
        g = ds[0].to(device)

        # DGL stores masks on the upstream graph as ndata.
        upstream_g = ds._upstream[0]  # noqa: SLF001
        if isinstance(upstream_g, tuple):
            upstream_g = upstream_g[0]
        train_mask = upstream_g.ndata.get("train_mask")
        val_mask = upstream_g.ndata.get("val_mask")
        test_mask = upstream_g.ndata.get("test_mask")
        if train_mask is None or val_mask is None or test_mask is None:
            return soft_skip(
                "DGL Cora upstream graph has no train/val/test masks; cannot validate",
                args.strict,
            )
        train_mask = train_mask.to(device).bool()
        val_mask = val_mask.to(device).bool()
        test_mask = test_mask.to(device).bool()

        write_run_provenance(
            run_dir, run_name="dgl_cora_smoke", device=str(device),
            task="node_classification", seed=args.seed,
            dataset="dgl:cora",
        )
        write_dataset_metadata(
            str(run_dir / "dataset_metadata.json"),
            **{**ds.metadata.to_dict(),
               "num_classes": int(g.node_labels.max().item()) + 1,
               "validation_note": "manual_run"},
        )

        num_classes = int(g.node_labels.max().item()) + 1
        model = build_model(
            task="node_classification", layer="linear",
            in_shape=(g.node_features.size(1),), hidden_shape=(32,),
            num_layers=2, num_classes=num_classes,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)

        state = RunState(run_dir=run_dir)
        csv_cb = CSVLoggerCallback()
        history = []
        for epoch in range(args.epochs):
            model.train()
            opt.zero_grad()
            logits = model(g.node_features, g.edge_index)
            loss = torch.nn.functional.cross_entropy(
                logits[train_mask], g.node_labels[train_mask].long(),
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                val_acc = float((preds[val_mask] == g.node_labels[val_mask]).float().mean())
            metrics = {"train_loss": float(loss.item()), "val_accuracy": val_acc}
            csv_cb.on_epoch_end(state, epoch, metrics)
            history.append({"epoch": epoch, **metrics})
            print(f"  epoch {epoch}  train_loss={metrics['train_loss']:.4f}  "
                  f"val_acc={val_acc:.3f}")

        with torch.no_grad():
            preds = model(g.node_features, g.edge_index).argmax(dim=-1)
            test_acc = float((preds[test_mask] == g.node_labels[test_mask]).float().mean())

        write_metrics_summary(
            str(run_dir / "metrics_summary.json"),
            final_train_loss=history[-1]["train_loss"],
            final_val_accuracy=history[-1]["val_accuracy"],
            test_accuracy_sanity=test_acc,
            note="Manual DGL Cora validation; not a benchmark / leaderboard result.",
        )
        mark_run_completed(run_dir, total_epochs=len(history),
                           final_train_loss=history[-1]["train_loss"])

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "device": str(device),
            "num_nodes": g.num_nodes,
            "epochs": len(history),
            "loss_start": history[0]["train_loss"],
            "loss_end": history[-1]["train_loss"],
            "test_accuracy_sanity": test_acc,
            "loss_decreased": history[-1]["train_loss"] < history[0]["train_loss"],
            "run_dir": str(run_dir),
        }
        write_summary_json(run_dir, summary)
        print("\n" + json.dumps(summary, indent=2))
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
