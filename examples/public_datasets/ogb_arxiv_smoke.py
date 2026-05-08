"""OGB ogbn-arxiv manual validation.

By default this only loads the dataset, exposes the official
``get_idx_split`` and evaluator, and reports shapes.  Pass
``--train-tiny`` to additionally train a small model on a capped
subset of nodes.

Usage::

    python examples/public_datasets/ogb_arxiv_smoke.py --download
    python examples/public_datasets/ogb_arxiv_smoke.py --download --train-tiny --max-nodes 5000
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
        "ogb_arxiv_smoke",
        "Manual OGB ogbn-arxiv validation via tgraphx.datasets.OGBDatasetAdapter.",
    )
    parser.add_argument(
        "--train-tiny", action="store_true",
        help="Optional capped training pass on a node subset.",
    )
    args = parser.parse_args(argv)

    try:
        import ogb  # noqa: F401
    except ImportError:
        return soft_skip("ogb not installed", args.strict)

    if not args.download:
        print("This script requires --download.")
        return 0

    import torch
    import tgraphx
    from tgraphx.datasets import OGBDatasetAdapter
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
        ds = OGBDatasetAdapter(name="ogbn-arxiv", root=args.root)
        g = ds[0]
        split_idx = ds.get_idx_split()
        evaluator = ds.get_evaluator()

        write_run_provenance(
            run_dir, run_name="ogb_arxiv_smoke", device=str(device),
            task="node_classification", seed=args.seed,
            dataset="ogb:ogbn-arxiv",
        )
        write_dataset_metadata(
            str(run_dir / "dataset_metadata.json"),
            **{**ds.metadata.to_dict(),
               "validation_note": "manual_run; load+split+evaluator smoke"},
        )

        summary = {
            "tgraphx_version": tgraphx.__version__,
            "num_nodes": int(g.num_nodes),
            "num_edges": int(g.num_edges),
            "num_classes": int(g.node_labels.max().item()) + 1,
            "split_idx_keys": sorted(list(split_idx.keys())),
            "split_train_size": int(len(split_idx["train"])),
            "split_valid_size": int(len(split_idx["valid"])),
            "split_test_size": int(len(split_idx["test"])),
            "has_evaluator": evaluator is not None,
        }

        if args.train_tiny:
            cap = min(args.max_nodes, g.num_nodes)
            sub_x = g.node_features[:cap].to(device)
            sub_labels = g.node_labels[:cap].to(device)
            ei = g.edge_index
            mask = (ei[0] < cap) & (ei[1] < cap)
            sub_ei = ei[:, mask].to(device)
            from tgraphx import build_model
            num_classes = summary["num_classes"]
            model = build_model(
                task="node_classification", layer="linear",
                in_shape=(sub_x.size(1),), hidden_shape=(64,),
                num_layers=2, num_classes=num_classes,
            ).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=5e-3)
            losses = []
            for epoch in range(args.epochs):
                model.train()
                opt.zero_grad()
                logits = model(sub_x, sub_ei)
                loss = torch.nn.functional.cross_entropy(logits, sub_labels.long().view(-1))
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                print(f"  epoch {epoch}  train_loss={losses[-1]:.4f}")
            summary["epochs"] = len(losses)
            summary["loss_start"] = losses[0]
            summary["loss_end"] = losses[-1]
            summary["loss_decreased"] = losses[-1] < losses[0]

        write_metrics_summary(
            str(run_dir / "metrics_summary.json"),
            **{k: v for k, v in summary.items() if k != "split_idx_keys"},
            note="Manual OGB validation; not a benchmark / leaderboard result.",
        )
        mark_run_completed(run_dir, **summary)
        write_summary_json(run_dir, summary)
        print("\n" + json.dumps(summary, indent=2))
        return 0
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
