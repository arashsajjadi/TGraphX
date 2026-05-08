"""FashionMNIST patch-graph benchmark for TGraphX (v0.3.2 foundation).

Same structure as ``mnist_patch_benchmark.py`` but targets
``torchvision.datasets.FashionMNIST`` (10 clothing-category classes).

Default mode uses :class:`~tgraphx.datasets.FakeDataPatchGraphDataset`
(no network).  Real FashionMNIST requires ``--download``.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # type: ignore  # noqa: E402
    env_block,
    make_parser,
    resolve_device,
    soft_skip,
    write_artefacts,
)


def _build_dataset(*, use_real: bool, root, max_samples: int, seed: int):
    if use_real:
        from tgraphx.datasets import FashionMNISTPatchGraphDataset
        return FashionMNISTPatchGraphDataset(
            root=root, train=True, download=True, patch_size=7, graph_builder="grid",
        )
    from tgraphx.datasets import FakeDataPatchGraphDataset
    return FakeDataPatchGraphDataset(
        upstream_kwargs={"size": int(max_samples), "image_size": (3, 28, 28),
                         "num_classes": 10, "random_offset": int(seed)},
        patch_size=7, graph_builder="grid",
    )


def main(argv=None) -> int:
    parser = make_parser(
        "fashionmnist_patch_benchmark",
        "FashionMNIST patch-graph training smoke benchmark.",
    )
    args = parser.parse_args(argv)
    seed = int(args.seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)

    try:
        ds = _build_dataset(
            use_real=bool(args.download),
            root=args.root,
            max_samples=int(args.max_samples),
            seed=seed,
        )
    except ImportError as exc:
        return soft_skip(f"Optional dependency missing: {exc}", strict=args.strict)
    except Exception as exc:
        if args.strict:
            raise
        return soft_skip(f"Could not build dataset: {exc}", strict=args.strict)

    from tgraphx import GraphBatch, build_model
    n = min(int(args.max_samples), len(ds))
    items = [ds[i] for i in range(n)]
    if not items:
        print("[skip] dataset produced 0 items")
        return 0
    batch = GraphBatch(items).to(device)
    in_shape = tuple(batch.node_features.shape[1:])

    model = build_model(
        task="graph_classification", layer="conv",
        in_shape=in_shape,
        hidden_shape=(8, in_shape[-2], in_shape[-1]),
        num_layers=2, num_classes=10, pooling="mean",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    targets = batch.graph_labels.long().view(-1)

    losses = []
    t0 = time.perf_counter()
    for _ in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(batch.node_features, batch.edge_index, batch=batch.batch)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    elapsed = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        final_acc = float(
            (model(batch.node_features, batch.edge_index,
                   batch=batch.batch).argmax(1) == targets).float().mean().item()
        )

    env = env_block(seed=seed, device=device)
    bench = {
        "benchmark": "fashionmnist_patch_benchmark",
        "data_source": "torchvision_fashionmnist" if args.download else "fake_data_synthetic",
        "elapsed_s": elapsed,
        "epochs": int(args.epochs),
        "num_graphs": len(items),
        "num_nodes": int(batch.node_features.size(0)),
        "num_edges": int(batch.edge_index.size(1)),
        "node_feature_shape": list(in_shape),
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "final_accuracy": final_acc,
        **env,
    }
    run_meta = {"run_name": bench["benchmark"], "status": "completed",
                "task": "graph_classification", **env}
    ds_meta = {
        "name": "fashionmnist_patch" if args.download else "fake_data_synthetic_28x28",
        "task": "graph_classification", "graph_type": "patch_grid",
        "upstream_library": "torchvision" if args.download else "tgraphx (synthetic)",
        "num_graphs": len(items), "num_classes": 10, "tgraphx_redistributes": False,
    }
    metrics = {
        "final_train_loss": float(losses[-1]) if losses else None,
        "final_train_accuracy": final_acc,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
    }

    if args.output_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix="tgraphx_fashion_bench_")
        out_dir = Path(ctx.name)
    else:
        ctx = None
        out_dir = Path(args.output_dir).expanduser()

    artefacts = write_artefacts(
        out_dir, benchmark=bench, run_metadata=run_meta,
        dataset_metadata=ds_meta, metrics_summary=metrics,
    )

    if args.json:
        print(json.dumps(
            {"artefacts": {k: str(v) for k, v in artefacts.items()},
             "summary": bench},
            indent=2, default=str,
        ))
    else:
        print(f"[fashionmnist_patch_benchmark] data={bench['data_source']} "
              f"graphs={bench['num_graphs']} "
              f"loss {bench['loss_start']:.4f}->{bench['loss_end']:.4f} "
              f"acc={bench['final_accuracy']:.3f} elapsed={elapsed:.3f}s")
        print(f"[fashionmnist_patch_benchmark] artefacts in {out_dir}")

    if ctx is not None:
        ctx.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
