"""MNIST patch-graph benchmark for TGraphX (v0.3.2 foundation).

Behaviour
---------
- ``--download`` not set (default): uses
  :class:`~tgraphx.datasets.FakeDataPatchGraphDataset` — pure synthetic
  random images, no network traffic.  Useful as a CI-safe smoke.
- ``--download`` set: uses
  :class:`~tgraphx.datasets.MNISTPatchGraphDataset` — torchvision
  downloads MNIST into ``--root`` (or the default TGraphX cache).

Both paths convert images to patch graphs, train a tiny TGraphX model
for ``--epochs`` epochs, and report training-loss start/end, training
time, accuracy on the same data (overfit-style smoke), and the standard
dashboard artefacts.

This is a benchmark **smoke**, not a leaderboard run.  It does not
publish accuracy numbers as quality claims.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

# Allow running the script directly (without -m); Python will then import
# ``_common`` from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # type: ignore  # noqa: E402
    env_block,
    make_parser,
    resolve_device,
    soft_skip,
    write_artefacts,
)


def _build_dataset(
    *,
    use_real_mnist: bool,
    root: str | None,
    max_samples: int,
    seed: int,
):
    """Return a TGraphX patch-graph dataset (FakeData or MNIST).

    FakeData generates random 3-channel images in memory (no network);
    MNIST is the real torchvision MNIST.  The downstream model is built
    from the resulting node-feature shape, so channel mismatch between
    the two paths is fine.
    """
    if use_real_mnist:
        from tgraphx.datasets import MNISTPatchGraphDataset
        ds = MNISTPatchGraphDataset(
            root=root,
            train=True,
            download=True,
            patch_size=7,
            graph_builder="grid",
        )
        # Cap the number of samples by trimming the underlying dataset.
        n = min(max_samples, len(ds))

        class _Capped:
            def __init__(self, base, n):
                self._base = base
                self._n = n
                self.metadata = base.metadata

            def __len__(self):
                return self._n

            def __getitem__(self, i):
                return self._base[i]

        return _Capped(ds, n)

    from tgraphx.datasets import FakeDataPatchGraphDataset
    return FakeDataPatchGraphDataset(
        upstream_kwargs={
            "size": int(max_samples),
            "image_size": (3, 28, 28),
            "num_classes": 10,
            "random_offset": int(seed),
        },
        patch_size=7,
        graph_builder="grid",
    )


def _build_model(in_shape: Tuple[int, ...], hidden: int, num_classes: int) -> torch.nn.Module:
    from tgraphx import build_model
    return build_model(
        task="graph_classification",
        layer="conv",
        in_shape=in_shape,
        hidden_shape=(hidden, in_shape[-2], in_shape[-1]),
        num_layers=2,
        num_classes=num_classes,
        pooling="mean",
    )


def _train_smoke(
    model, batch, *, epochs: int, lr: float, device: torch.device,
):
    """Train for ``epochs`` steps and return (start_loss, end_loss, accuracy)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list = []
    accuracies: list = []
    targets = batch.graph_labels.long().view(-1)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(
            batch.node_features, batch.edge_index, batch=batch.batch,
        )
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().detach().item()
            accuracies.append(acc)
    return losses[0], losses[-1], accuracies[-1]


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = make_parser(
        prog="mnist_patch_benchmark",
        description="MNIST patch-graph training smoke benchmark.",
    )
    args = parser.parse_args(argv)

    seed = int(args.seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)

    # ── Build dataset ────────────────────────────────────────────────────
    use_real_mnist = bool(args.download)
    try:
        ds = _build_dataset(
            use_real_mnist=use_real_mnist,
            root=args.root,
            max_samples=int(args.max_samples),
            seed=seed,
        )
    except ImportError as exc:
        return soft_skip(
            f"Optional dependency missing for MNIST patch benchmark: {exc}",
            strict=args.strict,
        )
    except Exception as exc:  # pragma: no cover  (real-MNIST IO failures only)
        if args.strict:
            raise
        return soft_skip(
            f"Could not build dataset (use --strict to fail instead): {exc}",
            strict=args.strict,
        )

    # ── Build batch ───────────────────────────────────────────────────────
    from tgraphx import GraphBatch
    items = [ds[i] for i in range(len(ds))]
    if not items:
        print("[skip] dataset produced 0 items")
        return 0
    batch = GraphBatch(items).to(device)
    in_shape = tuple(batch.node_features.shape[1:])

    # ── Build + train model ──────────────────────────────────────────────
    model = _build_model(
        in_shape=in_shape, hidden=8, num_classes=10,
    ).to(device)
    t0 = time.perf_counter()
    start_loss, end_loss, final_acc = _train_smoke(
        model, batch, epochs=int(args.epochs), lr=5e-3, device=device,
    )
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())

    # ── Compose artefacts ────────────────────────────────────────────────
    env = env_block(seed=seed, device=device)
    bench = {
        "benchmark": "mnist_patch_benchmark",
        "data_source": "torchvision_mnist" if use_real_mnist else "fake_data_synthetic",
        "elapsed_s": elapsed,
        "epochs": int(args.epochs),
        "num_graphs": len(items),
        "num_nodes": int(batch.node_features.size(0)),
        "num_edges": int(batch.edge_index.size(1)),
        "node_feature_shape": list(in_shape),
        "model_param_count": int(n_params),
        "loss_start": float(start_loss),
        "loss_end": float(end_loss),
        "loss_decreased": bool(end_loss < start_loss),
        "final_accuracy": float(final_acc),
        **env,
    }
    run_meta = {
        "run_name": bench["benchmark"],
        "status": "completed",
        "task": "graph_classification",
        **env,
    }
    ds_meta = {
        "name": "mnist_patch" if use_real_mnist else "fake_data_synthetic_28x28",
        "task": "graph_classification",
        "graph_type": "patch_grid",
        "upstream_library": "torchvision" if use_real_mnist else "tgraphx (synthetic)",
        "num_graphs": len(items),
        "num_classes": 10,
        "license": "Mixed (Yann LeCun / Sun)" if use_real_mnist else "synthetic",
        "tgraphx_redistributes": False,
    }
    metrics = {
        "final_train_loss": float(end_loss),
        "final_train_accuracy": float(final_acc),
        "loss_decreased": bool(end_loss < start_loss),
    }

    # ── Output artefacts ────────────────────────────────────────────────
    if args.output_dir is None:
        out_dir_ctx = tempfile.TemporaryDirectory(prefix="tgraphx_mnist_bench_")
        out_dir = Path(out_dir_ctx.name)
    else:
        out_dir_ctx = None
        out_dir = Path(args.output_dir).expanduser()

    artefacts = write_artefacts(
        out_dir,
        benchmark=bench,
        run_metadata=run_meta,
        dataset_metadata=ds_meta,
        metrics_summary=metrics,
    )

    if args.json:
        print(json.dumps({"artefacts": {k: str(v) for k, v in artefacts.items()},
                          "summary": bench}, indent=2, default=str))
    else:
        print(f"[mnist_patch_benchmark] data={bench['data_source']} "
              f"graphs={bench['num_graphs']} nodes={bench['num_nodes']} "
              f"loss {bench['loss_start']:.4f}->{bench['loss_end']:.4f} "
              f"acc={bench['final_accuracy']:.3f} "
              f"elapsed={bench['elapsed_s']:.3f}s")
        print(f"[mnist_patch_benchmark] artefacts in {out_dir}")

    if out_dir_ctx is not None:
        # Keep artefacts only when --output-dir was explicit.
        out_dir_ctx.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
