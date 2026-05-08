"""PyG Cora node-classification benchmark for TGraphX (v0.3.2 foundation).

Behaviour
---------
- Skips cleanly (exit 0) when ``torch_geometric`` is not installed,
  unless ``--strict`` is passed.
- Without ``--download``: prints an actionable instruction to use
  ``--download``.
- With ``--download``: PyG downloads the Planetoid Cora split into
  ``--root``, the dataset is converted to a TGraphX :class:`Graph`,
  a tiny `LinearMessagePassing` model is trained for ``--epochs``
  epochs on the official train mask, and val/test accuracies are
  reported.

This is a benchmark **smoke**, not a leaderboard run.
"""
from __future__ import annotations

import argparse
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


def _accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=1)
    return float((pred == target[mask]).float().mean().detach().item())


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = make_parser(
        prog="pyg_cora_benchmark",
        description="PyG Cora (Planetoid) node-classification smoke benchmark.",
    )
    args = parser.parse_args(argv)

    seed = int(args.seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)

    # ── Optional dependency check ────────────────────────────────────────
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return soft_skip(
            "torch_geometric is not installed.  "
            "Install with `pip install \"tgraphx[pyg]\"`.",
            strict=args.strict,
        )

    if not args.download:
        return soft_skip(
            "Pass --download to fetch Cora through PyG.  TGraphX never "
            "bundles datasets.",
            strict=args.strict,
        )

    # ── Build dataset via the existing TGraphX adapter ──────────────────
    try:
        from tgraphx.datasets import PyGPlanetoidDataset
    except ImportError as exc:  # pragma: no cover  (PyG installed but adapter missing)
        return soft_skip(f"PyG adapter unavailable: {exc}", strict=args.strict)

    try:
        ds = PyGPlanetoidDataset(
            root=args.root,
            name="Cora",
            download=True,
        )
    except Exception as exc:  # pragma: no cover  (network-only failures)
        if args.strict:
            raise
        return soft_skip(
            f"Could not load Cora (use --strict to fail instead): {exc}",
            strict=args.strict,
        )

    # The Planetoid adapter returns a single :class:`Graph` with masks.
    g = ds[0].to(device)
    masks = g.metadata.get("masks", {}) if g.metadata else {}
    train_mask = masks.get("train_mask")
    val_mask = masks.get("val_mask")
    test_mask = masks.get("test_mask")
    if train_mask is None:
        return soft_skip(
            "Cora returned no train_mask metadata; cannot run smoke benchmark.",
            strict=args.strict,
        )
    train_mask = train_mask.to(device)
    if val_mask is not None:
        val_mask = val_mask.to(device)
    if test_mask is not None:
        test_mask = test_mask.to(device)

    num_classes = int(g.node_labels.max().item()) + 1
    in_dim = int(g.node_features.size(1))

    # Cap node count for very large graphs (Cora is small, but safety).
    if g.num_nodes > int(args.max_nodes):
        return soft_skip(
            f"Graph has {g.num_nodes} nodes > --max-nodes={args.max_nodes}; "
            "raise --max-nodes to allow this.",
            strict=args.strict,
        )

    # ── Build a tiny vector model ────────────────────────────────────────
    from tgraphx import GCNConv

    class _TwoLayerGCN(torch.nn.Module):
        def __init__(self, in_dim: int, hidden: int, out_dim: int):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden, out_dim, add_self_loops=True, normalize=True)

        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = _TwoLayerGCN(in_dim, hidden=16, out_dim=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # ── Train ────────────────────────────────────────────────────────────
    losses: list = []
    accuracies: list = []
    t0 = time.perf_counter()
    targets = g.node_labels.long()
    for _ in range(int(args.epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(g.node_features, g.edge_index)
        loss = F.cross_entropy(logits[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
        with torch.no_grad():
            accuracies.append(_accuracy(logits, targets, train_mask))
    elapsed = time.perf_counter() - t0

    # ── Evaluate ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(g.node_features, g.edge_index)
        train_acc = _accuracy(logits, targets, train_mask)
        val_acc = _accuracy(logits, targets, val_mask) if val_mask is not None else None
        test_acc = _accuracy(logits, targets, test_mask) if test_mask is not None else None

    # ── Compose artefacts ────────────────────────────────────────────────
    env = env_block(seed=seed, device=device)
    bench = {
        "benchmark": "pyg_cora_benchmark",
        "data_source": "torch_geometric.datasets.Planetoid:Cora",
        "elapsed_s": elapsed,
        "epochs": int(args.epochs),
        "num_nodes": int(g.num_nodes),
        "num_edges": int(g.edge_index.size(1)),
        "num_classes": num_classes,
        "loss_start": float(losses[0]) if losses else None,
        "loss_end": float(losses[-1]) if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        **env,
    }
    run_meta = {
        "run_name": bench["benchmark"],
        "status": "completed",
        "task": "node_classification",
        **env,
    }
    ds_meta = {
        "name": "planetoid_cora",
        "task": "node_classification",
        "graph_type": "homogeneous",
        "upstream_library": "torch_geometric",
        "num_graphs": 1,
        "num_nodes": int(g.num_nodes),
        "num_edges": int(g.edge_index.size(1)),
        "num_classes": num_classes,
        "license": "See PyG Planetoid attribution",
        "tgraphx_redistributes": False,
    }
    metrics = {
        "final_train_loss": float(losses[-1]) if losses else None,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
        "final_test_accuracy": test_acc,
        "loss_decreased": bool(losses and losses[-1] < losses[0]),
    }

    if args.output_dir is None:
        out_dir_ctx = tempfile.TemporaryDirectory(prefix="tgraphx_cora_bench_")
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
        print(json.dumps(
            {"artefacts": {k: str(v) for k, v in artefacts.items()},
             "summary": bench},
            indent=2, default=str,
        ))
    else:
        print(f"[pyg_cora_benchmark] nodes={bench['num_nodes']} "
              f"edges={bench['num_edges']} classes={bench['num_classes']} "
              f"train_acc={train_acc:.3f} val_acc={val_acc} "
              f"test_acc={test_acc} elapsed={elapsed:.3f}s")
        print(f"[pyg_cora_benchmark] artefacts in {out_dir}")

    if out_dir_ctx is not None:
        out_dir_ctx.cleanup()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
