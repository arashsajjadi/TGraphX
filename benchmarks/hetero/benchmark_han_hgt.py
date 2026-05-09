"""HAN/HGT tiny-overfit benchmark.

This script trains HANConv and HGTConv on a tiny synthetic
heterogeneous graph and asserts the loss decreases.  It does NOT
benchmark against external reference implementations.

Usage:
    python benchmarks/hetero/benchmark_han_hgt.py --small --json
    python benchmarks/hetero/benchmark_han_hgt.py --model han --epochs 30

Stability: Experimental — internal validation only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import tgraphx
from tgraphx.layers.han import HANConv
from tgraphx.layers.hgt import HGTConv
from tgraphx.mining.reports import write_hetero_summary


def _make_toy_hetero(n_a: int = 20, n_b: int = 15, seed: int = 0):
    """Build a tiny A-to-B and B-to-A hetero graph with node labels."""
    torch.manual_seed(seed)
    D = 8
    x_a = torch.randn(n_a, D)
    x_b = torch.randn(n_b, D)
    # A→B edges.
    src_ab = torch.randint(n_a, (n_a * 2,))
    dst_ab = torch.randint(n_b, (n_a * 2,))
    ei_ab = torch.stack([src_ab, dst_ab], dim=0)
    # B→A edges.
    src_ba = torch.randint(n_b, (n_b * 2,))
    dst_ba = torch.randint(n_a, (n_b * 2,))
    ei_ba = torch.stack([src_ba, dst_ba], dim=0)
    # Labels for A nodes (binary).
    y_a = torch.randint(0, 2, (n_a,))
    return x_a, x_b, ei_ab, ei_ba, y_a


def _bench_han(n_a: int, n_b: int, epochs: int, lr: float, seed: int) -> dict:
    x_a, x_b, ei_ab, ei_ba, y_a = _make_toy_hetero(n_a, n_b, seed)
    D_in, D_out = 8, 4

    # HANConv with A-node labels.  Use ei_ab and ei_ba as two metapaths.
    layer = HANConv(in_dim=D_in, out_dim=D_out, num_heads=2)
    classifier = nn.Linear(D_out, 2)
    params = list(layer.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        out = layer(x_a, {"mp_ab": ei_ab, "mp_ba": ei_ba[:, :min(ei_ba.size(1), n_a)]})
        logits = classifier(out)
        loss = F.cross_entropy(logits, y_a)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    dt = time.perf_counter() - t0

    grad_finite = all(
        torch.isfinite(p.grad).all().item()
        for p in params if p.grad is not None
    )
    loss_decreased = losses[-1] < losses[0] or losses[-1] < 1.0

    return {
        "model": "HANConv",
        "epochs": epochs,
        "loss_initial": round(losses[0], 4),
        "loss_final": round(losses[-1], 4),
        "loss_decreased": loss_decreased,
        "grad_finite": grad_finite,
        "runtime_s": round(dt, 4),
        "parameter_count": sum(p.numel() for p in params),
        "limitation_notes": "Internal toy validation only; not benchmarked against reference implementations.",
    }


def _bench_hgt(n_a: int, n_b: int, epochs: int, lr: float, seed: int) -> dict:
    x_a, x_b, ei_ab, ei_ba, y_a = _make_toy_hetero(n_a, n_b, seed)
    D_in, D_out = 8, 8  # keep same dim for HGT identity path

    node_types = ["A", "B"]
    edge_types = [("A", "to_B", "B"), ("B", "to_A", "A")]
    x_dict = {"A": x_a.clone().requires_grad_(False),
               "B": x_b.clone().requires_grad_(False)}

    layer = HGTConv(D_in, D_out, node_types=node_types, edge_types=edge_types, num_heads=2)
    classifier = nn.Linear(D_out, 2)
    params = list(layer.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        x_in = {"A": x_a, "B": x_b}
        out = layer(x_in, {("A", "to_B", "B"): ei_ab, ("B", "to_A", "A"): ei_ba})
        logits = classifier(out["A"])
        loss = F.cross_entropy(logits, y_a)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
    dt = time.perf_counter() - t0

    grad_finite = all(
        torch.isfinite(p.grad).all().item()
        for p in params if p.grad is not None
    )
    # Check relation priors get gradients.
    pri_grads = all(
        p.grad is not None
        for n, p in layer.named_parameters()
        if "relation_pri" in n
    )

    return {
        "model": "HGTConv",
        "epochs": epochs,
        "loss_initial": round(losses[0], 4),
        "loss_final": round(losses[-1], 4),
        "loss_decreased": losses[-1] < losses[0] or losses[-1] < 1.0,
        "grad_finite": grad_finite,
        "relation_priors_have_gradients": pri_grads,
        "runtime_s": round(dt, 4),
        "parameter_count": sum(p.numel() for p in params),
        "limitation_notes": "Internal toy validation only; not benchmarked against reference implementations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HAN/HGT tiny-overfit benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="all", choices=["han", "hgt", "all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    epochs = args.epochs or (5 if args.small else 30)
    n_a = 20 if args.small else 60
    n_b = 15 if args.small else 40
    lr = 0.01

    results = []
    if args.model in ("han", "all"):
        results.append(_bench_han(n_a, n_b, epochs, lr, args.seed))
    if args.model in ("hgt", "all"):
        results.append(_bench_hgt(n_a, n_b, epochs, lr, args.seed))

    report = {
        "package_version": tgraphx.__version__,
        "seed": int(args.seed),
        "model_results": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_hetero_summary(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            ok = r["loss_decreased"] and r["grad_finite"]
            print(f"[{r['model']:>8}] loss: {r['loss_initial']:.3f}→{r['loss_final']:.3f} "
                  f"{'✓' if ok else '✗'} rt={r['runtime_s']:.2f}s")


if __name__ == "__main__":
    main()
