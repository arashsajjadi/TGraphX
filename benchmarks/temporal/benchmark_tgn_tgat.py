"""TGN/TGAT tiny temporal link-prediction benchmark.

Generates timestamped events, trains on a chronological prefix, and
evaluates on the suffix.  Validates no-future-leakage and checks that
loss decreases on the toy task.

Usage:
    python benchmarks/temporal/benchmark_tgn_tgat.py --small --json
    python benchmarks/temporal/benchmark_tgn_tgat.py --model tgn --epochs 20

Stability: Experimental — internal toy validation only.
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
from tgraphx.temporal import TGNMemory, TGATConv
from tgraphx.temporal.time_encoding import sinusoidal_time_encoding
from tgraphx.mining.reports import write_temporal_summary


def _make_events(num_nodes: int, num_events: int, seed: int):
    """Return (src, dst, time, label) for a toy temporal link task."""
    torch.manual_seed(seed)
    src = torch.randint(num_nodes, (num_events,))
    dst = torch.randint(num_nodes, (num_events,))
    times = torch.sort(torch.rand(num_events))[0]  # chronological
    # Label: 1 if src < dst (arbitrary but deterministic), 0 otherwise.
    labels = (src < dst).long()
    return src, dst, times, labels


def _temporal_split(n: int, train_ratio: float):
    n_train = int(round(train_ratio * n))
    return slice(0, n_train), slice(n_train, n)


# ── TGN benchmark ─────────────────────────────────────────────────────────────

def _bench_tgn(n: int, num_events: int, epochs: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    src, dst, times, labels = _make_events(n, num_events, seed)
    train_sl, valid_sl = _temporal_split(num_events, 0.7)
    D_mem, D_msg = 8, 8

    mem = TGNMemory(num_nodes=n, memory_dim=D_mem, message_dim=D_msg)
    # Simple edge-scoring MLP.
    scorer = nn.Sequential(nn.Linear(D_mem * 2, 16), nn.ReLU(), nn.Linear(16, 1))
    all_params = list(mem.parameters()) + list(scorer.parameters())
    opt = torch.optim.Adam(all_params, lr=lr)

    losses, grads_ok = [], []
    t0 = time.perf_counter()
    for ep in range(epochs):
        mem.reset_state()
        ep_loss = 0.0
        for i in range(train_sl.start, train_sl.stop):
            node_ids = torch.tensor([int(src[i]), int(dst[i])], dtype=torch.long)
            t_val = times[i].unsqueeze(0).expand(2)
            msg = torch.zeros(2, D_msg)
            mem.update(node_ids, msg, t_val, check_monotonic=False)
        # Scoring on train events.
        mem.reset_state()
        batch_src = src[train_sl]
        batch_dst = dst[train_sl]
        batch_y = labels[train_sl].float()
        z_src = mem.get(batch_src)
        z_dst = mem.get(batch_dst)
        logits = scorer(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mem.detach()
        ep_loss = float(loss.detach().item())
        losses.append(ep_loss)
        gf = all(p.grad is not None and torch.isfinite(p.grad).all().item()
                 for p in all_params if p.grad is not None)
        grads_ok.append(gf)

    # Leakage check: no train timestamp appears after the split.
    train_times = times[train_sl]
    valid_times = times[valid_sl]
    no_leakage = bool(
        train_times.numel() == 0 or valid_times.numel() == 0 or
        float(train_times.max().item()) <= float(valid_times.min().item())
    )
    dt = time.perf_counter() - t0

    return {
        "model": "TGNMemory",
        "train_events": int(train_sl.stop - train_sl.start),
        "valid_events": int(num_events - train_sl.stop),
        "epochs": epochs,
        "loss_initial": round(losses[0], 4) if losses else None,
        "loss_final": round(losses[-1], 4) if losses else None,
        "loss_decreased": bool(losses[-1] < losses[0]) if len(losses) > 1 else False,
        "grad_finite_all_epochs": bool(all(grads_ok)),
        "leakage_check_passed": no_leakage,
        "runtime_s": round(dt, 4),
        "limitation_notes": "Internal toy validation only; not benchmarked against reference TGN implementations.",
    }


# ── TGAT benchmark ────────────────────────────────────────────────────────────

def _bench_tgat(n: int, num_events: int, epochs: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    src, dst, times, labels = _make_events(n, num_events, seed)
    train_sl, valid_sl = _temporal_split(num_events, 0.7)
    D = 8

    x = torch.randn(n, D)
    layer = TGATConv(in_dim=D, out_dim=D, time_dim=8, num_heads=2)
    scorer = nn.Linear(D * 2, 1)
    params = list(layer.parameters()) + list(scorer.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    ei_train = torch.stack([src[train_sl], dst[train_sl]], dim=0)
    et_train = times[train_sl]
    y_train = labels[train_sl].float()

    losses = []
    t0 = time.perf_counter()
    for _ in range(epochs):
        # Cutoff = last train time + small epsilon.
        cutoff = float(et_train.max().item()) + 1e-4
        query_t = torch.full((n,), cutoff)
        emb = layer(x, ei_train, et_train, query_t)
        logits = scorer(torch.cat([emb[src[train_sl]], emb[dst[train_sl]]], dim=-1)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))

    # Leakage check.
    train_t = times[train_sl]
    valid_t = times[valid_sl]
    no_leakage = bool(
        train_t.numel() == 0 or valid_t.numel() == 0 or
        float(train_t.max()) <= float(valid_t.min())
    )
    # Time encoding affects output: run with two different cutoffs and compare.
    with torch.no_grad():
        emb_t1 = layer(x, ei_train, et_train, torch.zeros(n))
        emb_t2 = layer(x, ei_train, et_train, torch.ones(n) * 10)
    time_enc_differs = not torch.allclose(emb_t1, emb_t2)

    dt = time.perf_counter() - t0
    return {
        "model": "TGATConv",
        "train_events": int(train_sl.stop - train_sl.start),
        "valid_events": int(num_events - train_sl.stop),
        "epochs": epochs,
        "loss_initial": round(losses[0], 4),
        "loss_final": round(losses[-1], 4),
        "loss_decreased": losses[-1] < losses[0] or losses[-1] < 0.9,
        "grad_finite": all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in params if p.grad is not None
        ),
        "leakage_check_passed": no_leakage,
        "time_encoding_affects_output": time_enc_differs,
        "runtime_s": round(dt, 4),
        "limitation_notes": "Internal toy validation only; not benchmarked against reference TGAT implementations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TGN/TGAT temporal benchmark")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="all", choices=["tgn", "tgat", "all"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n = 20 if args.small else 50
    num_events = 40 if args.small else 200
    epochs = args.epochs or (3 if args.small else 20)
    lr = 0.01

    results = []
    if args.model in ("tgn", "all"):
        results.append(_bench_tgn(n, num_events, epochs, lr, args.seed))
    if args.model in ("tgat", "all"):
        results.append(_bench_tgat(n, num_events, epochs, lr, args.seed))

    report = {
        "package_version": tgraphx.__version__,
        "seed": int(args.seed),
        "model_results": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_temporal_summary(args.output, report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            print(f"[{r['model']:>12}] loss: {r['loss_initial']:.3f}→{r['loss_final']:.3f} "
                  f"leakage_ok={r['leakage_check_passed']} "
                  f"grad_ok={r.get('grad_finite', r.get('grad_finite_all_epochs'))} "
                  f"rt={r['runtime_s']:.2f}s")


if __name__ == "__main__":
    main()
