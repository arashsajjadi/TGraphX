"""benchmarks/benchmark_layers.py — TGraphX GNN layer throughput benchmark.

Usage examples
--------------
# GAT on a 3×3 patch graph, CPU
python benchmarks/benchmark_layers.py --layer gat --nodes 16 --edges 64 \\
    --shape 8,4,4 --device cpu --iters 10 --warmup 3

# ConvMessagePassing on CUDA with AMP
python benchmarks/benchmark_layers.py --layer conv --nodes 256 --edges 2048 \\
    --shape 32,8,8 --device cuda --amp 1 --backward 1

# Save JSON result
python benchmarks/benchmark_layers.py --layer gin --shape 8,4,4 \\
    --output results/gin_bench.json

No file output unless --output is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Tuple

# Allow running from project root without installing
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

import torch
import torch.nn as nn

from tgraphx.layers.factory import make_layer
from tgraphx.performance import env_report, estimate_message_memory
from tgraphx.training import count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _time_block_cuda(fn, n: int) -> Tuple[float, float]:
    """Time fn() n times using CUDA events. Returns (mean_ms, std_ms)."""
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
    return mean, std


def _time_block_cpu(fn, n: int) -> Tuple[float, float]:
    """Time fn() n times using perf_counter. Returns (mean_ms, std_ms)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
    return mean, std


def _time(fn, n: int, device: torch.device) -> Tuple[float, float]:
    if device.type == "cuda":
        return _time_block_cuda(fn, n)
    return _time_block_cpu(fn, n)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TGraphX GNN layer benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--layer",     default="gat",
                   choices=["conv", "gat", "sage", "gin", "linear", "legacy_attention"],
                   help="Layer type (default: gat)")
    p.add_argument("--nodes",     type=int, default=64,
                   help="Number of nodes (default: 64)")
    p.add_argument("--edges",     type=int, default=256,
                   help="Number of edges (default: 256)")
    p.add_argument("--shape",     default="8,4,4",
                   help="Node feature shape, comma-separated (default: 8,4,4)")
    p.add_argument("--out-shape", default=None,
                   help="Output shape override (default: same as --shape but doubled channels)")
    p.add_argument("--edge-features", default="none",
                   choices=["none", "vector", "spatial"],
                   help="Edge feature type (default: none)")
    p.add_argument("--edge-weight", type=int, default=0, choices=[0, 1],
                   help="Include per-edge scalar weights (default: 0)")
    p.add_argument("--device",    default="auto",
                   help="Device: auto/cpu/cuda/mps (default: auto)")
    p.add_argument("--iters",     type=int, default=10,
                   help="Timed iterations (default: 10)")
    p.add_argument("--warmup",    type=int, default=3,
                   help="Warmup iterations (default: 3)")
    p.add_argument("--backward",  type=int, default=0, choices=[0, 1],
                   help="Benchmark backward pass (default: 0)")
    p.add_argument("--amp",       type=int, default=0, choices=[0, 1],
                   help="Use torch.autocast (default: 0)")
    p.add_argument("--compile",   type=int, default=0, choices=[0, 1],
                   help="Use torch.compile (default: 0)")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--log-level", type=int, default=0, choices=[0, 1, 2],
                   help="0=minimal, 1=verbose, 2=debug (default: 0)")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Edge chunk size for ConvMessagePassing (sum/mean only). "
                        "None = no chunking (default). Ignored for other layers.")
    p.add_argument("--output",    default=None,
                   help="Optional path to write JSON results (default: stdout only)")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    torch.manual_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        info = env_report()
        device = torch.device(info["recommended_device"])
    else:
        device = torch.device(args.device)

    # ── Shapes ───────────────────────────────────────────────────────────────
    in_shape = tuple(int(x) for x in args.shape.split(","))
    if args.out_shape:
        out_shape = tuple(int(x) for x in args.out_shape.split(","))
    else:
        # Double the channel dimension
        out_shape = (in_shape[0] * 2,) + in_shape[1:]

    rank = len(in_shape)
    if rank == 1:
        # Vector shape; only 'linear' / 'legacy_attention' work
        if args.layer in ("conv", "gat", "sage", "gin"):
            print(
                f"[WARN] --layer {args.layer} requires spatial in_shape. "
                f"Use --shape C,H,W or --shape C,D,H,W, or --layer linear.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── Layer construction ────────────────────────────────────────────────────
    layer_kwargs = {}
    edge_dim = None

    if args.edge_features != "none":
        edge_dim = in_shape[0]
        layer_kwargs["use_edge_features"] = True
        layer_kwargs["edge_dim"] = edge_dim
        if args.edge_features == "vector":
            layer_kwargs["edge_features_kind"] = "vector"
        else:
            layer_kwargs["edge_features_kind"] = "spatial"

    try:
        layer = make_layer(args.layer, in_shape, out_shape, **layer_kwargs).to(device)
    except (ValueError, NotImplementedError) as exc:
        print(f"[ERROR] Cannot create layer: {exc}", file=sys.stderr)
        sys.exit(1)

    n_params = count_parameters(layer)

    # ── Compile ───────────────────────────────────────────────────────────────
    compile_status = "disabled"
    if args.compile:
        if not hasattr(torch, "compile"):
            compile_status = "skipped (torch.compile unavailable; PyTorch < 2.0)"
        else:
            try:
                layer = torch.compile(layer, mode="default")
                compile_status = "enabled"
            except Exception as e:
                compile_status = f"failed ({e})"

    # ── Synthetic data ────────────────────────────────────────────────────────
    N, E = args.nodes, args.edges
    x = torch.randn(N, *in_shape, device=device)
    ei = torch.stack([
        torch.randint(0, N, (E,), device=device),
        torch.randint(0, N, (E,), device=device),
    ])

    edge_weight = torch.rand(E, device=device) if args.edge_weight else None

    edge_features = None
    if args.edge_features == "vector":
        edge_features = torch.randn(E, edge_dim, device=device)
    elif args.edge_features == "spatial" and rank >= 3:
        edge_features = torch.randn(E, edge_dim, *in_shape[1:], device=device)
    elif args.edge_features == "spatial":
        print("[WARN] spatial edge features require rank-3+ in_shape.", file=sys.stderr)

    # ── AMP context ───────────────────────────────────────────────────────────
    amp_status = "disabled"
    amp_ctx = None
    if args.amp:
        if device.type == "cuda":
            amp_ctx = torch.autocast("cuda", dtype=torch.float16)
            amp_status = "enabled (cuda float16)"
        elif device.type == "cpu":
            try:
                amp_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
                amp_status = "enabled (cpu bfloat16)"
            except Exception:
                amp_status = "skipped (bfloat16 not supported on this CPU)"
        else:
            amp_status = f"skipped (unsupported device: {device.type})"

    chunk_size = getattr(args, "chunk_size", None)

    def _fwd():
        with (amp_ctx if amp_ctx else _null_ctx()):
            if chunk_size is not None and args.layer == "conv":
                return layer(x, ei, edge_features=edge_features,
                             edge_weight=edge_weight, chunk_size=chunk_size)
            return layer(x, ei, edge_features=edge_features, edge_weight=edge_weight)

    # ── Warmup ────────────────────────────────────────────────────────────────
    layer.eval()
    with torch.no_grad():
        for _ in range(args.warmup):
            out = _fwd()

    out_shape_actual = tuple(out.shape)

    # ── Time forward ──────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        fwd_mean, fwd_std = _time(lambda: _fwd(), args.iters, device)

    peak_cuda_mb = None
    if device.type == "cuda":
        peak_cuda_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── Time backward (optional) ──────────────────────────────────────────────
    bwd_mean = bwd_std = None
    if args.backward:
        layer.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def _fwd_bwd():
            with (amp_ctx if amp_ctx else _null_ctx()):
                o = layer(x, ei, edge_features=edge_features, edge_weight=edge_weight)
            o.sum().backward()
            layer.zero_grad(set_to_none=True)

        bwd_mean, bwd_std = _time(_fwd_bwd, args.iters, device)
        if device.type == "cuda":
            peak_cuda_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── Memory estimate ───────────────────────────────────────────────────────
    mem = estimate_message_memory(E, out_shape, dtype=torch.float32)

    # ── Build result dict ─────────────────────────────────────────────────────
    result = {
        "layer":          args.layer,
        "device":         str(device),
        "in_shape":       in_shape,
        "out_shape":      out_shape_actual,
        "nodes":          N,
        "edges":          E,
        "parameters":     n_params,
        "edge_features":  args.edge_features,
        "edge_weight":    bool(args.edge_weight),
        "fwd_mean_ms":    round(fwd_mean, 3),
        "fwd_std_ms":     round(fwd_std, 3),
        "bwd_mean_ms":    round(bwd_mean, 3) if bwd_mean is not None else None,
        "bwd_std_ms":     round(bwd_std,  3) if bwd_std  is not None else None,
        "peak_cuda_mb":   round(peak_cuda_mb, 1) if peak_cuda_mb is not None else None,
        "msg_mem_est_mb": mem["total_mb"],
        "amp":            amp_status,
        "compile":        compile_status,
        "chunk_size":     chunk_size,
        "seed":           args.seed,
    }

    return result


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(r: dict, log_level: int = 0) -> None:
    sep = "-" * 54
    def row(k, v): print(f"  {k:<22} {v}")

    print(sep)
    print("  TGraphX Layer Benchmark")
    print(sep)
    row("Layer",       r["layer"])
    row("Device",      r["device"])
    row("Node shape",  f"{r['in_shape']}   [{r['nodes']} nodes]")
    row("Edges",       r["edges"])
    row("Parameters",  f"{r['parameters']:,}")
    row("Edge features", r["edge_features"])
    row("Edge weight", "yes" if r["edge_weight"] else "no")
    print(sep)
    row("Forward",     f"{r['fwd_mean_ms']:.3f} ms  ±{r['fwd_std_ms']:.3f} ms")
    if r["bwd_mean_ms"] is not None:
        row("Backward", f"{r['bwd_mean_ms']:.3f} ms  ±{r['bwd_std_ms']:.3f} ms")
    row("Output shape", r["out_shape"])
    row("Peak CUDA mem",
        f"{r['peak_cuda_mb']:.1f} MB" if r["peak_cuda_mb"] is not None else "N/A")
    row("Msg mem est.", f"~{r['msg_mem_est_mb']:.2f} MB")
    row("AMP",         r["amp"])
    row("Compile",     r["compile"])
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    result = run_benchmark(args)
    _print_report(result, args.log_level)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result written to: {args.output}")


if __name__ == "__main__":
    main()
