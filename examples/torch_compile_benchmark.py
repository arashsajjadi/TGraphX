"""torch_compile_benchmark.py — eager vs compiled layer comparison.

Compares forward-pass output correctness and timing for a small GNN layer
under eager mode and torch.compile (PyTorch 2.0+).

* Skips gracefully if torch.compile is unavailable.
* Uses synthetic data — no GPU required.
* No file writes.
* Makes no universal speed claims: compile may add overhead for small
  graphs and benefit larger ones.
"""
import sys
import time

import torch
import torch.nn as nn

from tgraphx.graph_builders import build_grid_graph
from tgraphx.layers.factory import make_layer
from tgraphx.performance import env_report
from tgraphx.training import count_parameters


def _time_iters(fn, n: int, device: torch.device) -> float:
    """Return mean wall-clock time in ms over n iterations."""
    fn()  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    info = env_report()
    device = torch.device(info["recommended_device"])

    print(f"\nPyTorch  : {info['torch']}")
    print(f"Device   : {device}")

    if not hasattr(torch, "compile"):
        print("\ntorch.compile is not available (requires PyTorch >= 2.0).")
        print("Skipping compile benchmark.")
        return

    # ── Build a small spatial GNN layer ──────────────────────────────────────
    # GIN is generally compile-friendly (simple MLP structure)
    in_shape  = (8, 4, 4)
    out_shape = (8, 4, 4)
    layer = make_layer("gin", in_shape, out_shape, spatial_rank=2).to(device).eval()
    print(f"Layer    : TensorGINLayer  {in_shape} → {out_shape}")
    print(f"Params   : {count_parameters(layer):,}")

    # ── Synthetic graph data ──────────────────────────────────────────────────
    N = 25   # 5×5 grid
    x  = torch.randn(N, *in_shape, device=device)
    ei = build_grid_graph(5, 5, directed=False, self_loops=True).to(device)

    # ── Eager forward ─────────────────────────────────────────────────────────
    with torch.no_grad():
        eager_out = layer(x, ei)

    ITERS = 20
    eager_ms = _time_iters(lambda: layer(x, ei), ITERS, device)
    print(f"\nEager     : {eager_ms:.3f} ms  (output shape {tuple(eager_out.shape)})")

    # ── torch.compile ─────────────────────────────────────────────────────────
    try:
        compiled = torch.compile(layer, mode="default")

        # First call triggers compilation (may be slow)
        print("Compiling... (first call may take several seconds)")
        with torch.no_grad():
            compiled_out = compiled(x, ei)

        # Check correctness
        max_diff = (eager_out - compiled_out).abs().max().item()
        ok = max_diff < 1e-4
        print(f"Correctness check : max |eager - compiled| = {max_diff:.2e}  {'OK' if ok else 'WARN'}")
        if not ok:
            print("  [WARN] Outputs differ beyond tolerance — check model and data types.")

        # Timed compiled run (post-compilation)
        compiled_ms = _time_iters(lambda: compiled(x, ei), ITERS, device)
        speedup = eager_ms / compiled_ms if compiled_ms > 0 else float("nan")
        print(f"Compiled  : {compiled_ms:.3f} ms  (speedup {speedup:.2f}x)")

        if speedup < 1.0:
            print("\n  Note: compile overhead dominates for small graphs.")
            print("  Try larger graphs (--nodes 256+) for potential compile benefits.")
        elif speedup > 1.1:
            print(f"\n  Speedup: {speedup:.2f}x with torch.compile (mode='default').")

    except Exception as e:
        print(f"\ntorch.compile failed: {e}")
        print("This can happen on some platforms, PyTorch versions, or operator combinations.")
        print("Falling back is safe — eager mode is always used when compile is unavailable.")

    print()


if __name__ == "__main__":
    main()
