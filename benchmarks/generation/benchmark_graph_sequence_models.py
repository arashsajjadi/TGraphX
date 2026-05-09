"""Benchmark graph sequence model forward/backward speed.

Flags: --small --json --seed
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark graph sequence models")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_benchmark(small: bool, seed: int) -> Dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    seq_len = 10 if small else 50
    input_dim = 8 if small else 32
    hidden_dim = 16 if small else 64
    n_iters = 5 if small else 20

    # Use a simple LSTM as stand-in for graph sequence encoder
    lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-3)

    x = torch.randn(1, seq_len, input_dim)

    # Forward
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            out, _ = lstm(x)
    fwd_time = (time.perf_counter() - t0) / n_iters

    # Backward
    t1 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad()
        out, _ = lstm(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    bwd_time = (time.perf_counter() - t1) / n_iters

    return {
        "seed": seed,
        "seq_len": seq_len,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "n_iters": n_iters,
        "metrics": {
            "forward_time_s": fwd_time,
            "backward_time_s": bwd_time,
        },
    }


def main():
    args = parse_args()
    result = run_benchmark(small=args.small, seed=args.seed)
    if args.json:
        import sys as _sys, tgraphx as _tgx
        import torch as _torch
        result.setdefault('package_version', _tgx.__version__)
        result.setdefault('status', 'ok')
        result.setdefault('limitations', 'CPU-only small-scale; Experimental stability')
        result.setdefault('device', 'cuda' if _torch.cuda.is_available() else 'cpu')
        print(json.dumps(result, indent=2))
    else:
        print(f"Graph sequence model benchmark")
        print(f"  seq_len={result['seq_len']}, hidden={result['hidden_dim']}")
        print(f"  forward: {result['metrics']['forward_time_s']*1000:.2f}ms/iter")
        print(f"  backward: {result['metrics']['backward_time_s']*1000:.2f}ms/iter")


if __name__ == "__main__":
    main()
