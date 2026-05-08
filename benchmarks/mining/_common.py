"""Shared utilities for mining benchmarks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import torch


def make_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--small", action="store_true",
                   help="Small/fast mode for CI (default sizes are already small).")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON output.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--num-nodes", type=int, default=None,
                   help="Override default node count.")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_synthetic_graph(num_nodes: int, density: float = 0.1, seed: int = 0):
    """Return (edge_index, num_nodes) for a random sparse directed graph."""
    torch.manual_seed(seed)
    N = int(num_nodes)
    num_edges = max(1, int(N * (N - 1) * density))
    src = torch.randint(N, (num_edges,))
    dst = torch.randint(N, (num_edges,))
    # Deduplicate.
    pairs = torch.stack([src, dst], dim=0)
    pairs = torch.unique(pairs, dim=1)
    return pairs, N


def timer(fn, *args, n_warmup: int = 1, n_runs: int = 5, **kwargs) -> tuple:
    """Return (mean_time_s, result) of calling fn(*args, **kwargs)."""
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) / n_runs
    return elapsed, result


def print_result(data: Dict[str, Any], use_json: bool) -> None:
    if use_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"[{data.get('benchmark', 'benchmark')}]")
        for k, v in data.items():
            if k != "benchmark":
                print(f"  {k}: {v}")


__all__ = ["make_parser", "resolve_device", "make_synthetic_graph", "timer", "print_result"]
