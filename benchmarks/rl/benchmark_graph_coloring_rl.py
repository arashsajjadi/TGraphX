"""Benchmark REINFORCE and DQN on graph coloring env.

Flags: --small --json --seed --episodes
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark RL on graph coloring")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=None)
    return p.parse_args()


def run_benchmark(small: bool, seed: int, episodes: int = None) -> Dict[str, Any]:
    import torch
    from tgraphx.rl import run_graph_rl

    if episodes is None:
        episodes = 5 if small else 20

    n = 5 if small else 10

    results = {}
    for algo in ("reinforce", "dqn"):
        t0 = time.perf_counter()
        res = run_graph_rl(
            env="graph_coloring",
            algorithm=algo,
            episodes=episodes,
            seed=seed,
            num_nodes=n,
        )
        elapsed = time.perf_counter() - t0
        results[algo] = {
            "mean_return": res.metrics["mean_return"],
            "episodes_per_sec": episodes / max(elapsed, 1e-9),
            "time_s": elapsed,
        }

    return {
        "seed": seed,
        "episodes": episodes,
        "num_nodes": n,
        "metrics": results,
    }


def main():
    args = parse_args()
    result = run_benchmark(small=args.small, seed=args.seed, episodes=args.episodes)
    if args.json:
        import sys as _sys, tgraphx as _tgx
        import torch as _torch
        result.setdefault('package_version', _tgx.__version__)
        result.setdefault('status', 'ok')
        result.setdefault('limitations', 'CPU-only small-scale; Experimental stability')
        result.setdefault('device', 'cuda' if _torch.cuda.is_available() else 'cpu')
        print(json.dumps(result, indent=2))
    else:
        print(f"Graph coloring RL benchmark ({result['episodes']} episodes)")
        for algo, m in result["metrics"].items():
            print(f"  {algo}: mean_return={m['mean_return']:.2f}, {m['episodes_per_sec']:.1f} ep/s")


if __name__ == "__main__":
    main()
