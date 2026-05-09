"""Benchmark DQN throughput on GraphNavigationEnv.

Flags: --small --json --seed --episodes
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark DQN on GraphNavigationEnv")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=None)
    return p.parse_args()


def run_benchmark(small: bool, seed: int, episodes: int = None) -> Dict[str, Any]:
    import torch
    from tgraphx.rl import run_graph_rl

    if episodes is None:
        episodes = 5 if small else 30

    t0 = time.perf_counter()
    result = run_graph_rl(
        env="graph_navigation",
        algorithm="dqn",
        episodes=episodes,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0

    return {
        "seed": seed,
        "device": "cpu",
        "episodes": episodes,
        "metrics": {
            "mean_return": result.metrics["mean_return"],
            "success_rate": result.metrics["success_rate"],
            "episodes_per_sec": episodes / max(elapsed, 1e-9),
            "time_s": elapsed,
        },
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
        print(f"DQN benchmark ({result['episodes']} episodes)")
        for k, v in result["metrics"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
