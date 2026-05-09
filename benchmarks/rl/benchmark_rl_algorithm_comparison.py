"""Compare all RL algorithms on navigation env: episodes/sec, mean return.

Flags: --small --json --seed --episodes
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict


def parse_args():
    p = argparse.ArgumentParser(description="Compare RL algorithms on navigation env")
    p.add_argument("--small", action="store_true", help="Use small configuration")
    p.add_argument("--json", action="store_true", help="Output JSON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=None)
    return p.parse_args()


def run_benchmark(small: bool, seed: int, episodes: int = None) -> Dict[str, Any]:
    import torch
    from tgraphx.rl import run_graph_rl, list_graph_rl_algorithms

    if episodes is None:
        episodes = 5 if small else 20

    all_algos = list_graph_rl_algorithms()
    discrete_algos = [k for k, v in all_algos.items() if v["action_type"] == "discrete"]
    continuous_algos = [k for k, v in all_algos.items() if v["action_type"] == "continuous"]

    results = {}

    for algo in discrete_algos:
        try:
            t0 = time.perf_counter()
            res = run_graph_rl(
                env="graph_navigation",
                algorithm=algo,
                episodes=episodes,
                seed=seed,
            )
            elapsed = time.perf_counter() - t0
            results[algo] = {
                "action_type": "discrete",
                "mean_return": res.metrics["mean_return"],
                "episodes_per_sec": episodes / max(elapsed, 1e-9),
                "time_s": elapsed,
                "status": "ok",
            }
        except Exception as e:
            results[algo] = {"status": "error", "error": str(e)}

    for algo in continuous_algos:
        try:
            t0 = time.perf_counter()
            res = run_graph_rl(
                env="continuous_navigation",
                algorithm=algo,
                episodes=episodes,
                seed=seed,
            )
            elapsed = time.perf_counter() - t0
            results[algo] = {
                "action_type": "continuous",
                "mean_return": res.metrics["mean_return"],
                "episodes_per_sec": episodes / max(elapsed, 1e-9),
                "time_s": elapsed,
                "status": "ok",
            }
        except Exception as e:
            results[algo] = {"status": "error", "error": str(e)}

    return {
        "seed": seed,
        "device": "cpu",
        "episodes": episodes,
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
        print(f"Algorithm comparison ({result['episodes']} episodes)")
        print(f"{'Algorithm':<20} {'Action':<12} {'Return':>10} {'ep/s':>10}")
        print("-" * 56)
        for algo, m in sorted(result["metrics"].items()):
            if m.get("status") == "ok":
                print(f"{algo:<20} {m.get('action_type','?'):<12} {m['mean_return']:>10.2f} {m['episodes_per_sec']:>10.1f}")
            else:
                print(f"{algo:<20} {'ERROR':<12} {m.get('error', '')[:20]}")


if __name__ == "__main__":
    main()
