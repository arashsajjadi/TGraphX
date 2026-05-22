"""Unified benchmark runner.

    python -m backgammon_rlx.benchmarks.benchmark_all
    python -m backgammon_rlx.benchmarks.benchmark_all --quick
    python -m backgammon_rlx.benchmarks.benchmark_all --config configs/rtx5080.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from .benchmark_movegen import benchmark_movegen, benchmark_random_games
from .benchmark_selfplay import benchmark_neural_inference, benchmark_selfplay_throughput
from .benchmark_inference import benchmark_inference_throughput, benchmark_latency
from ..models.policy_value_net import BackgammonPolicyValueNet


def _load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="BackgammonX benchmark suite")
    parser.add_argument("--out",    default=None, help="Output JSON path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick",  action="store_true")
    parser.add_argument("--config", default=None, help="YAML config")
    args = parser.parse_args()

    cfg    = _load_config(args.config)
    device = (args.device if args.device != "auto" else
              ("cuda" if torch.cuda.is_available() else "cpu"))

    scale  = 0.1 if args.quick else 1.0
    model  = BackgammonPolicyValueNet(
        state_dim=cfg.get("state_dim", 256),
        act_dim=cfg.get("act_dim", 256),
    )

    results = {}
    t_start = time.perf_counter()

    print("=" * 60)
    print("BackgammonX Benchmark Suite")
    print(f"device={device}  quick={args.quick}")
    print("=" * 60)

    print("\n[1] Legal move generation...")
    r = benchmark_movegen(n_positions=int(10_000 * scale))
    results["movegen"] = r
    print(f"    {r['positions_per_s']:.0f} positions/s  "
          f"avg_turns={r['avg_turns']:.1f}")

    print("\n[2] Random game simulation...")
    r = benchmark_random_games(n_games=int(500 * scale))
    results["random_games"] = r
    print(f"    {r['games_per_s']:.1f} games/s  "
          f"avg_len={r['avg_length']:.1f}  steps/s={r['steps_per_s']:.0f}")

    print("\n[3] Neural inference throughput (batch sweep)...")
    r = benchmark_inference_throughput(
        model, device_str=device,
        batch_sizes=(1, 8, 64, 512) if args.quick else (1, 8, 32, 128, 512, 2048),
    )
    results["inference_throughput"] = r
    for B, row in r["results"].items():
        print(f"    B={B:4d}: {row['states_per_s']:>10.0f} states/s  "
              f"{row['actions_per_s']:>10.0f} actions/s  "
              f"ms/batch={row['ms_per_batch']:.2f}")

    print("\n[4] Single-sample inference latency...")
    r = benchmark_latency(model, device_str=device)
    results["inference_latency"] = r
    print(f"    mean={r['mean_ms']:.2f}ms  median={r['median_ms']:.2f}ms  "
          f"p99={r['p99_ms']:.2f}ms")

    print("\n[5] Self-play throughput...")
    r = benchmark_selfplay_throughput(model, n_games=int(20 * scale),
                                      device_str=device)
    results["selfplay"] = r
    print(f"    {r['games_per_s']:.2f} games/s  {r['steps_per_s']:.0f} steps/s")

    total = time.perf_counter() - t_start
    results["total_elapsed_s"] = total
    results["device"] = device

    print(f"\nTotal benchmark time: {total:.1f}s")
    print("\n[Bottleneck analysis]")
    mg_pos_s = results["movegen"]["positions_per_s"]
    sp_steps_s = results["selfplay"]["steps_per_s"]
    inf_s = max(results["inference_throughput"]["results"].values(),
                key=lambda x: x["states_per_s"])["states_per_s"]
    print(f"  Move generation:  {mg_pos_s:.0f} pos/s")
    print(f"  Self-play (end2end): {sp_steps_s:.0f} steps/s")
    print(f"  Neural inference (peak): {inf_s:.0f} states/s")
    if sp_steps_s < mg_pos_s * 0.1:
        print("  → Bottleneck likely: neural inference or IPC overhead")
    elif mg_pos_s < 1000:
        print("  → Bottleneck likely: move generation")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.out}")


from typing import Optional

if __name__ == "__main__":
    main()
