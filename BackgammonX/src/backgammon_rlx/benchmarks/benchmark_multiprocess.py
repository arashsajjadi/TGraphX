"""Benchmark multiprocess self-play throughput across worker counts and batch sizes.

    python -m backgammon_rlx.benchmarks.benchmark_multiprocess \\
      --workers 1,2,4,8,12 \\
      --batch-sizes 128,256,512,1024
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch

from ..models.policy_value_net import BackgammonPolicyValueNet
from ..env.encoding import ObservationEncoder, ActionEncoder


def benchmark_worker_scaling(
    n_workers_list: List[int],
    batch_sizes:    List[int],
    n_games:        int   = 16,
    device_str:     str   = "cuda",
    state_dim:      int   = 128,
    act_dim:        int   = 128,
    seed:           int   = 42,
) -> dict:
    device  = torch.device(device_str if torch.cuda.is_available() else "cpu")
    obs_enc = ObservationEncoder()
    act_enc = ActionEncoder()
    results = {}

    for n_workers in n_workers_list:
        for batch_size in batch_sizes:
            key = f"w{n_workers}_b{batch_size}"
            print(f"  {key}: {n_workers} workers, batch={batch_size}...", flush=True)

            model = BackgammonPolicyValueNet(
                state_dim=state_dim, act_dim=act_dim,
                n_point_res=2, n_action_res=1
            )

            try:
                from ..rl.multiprocess_rollout import MultiprocessRolloutCollector
                from ..rl.buffer import RolloutBuffer

                collector = MultiprocessRolloutCollector(
                    model=model,
                    n_workers=n_workers,
                    obs_enc=obs_enc,
                    act_enc=act_enc,
                    device=device,
                    seed=seed,
                    inference_batch_size=batch_size,
                    inference_max_wait_ms=5.0,
                    use_amp=(device.type == "cuda"),
                )
                collector.start()
                t0 = time.perf_counter()
                buf, stats = collector.collect(n_games=n_games)
                elapsed = time.perf_counter() - t0
                collector.stop()

                games_per_s = n_games / elapsed
                steps_per_s = stats.get("total_steps", 0) / elapsed

                results[key] = {
                    "n_workers":    n_workers,
                    "batch_size":   batch_size,
                    "n_games":      n_games,
                    "elapsed_s":    elapsed,
                    "games_per_s":  games_per_s,
                    "steps_per_s":  steps_per_s,
                    "inf_calls":    stats.get("inf_total", 0),
                    "device":       str(device),
                    "error":        None,
                }
                print(f"    → {games_per_s:.1f} games/s  {steps_per_s:.0f} steps/s")

            except Exception as e:
                results[key] = {
                    "n_workers": n_workers, "batch_size": batch_size,
                    "error": str(e), "games_per_s": 0.0
                }
                print(f"    → ERROR: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",     default="1,2,4",
                        help="Comma-separated worker counts")
    parser.add_argument("--batch-sizes", default="128,256,512",
                        help="Comma-separated inference batch sizes")
    parser.add_argument("--games",       type=int, default=16)
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    workers     = [int(x) for x in args.workers.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    device      = "cuda" if (args.device == "auto" and torch.cuda.is_available()) \
                  else args.device

    print(f"Multiprocess scaling benchmark")
    print(f"  workers={workers}  batch_sizes={batch_sizes}  device={device}")
    print()

    results = benchmark_worker_scaling(
        workers, batch_sizes, n_games=args.games, device_str=device
    )

    print("\nSummary:")
    print(f"{'Config':20s} {'games/s':>10s} {'steps/s':>10s} {'error':>20s}")
    for k, v in sorted(results.items()):
        print(f"  {k:20s} {v.get('games_per_s',0):>10.1f} "
              f"{v.get('steps_per_s',0):>10.0f} "
              f"{str(v.get('error','OK')):>20s}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
