"""CLI entry point for the v1.3 benchmark suite.

Usage::

    python -m tgraphx.benchmarks.run_v13_benchmark_suite --small --json
    python -m tgraphx.benchmarks.run_v13_benchmark_suite --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys

from .suite import run_v13_benchmark_suite


def main():
    p = argparse.ArgumentParser(description="TGraphX v1.3 benchmark suite")
    p.add_argument("--small", action="store_true", help="Use tiny configs (CI-safe).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="JSON output path.")
    p.add_argument("--repeat", type=int, default=1, help="Repeats per benchmark.")
    p.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    args = p.parse_args()

    data = run_v13_benchmark_suite(
        small=args.small,
        device=args.device,
        seed=args.seed,
        repeat=args.repeat,
        return_dict=True,
        out=args.out,
    )

    if args.out:
        print(f"[v13-suite] wrote {args.out}")

    if args.json or args.out is None:
        print(json.dumps(data, indent=2))

    failed = [r for r in data["benchmarks"] if r["status"] == "failed"]
    if failed:
        print(f"\n[v13-suite] {len(failed)} failed:", file=sys.stderr)
        for r in failed:
            print(f"  - {r['name']}: {r.get('error', '')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
