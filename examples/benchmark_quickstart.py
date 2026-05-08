"""benchmark_quickstart.py — run a few CI-safe benchmark scripts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

SCRIPTS = [
    "benchmark_dataset_loading.py",
    "benchmark_transforms.py",
    "benchmark_metrics.py",
    "benchmark_training_synthetic.py",
    "benchmark_tensor_vs_flatten.py",
]


def main() -> None:
    for s in SCRIPTS:
        print(f"\n=== {s} --small ===")
        subprocess.check_call([sys.executable, str(BENCH / s), "--small"])


if __name__ == "__main__":
    main()
