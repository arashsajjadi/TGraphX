"""Shared utilities for benchmarks/public/ scripts.

The conventions here are slightly different from
``examples/public_datasets/_common.py``: the public-dataset *examples*
are validation smokes that emit ``run_metadata.json`` plus dashboard
artefacts; the public *benchmarks* additionally emit
``benchmark_results.json``.  Both sets of scripts deliberately stop short
of leaderboard runs.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch


def make_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Standard CLI for benchmarks/public/ scripts."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "--root", type=str, default=None,
        help="Cache root for the upstream dataset (defaults to $TGRAPHX_DATA "
             "or ~/.cache/tgraphx).",
    )
    p.add_argument(
        "--download", action="store_true",
        help="Allow the upstream library to download data over the network.  "
             "Off by default.  TGraphX never bundles datasets.",
    )
    p.add_argument(
        "--max-samples", type=int, default=200,
        help="Cap on the number of training samples (default: 200).",
    )
    p.add_argument(
        "--max-nodes", type=int, default=10_000,
        help="Cap on the number of nodes for very large graphs (default: 10_000).",
    )
    p.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs for the smoke run (default: 5).",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device override (default: auto-detect; CPU when neither "
             "CUDA nor MPS is available).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory where benchmark JSON artefacts will be written.  "
             "Default: a TemporaryDirectory removed on exit.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--json", action="store_true",
        help="Print a machine-readable JSON summary to stdout in addition "
             "to writing artefacts to --output-dir.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Fail (exit 2) instead of skipping when an optional dependency "
             "is missing or the upstream dataset cannot be loaded.",
    )
    return p


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def soft_skip(msg: str, strict: bool) -> int:
    """Return an exit code suitable for ``raise SystemExit(...)``.

    ``--strict`` flips a soft skip into a hard failure (exit 2).
    """
    print(f"[skip] {msg}")
    return 2 if strict else 0


def env_block(seed: int, device: torch.device) -> Dict[str, Any]:
    import tgraphx
    return {
        "tgraphx_version": tgraphx.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": int(seed),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_artefacts(
    output_dir: Path,
    *,
    benchmark: Dict[str, Any],
    run_metadata: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
    metrics_summary: Dict[str, Any],
) -> Dict[str, Path]:
    """Write the standard four-file artefact set the dashboard reads."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "benchmark_results.json": write_json(
            output_dir / "benchmark_results.json", benchmark,
        ),
        "run_metadata.json": write_json(
            output_dir / "run_metadata.json", run_metadata,
        ),
        "dataset_metadata.json": write_json(
            output_dir / "dataset_metadata.json", dataset_metadata,
        ),
        "metrics_summary.json": write_json(
            output_dir / "metrics_summary.json", metrics_summary,
        ),
    }


__all__ = [
    "make_parser",
    "resolve_device",
    "soft_skip",
    "env_block",
    "write_json",
    "write_artefacts",
]
