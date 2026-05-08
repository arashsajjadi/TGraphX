"""Shared argparse + run-dir helpers for public-dataset validation scripts."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def make_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Return a parser with the standard public-validation flags."""
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "--root", type=str, default=None,
        help="Cache root for the upstream dataset (defaults to TGRAPHX_DATA / ~/.cache/tgraphx).",
    )
    p.add_argument(
        "--download", action="store_true",
        help="Allow the upstream library to download data over the network. Off by default.",
    )
    p.add_argument(
        "--max-samples", type=int, default=100,
        help="Cap on the number of samples (graph-level datasets).  Default 100.",
    )
    p.add_argument(
        "--max-nodes", type=int, default=5000,
        help="Cap on the number of nodes for very large graphs (e.g. ogbn-arxiv).",
    )
    p.add_argument(
        "--max-edges", type=int, default=64,
        help="Cap on the number of edges scored by edge_perturbation_attribution.",
    )
    p.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs for the optional tiny-training smoke.",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device override.",
    )
    p.add_argument(
        "--output-run-dir", type=str, default=None,
        help="Directory where dashboard artefacts will be written.  "
             "Default: a TemporaryDirectory that is removed on exit.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strict", action="store_true",
        help="Fail (exit-code 2) instead of skipping when an optional dependency is missing.",
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


def write_run_provenance(run_dir: Path, *, run_name: str, **extra: Any) -> Path:
    """Write a minimal ``run_metadata.json`` (status="running")."""
    import tgraphx

    run_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tgraphx_version": tgraphx.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    payload.update(extra)
    path = run_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def mark_run_completed(run_dir: Path, **extra: Any) -> None:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        payload = {}
    payload["status"] = "completed"
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))


def write_summary_json(run_dir: Path, summary: Dict[str, Any]) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


__all__ = [
    "make_parser",
    "resolve_device",
    "soft_skip",
    "write_run_provenance",
    "mark_run_completed",
    "write_summary_json",
]
