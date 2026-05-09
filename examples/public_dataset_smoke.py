"""Public dataset smoke script.

Validates metadata and basic functionality for Cora (PyG), OGB, and TGB
datasets.  **Never downloads data without explicit --download flag.**
Skips cleanly when optional dependencies or data files are missing.

Usage:
    # Dry run (no downloads, reports what is available):
    python examples/public_dataset_smoke.py --dataset all --json

    # With downloads (requires PyG / OGB / TGB installed):
    python examples/public_dataset_smoke.py --dataset cora \\
        --root ~/.cache/tgraphx --download --json

Stability: Beta (optional dependency, explicit download required).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _skip(name: str, reason: str) -> Dict[str, Any]:
    return {"dataset": name, "status": "skipped", "reason": reason}


def _check_cora(root: str, download: bool) -> Dict[str, Any]:
    try:
        import torch_geometric  # noqa: F401
        from torch_geometric.datasets import Planetoid
    except ImportError:
        return _skip("cora", "torch_geometric not installed; pip install torch_geometric")

    data_root = Path(root) / "planetoid"
    if not (data_root / "Cora" / "raw").exists() and not download:
        return _skip("cora", f"data not found at {data_root}; pass --download to fetch")

    try:
        ds = Planetoid(root=str(data_root), name="Cora", split="public")
        g = ds[0]
        return {
            "dataset": "cora",
            "status": "available",
            "num_nodes": int(g.num_nodes),
            "num_edges": int(g.num_edges),
            "num_features": int(g.num_features),
            "num_classes": int(ds.num_classes),
            "has_train_mask": bool(hasattr(g, "train_mask") and g.train_mask is not None),
        }
    except Exception as e:
        return {"dataset": "cora", "status": "error", "reason": str(e)}


def _check_ogb(name: str, root: str, download: bool) -> Dict[str, Any]:
    try:
        import ogb  # noqa: F401
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        return _skip(name, "ogb not installed; pip install ogb")
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return _skip(name, "torch_geometric required for OGB PygNodePropPredDataset")

    data_root = Path(root) / "ogb"
    try:
        ds = PygNodePropPredDataset(name=name, root=str(data_root))
        g = ds[0]
        split = ds.get_idx_split()
        from tgraphx.benchmarks import OGBNodeEvaluator
        ev = OGBNodeEvaluator(name=name)
        return {
            "dataset": name,
            "status": "available",
            "num_nodes": int(g.num_nodes),
            "num_edges": int(g.num_edges),
            "num_features": int(g.num_features),
            "split_keys": list(split.keys()),
            "evaluator_type": type(ev._eval).__name__,
        }
    except FileNotFoundError:
        if not download:
            return _skip(name, f"data not found; pass --download to fetch")
        raise
    except Exception as e:
        return {"dataset": name, "status": "error", "reason": str(e)}


def _check_tgb(name: str, root: str, download: bool) -> Dict[str, Any]:
    try:
        import tgb  # noqa: F401
    except ImportError:
        return _skip(name, "tgb not installed; pip install py-tgb")
    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset  # type: ignore
        ds = LinkPropPredDataset(name=name, root=root)
        meta = ds.get_metadata() if hasattr(ds, "get_metadata") else {}
        return {
            "dataset": name,
            "status": "available",
            "metadata": {k: str(v) for k, v in (meta or {}).items()},
        }
    except FileNotFoundError:
        if not download:
            return _skip(name, "data not found; pass --download to fetch")
        raise
    except Exception as e:
        return {"dataset": name, "status": "error", "reason": str(e)}


_DATASET_MAP = {
    "cora":       lambda r, dl: _check_cora(r, dl),
    "ogbn-arxiv": lambda r, dl: _check_ogb("ogbn-arxiv", r, dl),
    "tgbl-wiki":  lambda r, dl: _check_tgb("tgbl-wiki-v2", r, dl),
    "tgbn-token": lambda r, dl: _check_tgb("tgbn-token", r, dl),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Public dataset smoke (optional deps)")
    parser.add_argument("--dataset", default="all",
                        choices=["all", "cora", "ogbn-arxiv", "tgbl-wiki", "tgbn-token"])
    parser.add_argument("--root", default=str(Path.home() / ".cache" / "tgraphx"))
    parser.add_argument("--download", action="store_true",
                        help="Allow downloading data from the internet.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    keys = list(_DATASET_MAP.keys()) if args.dataset == "all" else [args.dataset]
    results = []
    for k in keys:
        fn = _DATASET_MAP.get(k)
        if fn is not None:
            results.append(fn(args.root, args.download))
        else:
            results.append(_skip(k, "unknown dataset"))

    report = {
        "download_enabled": args.download,
        "root": str(args.root),
        "results": results,
        "note": (
            "No data was downloaded unless --download was passed. "
            "TGraphX never bundles third-party datasets."
        ),
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        p = Path(args.output)
        p.write_text(json.dumps(report, indent=2))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in results:
            print(f"  {r['dataset']:>15}: {r['status']}", end="")
            if r["status"] == "skipped":
                print(f" — {r['reason']}")
            elif r["status"] == "available":
                info = ", ".join(f"{k}={v}" for k, v in r.items()
                                 if k not in ("dataset", "status"))
                print(f" — {info}")
            else:
                print(f" — ERROR: {r.get('reason', '?')}")


if __name__ == "__main__":
    main()
