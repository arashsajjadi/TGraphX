"""Benchmark metrics throughput and correctness."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

import tgraphx
from tgraphx.metrics import (
    accuracy,
    average_precision,
    classification_report,
    hits_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    regression_report,
    roc_auc,
)


def _bench(label: str, fn) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    return {"metric": label, "elapsed_s": elapsed, "value": float(out) if isinstance(out, (int, float)) else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--small", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    N = 256 if args.small else 4096
    C = 5 if args.small else 50
    M = 16 if args.small else 256

    torch.manual_seed(0)
    logits = torch.randn(N, C)
    labels = torch.randint(0, C, (N,))
    scores_2d = torch.randn(N, M)
    target_idx = torch.randint(0, M, (N,))
    pos = torch.randn(N // 2)
    neg = torch.randn(N // 2)
    targets_reg = torch.randn(N)
    preds_reg = targets_reg + 0.1 * torch.randn(N)

    results: List[Dict[str, Any]] = [
        _bench("accuracy", lambda: accuracy(logits, labels)),
        _bench("classification_report", lambda: classification_report(logits, labels, num_classes=C)["accuracy"]),
        _bench("hits@10", lambda: hits_at_k(scores_2d, target_idx, k=10)),
        _bench("mrr", lambda: mean_reciprocal_rank(scores_2d, target_idx)),
        _bench("ndcg@10", lambda: ndcg_at_k(scores_2d, target_idx, k=10)),
        _bench("roc_auc", lambda: roc_auc(pos, neg)),
        _bench("average_precision", lambda: average_precision(pos, neg)),
        _bench("regression_report", lambda: regression_report(preds_reg, targets_reg)["rmse"]),
    ]
    print(f"\nTGraphX metrics benchmark "
          f"(version={tgraphx.__version__}, N={N}, C={C}, M={M})\n")
    print(f"  {'Metric':<28} {'Time (s)':>10}")
    print("  " + "-" * 42)
    for r in results:
        print(f"  {r['metric']:<28} {r['elapsed_s']:>10.5f}")
    if args.output:
        Path(args.output).write_text(json.dumps({
            "version": tgraphx.__version__,
            "small": args.small,
            "results": results,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
