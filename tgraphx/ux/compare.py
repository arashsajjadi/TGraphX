"""Functionality comparison helper (NOT throughput SOTA)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch


def compare(
    workflows: Sequence[Dict[str, Any]],
    *,
    runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset: Optional[Any] = None,
    metric: str = "accuracy",
    fast_mode: bool = True,
    seed: int = 42,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run a small functionality comparison across labeled workflows.

    This is **not** a throughput benchmark and does not make SOTA claims.
    It is a reproducibility utility for "is method A's metric finite and
    comparable to method B's on the same tiny seeded dataset" questions.

    Args:
        workflows: List of ``{"name": str, ...}`` dicts. If ``runner`` is None,
            each dict must have either a ``"call"`` callable or a ``"task"`` for
            :func:`tgraphx.workflow`.
        runner: Optional function ``runner(workflow_dict) -> metrics_dict``.
        dataset: Optional dataset id (forwarded to runner / workflow).
        metric: Metric name to highlight in the result.
        fast_mode: Forwarded to workflow if used.
        seed: Reproducibility seed.
        out_dir: Optional path to write benchmark_summary.json.

    Returns:
        Dict with ``results`` (list of per-workflow dicts) and metadata.
    """
    from .workflow import workflow as _wf
    from .. import __version__

    results = []
    t0 = time.time()
    for wf in workflows:
        name = wf.get("name", wf.get("task", "workflow"))
        t = time.time()
        try:
            if runner is not None:
                metrics = runner(wf)
            elif "call" in wf and callable(wf["call"]):
                metrics = wf["call"]()
            elif "task" in wf:
                r = _wf(task=wf["task"], fast_mode=fast_mode, seed=seed,
                        dataset=dataset, **{k: v for k, v in wf.items()
                                              if k not in ("name", "task", "call")})
                metrics = r.metrics if hasattr(r, "metrics") else r
            else:
                raise ValueError(
                    f"workflow {name!r}: must provide 'call' or 'task'"
                )
            status = "ok"
            error = None
        except Exception as exc:
            metrics = {}
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        results.append({
            "name": name,
            "metrics": metrics if isinstance(metrics, dict) else {"value": metrics},
            "runtime_s": round(time.time() - t, 3),
            "status": status,
            "error": error,
        })

    total = time.time() - t0
    out = {
        "results": results,
        "metric": metric,
        "fast_mode": fast_mode,
        "seed": seed,
        "tgraphx_version": __version__,
        "total_runtime_s": round(total, 3),
        "note": "Functionality / reproducibility comparison. NOT a SOTA or throughput claim.",
    }
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "benchmark_summary.json", "w") as f:
            json.dump(out, f, indent=2)
    return out
