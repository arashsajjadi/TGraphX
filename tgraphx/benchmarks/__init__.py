"""Public-benchmark integration helpers (OGB / TGB) and v1.3 benchmark suite.

Optional, dependency-light evaluator wrappers around the official
benchmark packages.  Importing this module never triggers a network
call; callers must explicitly download datasets via the upstream
packages.

The v1.3 benchmark suite is available as a package-level function so it
can be called without a cloned repository::

    from tgraphx.benchmarks import run_v13_benchmark_suite
    results = run_v13_benchmark_suite(small=True, return_dict=True)

Stability: Beta (v0.5.0+). ``run_v13_benchmark_suite`` — Beta (v1.3.4+).
"""
from __future__ import annotations

from .public import (
    OGBNodeEvaluator,
    OGBLinkEvaluator,
    OGBGraphEvaluator,
    TGBLinkEvaluator,
)
from .suite import run_v13_benchmark_suite

__all__ = [
    "OGBNodeEvaluator",
    "OGBLinkEvaluator",
    "OGBGraphEvaluator",
    "TGBLinkEvaluator",
    "run_v13_benchmark_suite",
]
