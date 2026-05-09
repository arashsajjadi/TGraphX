"""Public-benchmark integration helpers (OGB / TGB).

Optional, dependency-light evaluator wrappers around the official
benchmark packages.  Importing this module never triggers a network
call; callers must explicitly download datasets via the upstream
packages.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from .public import (
    OGBNodeEvaluator,
    OGBLinkEvaluator,
    OGBGraphEvaluator,
    TGBLinkEvaluator,
)

__all__ = [
    "OGBNodeEvaluator",
    "OGBLinkEvaluator",
    "OGBGraphEvaluator",
    "TGBLinkEvaluator",
]
