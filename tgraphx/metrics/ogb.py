"""Lightweight OGB evaluator helpers.

If you only need :class:`OGBEvaluatorWrapper`, prefer the import from
:mod:`tgraphx.datasets.ogb_wrappers`.  This module re-exports it for
ergonomic ``from tgraphx.metrics import OGBEvaluatorWrapper``.
"""
from __future__ import annotations

from ..datasets.ogb_wrappers import OGBEvaluatorWrapper

__all__ = ["OGBEvaluatorWrapper"]
