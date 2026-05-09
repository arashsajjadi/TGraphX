"""Tests for OGB/TGB evaluator wrappers.

Validates:
  - importing tgraphx.benchmarks does NOT import ogb or tgb
  - missing dependency raises OptionalDependencyError cleanly
  - is_available flag correctly reflects install state
"""
from __future__ import annotations

import sys

import pytest


def test_import_benchmarks_does_not_pull_ogb_tgb():
    """Importing tgraphx.benchmarks must not transitively import ogb or tgb."""
    # Remove stale module refs if present.
    for mod in list(sys.modules.keys()):
        if mod.startswith("ogb") or mod.startswith("tgb"):
            del sys.modules[mod]
    import importlib
    # Force fresh import.
    if "tgraphx.benchmarks" in sys.modules:
        del sys.modules["tgraphx.benchmarks"]
    if "tgraphx.benchmarks.public" in sys.modules:
        del sys.modules["tgraphx.benchmarks.public"]
    importlib.import_module("tgraphx.benchmarks")
    # OGB and TGB should NOT have been imported.
    assert "ogb" not in sys.modules, "ogb was unexpectedly imported at benchmarks import time"
    assert "tgb" not in sys.modules, "tgb was unexpectedly imported at benchmarks import time"


def test_ogb_evaluators_is_available_flag():
    from tgraphx.benchmarks import OGBNodeEvaluator, OGBLinkEvaluator, OGBGraphEvaluator
    try:
        import ogb  # noqa: F401
        ogb_installed = True
    except ImportError:
        ogb_installed = False
    assert OGBNodeEvaluator.is_available == ogb_installed
    assert OGBLinkEvaluator.is_available == ogb_installed
    assert OGBGraphEvaluator.is_available == ogb_installed


def test_tgb_evaluator_is_available_flag():
    from tgraphx.benchmarks import TGBLinkEvaluator
    try:
        import tgb  # noqa: F401
        tgb_installed = True
    except ImportError:
        tgb_installed = False
    assert TGBLinkEvaluator.is_available == tgb_installed


def test_ogb_evaluator_raises_when_missing():
    """When ogb is not installed, instantiation raises OptionalDependencyError."""
    try:
        import ogb  # noqa: F401
        pytest.skip("ogb is installed; skipping missing-dep test")
    except ImportError:
        pass
    from tgraphx.benchmarks import OGBNodeEvaluator
    from tgraphx.datasets.errors import OptionalDependencyError
    with pytest.raises(OptionalDependencyError):
        OGBNodeEvaluator("ogbn-arxiv")


def test_tgb_evaluator_raises_when_missing():
    """When tgb is not installed, instantiation raises OptionalDependencyError."""
    try:
        import tgb  # noqa: F401
        pytest.skip("tgb is installed; skipping missing-dep test")
    except ImportError:
        pass
    from tgraphx.benchmarks import TGBLinkEvaluator
    from tgraphx.datasets.errors import OptionalDependencyError
    with pytest.raises(OptionalDependencyError):
        TGBLinkEvaluator(name="tgbl-wiki-v2")
