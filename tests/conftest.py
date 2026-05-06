"""Shared pytest configuration for TGraphX.

Registers custom markers and auto-skips device-specific tests when the
required hardware is not present.  No fixtures are defined here — helpers
live in each test module to keep each file self-contained.
"""

import pytest
import torch


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: test requires a CUDA GPU")
    config.addinivalue_line("markers", "mps: test requires Apple Silicon MPS")


def pytest_collection_modifyitems(items: list) -> None:
    skip_cuda = pytest.mark.skip(reason="CUDA not available on this machine")
    skip_mps  = pytest.mark.skip(reason="Apple MPS not available on this machine")
    _mps_ok = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_cuda)
        if "mps" in item.keywords and not _mps_ok:
            item.add_marker(skip_mps)
