"""Tests for the sparse backend registry."""
from __future__ import annotations

import torch

from tgraphx.sparse import (
    backend_info, select_backend, active_backend,
    segment_sum, segment_mean,
)


def test_backend_info_has_pure_torch():
    info = backend_info()
    assert info["pure_torch"] is True
    assert "torch_scatter" in info
    assert "active" in info


def test_select_backend_fallback():
    chosen = select_backend("torch_scatter")
    # Either torch_scatter or pure_torch (with warning).
    assert chosen in ("torch_scatter", "pure_torch")
    select_backend("pure_torch")
    assert active_backend() == "pure_torch"


def test_segment_sum_pure_torch():
    select_backend("pure_torch")
    src = torch.arange(6, dtype=torch.float)
    idx = torch.tensor([0, 0, 1, 1, 2, 2])
    out = segment_sum(src.unsqueeze(-1), idx, num_segments=3).squeeze(-1)
    expected = torch.tensor([1.0, 5.0, 9.0])
    assert torch.allclose(out, expected)


def test_segment_mean_pure_torch():
    src = torch.arange(6, dtype=torch.float)
    idx = torch.tensor([0, 0, 1, 1, 2, 2])
    out = segment_mean(src.unsqueeze(-1), idx, num_segments=3).squeeze(-1)
    expected = torch.tensor([0.5, 2.5, 4.5])
    assert torch.allclose(out, expected)


def test_invalid_backend_name_raises():
    try:
        select_backend("not_real")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
