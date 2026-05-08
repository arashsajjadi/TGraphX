"""Distributed-helper tests (v0.2.6).

These verify the TGraphX-side helpers work safely in single-process
(non-distributed) environments.  No actual distributed init is performed
in CI.
"""
from __future__ import annotations

import torch

from tgraphx.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_distributed_available_and_initialized,
    is_rank_zero,
    rank_zero_only,
    rank_zero_print,
)


class TestSingleProcess:
    def test_not_initialized_by_default(self):
        # In a fresh interpreter with no DDP, distributed is not initialized.
        assert is_distributed_available_and_initialized() is False

    def test_get_rank_default(self):
        assert get_rank() == 0
        assert get_rank(default=42) == 42  # default returned when not init'd

    def test_get_world_size_default(self):
        assert get_world_size() == 1

    def test_is_rank_zero(self):
        assert is_rank_zero() is True

    def test_rank_zero_print_no_crash(self):
        rank_zero_print("hello from rank-zero helper")

    def test_rank_zero_only_decorator_executes_in_single_process(self):
        @rank_zero_only
        def f(x):
            return x + 1
        assert f(5) == 6

    def test_barrier_is_noop_when_not_initialized(self):
        # Must not raise.
        barrier()


class TestImportSmoke:
    def test_no_torch_distributed_init_at_import(self):
        # Importing tgraphx.distributed must not call init_process_group.
        import tgraphx.distributed  # noqa: F401
        # Cheap check: distributed is not initialized after import.
        if hasattr(torch, "distributed") and torch.distributed.is_available():
            assert not torch.distributed.is_initialized()
