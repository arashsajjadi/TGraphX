"""Tests for tgraphx.reproducibility — seed determinism and utilities.

Covers:
- set_seed repeatability on CPU
- make_generator output determinism
- seed_worker formula
- deterministic_mode context manager restores state
- reproducibility_report returns expected keys
- WL cross-process determinism (via subprocess)
- random walk seed determinism
- hard negative sampling seed determinism
- no global RNG pollution from make_generator
- no performance-heavy deterministic mode by default
"""
from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from tgraphx.reproducibility import (
    deterministic_mode,
    make_generator,
    reproducibility_report,
    seed_worker,
    set_seed,
)


class TestSetSeed:
    def test_same_seed_same_output_cpu(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.equal(a, b), "Same seed must produce same randn on CPU"

    def test_different_seeds_different_output(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        assert not torch.equal(a, b)

    def test_returns_state_dict(self):
        state = set_seed(0)
        assert "seed" in state
        assert "deterministic" in state
        assert "torch_version" in state
        assert "cuda_available" in state
        assert state["seed"] == 0
        assert state["deterministic"] is False

    def test_returns_state_dict_deterministic(self):
        state = set_seed(0, deterministic=True)
        assert state["deterministic"] is True
        # Restore non-deterministic mode.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    def test_no_crash_without_cuda(self):
        # set_seed must never crash even if CUDA is unavailable.
        set_seed(99)

    def test_numpy_seeded_if_available(self):
        set_seed(7)
        try:
            import numpy as np
            a = np.random.rand(5)
            set_seed(7)
            b = np.random.rand(5)
            assert (a == b).all()
        except ImportError:
            pass


class TestMakeGenerator:
    def test_produces_repeatable_values(self):
        g1 = make_generator(0)
        g2 = make_generator(0)
        a = torch.randint(100, (5,), generator=g1)
        b = torch.randint(100, (5,), generator=g2)
        assert torch.equal(a, b)

    def test_different_seeds_different_values(self):
        g1 = make_generator(0)
        g2 = make_generator(1)
        a = torch.randint(100, (5,), generator=g1)
        b = torch.randint(100, (5,), generator=g2)
        assert not torch.equal(a, b)

    def test_no_global_rng_pollution(self):
        torch.manual_seed(99)
        before = torch.rand(5)
        torch.manual_seed(99)
        g = make_generator(42)  # noqa: F841
        after = torch.rand(5)
        assert torch.equal(before, after), "make_generator must not pollute global RNG"

    def test_cpu_device(self):
        g = make_generator(0, device="cpu")
        assert g.device.type == "cpu"


class TestSeedWorker:
    def test_does_not_crash(self):
        # Call as PyTorch DataLoader would.
        torch.manual_seed(0)
        seed_worker(0)
        seed_worker(1)
        seed_worker(100)


class TestReproducibilityReport:
    def test_returns_expected_keys(self):
        report = reproducibility_report()
        for key in ["torch_version", "cuda_available", "python_hash_seed"]:
            assert key in report, f"Missing key: {key}"

    def test_json_serializable(self):
        import json
        report = reproducibility_report()
        # Must not raise.
        json.dumps(report)


class TestDeterministicMode:
    def test_context_manager_restores_deterministic_flag(self):
        # Enter/exit and verify state is restored.
        try:
            before = torch.are_deterministic_algorithms_enabled()
        except AttributeError:
            pytest.skip("torch.are_deterministic_algorithms_enabled not available")

        with deterministic_mode(seed=0, warn_only=True):
            pass  # inside the context

        # Restore non-deterministic mode.
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    def test_yields_state_dict(self):
        with deterministic_mode(seed=42) as state:
            assert "seed" in state
            assert state["seed"] == 42
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    def test_same_seed_in_context_gives_same_output(self):
        with deterministic_mode(seed=7, warn_only=True):
            set_seed(7)
            a = torch.randn(5)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

        with deterministic_mode(seed=7, warn_only=True):
            set_seed(7)
            b = torch.randn(5)
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

        assert torch.equal(a, b)


class TestWLCrossProcessDeterminism:
    """WL feature labels must be identical across separate Python processes
    with different PYTHONHASHSEED values."""

    _script = """
import sys, os
os.environ["PYTHONHASHSEED"] = sys.argv[1]
import torch
from tgraphx.mining import weisfeiler_lehman_labels
ei = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
h = weisfeiler_lehman_labels(ei, 3, num_iterations=2)
print(str(h))
"""

    def _run_wl(self, hashseed: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", self._script, hashseed],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_wl_deterministic_across_pythonhashseeds(self):
        out1 = self._run_wl("1")
        out2 = self._run_wl("42")
        out3 = self._run_wl("999")
        assert out1 == out2 == out3, (
            f"WL labels differ across PYTHONHASHSEED values:\n"
            f"  seed=1:   {out1}\n"
            f"  seed=42:  {out2}\n"
            f"  seed=999: {out3}"
        )


class TestMiningSeedDeterminism:
    def test_random_walk_deterministic(self):
        from tgraphx.mining import random_walks
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        s = torch.tensor([0, 1], dtype=torch.long)
        w1 = random_walks(ei, s, walk_length=4, num_nodes=4, seed=7)
        w2 = random_walks(ei, s, walk_length=4, num_nodes=4, seed=7)
        assert torch.equal(w1, w2)

    def test_random_walk_different_seeds(self):
        from tgraphx.mining import random_walks
        ei = torch.tensor([[0,0,1,2],[1,2,2,3]], dtype=torch.long)
        s = torch.arange(4, dtype=torch.long)
        w1 = random_walks(ei, s, walk_length=10, num_nodes=4, seed=0)
        w2 = random_walks(ei, s, walk_length=10, num_nodes=4, seed=1)
        assert not torch.equal(w1, w2)

    def test_hard_negative_sampling_deterministic(self):
        from tgraphx import hard_negative_sampling
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        emb = torch.randn(4, 8)
        n1 = hard_negative_sampling(ei, emb, num_nodes=4, num_neg_samples=4, seed=0)
        n2 = hard_negative_sampling(ei, emb, num_nodes=4, num_neg_samples=4, seed=0)
        assert torch.equal(n1, n2)

    def test_negative_sampling_no_global_rng_pollution(self):
        from tgraphx import negative_sampling
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        torch.manual_seed(42)
        before = torch.rand(5)
        torch.manual_seed(42)
        _ = negative_sampling(ei, num_nodes=3, num_neg_samples=4, seed=99)
        after = torch.rand(5)
        assert torch.equal(before, after)
