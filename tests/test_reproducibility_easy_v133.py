"""Reproducibility tests for Easy Mode (v1.3.3).

Tests that:
1. Synthetic data creation is seeded deterministically.
2. NeighborLoader with same seed produces identical batches.
3. CPU + deterministic=True gives exact same loss across two runs.
4. Reproducibility state is recorded in result.config.
5. CUDA (if available) at least produces finite results in deterministic mode.
"""
from __future__ import annotations

import pytest
import torch
import tgraphx as tgx
from tgraphx.reproducibility import set_seed


# ── 1. Synthetic data reproducibility ────────────────────────────────────────


class TestSyntheticDataReproducibility:
    def test_same_seed_same_features(self):
        d1 = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=3, num_edges=100, seed=42,
        )
        d2 = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=3, num_edges=100, seed=42,
        )
        assert torch.equal(d1.node_features, d2.node_features)
        assert torch.equal(d1.edge_index, d2.edge_index)
        assert torch.equal(d1.node_labels, d2.node_labels)

    def test_different_seed_different_output(self):
        d1 = tgx.easy.synthetic_tensor_node_classification(num_nodes=50, seed=0)
        d2 = tgx.easy.synthetic_tensor_node_classification(num_nodes=50, seed=99)
        # With different seeds the features should differ (with overwhelmingly high probability).
        assert not torch.equal(d1.node_features, d2.node_features)


# ── 2. NeighborLoader batch order reproducibility ────────────────────────────


class TestNeighborLoaderReproducibility:
    def test_same_seed_same_first_batch_seed_nodes(self):
        from tgraphx import Graph, NeighborLoader
        x = torch.randn(100, 8)
        ei = torch.randint(0, 100, (2, 400))
        y = torch.randint(0, 3, (100,))
        g = Graph(node_features=x, edge_index=ei, y=y)

        loader_a = NeighborLoader(g, fanouts=[5, 3], batch_size=8, shuffle=True, seed=7)
        loader_b = NeighborLoader(g, fanouts=[5, 3], batch_size=8, shuffle=True, seed=7)

        seeds_a = [b.seed_node_ids.clone() for b in loader_a]
        seeds_b = [b.seed_node_ids.clone() for b in loader_b]
        assert len(seeds_a) == len(seeds_b)
        for a, b in zip(seeds_a, seeds_b):
            assert torch.equal(a, b)


# ── 3. Easy Mode CPU deterministic reproducibility ───────────────────────────


class TestEasyModeDeterministicCPU:
    def _run(self):
        set_seed(42, deterministic=True)
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=3, num_edges=100, seed=42,
        )
        return tgx.easy.train_node_classifier(
            data, epochs=2, batch_size=8, fanouts=[4, 2],
            verbose=False, seed=42, deterministic=True, device="cpu",
        )

    def test_deterministic_cpu_exact_match(self):
        r1 = self._run()
        r2 = self._run()
        diff = abs(r1.metrics["loss"] - r2.metrics["loss"])
        assert diff < 1e-7, f"CPU deterministic diff too large: {diff:.2e}"

    def test_deterministic_flag_in_config(self):
        r = self._run()
        assert r.config.get("deterministic") is True

    def test_reproducibility_state_recorded(self):
        r = self._run()
        state = r.config.get("reproducibility_state", {})
        assert state.get("seed") == 42
        assert state.get("deterministic") is True
        assert "torch_version" in state
        assert "cuda_available" in state

    def test_default_non_deterministic_still_runs(self):
        """Default deterministic=False must not crash."""
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=2, num_edges=100, seed=0,
        )
        r = tgx.easy.train_node_classifier(
            data, epochs=1, batch_size=8, fanouts=[3, 2],
            verbose=False, seed=0, device="cpu",
        )
        assert "loss" in r.metrics
        assert r.config.get("deterministic") is False


# ── 4. set_seed return value ──────────────────────────────────────────────────


class TestSetSeedReturnValue:
    def test_returns_dict_with_required_keys(self):
        state = set_seed(42, deterministic=True)
        assert isinstance(state, dict)
        assert state["seed"] == 42
        assert state["deterministic"] is True
        assert "torch_version" in state
        assert "cuda_available" in state

    def test_non_deterministic_mode(self):
        state = set_seed(7, deterministic=False)
        assert state["deterministic"] is False


# ── 5. CUDA smoke (if available) ──────────────────────────────────────────────


class TestEasyModeDeviceCUDA:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_deterministic_cuda_finite_loss(self):
        set_seed(42, deterministic=True)
        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=32, node_shape=(4, 4, 4), num_classes=3, num_edges=100, seed=42,
        )
        r = tgx.easy.train_node_classifier(
            data, epochs=1, batch_size=8, fanouts=[4, 2],
            verbose=False, seed=42, deterministic=True, device="cuda",
        )
        assert torch.isfinite(torch.tensor(r.metrics["loss"]))
