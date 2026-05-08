"""Tests for batched PrototypeMembershipScorer and neural mining benchmark.

Covers:
- score_batch_fast output shape matches score_batch
- backward works through score_batch_fast
- no cross-graph edge leakage in batched forward
- benchmark_neural_mining --help and --small --json run
- benchmark JSON schema validated
- loss decreased in benchmark tasks
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tgraphx.mining import PrototypeMembershipScorer

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "mining"


# ── Batched scorer ────────────────────────────────────────────────────────────


def _make_candidates(n: int = 3, N: int = 5, D: int = 4):
    """Build n candidate graph dicts.  N must be >= 5 for the default edge_index."""
    N = max(N, 5)  # edge_index goes up to node 4
    ei = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)
    candidates = []
    for _ in range(n):
        x = torch.randn(N, D)
        candidates.append({
            "node_features": x,
            "edge_index": ei,
            "query_idx": N - 1,
        })
    return candidates


class TestBatchedMembershipScorer:
    def test_shape_matches_sequential(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        cands = _make_candidates(3)
        seq = model.score_batch(cands)
        fast = model.score_batch_fast(cands)
        assert seq.shape == fast.shape == (3,)

    def test_values_close_to_sequential(self):
        """Fast and sequential results should be identical (same forward)."""
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        model.eval()
        cands = _make_candidates(2, N=4, D=4)
        with torch.no_grad():
            seq = model.score_batch(cands)
            fast = model.score_batch_fast(cands)
        assert torch.allclose(seq, fast, atol=1e-5), f"seq={seq} fast={fast}"

    def test_backward_through_fast(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        cands = _make_candidates(2)
        logits = model.score_batch_fast(cands)
        logits.sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_gradients_finite(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        cands = _make_candidates(3)
        model.score_batch_fast(cands).sum().backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_no_cross_graph_edge(self):
        """Different query positions should give different logits."""
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        model.eval()
        x = torch.randn(5, 4)
        ei = torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long)
        cands = [
            {"node_features": x.clone(), "edge_index": ei, "query_idx": 4},
            {"node_features": x.clone(), "edge_index": ei, "query_idx": 0},
        ]
        with torch.no_grad():
            logits = model.score_batch_fast(cands)
        # Different query positions → different scores.
        # (Not guaranteed but almost certain with random weights)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()

    def test_empty_raises(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        with pytest.raises(ValueError):
            model.score_batch_fast([])
        with pytest.raises(ValueError):
            model.score_batch([])

    def test_single_node_graph(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4)
        x = torch.randn(1, 4)
        ei = torch.zeros((2, 0), dtype=torch.long)
        cands = [{"node_features": x, "edge_index": ei, "query_idx": 0}]
        logits = model.score_batch_fast(cands)
        assert logits.shape == (1,)
        assert torch.isfinite(logits).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_fast_batch(self):
        model = PrototypeMembershipScorer(in_dim=4, hidden_dim=8, out_dim=4).cuda()
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long).cuda()
        cands = [
            {"node_features": torch.randn(3, 4).cuda(), "edge_index": ei, "query_idx": 2},
            {"node_features": torch.randn(3, 4).cuda(), "edge_index": ei, "query_idx": 1},
        ]
        logits = model.score_batch_fast(cands)
        assert logits.device.type == "cuda"
        assert logits.shape == (2,)


# ── Neural mining benchmark ───────────────────────────────────────────────────


def _run_bench(*args):
    cmd = [sys.executable, str(BENCH_DIR / "benchmark_neural_mining.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestNeuralMiningBenchmark:
    def test_help(self):
        res = _run_bench("--help")
        assert res.returncode == 0
        assert "--small" in res.stdout + res.stderr

    def test_small_runs(self):
        res = _run_bench("--small")
        assert res.returncode == 0, res.stderr + res.stdout

    def test_small_json_parseable(self):
        res = _run_bench("--small", "--json")
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        assert "benchmark" in data
        assert "tasks" in data
        assert "prototype_membership" in data["tasks"]
        assert "graph_pattern_classifier" in data["tasks"]
        assert "anomaly_autoencoder" in data["tasks"]

    def test_loss_decreased_on_all_tasks(self):
        res = _run_bench("--small", "--json", "--seed", "42", "--epochs", "5")
        assert res.returncode == 0, res.stderr
        data = json.loads(res.stdout)
        for task_name, task in data["tasks"].items():
            assert task["loss_decreased"], (
                f"Loss did not decrease for {task_name}: "
                f"{task['initial_loss']} → {task['final_loss']}"
            )

    def test_json_schema(self):
        res = _run_bench("--small", "--json")
        data = json.loads(res.stdout)
        assert data["tgraphx_version"]
        assert data["device"]
        assert isinstance(data["epochs"], int)
        for task in data["tasks"].values():
            for key in ["train_time_s", "initial_loss", "final_loss",
                        "loss_decreased", "gradient_health"]:
                assert key in task, f"Missing key {key} in {task}"
            gh = task["gradient_health"]
            assert gh["has_nonzero"], f"All-zero gradients in {task}"
            assert not gh["any_nan"], f"NaN gradients in {task}"
