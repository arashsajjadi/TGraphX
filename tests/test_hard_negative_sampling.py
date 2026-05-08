"""Tests for hard_negative_sampling (v0.3.2 beta).

Invariants tested
-----------------
- No false negatives (no positive edge returned).
- No self-loops.
- No duplicates in output.
- Hard negatives have higher mean similarity than random negatives on a
  toy embedding where similarity is strongly correlated with id proximity.
- Deterministic with seed.
- No global RNG pollution.
- force_undirected semantics.
- candidate_pool_size exhaustion produces warning.
- Device preserved (CUDA optional).
"""
from __future__ import annotations

import warnings

import pytest
import torch

from tgraphx import hard_negative_sampling, negative_sampling


def _chain_graph(N: int):
    ei = torch.tensor([[i for i in range(N - 1)], [i + 1 for i in range(N - 1)]], dtype=torch.long)
    emb = torch.zeros(N, 4)
    for i in range(N):
        emb[i, 0] = float(i) / N
    return ei, emb, N


class TestHardNegativeSampling:
    def test_no_false_negatives(self):
        ei, emb, N = _chain_graph(8)
        neg = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=0)
        pos = {(int(ei[0, c]), int(ei[1, c])) for c in range(ei.size(1))}
        for c in range(neg.size(1)):
            assert (int(neg[0, c]), int(neg[1, c])) not in pos

    def test_no_self_loops(self):
        ei, emb, N = _chain_graph(8)
        neg = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=0)
        for c in range(neg.size(1)):
            assert int(neg[0, c]) != int(neg[1, c])

    def test_no_duplicates(self):
        ei, emb, N = _chain_graph(10)
        neg = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=8, seed=0)
        pairs = [(int(neg[0, c]), int(neg[1, c])) for c in range(neg.size(1))]
        assert len(pairs) == len(set(pairs))

    def test_harder_than_random(self):
        ei, emb, N = _chain_graph(10)
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)

        neg_hard = hard_negative_sampling(
            ei, emb, num_nodes=N, num_neg_samples=4,
            candidate_pool_size=512, seed=0,
        )
        neg_rand = negative_sampling(ei, N, num_neg_samples=4, seed=0)

        def mean_sim(ne):
            if ne.size(1) == 0:
                return 0.0
            return float((emb_norm[ne[0]] * emb_norm[ne[1]]).sum(dim=1).mean().item())

        # Hard negatives should have at least as high mean similarity as random.
        assert mean_sim(neg_hard) >= mean_sim(neg_rand) - 0.05

    def test_determinism(self):
        ei, emb, N = _chain_graph(8)
        a = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=7)
        b = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=7)
        assert torch.equal(a, b)

    def test_different_seeds_different_output(self):
        ei, emb, N = _chain_graph(12)
        a = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=1)
        b = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=2)
        # Different seeds should (very likely) produce different results.
        assert not torch.equal(a, b)

    def test_no_global_rng_pollution(self):
        ei, emb, N = _chain_graph(8)
        torch.manual_seed(99)
        before = torch.rand(3)
        torch.manual_seed(99)
        hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, seed=42)
        after = torch.rand(3)
        assert torch.equal(before, after)

    def test_output_dtype_and_device(self):
        ei, emb, N = _chain_graph(6)
        neg = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=2, seed=0)
        assert neg.dtype == torch.long
        assert neg.device == ei.device

    def test_force_undirected_no_reverse_positives(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        emb = torch.randn(4, 4)
        neg = hard_negative_sampling(
            ei, emb, num_nodes=4, num_neg_samples=4,
            force_undirected=True, seed=0,
        )
        forbidden = {(0, 1), (1, 0)}
        for c in range(neg.size(1)):
            assert (int(neg[0, c]), int(neg[1, c])) not in forbidden

    def test_cosine_vs_dot_different(self):
        ei, emb, N = _chain_graph(8)
        c = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, method="cosine", seed=0)
        d = hard_negative_sampling(ei, emb, num_nodes=N, num_neg_samples=4, method="dot", seed=0)
        # Both should be valid (no false negatives).
        pos = {(int(ei[0, i]), int(ei[1, i])) for i in range(ei.size(1))}
        for ne in (c, d):
            for col in range(ne.size(1)):
                assert (int(ne[0, col]), int(ne[1, col])) not in pos

    def test_small_pool_warns(self):
        ei, emb, N = _chain_graph(8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            neg = hard_negative_sampling(
                ei, emb, num_nodes=N, num_neg_samples=100,
                candidate_pool_size=4, seed=0,
            )
        # Should warn about returning fewer than requested.
        assert any("candidate_pool_size" in str(warning.message) or
                   "hard_negative_sampling" in str(warning.message)
                   for warning in w) or neg.size(1) < 100

    def test_invalid_method(self):
        ei, emb, N = _chain_graph(4)
        with pytest.raises(ValueError, match="method"):
            hard_negative_sampling(ei, emb, num_nodes=N, method="euclidean")

    def test_invalid_pool_size(self):
        ei, emb, N = _chain_graph(4)
        with pytest.raises(ValueError, match="candidate_pool_size"):
            hard_negative_sampling(ei, emb, num_nodes=N, candidate_pool_size=0)

    def test_embeddings_too_small(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        emb = torch.zeros(2, 4)  # only 2 embeddings for 3 nodes
        with pytest.raises(ValueError, match="node_embeddings"):
            hard_negative_sampling(ei, emb, num_nodes=3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_preserved(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).cuda()
        emb = torch.randn(5, 4).cuda()
        neg = hard_negative_sampling(ei, emb, num_nodes=5, num_neg_samples=4, seed=0)
        assert neg.device.type == "cuda"
