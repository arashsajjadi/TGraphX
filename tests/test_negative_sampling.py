"""Tests for tgraphx.sampling_negative — v0.3.2 beta primitives.

Invariants tested
-----------------
- No false negatives (a "negative" is never a positive edge).
- No self-loops in the output.
- No duplicates within the output.
- ``force_undirected`` blocks both directions.
- Determinism with ``seed``.
- ``num_neg_samples`` semantics.
- Empty graph and degenerate inputs.
- Batched sampling never crosses graph boundaries.
- Structured sampling: ``(i, k)`` is never a positive edge.
"""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    negative_sampling,
    structured_negative_sampling,
    batched_negative_sampling,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_set(edge_index: torch.Tensor) -> set:
    return {(int(edge_index[0, i]), int(edge_index[1, i]))
            for i in range(edge_index.size(1))}


def _has_self_loops(edge_index: torch.Tensor) -> bool:
    if edge_index.numel() == 0:
        return False
    return bool((edge_index[0] == edge_index[1]).any().item())


# ── basic negative_sampling ──────────────────────────────────────────────────

class TestBasicNegativeSampling:
    def test_no_false_negatives(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=0)
        pos = _to_set(edge_index)
        neg_set = _to_set(neg)
        assert not (pos & neg_set), f"false negatives: {pos & neg_set}"

    def test_no_self_loops(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=5, num_neg_samples=8, seed=0)
        assert not _has_self_loops(neg)

    def test_no_duplicates(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=5, num_neg_samples=10, seed=0)
        assert len(_to_set(neg)) == neg.size(1)

    def test_default_count_matches_positives(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=5, seed=0)  # default
        # The function targets ``E`` negatives; for sparse graphs it usually
        # hits the target, but we accept up to E because dense graphs could
        # short us a few.
        assert 0 < neg.size(1) <= 4

    def test_explicit_zero(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=3, num_neg_samples=0, seed=0)
        assert neg.shape == (2, 0)

    def test_empty_edge_index(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=0)
        assert neg.size(0) == 2
        assert neg.size(1) <= 4
        assert not _has_self_loops(neg)

    def test_returned_dtype_and_device(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=3, num_neg_samples=2, seed=0)
        assert neg.dtype == torch.long
        assert neg.device == edge_index.device


class TestDeterminism:
    def test_same_seed_same_output(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        a = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=42)
        b = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=42)
        assert torch.equal(a, b)

    def test_different_seed_different_output(self):
        edge_index = torch.tensor(
            [[i for i in range(20)], [(i + 1) % 20 for i in range(20)]],
            dtype=torch.long,
        )
        a = negative_sampling(edge_index, num_nodes=20, num_neg_samples=10, seed=1)
        b = negative_sampling(edge_index, num_nodes=20, num_neg_samples=10, seed=2)
        assert not torch.equal(a, b)

    def test_no_global_rng_pollution(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        torch.manual_seed(99)
        before = torch.rand(3)
        torch.manual_seed(99)
        _ = negative_sampling(edge_index, num_nodes=5, num_neg_samples=4, seed=42)
        after = torch.rand(3)
        assert torch.equal(before, after), "negative_sampling polluted global RNG"


class TestForceUndirected:
    def test_excludes_reverse_positive(self):
        # Tightly packed undirected pair (0, 1) and (1, 0).
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        neg = negative_sampling(
            edge_index, num_nodes=4, num_neg_samples=10,
            force_undirected=True, seed=0,
        )
        # Neither (0,1) nor (1,0) may appear in negatives.
        s = _to_set(neg)
        assert (0, 1) not in s and (1, 0) not in s

    def test_no_reverse_within_output(self):
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        neg = negative_sampling(
            edge_index, num_nodes=6, num_neg_samples=12,
            force_undirected=True, seed=0,
        )
        s = _to_set(neg)
        # Output should not contain both (u,v) and (v,u).
        for (u, v) in s:
            assert (v, u) not in s


class TestDenseMethod:
    def test_dense_no_false_negatives(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        neg = negative_sampling(
            edge_index, num_nodes=4, num_neg_samples=4,
            method="dense", seed=0,
        )
        assert not (_to_set(edge_index) & _to_set(neg))

    def test_dense_rejects_huge_n(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="O\\(N\\^2\\)"):
            negative_sampling(
                edge_index, num_nodes=10_000, num_neg_samples=10,
                method="dense",
            )

    def test_invalid_method(self):
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        with pytest.raises(ValueError, match="method"):
            negative_sampling(edge_index, num_nodes=2, method="quantum")


class TestValidation:
    def test_bad_edge_index_shape(self):
        with pytest.raises(ValueError, match="\\[2, E\\]"):
            negative_sampling(torch.zeros(3, 4, dtype=torch.long), num_nodes=4)

    def test_edge_id_out_of_range(self):
        edge_index = torch.tensor([[0, 5], [1, 2]], dtype=torch.long)
        with pytest.raises(ValueError, match="num_nodes"):
            negative_sampling(edge_index, num_nodes=4, num_neg_samples=2)

    def test_zero_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes"):
            negative_sampling(torch.zeros(2, 0, dtype=torch.long), num_nodes=0)


# ── structured_negative_sampling ─────────────────────────────────────────────


class TestStructuredNegativeSampling:
    def test_shapes(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        i, j, k = structured_negative_sampling(edge_index, num_nodes=5, seed=0)
        assert i.shape == j.shape == k.shape == (4,)

    def test_aligned_with_positives(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        i, j, k = structured_negative_sampling(edge_index, num_nodes=5, seed=0)
        # i, j must equal the positive edge endpoints in order.
        assert torch.equal(i, edge_index[0])
        assert torch.equal(j, edge_index[1])

    def test_k_not_a_positive_edge(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        pos = _to_set(edge_index)
        i, j, k = structured_negative_sampling(edge_index, num_nodes=5, seed=0)
        for col in range(len(i)):
            assert (int(i[col]), int(k[col])) not in pos

    def test_no_self_loop_when_disallowed(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        i, _, k = structured_negative_sampling(
            edge_index, num_nodes=4, contains_neg_self_loops=False, seed=0,
        )
        for col in range(len(i)):
            assert int(i[col]) != int(k[col])

    def test_empty(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        i, j, k = structured_negative_sampling(edge_index, num_nodes=4)
        assert i.numel() == 0 and j.numel() == 0 and k.numel() == 0


# ── batched_negative_sampling ────────────────────────────────────────────────


class TestBatchedNegativeSampling:
    def test_no_cross_graph_negatives(self):
        # Two graphs: graph 0 has nodes {0,1,2}; graph 1 has nodes {3,4,5}.
        edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        neg = batched_negative_sampling(
            edge_index, batch, num_neg_samples=4, seed=0,
        )
        for c in range(neg.size(1)):
            assert int(batch[neg[0, c]]) == int(batch[neg[1, c]])

    def test_no_false_negatives_per_graph(self):
        edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        pos = _to_set(edge_index)
        neg = batched_negative_sampling(
            edge_index, batch, num_neg_samples=4, seed=0,
        )
        assert not (pos & _to_set(neg))

    def test_determinism(self):
        edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        a = batched_negative_sampling(edge_index, batch, num_neg_samples=4, seed=7)
        b = batched_negative_sampling(edge_index, batch, num_neg_samples=4, seed=7)
        assert torch.equal(a, b)

    def test_empty_graph_batch(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        batch = torch.zeros((0,), dtype=torch.long)
        neg = batched_negative_sampling(edge_index, batch, num_neg_samples=2, seed=0)
        assert neg.shape == (2, 0)


# ── Edge-case / degenerate inputs ────────────────────────────────────────────


class TestDegenerateCases:
    def test_complete_directed_graph_has_no_negatives(self):
        # All 6 directed non-self-loop edges on 3 nodes.
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long,
        )
        neg = negative_sampling(edge_index, num_nodes=3, num_neg_samples=4, seed=0)
        # Only self-loops could be negatives but they're also excluded.
        assert neg.shape[0] == 2
        assert neg.shape[1] == 0

    def test_single_node_no_negatives(self):
        neg = negative_sampling(
            torch.zeros((2, 0), dtype=torch.long), num_nodes=1,
            num_neg_samples=10, seed=0,
        )
        assert neg.shape == (2, 0)

    def test_two_nodes_one_edge_max_one_negative(self):
        # Only possible negative in a 2-node directed graph with edge 0→1 is 1→0.
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        neg = negative_sampling(edge_index, num_nodes=2, num_neg_samples=10, seed=0)
        assert neg.shape == (2, 1)
        assert neg[0, 0].item() == 1 and neg[1, 0].item() == 0

    def test_all_but_one_edge(self):
        # 3 nodes: all directed edges except 2→0.
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2], [1, 2, 0, 2, 1]], dtype=torch.long,
        )
        neg = negative_sampling(edge_index, num_nodes=3, num_neg_samples=5, seed=0)
        assert neg.shape == (2, 1)
        assert neg[0, 0].item() == 2 and neg[1, 0].item() == 0


# ── CUDA device tests ────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDADevice:
    def test_negative_sampling_output_on_cuda(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).cuda()
        neg = negative_sampling(ei, num_nodes=5, num_neg_samples=4, seed=0)
        assert neg.device.type == "cuda"
        assert neg.dtype == torch.long

    def test_structured_negative_sampling_output_on_cuda(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).cuda()
        i, j, k = structured_negative_sampling(ei, num_nodes=4, seed=0)
        assert i.device.type == "cuda"
        assert j.device.type == "cuda"
        assert k.device.type == "cuda"

    def test_batched_negative_sampling_output_on_cuda(self):
        ei = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long).cuda()
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long).cuda()
        neg = batched_negative_sampling(ei, batch, num_neg_samples=2, seed=0)
        assert neg.device.type == "cuda"
