"""Tests for KG negative samplers."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    generate_synthetic_kg,
    UniformNegativeSampler,
    BernoulliNegativeSampler,
    FilteredNegativeSampler,
    TypedNegativeSampler,
)


def _pos_triples():
    return torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 0]], dtype=torch.long)


class TestUniformNegativeSampler:

    def test_output_shape(self):
        s = UniformNegativeSampler(num_entities=10, num_negatives=3)
        neg = s.sample(_pos_triples())
        assert neg.shape == (3, 3, 3)  # [B, K, 3]

    def test_relation_ids_preserved(self):
        s = UniformNegativeSampler(num_entities=20, num_negatives=2)
        pos = _pos_triples()
        neg = s.sample(pos)
        for i in range(pos.size(0)):
            assert torch.equal(neg[i, :, 1], pos[i:i+1, 1].expand(2))

    def test_deterministic_with_seed(self):
        s = UniformNegativeSampler(10, 2)
        gen = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        a = s.sample(_pos_triples(), generator=gen)
        b = s.sample(_pos_triples(), generator=gen2)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        s = UniformNegativeSampler(20, 2)
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        a = s.sample(_pos_triples(), generator=g1)
        b = s.sample(_pos_triples(), generator=g2)
        assert not torch.equal(a, b)

    def test_all_head_corruption(self):
        s = UniformNegativeSampler(10, 1, corrupt_head_prob=1.0)
        pos = _pos_triples()
        neg = s.sample(pos)
        # Tails must be unchanged.
        assert torch.equal(neg[:, 0, 2], pos[:, 2])

    def test_all_tail_corruption(self):
        s = UniformNegativeSampler(10, 1, corrupt_head_prob=0.0)
        pos = _pos_triples()
        neg = s.sample(pos)
        # Heads must be unchanged.
        assert torch.equal(neg[:, 0, 0], pos[:, 0])

    def test_entities_in_range(self):
        N = 15
        s = UniformNegativeSampler(N, 5)
        neg = s.sample(_pos_triples())
        assert int(neg[:, :, 0].min()) >= 0
        assert int(neg[:, :, 0].max()) < N
        assert int(neg[:, :, 2].min()) >= 0
        assert int(neg[:, :, 2].max()) < N


class TestBernoulliNegativeSampler:

    def test_output_shape(self):
        kg = generate_synthetic_kg(10, 3, 20, seed=0)
        s = BernoulliNegativeSampler(10, 2, train_triples=kg.triples)
        neg = s.sample(kg.triples[:4])
        assert neg.shape == (4, 2, 3)

    def test_probabilities_in_range(self):
        kg = generate_synthetic_kg(10, 3, 20, seed=0)
        probs = BernoulliNegativeSampler._estimate_bernoulli_probs(kg.triples)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_deterministic_with_seed(self):
        kg = generate_synthetic_kg(15, 3, 30, seed=0)
        s = BernoulliNegativeSampler(15, 1, train_triples=kg.triples)
        g1 = torch.Generator().manual_seed(99)
        g2 = torch.Generator().manual_seed(99)
        a = s.sample(kg.triples[:3], generator=g1)
        b = s.sample(kg.triples[:3], generator=g2)
        assert torch.equal(a, b)


class TestFilteredNegativeSampler:

    def test_no_known_positives_in_negatives(self):
        kg = generate_synthetic_kg(15, 3, 30, seed=0)
        pos_set = kg.positive_triple_set()
        base = UniformNegativeSampler(15, 1)
        s = FilteredNegativeSampler(15, 1, positive_set=pos_set, base_sampler=base)
        pos = kg.triples[:6]
        neg = s.sample(pos)
        for row in neg.view(-1, 3).tolist():
            triple = (int(row[0]), int(row[1]), int(row[2]))
            assert triple not in pos_set, f"Known positive appeared as negative: {triple}"

    def test_empty_positive_set_no_warning(self):
        # Empty positive set: every candidate is valid → no rejection → no warning.
        base = UniformNegativeSampler(10, 1)
        s = FilteredNegativeSampler(10, 1, positive_set=set(), base_sampler=base)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            neg = s.sample(torch.tensor([[0, 0, 1]], dtype=torch.long))
        assert neg.shape == (1, 1, 3)

    def test_max_attempts_respected(self):
        # Create an artificial scenario where every sample hits the positive set.
        # We do this by making a FilteredNegativeSampler with max_attempts=1,
        # and a positive set that includes every possible triple, then check it returns.
        # (In practice this falls back with a warning.)
        all_pos = {(0, 0, e) for e in range(10)} | {(e, 0, 1) for e in range(10)}
        base = UniformNegativeSampler(2, 1)  # only 2 entities, all combos in pos
        s = FilteredNegativeSampler(2, 1, positive_set=all_pos, base_sampler=base,
                                    max_attempts=2)
        # Should run without raising; may warn.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neg = s.sample(torch.tensor([[0, 0, 1]], dtype=torch.long))
        assert neg.shape == (1, 1, 3)


class TestTypedNegativeSampler:

    def test_domain_constraint_respected(self):
        N = 10
        # Entity types: 0-4 type A, 5-9 type B.
        entity_types = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        # Relation 0 domain: type A entities only.
        domains = {0: set(range(5))}
        ranges = {0: set(range(5, 10))}
        s = TypedNegativeSampler(N, 5, entity_types=entity_types,
                                 domains=domains, ranges=ranges, corrupt_head_prob=1.0)
        pos = torch.tensor([[0, 0, 5], [1, 0, 6]], dtype=torch.long)
        neg = s.sample(pos)
        # Head corruptions must be in domain (0-4).
        for nh in neg[:, :, 0].view(-1).tolist():
            assert 0 <= nh < 5, f"Head {nh} not in domain"
