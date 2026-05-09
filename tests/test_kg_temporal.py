"""Tests for temporal KG negative sampling and evaluation."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import generate_synthetic_kg
from tgraphx.kg.data import TemporalKnowledgeGraph
from tgraphx.kg import TemporalKGNegativeSampler
from tgraphx.kg.temporal import evaluate_temporal_filtered_ranking
from tgraphx.kg import DistMultModel


def _temporal_chain_kg(n: int = 10, m: int = 20, seed: int = 0):
    torch.manual_seed(seed)
    triples = torch.tensor([[i % n, i % 3, (i + 1) % n] for i in range(m)], dtype=torch.long)
    timestamps = torch.sort(torch.rand(m))[0] * 100
    return TemporalKnowledgeGraph(triples, timestamps, num_entities=n, num_relations=3)


class TestTemporalKGNegativeSampler:

    def test_output_shape(self):
        tkg = _temporal_chain_kg()
        sampler = TemporalKGNegativeSampler(num_entities=10, num_negatives=2, temporal_kg=tkg)
        pos = tkg.triples[:4]
        ts = tkg.timestamp[:4]
        gen = torch.Generator().manual_seed(0)
        neg = sampler.sample(pos, ts, generator=gen, filtered=False)
        assert neg.shape == (4, 2, 3)

    def test_relation_preserved(self):
        tkg = _temporal_chain_kg()
        sampler = TemporalKGNegativeSampler(num_entities=10, num_negatives=3, temporal_kg=tkg)
        pos = tkg.triples[:3]
        ts = tkg.timestamp[:3]
        gen = torch.Generator().manual_seed(1)
        neg = sampler.sample(pos, ts, generator=gen, filtered=False)
        for i in range(3):
            assert torch.equal(neg[i, :, 1], pos[i:i+1, 1].expand(3))

    def test_no_future_leakage_filtered(self):
        """Filtered negatives must not be in the positive set at or before event time."""
        tkg = _temporal_chain_kg(5, 10, seed=0)
        sampler = TemporalKGNegativeSampler(num_entities=5, num_negatives=1, temporal_kg=tkg)
        pos = tkg.triples[:3]
        ts = tkg.timestamp[:3]
        gen = torch.Generator().manual_seed(0)
        neg = sampler.sample(pos, ts, generator=gen, filtered=True)
        for i in range(3):
            tau = float(ts[i].item())
            for k in range(neg.shape[1]):
                nh, nr, nt = int(neg[i, k, 0]), int(neg[i, k, 1]), int(neg[i, k, 2])
                # Check: (nh, nr, nt) must not exist at or before tau.
                assert not sampler._is_positive_at_or_before(nh, nr, nt, tau), \
                    f"Future leakage: ({nh},{nr},{nt}) found at tau={tau}"

    def test_deterministic(self):
        tkg = _temporal_chain_kg()
        sampler = TemporalKGNegativeSampler(num_entities=10, num_negatives=1, temporal_kg=tkg)
        pos, ts = tkg.triples[:3], tkg.timestamp[:3]
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        a = sampler.sample(pos, ts, generator=g1, filtered=False)
        b = sampler.sample(pos, ts, generator=g2, filtered=False)
        assert torch.equal(a, b)


class TestTemporalFilteredEval:

    def test_temporal_eval_runs(self):
        tkg = _temporal_chain_kg(8, 20, seed=0)
        tr, va, te = tkg.chronological_split(0.6, 0.2, 0.2)
        model = DistMultModel(8, 3, embedding_dim=4)
        if te.num_triples == 0:
            return
        result = evaluate_temporal_filtered_ranking(
            model, test_kg=te, train_kg=tr,
            num_entities=8, chunk_size=8, hits_at=(1, 3),
        )
        assert 0 < result.filt_mrr <= 1.0

    def test_temporal_rank_finite(self):
        tkg = _temporal_chain_kg(6, 12, seed=1)
        tr, _, te = tkg.chronological_split(0.7, 0.15, 0.15)
        model = DistMultModel(6, 3, embedding_dim=4)
        if te.num_triples == 0:
            return
        result = evaluate_temporal_filtered_ranking(
            model, te, tr, num_entities=6, chunk_size=6,
        )
        assert result.filt_mr >= 1.0


class TestHornRuleMiningChain:

    def test_chain_kg_rule(self):
        """Chain KG: 0->1->2 with rel 0 gives body (0,0)=>0 rule."""
        from tgraphx.kg import KnowledgeGraph
        from tgraphx.kg.reasoning import PathExtractor, mine_horn_rules
        triples = torch.tensor([
            [0, 0, 1],
            [1, 0, 2],
            [0, 1, 2],  # shortcut
        ], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=3, num_relations=2)
        extractor = PathExtractor(kg, max_path_length=2)
        paths_0_to_2 = extractor.paths(0, 2)
        # Should find path (0, 0) = rel0 then rel0.
        assert (0, 0) in paths_0_to_2

    def test_rule_confidence_in_unit_interval(self):
        from tgraphx.kg.reasoning import mine_horn_rules, HornRuleCandidate
        from tgraphx.kg import FamilyKG
        ds = FamilyKG(num_persons=15, seed=0)
        rules = mine_horn_rules(ds.kg, max_body_length=1, min_support=1, max_rules=5)
        for rule in rules:
            assert 0.0 <= rule.confidence <= 1.0
            assert rule.lift >= 0.0
