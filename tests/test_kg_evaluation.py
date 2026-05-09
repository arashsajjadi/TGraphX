"""Tests for KG filtered ranking evaluation."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    generate_synthetic_kg,
    TransEModel,
    DistMultModel,
    evaluate_filtered_ranking,
    KGEvaluator,
)


def _dummy_model():
    """Simple model that scores triple (h, r, t) = -(h + t) so ranks are deterministic."""

    class DummyModel(torch.nn.Module):
        def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
            h, t = triples[:, 0].float(), triples[:, 2].float()
            return -(h + t)  # lower h+t → higher score

    return DummyModel()


class TestHandComputedRanks:

    def setup_method(self):
        # Tiny KG: 5 entities, 1 relation.
        # Train: (0,0,1), (0,0,2), (1,0,2)
        # Test:  (0,0,3)
        # The dummy model scores -(h+t), so for (0,0,?):
        # score(0,0,0)=-0, (0,0,1)=-1, (0,0,2)=-2, (0,0,3)=-3, (0,0,4)=-4
        # Tail rank of 3 (raw): higher scores: 0,1,2 → rank = 4
        # Filtered: (0,0,1) and (0,0,2) are in train → remove → rank = 2
        self.model = _dummy_model()
        self.num_entities = 5
        self.train = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 2]], dtype=torch.long)
        self.test = torch.tensor([[0, 0, 3]], dtype=torch.long)
        self.all_pos = {
            (0, 0, 1), (0, 0, 2), (1, 0, 2), (0, 0, 3)
        }

    def test_raw_tail_rank(self):
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos,
            self.num_entities, filtered=False, hits_at=(1, 3, 10),
        )
        # raw rank of t=3: entities 0,1,2 have higher score (-0,-1,-2 > -3) → rank = 4
        assert result.raw_mr_tail == pytest.approx(4.0, abs=0.6)

    def test_filtered_tail_rank(self):
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos,
            self.num_entities, filtered=True, hits_at=(1, 3, 10),
        )
        # Filtered: entities 1 and 2 removed. Remaining higher: entity 0 → rank = 2.
        assert result.filt_mr_tail == pytest.approx(2.0, abs=0.6)

    def test_filtered_better_than_raw(self):
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos,
            self.num_entities, filtered=True, hits_at=(1, 3, 10),
        )
        # Filtered rank should be <= raw rank.
        assert result.filt_mr_tail <= result.raw_mr_tail

    def test_mrr_in_unit_interval(self):
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos,
            self.num_entities, hits_at=(1, 3, 10),
        )
        assert 0.0 < result.filt_mrr <= 1.0
        assert 0.0 < result.filt_mrr_tail <= 1.0

    def test_hits_monotone(self):
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos,
            self.num_entities, hits_at=(1, 3, 10),
        )
        h1 = result.filt_hits[1]
        h3 = result.filt_hits[3]
        h10 = result.filt_hits[10]
        assert h1 <= h3 <= h10

    def test_target_not_filtered_out(self):
        # rank must always be >= 1 (target kept).
        result = evaluate_filtered_ranking(
            self.model, self.test, self.all_pos, self.num_entities,
        )
        assert result.filt_mr_tail >= 1.0

    def test_no_autograd_retention(self):
        # Evaluator must not retain computation graph.
        model = DistMultModel(5, 1, embedding_dim=4)
        result = evaluate_filtered_ranking(
            model, self.test, self.all_pos, self.num_entities,
        )
        # result attributes are plain floats, not tensors.
        assert isinstance(result.filt_mrr, float)


class TestKGEvaluatorSingleTriple:

    def test_evaluator_integration(self):
        kg = generate_synthetic_kg(15, 3, 40, seed=0)
        tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=0)
        model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=8)
        evaluator = KGEvaluator(tr.triples, va.triples, te.triples, kg.num_entities)
        result = evaluator.evaluate(model, triples=te.triples, batch_size=8)
        assert 0.0 < result.filt_mrr <= 1.0
        assert result.filt_mr >= 1.0

    def test_chunked_equals_unchunked_on_toy(self):
        kg = generate_synthetic_kg(8, 2, 15, seed=0)
        tr, va, te = kg.train_valid_test_split(0.6, 0.2, 0.2, seed=0)
        if te.num_triples == 0:
            return  # skip if no test triples
        pos_set = kg.positive_triple_set()
        model = TransEModel(kg.num_entities, kg.num_relations, embedding_dim=4)
        r1 = evaluate_filtered_ranking(model, te.triples, pos_set, kg.num_entities, chunk_size=100000)
        r2 = evaluate_filtered_ranking(model, te.triples, pos_set, kg.num_entities, chunk_size=2)
        assert abs(r1.filt_mrr - r2.filt_mrr) < 1e-4
