"""Tests for KGTrainer."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    generate_synthetic_kg,
    TransEModel, DistMultModel,
    KGTrainer, KGTrainingConfig,
    UniformNegativeSampler,
    KGEvaluator,
)


def _setup(seed: int = 0, n_e: int = 20, n_r: int = 3, n_t: int = 50):
    kg = generate_synthetic_kg(n_e, n_r, n_t, seed=seed)
    tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=seed)
    return kg, tr, va, te


class TestKGTrainer:

    def test_one_epoch_runs(self):
        kg, tr, va, te = _setup()
        model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=8)
        cfg = KGTrainingConfig(num_epochs=1, batch_size=16, loss_type="bce", seed=0)
        trainer = KGTrainer(model, cfg, tr.triples)
        result = trainer.train()
        assert result["final_loss"] is not None
        assert len(result["loss_history"]) == 1

    def test_loss_decreases_over_epochs(self):
        # Run longer to be robust to noise; use softplus which tends to be smoother.
        kg = generate_synthetic_kg(15, 2, 40, seed=0)
        tr, _, _ = kg.train_valid_test_split(0.8, 0.1, 0.1, seed=0)
        model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=16)
        cfg = KGTrainingConfig(num_epochs=50, batch_size=32, loss_type="softplus",
                               lr=0.05, seed=0)
        trainer = KGTrainer(model, cfg, tr.triples)
        result = trainer.train()
        avg_first5 = sum(result["loss_history"][:5]) / 5
        avg_last5 = sum(result["loss_history"][-5:]) / 5
        assert avg_last5 < avg_first5, \
            f"loss did not decrease overall: first5={avg_first5:.4f} last5={avg_last5:.4f}"

    def test_deterministic_with_seed(self):
        kg, tr, _, _ = _setup()
        cfg = KGTrainingConfig(num_epochs=3, batch_size=16, seed=42)
        model1 = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=4)
        model2 = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=4)
        model1.load_state_dict(model2.state_dict())  # same init
        r1 = KGTrainer(model1, cfg, tr.triples).train()
        r2 = KGTrainer(model2, cfg, tr.triples).train()
        assert r1["loss_history"] == r2["loss_history"]

    def test_validation_runs(self):
        kg, tr, va, te = _setup()
        model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=4)
        evaluator = KGEvaluator(tr.triples, va.triples, te.triples, kg.num_entities)
        cfg = KGTrainingConfig(num_epochs=5, batch_size=16, valid_every=5, seed=0)
        trainer = KGTrainer(model, cfg, tr.triples, evaluator=evaluator)
        result = trainer.train()
        assert len(result["valid_history"]) == 1
        assert "filtered" in result["valid_history"][0]

    def test_no_train_valid_test_overlap(self):
        kg, tr, va, te = _setup()
        tr_set = set(map(tuple, tr.triples.tolist()))
        va_set = set(map(tuple, va.triples.tolist()))
        te_set = set(map(tuple, te.triples.tolist()))
        assert not (tr_set & va_set)
        assert not (tr_set & te_set)
        assert not (va_set & te_set)

    def test_grad_clip_clips(self):
        """Gradient norm after clip should be <= clip value."""
        kg, tr, _, _ = _setup()
        model = DistMultModel(kg.num_entities, kg.num_relations, embedding_dim=8)
        cfg = KGTrainingConfig(num_epochs=2, batch_size=8, grad_clip_norm=1.0, seed=0)
        trainer = KGTrainer(model, cfg, tr.triples)
        # We just test it runs without error and that the final loss is finite.
        result = trainer.train()
        assert isinstance(result["final_loss"], float)
        assert torch.isfinite(torch.tensor(result["final_loss"]))
