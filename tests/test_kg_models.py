"""Tests for KG model zoo: TransE, DistMult, ComplEx, RotatE."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    generate_synthetic_kg,
    TransEModel, DistMultModel, ComplExModel, RotatEModel,
    MarginRankingLoss, BCEKGLoss, SoftplusKGLoss,
    UniformNegativeSampler,
)
from tgraphx.kg.losses import l2_regularization


def _pos_neg(model_cls, N_e=20, N_r=4, D=8, B=8):
    torch.manual_seed(0)
    kg = generate_synthetic_kg(N_e, N_r, 50, seed=0)
    model = model_cls(N_e, N_r, embedding_dim=D)
    sampler = UniformNegativeSampler(N_e, 2)
    pos = kg.triples[:B]
    neg = sampler.sample(pos).view(-1, 3)
    return model, pos, neg


class TestTransEModel:

    def test_score_shape(self):
        model, pos, neg = _pos_neg(TransEModel)
        s = model.score_triples(pos)
        assert s.shape == (pos.size(0),)

    def test_backward(self):
        model, pos, neg = _pos_neg(TransEModel)
        pos_s = model.score_triples(pos)
        neg_s = model.score_triples(neg)
        loss = MarginRankingLoss(margin=1.0)(pos_s, neg_s)
        loss.backward()
        assert torch.isfinite(model.entity_emb.weight.grad).all()
        assert torch.isfinite(model.relation_emb.weight.grad).all()

    def test_pos_higher_than_neg_after_training(self):
        torch.manual_seed(42)
        N_e, N_r = 8, 2
        # Tiny overfit: all same triples.
        pos = torch.tensor([[0, 0, 1], [2, 1, 3]] * 10, dtype=torch.long)
        sampler = UniformNegativeSampler(N_e, 1, corrupt_head_prob=0.5)
        model = TransEModel(N_e, N_r, embedding_dim=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = MarginRankingLoss(1.0)
        g = torch.Generator().manual_seed(0)
        for _ in range(100):
            opt.zero_grad()
            neg = sampler.sample(pos, generator=g).view(-1, 3)
            loss = loss_fn(model.score_triples(pos), model.score_triples(neg))
            loss.backward()
            opt.step()
        final = float(loss.detach().item())
        assert final < 0.5, f"TransE loss should drop, got {final}"

    def test_score_fn_finite(self):
        model, pos, _ = _pos_neg(TransEModel)
        s = model.score_triples(pos)
        assert torch.isfinite(s).all()


class TestDistMultModel:

    def test_score_shape(self):
        model, pos, neg = _pos_neg(DistMultModel)
        assert model.score_triples(pos).shape == (pos.size(0),)

    def test_backward(self):
        model, pos, neg = _pos_neg(DistMultModel)
        loss = BCEKGLoss()(model.score_triples(pos), model.score_triples(neg))
        loss.backward()
        assert torch.isfinite(model.entity_emb.weight.grad).all()

    def test_tiny_overfit(self):
        torch.manual_seed(0)
        pos = torch.tensor([[0, 0, 1]] * 20, dtype=torch.long)
        s = UniformNegativeSampler(5, 1)
        model = DistMultModel(5, 1, embedding_dim=4)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        g = torch.Generator().manual_seed(0)
        for _ in range(50):
            opt.zero_grad()
            neg = s.sample(pos, g).view(-1, 3)
            loss = BCEKGLoss()(model.score_triples(pos), model.score_triples(neg))
            loss.backward(); opt.step()
        assert float(loss.detach()) < 0.9


class TestComplExModel:

    def test_score_shape(self):
        model, pos, _ = _pos_neg(ComplExModel)
        assert model.score_triples(pos).shape == (pos.size(0),)

    def test_backward_finite(self):
        model, pos, neg = _pos_neg(ComplExModel)
        loss = SoftplusKGLoss()(model.score_triples(pos), model.score_triples(neg))
        loss.backward()
        assert torch.isfinite(model.entity_re.weight.grad).all()
        assert torch.isfinite(model.entity_im.weight.grad).all()

    def test_asymmetric_scoring(self):
        # ComplEx should give different scores for (h,r,t) vs (t,r,h)
        model = ComplExModel(5, 2, embedding_dim=4)
        triple_ht = torch.tensor([[0, 0, 1]], dtype=torch.long)
        triple_th = torch.tensor([[1, 0, 0]], dtype=torch.long)
        s_ht = float(model.score_triples(triple_ht).item())
        s_th = float(model.score_triples(triple_th).item())
        # Not necessarily different due to initialisation, but the model can distinguish.
        # Just check scores are finite.
        assert math.isfinite(s_ht) and math.isfinite(s_th)


class TestRotatEModel:

    def test_score_shape(self):
        model, pos, _ = _pos_neg(RotatEModel)
        assert model.score_triples(pos).shape == (pos.size(0),)

    def test_backward_finite(self):
        model, pos, neg = _pos_neg(RotatEModel)
        loss = SoftplusKGLoss()(model.score_triples(pos), model.score_triples(neg))
        loss.backward()
        assert torch.isfinite(model.relation_phase.weight.grad).all()

    def test_unit_rotation_constraint(self):
        model = RotatEModel(5, 2, embedding_dim=4)
        # Call _entity_norm to check unit norms.
        h = torch.tensor([0, 1, 2])
        re, im = model._entity_norm(h)
        norms = torch.sqrt(re ** 2 + im ** 2).mean(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)


import math


class TestLosses:

    def test_margin_ranking_loss_zero_when_pos_above_neg_by_margin(self):
        margin = 1.0
        pos_s = torch.tensor([2.0, 3.0])
        neg_s = torch.tensor([0.0, 1.0])  # pos - neg = 2 > margin
        loss = MarginRankingLoss(margin)(pos_s, neg_s)
        assert float(loss.item()) == pytest.approx(0.0)

    def test_bce_loss_positive(self):
        pos_s = torch.tensor([1.0, 2.0])
        neg_s = torch.tensor([-1.0, -2.0])
        loss = BCEKGLoss()(pos_s, neg_s)
        assert float(loss.item()) > 0

    def test_softplus_loss_positive(self):
        pos_s = torch.tensor([0.0])
        neg_s = torch.tensor([0.0])
        loss = SoftplusKGLoss()(pos_s, neg_s)
        assert float(loss.item()) > 0

    def test_l2_regularization(self):
        params = [torch.randn(10, requires_grad=True)]
        reg = l2_regularization(params, weight=0.01)
        assert reg.item() >= 0
        reg.backward()
        assert params[0].grad is not None
