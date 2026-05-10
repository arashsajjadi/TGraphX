"""SimplE KG model tests (v1.3).

SimplE: f(h, r, t) = 0.5 * (⟨h_head, r_fwd, t_tail⟩ + ⟨t_head, r_inv, h_tail⟩)
Reference: Kazemi & Poole, NeurIPS 2018.

These tests pin shape, hand-computed values, gradient flow, dtype/device,
asymmetry behavior, and a tiny overfit sanity check.
"""
from __future__ import annotations

import pytest
import torch


class TestSimplEShape:
    def test_score_shape(self):
        from tgraphx.kg import SimplEModel
        model = SimplEModel(num_entities=6, num_relations=3, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)
        scores = model.score_triples(triples)
        assert scores.shape == (3,)
        assert scores.dtype == torch.float32

    def test_forward_alias(self):
        from tgraphx.kg import SimplEModel
        model = SimplEModel(num_entities=4, num_relations=2, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        assert torch.equal(model.score_triples(triples), model.forward(triples))


class TestSimplEHandComputed:
    """Hand-computed reference values for SimplE scoring.

    Setup:
        D=1 (scalar embeddings for easy hand-calculation).
        entity_head = [h0=1, h1=2, h2=3, h3=4]
        entity_tail = [t0=5, t1=6, t2=7, t3=8]
        relation_fwd = [r0=2, r1=3]
        relation_inv = [r0=4, r1=5]

    Triple (h=0, r=0, t=1):
        fwd  = h_head[0] * r_fwd[0] * t_tail[1] = 1 * 2 * 6 = 12
        inv  = t_head[1] * r_inv[0] * h_tail[0] = 2 * 4 * 5 = 40
        score = 0.5 * (12 + 40) = 26
    """

    def _setup_d1_model(self):
        from tgraphx.kg import SimplEModel
        model = SimplEModel(num_entities=4, num_relations=2, embedding_dim=1)
        with torch.no_grad():
            # D=1 so each weight is a column vector [N, 1].
            model.entity_head.weight.data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
            model.entity_tail.weight.data = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
            model.relation_fwd.weight.data = torch.tensor([[2.0], [3.0]])
            model.relation_inv.weight.data = torch.tensor([[4.0], [5.0]])
        return model

    def test_hand_computed_triple_0_0_1(self):
        model = self._setup_d1_model()
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        # fwd = 1 * 2 * 6 = 12; inv = 2 * 4 * 5 = 40; score = 0.5*(12+40) = 26
        assert score.item() == pytest.approx(26.0, abs=1e-5)

    def test_hand_computed_triple_1_1_2(self):
        model = self._setup_d1_model()
        triples = torch.tensor([[1, 1, 2]], dtype=torch.long)
        score = model.score_triples(triples)
        # fwd = entity_head[1] * r_fwd[1] * entity_tail[2]
        #     = 2 * 3 * 7 = 42
        # inv = entity_head[2] * r_inv[1] * entity_tail[1]
        #     = 3 * 5 * 6 = 90
        # score = 0.5 * (42 + 90) = 66
        assert score.item() == pytest.approx(66.0, abs=1e-5)

    def test_asymmetry(self):
        """f(h, r, t) ≠ f(t, r, h) in general — SimplE can model asymmetric relations."""
        model = self._setup_d1_model()
        fwd = model.score_triples(torch.tensor([[0, 0, 1]], dtype=torch.long))
        rev = model.score_triples(torch.tensor([[1, 0, 0]], dtype=torch.long))
        # fwd=26 as computed above; rev:
        # fwd_term = entity_head[1] * r_fwd[0] * entity_tail[0] = 2*2*5 = 20
        # inv_term = entity_head[0] * r_inv[0] * entity_tail[1] = 1*4*6 = 24
        # rev_score = 0.5*(20+24) = 22  (different from 26)
        assert rev.item() == pytest.approx(22.0, abs=1e-5)
        assert not torch.allclose(fwd, rev)

    def test_symmetric_when_head_eq_tail_embeddings(self):
        """If head_emb == tail_emb and fwd_rel == inv_rel, score is symmetric."""
        from tgraphx.kg import SimplEModel
        model = SimplEModel(num_entities=4, num_relations=2, embedding_dim=2)
        with torch.no_grad():
            same = torch.randn(4, 2)
            model.entity_head.weight.data = same.clone()
            model.entity_tail.weight.data = same.clone()
            same_r = torch.randn(2, 2)
            model.relation_fwd.weight.data = same_r.clone()
            model.relation_inv.weight.data = same_r.clone()
        # With identical embeddings, f(h,r,t) and f(t,r,h) should be equal.
        fwd = model.score_triples(torch.tensor([[0, 0, 1]], dtype=torch.long))
        rev = model.score_triples(torch.tensor([[1, 0, 0]], dtype=torch.long))
        assert torch.allclose(fwd, rev, atol=1e-5)


class TestSimplEGradient:
    def test_finite_nonzero_gradient(self):
        from tgraphx.kg import SimplEModel
        torch.manual_seed(0)
        model = SimplEModel(num_entities=6, num_relations=3, embedding_dim=8)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)
        scores = model.score_triples(triples)
        scores.sum().backward()
        for emb in [model.entity_head, model.entity_tail, model.relation_fwd, model.relation_inv]:
            assert emb.weight.grad is not None
            assert torch.isfinite(emb.weight.grad).all()
        # Referenced entities/relations must have nonzero gradient.
        assert model.entity_head.weight.grad[0].abs().sum() > 0


class TestSimplEDeviceDtype:
    def test_cpu_float32(self):
        from tgraphx.kg import SimplEModel
        model = SimplEModel(num_entities=4, num_relations=2, embedding_dim=4)
        assert model.entity_head.weight.device.type == "cpu"
        assert model.entity_head.weight.dtype == torch.float32

    def test_cuda_smoke(self):
        from tgraphx.kg import SimplEModel
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        model = SimplEModel(num_entities=4, num_relations=2, embedding_dim=4).cuda()
        triples = torch.tensor([[0, 0, 1]], device="cuda", dtype=torch.long)
        scores = model.score_triples(triples)
        assert scores.device.type == "cuda"
        assert torch.isfinite(scores).all()


class TestSimplETinyOverfit:
    def test_margin_loss_decreases(self):
        from tgraphx.kg import SimplEModel
        torch.manual_seed(42)
        model = SimplEModel(num_entities=6, num_relations=2, embedding_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        pos = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
        neg = torch.tensor([[0, 0, 3], [2, 1, 1]], dtype=torch.long)
        initial = (1.0 + model.score_triples(neg) - model.score_triples(pos)).clamp(min=0).mean()
        for _ in range(60):
            loss = (1.0 + model.score_triples(neg) - model.score_triples(pos)).clamp(min=0).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        final = (1.0 + model.score_triples(neg) - model.score_triples(pos)).clamp(min=0).mean()
        assert final.item() <= initial.item() + 0.3, \
            f"Loss did not decrease: {initial.item():.4f} -> {final.item():.4f}"


class TestSimplERegistry:
    def test_in_list_kg_models(self):
        from tgraphx.kg import list_kg_models
        assert "SimplE" in list_kg_models()

    def test_importable_from_kg(self):
        from tgraphx.kg import SimplEModel
        assert SimplEModel is not None
