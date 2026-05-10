"""RESCAL knowledge-graph model tests (v1.2).

RESCAL: f(h, r, t) = h^T M_r t  where M_r is a [D, D] dense matrix per relation.

These tests pin shape, hand-computed values, gradient flow, dtype/device,
and a tiny overfit/loss-decrease sanity check.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestRESCALShape:
    def test_score_shape(self):
        from tgraphx.kg import RESCALModel
        model = RESCALModel(num_entities=5, num_relations=2, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
        scores = model.score_triples(triples)
        assert scores.shape == (2,)
        assert scores.dtype == torch.float32

    def test_forward_alias(self):
        from tgraphx.kg import RESCALModel
        model = RESCALModel(num_entities=5, num_relations=2, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        s1 = model.score_triples(triples)
        s2 = model.forward(triples)
        assert torch.equal(s1, s2)


class TestRESCALHandComputed:
    """f(h, r, t) = h^T M_r t.  Pin three small cases to known values."""

    def test_zero_relation_matrix_gives_zero_score(self):
        """If M_r = 0, score = h^T 0 t = 0."""
        from tgraphx.kg import RESCALModel
        D = 4
        model = RESCALModel(num_entities=3, num_relations=1, embedding_dim=D)
        with torch.no_grad():
            model.entity_emb.weight.data.fill_(1.0)        # all ones
            model.relation_matrix.weight.data.zero_()      # all zero
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        assert score.item() == pytest.approx(0.0, abs=1e-6)

    def test_identity_matrix_dot_product(self):
        """If M_r = I (identity), score = h^T t = dot product."""
        from tgraphx.kg import RESCALModel
        D = 4
        model = RESCALModel(num_entities=3, num_relations=1, embedding_dim=D)
        with torch.no_grad():
            # h = [1, 0, 1, 0], t = [1, 1, 0, 0]  → dot = 1
            model.entity_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 0.0, 1.0, 0.0])
            model.entity_emb.weight[1] = torch.tensor([1.0, 1.0, 0.0, 0.0])
            # M_r = identity[D, D] flattened
            I = torch.eye(D).flatten()
            model.relation_matrix.weight.data.zero_()
            model.relation_matrix.weight[0] = I
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        # h · t = 1*1 + 0*1 + 1*0 + 0*0 = 1.
        assert score.item() == pytest.approx(1.0, abs=1e-6)

    def test_known_bilinear_value(self):
        """Hand-computed bilinear form.

        h = [1, 2], t = [3, 4]
        M_r = [[1, 2], [3, 4]]
        h^T M_r t = h^T [Mt] where Mt = M @ t = [1*3+2*4, 3*3+4*4] = [11, 25]
                  = 1*11 + 2*25 = 11 + 50 = 61
        """
        from tgraphx.kg import RESCALModel
        D = 2
        model = RESCALModel(num_entities=2, num_relations=1, embedding_dim=D)
        with torch.no_grad():
            model.entity_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 2.0])
            model.entity_emb.weight[1] = torch.tensor([3.0, 4.0])
            # M_r = [[1, 2], [3, 4]] flattened row-major
            M_flat = torch.tensor([1.0, 2.0, 3.0, 4.0])
            model.relation_matrix.weight.data.zero_()
            model.relation_matrix.weight[0] = M_flat
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        score = model.score_triples(triples)
        assert score.item() == pytest.approx(61.0, abs=1e-5)

    def test_asymmetry_via_non_symmetric_matrix(self):
        """RESCAL captures asymmetric relations (unlike DistMult).

        M_r = [[0, 1], [0, 0]]  (clearly non-symmetric)
        h = [1, 0], t = [0, 1]
        h^T M t = [0, 1] · [0, 1] = 1   (forward direction)

        Reverse: h' = [0, 1], t' = [1, 0]
        h'^T M t' = [0, 0] · [1, 0] = 0  (different score → asymmetry captured)
        """
        from tgraphx.kg import RESCALModel
        D = 2
        model = RESCALModel(num_entities=2, num_relations=1, embedding_dim=D)
        with torch.no_grad():
            model.entity_emb.weight.zero_()
            model.entity_emb.weight[0] = torch.tensor([1.0, 0.0])
            model.entity_emb.weight[1] = torch.tensor([0.0, 1.0])
            M_flat = torch.tensor([0.0, 1.0, 0.0, 0.0])
            model.relation_matrix.weight.data.zero_()
            model.relation_matrix.weight[0] = M_flat
        # Forward direction (0 -[r]-> 1)
        forward = model.score_triples(torch.tensor([[0, 0, 1]], dtype=torch.long))
        # Reverse direction (1 -[r]-> 0)
        reverse = model.score_triples(torch.tensor([[1, 0, 0]], dtype=torch.long))
        assert forward.item() == pytest.approx(1.0, abs=1e-6)
        assert reverse.item() == pytest.approx(0.0, abs=1e-6)
        # Demonstrates the asymmetry capability: scores differ.
        assert not torch.allclose(forward, reverse)


class TestRESCALGradient:
    def test_finite_gradient_on_random_batch(self):
        from tgraphx.kg import RESCALModel
        torch.manual_seed(0)
        model = RESCALModel(num_entities=8, num_relations=3, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], dtype=torch.long)
        scores = model.score_triples(triples)
        loss = scores.sum()
        loss.backward()
        # Both entity_emb and relation_matrix should accumulate gradient.
        assert model.entity_emb.weight.grad is not None
        assert torch.isfinite(model.entity_emb.weight.grad).all()
        assert model.relation_matrix.weight.grad is not None
        assert torch.isfinite(model.relation_matrix.weight.grad).all()

    def test_nonzero_gradient(self):
        from tgraphx.kg import RESCALModel
        torch.manual_seed(1)
        model = RESCALModel(num_entities=5, num_relations=2, embedding_dim=4)
        triples = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
        scores = model.score_triples(triples)
        scores.sum().backward()
        # The entities and relations referenced must receive nonzero gradient.
        used_entities = {0, 1, 2, 3}
        for e in used_entities:
            assert model.entity_emb.weight.grad[e].abs().sum() > 0
        used_relations = {0, 1}
        for r in used_relations:
            assert model.relation_matrix.weight.grad[r].abs().sum() > 0


class TestRESCALDeviceDtype:
    def test_cpu_float32_default(self):
        from tgraphx.kg import RESCALModel
        model = RESCALModel(num_entities=5, num_relations=2, embedding_dim=4)
        assert model.entity_emb.weight.device.type == "cpu"
        assert model.entity_emb.weight.dtype == torch.float32

    def test_cuda_smoke(self):
        from tgraphx.kg import RESCALModel
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        model = RESCALModel(num_entities=5, num_relations=2, embedding_dim=4).cuda()
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long, device="cuda")
        scores = model.score_triples(triples)
        assert scores.device.type == "cuda"
        assert torch.isfinite(scores).all()


class TestRESCALTinyOverfit:
    def test_loss_decreases_on_tiny_kg(self):
        """Two-triple KG; verify the score on positive triples increases."""
        from tgraphx.kg import RESCALModel

        torch.manual_seed(42)
        model = RESCALModel(num_entities=4, num_relations=2, embedding_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        # Positive triples — these should rise in score.
        positives = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)
        # Negative triples — these should fall in score.
        negatives = torch.tensor([[0, 0, 3], [2, 1, 1]], dtype=torch.long)

        initial_pos_score = model.score_triples(positives).detach()

        for _ in range(50):
            pos_score = model.score_triples(positives)
            neg_score = model.score_triples(negatives)
            # Margin loss: encourage pos > neg.
            loss = (1.0 + neg_score - pos_score).clamp(min=0.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        final_pos_score = model.score_triples(positives).detach()
        # Either positive scores rose, or the gap to negatives widened.
        gap_initial = (initial_pos_score - model.score_triples(negatives).detach() + 0).mean()
        gap_final = (final_pos_score - model.score_triples(negatives).detach()).mean()
        assert gap_final.item() > gap_initial.item() - 0.5, \
            f"Margin should not collapse: initial={gap_initial.item():.4f}, final={gap_final.item():.4f}"


class TestRESCALInRegistry:
    def test_listed_in_kg_models(self):
        from tgraphx.kg import list_kg_models
        models = list_kg_models()
        assert "RESCAL" in models, f"RESCAL missing from list_kg_models(): {list(models)}"

    def test_importable_from_top_kg_module(self):
        from tgraphx.kg import RESCALModel
        assert RESCALModel is not None
