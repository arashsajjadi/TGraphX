"""Regression tests for feature-aware KG scoring (v1.3.1 bugfix).

Bug fixed: TransEModel._embed_entities called self.entity_proj(feat) instead of
self.entity_proj(emb, feat), causing:
  TypeError: _FeatureProjector.forward() missing 1 required positional argument: 'feat'

These tests pin the correct behaviour and prevent regression.
"""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn


class TestFeatureAwareTransE:
    """TransEModel with entity_feature_dim and relation_feature_dim."""

    def test_entity_features_score_shape(self):
        from tgraphx.kg import TransEModel
        N_e, N_r, D, F_e = 10, 3, 16, 32
        model = TransEModel(N_e, N_r, embedding_dim=D, entity_feature_dim=F_e)
        triples = torch.tensor([[0,0,1],[2,1,3]], dtype=torch.long)
        feats = torch.randn(N_e, F_e)
        scores = model.score_triples(triples, entity_features=feats)
        assert scores.shape == (2,)

    def test_entity_features_gradient_flows_into_projector(self):
        from tgraphx.kg import TransEModel
        N_e, N_r, D, F_e = 10, 3, 16, 32
        model = TransEModel(N_e, N_r, embedding_dim=D, entity_feature_dim=F_e)
        triples = torch.tensor([[0,0,1],[2,1,3]], dtype=torch.long)
        feats = torch.randn(N_e, F_e)
        scores = model.score_triples(triples, entity_features=feats)
        scores.mean().backward()
        assert model.entity_proj.proj.weight.grad is not None
        assert torch.isfinite(model.entity_proj.proj.weight.grad).all()
        assert model.entity_proj.proj.weight.grad.abs().sum() > 0

    def test_relation_features_score_shape(self):
        from tgraphx.kg import TransEModel
        N_e, N_r, D, F_r = 10, 3, 16, 8
        model = TransEModel(N_e, N_r, embedding_dim=D, relation_feature_dim=F_r)
        triples = torch.tensor([[0,0,1],[2,1,3]], dtype=torch.long)
        r_feats = torch.randn(N_r, F_r)
        scores = model.score_triples(triples, relation_features=r_feats)
        assert scores.shape == (2,)

    def test_both_entity_and_relation_features(self):
        from tgraphx.kg import TransEModel
        N_e, N_r, D = 8, 2, 12
        model = TransEModel(N_e, N_r, embedding_dim=D,
                            entity_feature_dim=16, relation_feature_dim=8)
        triples = torch.tensor([[0,0,1],[2,1,3]], dtype=torch.long)
        scores = model.score_triples(
            triples,
            entity_features=torch.randn(N_e, 16),
            relation_features=torch.randn(N_r, 8),
        )
        assert scores.shape == (2,)
        scores.mean().backward()
        assert model.entity_proj.proj.weight.grad is not None
        assert model.relation_proj.proj.weight.grad is not None

    def test_no_features_unchanged(self):
        """Without entity_features, behaviour is unchanged (no regression)."""
        from tgraphx.kg import TransEModel
        model = TransEModel(5, 2, embedding_dim=8)
        triples = torch.tensor([[0,0,1]], dtype=torch.long)
        scores = model.score_triples(triples)
        assert scores.shape == (1,)

    def test_colab_exact_user_repro(self):
        """Exact code from the Colab bug report."""
        from tgraphx.kg import KnowledgeGraph, TransEModel
        torch.manual_seed(0)
        N_e, N_r, N_t = 10, 3, 30
        heads = torch.randint(0, N_e, (N_t,))
        rels  = torch.randint(0, N_r, (N_t,))
        tails = torch.randint(0, N_e, (N_t,))
        entity_features = {"visual": torch.randn(N_e, 32)}
        kg = KnowledgeGraph.from_hrt(
            heads, rels, tails,
            num_entities=N_e, num_relations=N_r,
            entity_features=entity_features,
        )
        model = TransEModel(N_e, N_r, embedding_dim=16, entity_feature_dim=32)
        triples = torch.stack([heads, rels, tails], dim=1)
        scores = model.score_triples(triples, entity_features=kg.entity_features["visual"])
        assert scores.shape == (N_t,)
        scores.mean().backward()
        assert model.entity_proj.proj.weight.grad is not None


class TestFeatureAwareDistMult:
    """DistMult has entity/relation projector support via _emb_e/_emb_r helpers.
    Its public score_triples() does not expose entity_features= kwarg, but the
    projectors must be callable with (emb, feat) — pin this against regression.
    """

    def test_projector_callable_correctly(self):
        """The _FeatureProjector must accept (emb, feat) — not just (feat,)."""
        from tgraphx.kg import DistMultModel
        N_e, N_r, D, F_e = 8, 2, 16, 24
        model = DistMultModel(N_e, N_r, embedding_dim=D, entity_feature_dim=F_e)
        idx = torch.tensor([0, 1], dtype=torch.long)
        feat = torch.randn(2, F_e)
        # Call the internal _emb_e with a feature slice — must not raise.
        out = model._emb_e(idx, feat)
        assert out.shape == (2, D)
