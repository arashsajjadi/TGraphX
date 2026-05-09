"""Tests for KG synthetic datasets."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    FamilyKG, AcademicKG, MultimodalKG, generate_synthetic_kg
)


class TestFamilyKG:

    def test_construction(self):
        ds = FamilyKG(num_persons=20, seed=0)
        assert ds.kg.num_entities == 20
        assert ds.kg.num_relations == FamilyKG.NUM_RELATIONS

    def test_splits_disjoint(self):
        ds = FamilyKG(num_persons=20, seed=0)
        tr = set(map(tuple, ds.train.triples.tolist()))
        va = set(map(tuple, ds.valid.triples.tolist()))
        te = set(map(tuple, ds.test.triples.tolist()))
        assert not (tr & va)
        assert not (tr & te)

    def test_deterministic(self):
        ds1 = FamilyKG(num_persons=20, seed=42)
        ds2 = FamilyKG(num_persons=20, seed=42)
        assert torch.equal(ds1.kg.triples.sort(dim=0).values,
                           ds2.kg.triples.sort(dim=0).values)

    def test_parentof_relation(self):
        ds = FamilyKG(num_persons=20, seed=0)
        # parentOf has ID 0.
        parent_triples = ds.kg.triples[ds.kg.relations == 0]
        assert parent_triples.numel() > 0


class TestAcademicKG:

    def test_construction(self):
        ds = AcademicKG(num_authors=10, num_papers=15, num_venues=3,
                        num_institutions=2, seed=0)
        assert ds.kg.num_entities == 10 + 15 + 3 + 2

    def test_splits_disjoint(self):
        ds = AcademicKG(seed=0)
        tr = set(map(tuple, ds.train.triples.tolist()))
        va = set(map(tuple, ds.valid.triples.tolist()))
        assert not (tr & va)

    def test_deterministic(self):
        ds1 = AcademicKG(seed=7)
        ds2 = AcademicKG(seed=7)
        assert ds1.kg.num_triples == ds2.kg.num_triples


class TestMultimodalKG:

    def test_entity_features_shape(self):
        ds = MultimodalKG(num_entities=20, num_relations=3, num_triples=50,
                           entity_feature_dim=16, seed=0)
        assert ds.kg.entity_features["x"].shape == (20, 16)

    def test_relation_features_shape(self):
        ds = MultimodalKG(num_entities=20, num_relations=3, num_triples=50,
                           relation_feature_dim=8, seed=0)
        assert ds.kg.relation_features["r"].shape == (3, 8)

    def test_no_feature_flattening(self):
        # Features should be stored exactly as-is.
        ds = MultimodalKG(entity_feature_dim=16)
        assert ds.kg.entity_features["x"].dim() == 2

    def test_deterministic(self):
        ds1 = MultimodalKG(seed=3)
        ds2 = MultimodalKG(seed=3)
        assert torch.equal(ds1.kg.entity_features["x"], ds2.kg.entity_features["x"])


class TestGenerateSyntheticKG:

    def test_basic(self):
        kg = generate_synthetic_kg(30, 4, 80, seed=0)
        assert kg.num_entities == 30
        assert kg.num_relations == 4
        assert kg.num_triples <= 80

    def test_with_weights(self):
        kg = generate_synthetic_kg(10, 2, 20, seed=0, with_weights=True)
        assert kg.edge_weight is not None
        assert kg.edge_weight.shape == (kg.num_triples,)
        assert (kg.edge_weight >= 0).all() and (kg.edge_weight <= 1).all()

    def test_with_timestamps_returns_temporal(self):
        from tgraphx.kg.data import TemporalKnowledgeGraph
        kg = generate_synthetic_kg(10, 2, 20, seed=0, with_timestamps=True)
        assert isinstance(kg, TemporalKnowledgeGraph)
        assert kg.timestamp.shape == (kg.num_triples,)

    def test_no_self_loops(self):
        kg = generate_synthetic_kg(20, 3, 60, seed=0)
        # heads != tails for all triples.
        assert (kg.heads != kg.tails).all()

    def test_deterministic(self):
        kg1 = generate_synthetic_kg(20, 3, 50, seed=0)
        kg2 = generate_synthetic_kg(20, 3, 50, seed=0)
        assert torch.equal(kg1.triples, kg2.triples)
