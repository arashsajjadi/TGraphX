"""Tests for KG+GNN integration (RGCN KG completion)."""
from __future__ import annotations

import torch
import pytest

from tgraphx.kg import (
    generate_synthetic_kg,
    KGRGCNModel,
    kg_to_edge_index,
    DistMultModel,
)
from tgraphx.kg.gnn import kg_to_edge_index


class TestKGToEdgeIndex:

    def test_basic_conversion(self):
        kg = generate_synthetic_kg(10, 3, 20, seed=0)
        ei, et, ea, ew = kg_to_edge_index(kg)
        assert ei.shape == (2, kg.num_triples)
        assert et.shape == (kg.num_triples,)
        assert ea is None  # no triple features
        assert ew is None  # no edge weight

    def test_edge_weight_preserved(self):
        import torch
        triples = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        ew = torch.tensor([0.5, 1.0])
        from tgraphx.kg import KnowledgeGraph
        kg = KnowledgeGraph(triples, edge_weight=ew, num_entities=3, num_relations=2)
        ei, et, ea, ew2 = kg_to_edge_index(kg)
        assert ew2 is not None
        assert torch.allclose(ew2, ew)

    def test_edge_attr_preserved(self):
        from tgraphx.kg import KnowledgeGraph
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        attr = torch.randn(1, 4)
        kg = KnowledgeGraph(triples, triple_features={"edge_attr": attr}, num_entities=2, num_relations=1)
        ei, et, ea, _ = kg_to_edge_index(kg, include_edge_attr=True)
        assert ea is not None
        assert ea.shape == (1, 4)

    def test_head_entity_in_first_row(self):
        triples = torch.tensor([[3, 1, 7]], dtype=torch.long)
        from tgraphx.kg import KnowledgeGraph
        kg = KnowledgeGraph(triples, num_entities=10, num_relations=2)
        ei, et, _, _ = kg_to_edge_index(kg)
        assert int(ei[0, 0]) == 3  # head
        assert int(ei[1, 0]) == 7  # tail
        assert int(et[0]) == 1     # relation


class TestKGRGCNModel:

    def test_forward_backward(self):
        kg = generate_synthetic_kg(12, 3, 30, seed=0)
        model = KGRGCNModel(
            kg.num_entities, kg.num_relations,
            embedding_dim=8, num_rgcn_layers=1,
        )
        triples = kg.triples[:4]
        scores = model.forward_kg(kg, triples)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()
        scores.sum().backward()
        # Entity embeddings should have gradients.
        assert model.entity_emb is not None
        assert model.entity_emb.weight.grad is not None

    def test_relation_embeddings_receive_gradients(self):
        kg = generate_synthetic_kg(10, 2, 20, seed=0)
        model = KGRGCNModel(kg.num_entities, kg.num_relations, embedding_dim=4)
        scores = model.forward_kg(kg, kg.triples[:3])
        scores.sum().backward()
        assert model.relation_emb.weight.grad is not None

    def test_encode_with_entity_features(self):
        kg = generate_synthetic_kg(10, 2, 20, seed=0)
        entity_feat = torch.randn(10, 8)
        model = KGRGCNModel(10, 2, in_dim=8, embedding_dim=8, num_rgcn_layers=1)
        ei, et, _, _ = kg_to_edge_index(kg)
        edge_index_by_rel = {}
        for r in range(2):
            mask = et == r
            if mask.any():
                edge_index_by_rel[r] = ei[:, mask]
        embs = model.encode(edge_index_by_rel, entity_features=entity_feat)
        assert embs.shape == (10, 8)
        assert torch.isfinite(embs).all()
