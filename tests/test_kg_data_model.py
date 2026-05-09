"""Tests for KnowledgeGraph data model."""
from __future__ import annotations

import json
import torch
import pytest

from tgraphx.kg import KnowledgeGraph, generate_synthetic_kg
from tgraphx.kg.data import TemporalKnowledgeGraph


def _tiny_kg():
    return KnowledgeGraph.from_triples(
        [("alice", "knows", "bob"), ("bob", "knows", "carol"), ("alice", "likes", "carol")],
    )


class TestKnowledgeGraph:

    def test_basic_construction(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=3, num_relations=1)
        assert kg.num_entities == 3
        assert kg.num_relations == 1
        assert kg.num_triples == 2

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            KnowledgeGraph(torch.zeros(5, 2, dtype=torch.long))

    def test_from_strings_deterministic_mapping(self):
        kg = _tiny_kg()
        assert kg.num_entities == 3
        assert kg.entity_to_id is not None
        assert "alice" in kg.entity_to_id
        # Same call → same IDs.
        kg2 = _tiny_kg()
        assert kg.entity_to_id == kg2.entity_to_id

    def test_from_triples_tensor(self):
        tri = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        kg = KnowledgeGraph.from_triples(tri)
        assert kg.num_triples == 2

    def test_heads_relations_tails(self):
        kg = _tiny_kg()
        assert kg.heads.shape == (3,)
        assert kg.relations.shape == (3,)
        assert kg.tails.shape == (3,)

    def test_entity_feature_shape_preserved(self):
        N_e = 5
        feat = torch.randn(N_e, 8)
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=N_e, entity_features={"x": feat})
        assert kg.entity_features["x"].shape == (N_e, 8)

    def test_image_feature_not_flattened(self):
        N_e = 4
        img_feat = torch.randn(N_e, 3, 8, 8)  # [N, C, H, W]
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=N_e, entity_features={"img": img_feat})
        assert kg.entity_features["img"].shape == (N_e, 3, 8, 8)

    def test_entity_feature_wrong_dim_raises(self):
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match="first dimension"):
            KnowledgeGraph(triples, num_entities=4, entity_features={"x": torch.zeros(3, 8)})

    def test_relation_feature_preserved(self):
        N_r = 3
        r_feat = torch.randn(N_r, 4)
        triples = torch.tensor([[0, 1, 2]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=3, num_relations=N_r,
                             relation_features={"r": r_feat})
        assert kg.relation_features["r"].shape == (N_r, 4)

    def test_triple_feature_preserved(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        ef = torch.randn(2, 6)
        kg = KnowledgeGraph(triples, triple_features={"edge_attr": ef})
        assert kg.triple_features["edge_attr"].shape == (2, 6)

    def test_edge_weight_preserved(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        ew = torch.tensor([0.5, 1.0])
        kg = KnowledgeGraph(triples, edge_weight=ew)
        assert kg.edge_weight is not None
        assert torch.allclose(kg.edge_weight, ew)

    def test_has_triple(self):
        kg = _tiny_kg()
        eid = kg.entity_to_id
        rid = kg.relation_to_id
        assert kg.has_triple(eid["alice"], rid["knows"], eid["bob"])
        assert not kg.has_triple(eid["bob"], rid["knows"], eid["alice"])

    def test_to_device_moves_all_tensors(self):
        N_e = 3
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        feat = torch.randn(N_e, 4)
        ew = torch.rand(1)
        kg = KnowledgeGraph(triples, num_entities=N_e, entity_features={"x": feat}, edge_weight=ew)
        kg.to("cpu")
        assert kg.triples.device.type == "cpu"
        assert kg.entity_features["x"].device.type == "cpu"
        assert kg.edge_weight.device.type == "cpu"

    def test_train_valid_test_no_overlap(self):
        kg = generate_synthetic_kg(50, 5, 100, seed=0)
        tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=0)
        tr_set = set(map(tuple, tr.triples.tolist()))
        va_set = set(map(tuple, va.triples.tolist()))
        te_set = set(map(tuple, te.triples.tolist()))
        assert not (tr_set & va_set), "train/valid overlap"
        assert not (tr_set & te_set), "train/test overlap"
        assert not (va_set & te_set), "valid/test overlap"
        assert len(tr_set) + len(va_set) + len(te_set) == kg.num_triples

    def test_train_valid_test_preserves_entity_features(self):
        kg = generate_synthetic_kg(20, 3, 40, seed=0)
        ef = torch.randn(20, 8)
        kg.entity_features["x"] = ef
        tr, va, te = kg.train_valid_test_split(0.7, 0.15, 0.15, seed=0)
        assert "x" in tr.entity_features
        assert tr.entity_features["x"].shape == (20, 8)

    def test_add_inverse_relations_doubles(self):
        kg = _tiny_kg()
        inv = kg.add_inverse_relations()
        assert inv.num_triples == 2 * kg.num_triples
        assert inv.num_relations == 2 * kg.num_relations

    def test_to_edge_index_correct(self):
        triples = torch.tensor([[0, 2, 1], [1, 0, 2]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=3, num_relations=3)
        ei, et = kg.to_edge_index()
        assert ei.shape == (2, 2)
        assert et.shape == (2,)
        assert int(et[0]) == 2
        assert int(ei[0, 0]) == 0 and int(ei[1, 0]) == 1

    def test_from_edge_index(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        et = torch.tensor([0, 1], dtype=torch.long)
        kg = KnowledgeGraph.from_edge_index(ei, et, num_entities=3, num_relations=2)
        assert kg.num_triples == 2

    def test_json_roundtrip(self, tmp_path):
        kg = _tiny_kg()
        path = str(tmp_path / "kg.json")
        kg.save_json(path)
        kg2 = KnowledgeGraph.load_json(path)
        assert kg2.num_entities == kg.num_entities
        assert kg2.num_triples == kg.num_triples
        assert torch.equal(kg2.triples.sort(dim=0).values, kg.triples.sort(dim=0).values)

    def test_tsv_roundtrip(self, tmp_path):
        kg = generate_synthetic_kg(10, 3, 20, seed=0)
        path = str(tmp_path / "kg.tsv")
        kg.save_tsv(path)
        kg2 = KnowledgeGraph.load_tsv(path, num_entities=kg.num_entities,
                                       num_relations=kg.num_relations)
        assert kg2.num_triples == kg.num_triples

    def test_detach_for_report_no_autograd(self):
        kg = generate_synthetic_kg(10, 2, 20, seed=0)
        ef = torch.randn(10, 4, requires_grad=True)
        kg.entity_features["x"] = ef
        kg2 = kg.detach_for_report()
        # Must not retain autograd.
        assert not kg2.entity_features["x"].requires_grad

    def test_summary_json_safe(self):
        kg = generate_synthetic_kg(20, 3, 40, seed=0)
        s = kg.summary()
        json.dumps(s)  # must not raise

    def test_no_unsafe_pickle(self, tmp_path):
        kg = _tiny_kg()
        path = str(tmp_path / "kg.json")
        kg.save_json(path)
        content = (tmp_path / "kg.json").read_text()
        assert "pickle" not in content.lower()


class TestTemporalKG:

    def test_construction(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
        ts = torch.tensor([1.0, 2.0])
        tkg = TemporalKnowledgeGraph(triples, ts)
        assert torch.equal(tkg.timestamp, ts.float())

    def test_timestamp_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="timestamp"):
            TemporalKnowledgeGraph(
                torch.tensor([[0, 0, 1]]), torch.tensor([1.0, 2.0])
            )

    def test_sort_by_time(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 2], [0, 0, 2]], dtype=torch.long)
        ts = torch.tensor([3.0, 1.0, 2.0])
        tkg = TemporalKnowledgeGraph(triples, ts)
        sorted_kg = tkg.sort_by_time()
        assert torch.equal(sorted_kg.timestamp, torch.tensor([1.0, 2.0, 3.0]))

    def test_chronological_split_no_leakage(self):
        triples = torch.tensor([[i, 0, (i + 1) % 10] for i in range(20)], dtype=torch.long)
        ts = torch.arange(20, dtype=torch.float)
        tkg = TemporalKnowledgeGraph(triples, ts, num_entities=10, num_relations=1)
        tr, va, te = tkg.chronological_split(0.6, 0.2, 0.2)
        assert float(tr.timestamp.max()) <= float(va.timestamp.min())
        assert float(va.timestamp.max()) <= float(te.timestamp.min())
