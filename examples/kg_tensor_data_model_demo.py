"""Tensor-aware KnowledgeGraph data model demonstration.

Shows entity vector features, relation features, triple features,
edge weights, ID mapping, and JSON/TSV IO.

Usage:
    python examples/kg_tensor_data_model_demo.py
"""
from __future__ import annotations

import os
import tempfile

import torch

from tgraphx.kg import KnowledgeGraph, generate_synthetic_kg
from tgraphx.kg.data import TemporalKnowledgeGraph
from tgraphx.kg.reports import write_kg_summary


def main() -> None:
    print("=== KnowledgeGraph data model demo ===\n")

    # 1. Build from string triples.
    kg = KnowledgeGraph.from_triples([
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("alice", "likes", "carol"),
        ("carol", "likes", "alice"),
    ])
    print(f"KG from strings: {kg}")
    print(f"  entity_to_id: {kg.entity_to_id}")
    print(f"  relation_to_id: {kg.relation_to_id}")
    eid = kg.entity_to_id or {}
    rid = kg.relation_to_id or {}
    alice_id = eid.get("alice", 0)
    knows_id = rid.get("knows", 0)
    bob_id = eid.get("bob", 1)
    print(f"  has_triple(alice, knows, bob): {kg.has_triple(alice_id, knows_id, bob_id)}")

    # 2. Attach tensor features (NOT flattened).
    N_e = kg.num_entities
    entity_feat = torch.randn(N_e, 16)           # vector features
    image_feat = torch.randn(N_e, 3, 8, 8)       # image-like features [N, C, H, W]
    kg.entity_features["x"] = entity_feat
    kg.entity_features["img"] = image_feat
    print(f"\nEntity features added:")
    print(f"  'x' shape: {kg.entity_features['x'].shape}")
    print(f"  'img' shape: {kg.entity_features['img'].shape} (not flattened)")

    # 3. Relation + triple features.
    kg.relation_features["r"] = torch.randn(kg.num_relations, 8)
    kg.triple_features["edge_attr"] = torch.randn(kg.num_triples, 4)
    kg.edge_weight = torch.rand(kg.num_triples)
    kg.confidence = torch.rand(kg.num_triples)
    print(f"\nRelation features: {kg.relation_features['r'].shape}")
    print(f"Triple features: {kg.triple_features['edge_attr'].shape}")
    print(f"Edge weight: {kg.edge_weight.shape}")

    # 4. Split.
    kg2 = generate_synthetic_kg(50, 5, 120, seed=0)
    tr, va, te = kg2.train_valid_test_split(0.7, 0.15, 0.15, seed=0)
    print(f"\nSplit KG (50 entities, 5 relations, 120 triples):")
    print(f"  train: {tr.num_triples} | valid: {va.num_triples} | test: {te.num_triples}")

    # 5. Inverse relations.
    inv_kg = kg2.add_inverse_relations()
    print(f"\nWith inverse: {inv_kg.num_triples} triples, {inv_kg.num_relations} relations")

    # 6. JSON roundtrip.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    kg2.save_json(path)
    kg_loaded = KnowledgeGraph.load_json(path)
    assert kg_loaded.num_triples == kg2.num_triples
    print(f"\nJSON roundtrip: OK ({kg2.num_triples} triples, no unsafe pickle)")
    os.unlink(path)

    # 7. Temporal KG.
    n_events = 30
    triples = torch.tensor([[i % 10, i % 3, (i + 1) % 10] for i in range(n_events)], dtype=torch.long)
    timestamps = torch.sort(torch.rand(n_events))[0] * 100
    tkg = TemporalKnowledgeGraph(triples, timestamps, num_entities=10, num_relations=3)
    tr_tkg, va_tkg, te_tkg = tkg.chronological_split(0.6, 0.2, 0.2)
    print(f"\nTemporal KG ({n_events} events):")
    print(f"  train max time: {tr_tkg.timestamp.max():.2f}")
    print(f"  valid min time: {va_tkg.timestamp.min():.2f} (no leakage)")

    # 8. Summary.
    summary = kg.summary()
    print(f"\nKG summary (JSON-safe):")
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
