"""Tests for KG reasoning: path extraction, Horn rules, constraint checker."""
from __future__ import annotations

import torch

from tgraphx.kg import (
    generate_synthetic_kg,
    PathExtractor,
    HornRuleCandidate,
    LogicalConstraintChecker,
)
from tgraphx.kg import KnowledgeGraph


def _chain_kg():
    """0 -[0]-> 1 -[0]-> 2: path (0,) from 0 to 1, (0,) from 1 to 2, (0,0) from 0 to 2."""
    triples = torch.tensor([[0, 0, 1], [1, 0, 2]], dtype=torch.long)
    return KnowledgeGraph(triples, num_entities=3, num_relations=1)


class TestPathExtractor:

    def test_direct_path(self):
        kg = _chain_kg()
        ex = PathExtractor(kg, max_path_length=2)
        paths = ex.paths(0, 1)
        assert (0,) in paths

    def test_two_hop_path(self):
        kg = _chain_kg()
        ex = PathExtractor(kg, max_path_length=2)
        paths = ex.paths(0, 2)
        assert (0, 0) in paths

    def test_no_path_to_unreachable(self):
        kg = _chain_kg()
        ex = PathExtractor(kg, max_path_length=2)
        paths = ex.paths(2, 0)  # reverse, directed
        assert len(paths) == 0

    def test_max_path_length_respected(self):
        # Linear chain of length 3.
        triples = torch.tensor([[0, 0, 1], [1, 0, 2], [2, 0, 3]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=4, num_relations=1)
        ex = PathExtractor(kg, max_path_length=2)
        paths = ex.paths(0, 3)
        # Would need length 3, but max=2.
        assert (0, 0, 0) not in paths

    def test_max_paths_per_pair_guard(self):
        kg = generate_synthetic_kg(10, 3, 30, seed=0)
        ex = PathExtractor(kg, max_path_length=2, max_paths_per_pair=2)
        paths = ex.paths(0, 5)
        assert len(paths) <= 2

    def test_extract_all_paths_has_support(self):
        kg = _chain_kg()
        ex = PathExtractor(kg, max_path_length=2)
        # Target relation 0: (0,0,1) and (1,0,2).
        # No non-trivial body paths here since only 1 relation.
        path_support = ex.extract_all_paths(target_relation=0)
        assert isinstance(path_support, dict)


class TestLogicalConstraintChecker:

    def test_symmetric_violation_detected(self):
        # (0,0,1) exists but (1,0,0) does not.
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=2, num_relations=1)
        checker = LogicalConstraintChecker(kg, symmetric_relations={0})
        violations = checker.check_all()
        assert len(violations["symmetric"]) == 1

    def test_no_symmetric_violation_when_both_exist(self):
        triples = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=2, num_relations=1)
        checker = LogicalConstraintChecker(kg, symmetric_relations={0})
        violations = checker.check_all()
        assert len(violations["symmetric"]) == 0

    def test_antisymmetric_violation_detected(self):
        # Both (0,0,1) and (1,0,0) exist but relation 0 is antisymmetric.
        triples = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=2, num_relations=1)
        checker = LogicalConstraintChecker(kg, antisymmetric_relations={0})
        violations = checker.check_all()
        assert len(violations["antisymmetric"]) > 0

    def test_inverse_violation_detected(self):
        # Declare r=0 and r=1 as inverses. (0,0,1) exists but (1,1,0) does not.
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=2, num_relations=2)
        checker = LogicalConstraintChecker(kg, inverse_pairs={0: 1})
        violations = checker.check_all()
        assert len(violations["inverse"]) == 1

    def test_violation_summary_json_safe(self):
        import json
        triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
        kg = KnowledgeGraph(triples, num_entities=2, num_relations=1)
        checker = LogicalConstraintChecker(kg, symmetric_relations={0})
        summary = checker.violation_summary()
        json.dumps(summary)  # must not raise
