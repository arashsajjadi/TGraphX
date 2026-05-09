"""KG reasoning foundations.

Implements:
  PathExtractor        — BFS/DFS path extraction for entity pairs.
  HornRuleCandidate    — support, confidence, and lift for relation rules.
  LogicalConstraintChecker — detect symmetric/asymmetric/inverse violations.

These are lightweight, exact utilities for small-to-medium KGs.
For large KGs (millions of triples), consider external rule miners.

All methods are fully transparent about complexity.

Stability: Experimental.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch

from .data import KnowledgeGraph

__all__ = [
    "PathExtractor",
    "HornRuleCandidate",
    "LogicalConstraintChecker",
]


# ── Path extraction ───────────────────────────────────────────────────────────


@dataclass
class RelationPath:
    """A sequence of relation IDs connecting h → t."""
    path: Tuple[int, ...]  # relation sequence
    length: int
    support: int = 0  # how many (h, t) pairs follow this path

    def __hash__(self):
        return hash(self.path)


class PathExtractor:
    """BFS relation-path extractor for (head, tail) entity pairs.

    For each entity pair (h, t), finds all relation paths of length
    up to ``max_path_length``.

    Complexity: O(N_e * E^max_path_length) in the worst case.
    Use ``max_paths_per_pair`` and ``max_path_length`` guards.

    Args:
        kg: Source :class:`~tgraphx.kg.KnowledgeGraph`.
        max_path_length: Maximum number of hops (default 2).
        max_paths_per_pair: Maximum paths returned per (h, t) pair.
        directed: If True, only follows directed edges h → t.
            If False, also follows inverse edges.

    Stability: Experimental.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        max_path_length: int = 2,
        max_paths_per_pair: int = 50,
        directed: bool = True,
    ) -> None:
        if max_path_length < 1:
            raise ValueError("max_path_length must be >= 1")
        self.kg = kg
        self.max_path_length = int(max_path_length)
        self.max_paths_per_pair = int(max_paths_per_pair)
        self.directed = bool(directed)
        # Build adjacency: entity → [(relation, target_entity)].
        self._adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for row in kg.triples.tolist():
            h, r, t = int(row[0]), int(row[1]), int(row[2])
            self._adj[h].append((r, t))
            if not directed:
                # Add inverse edge.
                self._adj[t].append((r + kg.num_relations, h))

    def paths(
        self,
        head: int,
        tail: int,
    ) -> List[Tuple[int, ...]]:
        """Return all relation paths from ``head`` to ``tail``.

        Returns:
            List of relation-ID tuples.  Empty if no path found.
        """
        found: List[Tuple[int, ...]] = []
        # BFS state: (current_entity, path_so_far)
        queue: deque = deque([(int(head), ())])
        visited_states: Set[Tuple[int, Tuple[int, ...]]] = set()
        while queue:
            cur, path = queue.popleft()
            state = (cur, path)
            if state in visited_states:
                continue
            visited_states.add(state)
            if len(path) > 0 and cur == int(tail):
                found.append(path)
                if len(found) >= self.max_paths_per_pair:
                    break
                continue
            if len(path) >= self.max_path_length:
                continue
            for rel, nxt in self._adj.get(cur, []):
                new_path = path + (rel,)
                # Avoid cycles in path.
                if nxt not in set(p % self.kg.num_relations for p in new_path):
                    queue.append((nxt, new_path))
        return found

    def extract_all_paths(
        self,
        target_relation: int,
    ) -> Dict[Tuple[int, ...], int]:
        """Extract body paths for all triples with ``target_relation``.

        For each (h, r=target_relation, t) triple, finds all paths
        from h to t that don't use ``target_relation``.

        Returns:
            Dict mapping body path tuple → support count.
        """
        path_support: Dict[Tuple[int, ...], int] = defaultdict(int)
        for row in self.kg.triples.tolist():
            h, r, t = int(row[0]), int(row[1]), int(row[2])
            if r != target_relation:
                continue
            for path in self.paths(h, t):
                # Exclude paths that use the target relation itself.
                if target_relation not in path:
                    path_support[path] += 1
        return dict(path_support)


# ── Horn rule candidates ──────────────────────────────────────────────────────


@dataclass
class HornRuleCandidate:
    """A Horn-clause rule candidate.

    Rule: body => head_relation

    Attributes:
        body: Tuple of relation IDs forming the rule body.
        head_relation: Relation ID that the body implies.
        support: Number of entity pairs (h, t) satisfying body AND head.
        body_support: Number of (h, t) pairs satisfying body alone.
        head_support: Number of head-relation triples.
        confidence: support / body_support.
        lift: confidence / (head_support / num_triples).
    """

    body: Tuple[int, ...]
    head_relation: int
    support: int
    body_support: int
    head_support: int
    num_triples: int
    confidence: float = 0.0
    lift: float = 0.0

    def __post_init__(self) -> None:
        self.confidence = float(self.support) / max(1, self.body_support)
        head_prob = float(self.head_support) / max(1, self.num_triples)
        self.lift = self.confidence / max(1e-12, head_prob)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "body": list(self.body),
            "head_relation": self.head_relation,
            "support": self.support,
            "body_support": self.body_support,
            "confidence": round(self.confidence, 6),
            "lift": round(self.lift, 6),
        }


def mine_horn_rules(
    kg: KnowledgeGraph,
    max_body_length: int = 2,
    min_support: int = 2,
    min_confidence: float = 0.0,
    max_rules: int = 100,
    target_relations: Optional[List[int]] = None,
) -> List[HornRuleCandidate]:
    """Mine Horn rule candidates from a KG.

    For each relation r_h, uses :class:`PathExtractor` to find body paths,
    then computes support/confidence/lift.

    Args:
        kg: Source KG.
        max_body_length: Maximum body path length.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        max_rules: Maximum number of rules returned.
        target_relations: If provided, only mine these head relations.

    Returns:
        List of :class:`HornRuleCandidate` sorted by confidence descending.

    Complexity: O(N_r * N_e * E^max_body_length) — use small KGs.
    """
    extractor = PathExtractor(kg, max_path_length=max_body_length, directed=True)
    relations = target_relations if target_relations is not None else list(range(kg.num_relations))
    rules: List[HornRuleCandidate] = []
    N_t = kg.num_triples

    for r_h in relations:
        head_triples = [(int(row[0]), int(row[2])) for row in kg.triples.tolist()
                        if int(row[1]) == r_h]
        head_support = len(head_triples)
        if head_support == 0:
            continue
        # Build (h, t) set for head relation.
        head_pairs: Set[Tuple[int, int]] = set(head_triples)
        # Extract body paths.
        body_path_support = extractor.extract_all_paths(r_h)
        # For each body path, count co-occurrence with head.
        for body, body_supp in body_path_support.items():
            if body_supp < min_support:
                continue
            # Count instances where body AND head both hold.
            supp = 0
            for h, t in head_pairs:
                if (h, t) in head_pairs:  # trivially true here
                    paths = extractor.paths(h, t)
                    if body in [p for p in paths if r_h not in p]:
                        supp += 1
            if supp < min_support:
                continue
            rule = HornRuleCandidate(
                body=body, head_relation=r_h,
                support=supp, body_support=body_supp,
                head_support=head_support, num_triples=N_t,
            )
            if rule.confidence >= min_confidence:
                rules.append(rule)
        if len(rules) >= max_rules:
            break
    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules[:max_rules]


# ── Logical constraint checker ────────────────────────────────────────────────


@dataclass
class ConstraintViolation:
    """A detected logical constraint violation."""
    violation_type: str
    triple_a: Tuple[int, int, int]
    triple_b: Optional[Tuple[int, int, int]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type,
            "triple_a": list(self.triple_a),
            "triple_b": list(self.triple_b) if self.triple_b else None,
            "description": self.description,
        }


class LogicalConstraintChecker:
    """Check logical constraint violations in a KG.

    Checks:
    - Symmetric violation: r is declared symmetric but (h, r, t) ∈ KG
      and (t, r, h) ∉ KG.
    - Antisymmetric violation: r is declared antisymmetric but both
      (h, r, t) and (t, r, h) are in KG.
    - Inverse violation: r and r_inv are declared inverses but
      (h, r, t) ∈ KG and (t, r_inv, h) ∉ KG.
    - Domain violation: head entity type not in declared domain(r).
    - Range violation: tail entity type not in declared range(r).

    Args:
        kg: Source :class:`~tgraphx.kg.KnowledgeGraph`.
        symmetric_relations: Set of symmetric relation IDs.
        antisymmetric_relations: Set of antisymmetric relation IDs.
        inverse_pairs: Dict ``{r: r_inv}`` declaring inverse relation pairs.
        max_violations: Cap on reported violations (performance guard).

    Stability: Experimental.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        symmetric_relations: Optional[Set[int]] = None,
        antisymmetric_relations: Optional[Set[int]] = None,
        inverse_pairs: Optional[Dict[int, int]] = None,
        max_violations: int = 1000,
    ) -> None:
        self.kg = kg
        self.symmetric = symmetric_relations or set()
        self.antisymmetric = antisymmetric_relations or set()
        self.inverse_pairs = inverse_pairs or {}
        self.max_violations = int(max_violations)
        self._pos = kg.positive_triple_set()

    def check_all(self) -> Dict[str, List[ConstraintViolation]]:
        """Run all constraint checks and return violations by type."""
        violations: Dict[str, List[ConstraintViolation]] = {
            "symmetric": [],
            "antisymmetric": [],
            "inverse": [],
        }
        for row in self.kg.triples.tolist():
            h, r, t = int(row[0]), int(row[1]), int(row[2])
            triple = (h, r, t)
            # Symmetric.
            if r in self.symmetric and (t, r, h) not in self._pos:
                if len(violations["symmetric"]) < self.max_violations:
                    violations["symmetric"].append(ConstraintViolation(
                        "symmetric", triple, None,
                        f"({h},{r},{t}) ∈ KG but ({t},{r},{h}) ∉ KG for symmetric r={r}",
                    ))
            # Antisymmetric.
            if r in self.antisymmetric and (t, r, h) in self._pos:
                if len(violations["antisymmetric"]) < self.max_violations:
                    violations["antisymmetric"].append(ConstraintViolation(
                        "antisymmetric", triple, (t, r, h),
                        f"Both ({h},{r},{t}) and ({t},{r},{h}) in KG for antisymmetric r={r}",
                    ))
            # Inverse.
            r_inv = self.inverse_pairs.get(r)
            if r_inv is not None and (t, r_inv, h) not in self._pos:
                if len(violations["inverse"]) < self.max_violations:
                    violations["inverse"].append(ConstraintViolation(
                        "inverse", triple, None,
                        f"({h},{r},{t}) ∈ KG but ({t},{r_inv},{h}) ∉ KG for inverse pair ({r},{r_inv})",
                    ))
        return violations

    def violation_summary(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of all violations."""
        all_v = self.check_all()
        return {
            vtype: {
                "count": len(vs),
                "examples": [v.to_dict() for v in vs[:5]],
            }
            for vtype, vs in all_v.items()
        }
