"""Synthetic Knowledge Graph datasets for testing and tutorials.

All datasets are generated programmatically — no network access, no
hidden downloads, no data committed to the repository.

Available:
  FamilyKG         — parentOf, childOf, siblingOf, spouseOf, grandparentOf
  AcademicKG       — author/paper/venue/institution relations
  MultimodalKG     — entity vector + relation features
  generate_synthetic_kg — random toy KG

All datasets return a KnowledgeGraph with correct train/valid/test splits
and deterministic generation under a fixed seed.

Stability: Beta (synthetic datasets).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .data import KnowledgeGraph, TemporalKnowledgeGraph

__all__ = [
    "FamilyKG",
    "AcademicKG",
    "MultimodalKG",
    "generate_synthetic_kg",
]


# ── Family KG ─────────────────────────────────────────────────────────────────


class FamilyKG:
    """Synthetic family knowledge graph.

    Entities: persons 0..num_persons-1.
    Relations:
      0: parentOf
      1: childOf       (inverse of parentOf)
      2: siblingOf     (symmetric)
      3: spouseOf      (symmetric)
      4: grandparentOf (composed from parentOf ∘ parentOf)

    Args:
        num_persons: Number of person entities (must be >= 6).
        seed: Deterministic RNG seed.
        train_ratio: Train split ratio.
        valid_ratio: Valid split ratio.

    Usage::
        ds = FamilyKG(num_persons=30, seed=0)
        kg = ds.kg
        train, valid, test = ds.train, ds.valid, ds.test

    Stability: Beta.
    """

    RELATIONS = {
        "parentOf": 0,
        "childOf": 1,
        "siblingOf": 2,
        "spouseOf": 3,
        "grandparentOf": 4,
    }
    RELATION_NAMES = {v: k for k, v in RELATIONS.items()}
    NUM_RELATIONS = 5

    def __init__(
        self,
        num_persons: int = 30,
        seed: int = 0,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
    ) -> None:
        if num_persons < 6:
            raise ValueError("num_persons must be >= 6")
        self.num_entities = int(num_persons)
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        triples: list = []
        N = self.num_entities

        # Generate parent-child pairs (N//3 families).
        num_families = N // 3
        for i in range(num_families):
            parent = i
            child1 = num_families + i * 2 % (N - num_families)
            child2 = (child1 + 1) % N
            if child1 == parent or child2 == parent:
                continue
            triples.append((parent, 0, child1))  # parentOf
            triples.append((child1, 1, parent))  # childOf
            if child1 != child2:
                triples.append((parent, 0, child2))
                triples.append((child2, 1, parent))
                triples.append((child1, 2, child2))  # siblingOf
                triples.append((child2, 2, child1))

        # Spouse pairs.
        num_couples = N // 4
        for i in range(num_couples):
            a = i * 2 % N
            b = (a + 1) % N
            if a != b and (a, 3, b) not in set(map(tuple, triples)):
                triples.append((a, 3, b))
                triples.append((b, 3, a))

        # Grandparent (if depth allows).
        for h, r, t in list(triples):
            if r == 0:  # parentOf
                for h2, r2, t2 in list(triples):
                    if h2 == t and r2 == 0 and t2 != h:
                        triples.append((h, 4, t2))

        # Deduplicate.
        triples = list(dict.fromkeys(map(tuple, triples)))
        if not triples:
            raise ValueError("Could not generate any triples; increase num_persons")

        tri_t = torch.tensor(triples, dtype=torch.long)
        entity_to_id = {f"person_{i}": i for i in range(N)}
        self.kg = KnowledgeGraph(
            tri_t,
            num_entities=N,
            num_relations=self.NUM_RELATIONS,
            entity_to_id=entity_to_id,
            relation_to_id=self.RELATIONS,
            metadata={"dataset": "FamilyKG", "seed": seed},
        )
        self.train, self.valid, self.test = self.kg.train_valid_test_split(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=1.0 - train_ratio - valid_ratio,
            seed=seed,
        )


# ── Academic KG ───────────────────────────────────────────────────────────────


class AcademicKG:
    """Synthetic academic knowledge graph.

    Entities: authors (0..A-1), papers (A..A+P-1), venues (A+P..A+P+V-1),
              institutions (A+P+V..A+P+V+I-1).
    Relations:
      0: wrote          (author → paper)
      1: publishedIn    (paper → venue)
      2: cites          (paper → paper)
      3: affiliatedWith (author → institution)
      4: hostedBy       (venue → institution)  [optional]

    Args:
        num_authors: A.
        num_papers: P.
        num_venues: V.
        num_institutions: I.
        seed: RNG seed.

    Stability: Beta.
    """

    NUM_RELATIONS = 5

    def __init__(
        self,
        num_authors: int = 20,
        num_papers: int = 30,
        num_venues: int = 5,
        num_institutions: int = 4,
        seed: int = 0,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
    ) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        A, P, V, I = num_authors, num_papers, num_venues, num_institutions
        N = A + P + V + I
        offset_p = A
        offset_v = A + P
        offset_i = A + P + V
        triples: list = []

        # Author wrote paper (each paper has 1-2 authors).
        for p in range(P):
            author = int(torch.randint(A, (1,), generator=gen).item())
            triples.append((author, 0, offset_p + p))
            if torch.rand(1, generator=gen).item() > 0.5:
                author2 = int(torch.randint(A, (1,), generator=gen).item())
                if author2 != author:
                    triples.append((author2, 0, offset_p + p))

        # Paper publishedIn venue.
        for p in range(P):
            venue = int(torch.randint(V, (1,), generator=gen).item())
            triples.append((offset_p + p, 1, offset_v + venue))

        # Paper cites paper (sparse).
        for p in range(P):
            if torch.rand(1, generator=gen).item() > 0.6:
                p2 = int(torch.randint(P, (1,), generator=gen).item())
                if p2 != p:
                    triples.append((offset_p + p, 2, offset_p + p2))

        # Author affiliatedWith institution.
        for a in range(A):
            inst = int(torch.randint(I, (1,), generator=gen).item())
            triples.append((a, 3, offset_i + inst))

        # Venue hostedBy institution.
        for v in range(V):
            inst = int(torch.randint(I, (1,), generator=gen).item())
            triples.append((offset_v + v, 4, offset_i + inst))

        triples = list(dict.fromkeys(map(tuple, triples)))
        tri_t = torch.tensor(triples, dtype=torch.long)
        self.kg = KnowledgeGraph(
            tri_t,
            num_entities=N,
            num_relations=self.NUM_RELATIONS,
            metadata={"dataset": "AcademicKG", "seed": seed,
                      "num_authors": A, "num_papers": P,
                      "num_venues": V, "num_institutions": I},
        )
        self.train, self.valid, self.test = self.kg.train_valid_test_split(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=1.0 - train_ratio - valid_ratio,
            seed=seed,
        )


# ── Multimodal KG ─────────────────────────────────────────────────────────────


class MultimodalKG:
    """Synthetic KG with entity vector features and relation text embeddings.

    Generates a random KG where:
    - Entity features: ``FloatTensor[N_e, entity_feature_dim]``
    - Relation features: ``FloatTensor[N_r, relation_feature_dim]``

    Image and volume features are NOT generated here; see docs for
    how to attach arbitrary tensor features using ``entity_features`` dict.

    Args:
        num_entities: N_e.
        num_relations: N_r.
        num_triples: N_t.
        entity_feature_dim: Vector feature size per entity.
        relation_feature_dim: Vector feature size per relation.
        seed: RNG seed.

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int = 50,
        num_relations: int = 5,
        num_triples: int = 200,
        entity_feature_dim: int = 16,
        relation_feature_dim: int = 8,
        seed: int = 0,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
    ) -> None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        N_e, N_r, N_t = num_entities, num_relations, num_triples

        # Random triples (deduplicated).
        heads = torch.randint(N_e, (N_t * 2,), generator=gen)
        rels = torch.randint(N_r, (N_t * 2,), generator=gen)
        tails = torch.randint(N_e, (N_t * 2,), generator=gen)
        # Exclude self-loops.
        valid = heads != tails
        raw = torch.stack([heads[valid], rels[valid], tails[valid]], dim=1)
        unique_triples = torch.unique(raw, dim=0)[:N_t]

        entity_features = torch.randn(N_e, entity_feature_dim, generator=gen)
        relation_features = torch.randn(N_r, relation_feature_dim, generator=gen)

        self.kg = KnowledgeGraph(
            unique_triples,
            num_entities=N_e,
            num_relations=N_r,
            entity_features={"x": entity_features},
            relation_features={"r": relation_features},
            metadata={"dataset": "MultimodalKG", "seed": seed},
        )
        self.train, self.valid, self.test = self.kg.train_valid_test_split(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=1.0 - train_ratio - valid_ratio,
            seed=seed,
        )


# ── Generic synthetic ─────────────────────────────────────────────────────────


def generate_synthetic_kg(
    num_entities: int = 50,
    num_relations: int = 5,
    num_triples: int = 200,
    seed: int = 0,
    with_weights: bool = False,
    with_timestamps: bool = False,
) -> KnowledgeGraph:
    """Generate a deterministic random KG.

    Args:
        num_entities: N_e.
        num_relations: N_r.
        num_triples: Target triple count (may be fewer after deduplication).
        seed: RNG seed.
        with_weights: Include random edge weights in [0, 1].
        with_timestamps: If True, returns a :class:`TemporalKnowledgeGraph`
            with random timestamps in [0, 100].

    Returns:
        :class:`KnowledgeGraph` or :class:`TemporalKnowledgeGraph`.

    Stability: Beta.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    heads = torch.randint(num_entities, (num_triples * 2,), generator=gen)
    rels = torch.randint(num_relations, (num_triples * 2,), generator=gen)
    tails = torch.randint(num_entities, (num_triples * 2,), generator=gen)
    mask = heads != tails
    raw = torch.unique(torch.stack([heads[mask], rels[mask], tails[mask]], dim=1), dim=0)
    triples = raw[:num_triples]

    ew = torch.rand(triples.size(0), generator=gen) if with_weights else None

    if with_timestamps:
        ts = torch.sort(torch.rand(triples.size(0), generator=gen) * 100)[0]
        return TemporalKnowledgeGraph(
            triples, ts,
            num_entities=num_entities,
            num_relations=num_relations,
            edge_weight=ew,
            metadata={"dataset": "synthetic", "seed": seed},
        )
    return KnowledgeGraph(
        triples,
        num_entities=num_entities,
        num_relations=num_relations,
        edge_weight=ew,
        metadata={"dataset": "synthetic", "seed": seed},
    )
