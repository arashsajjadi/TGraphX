"""Tensor-aware Knowledge Graph data model.

Formal definition:
  G_KG = (E, R, T, X_E, X_R, X_T, W, τ, C, M)

where:
  E   = entity set, |E| = N_e
  R   = relation set, |R| = N_r
  T   = triple set  T ⊆ E × R × E, |T| = N_t
  X_E = optional entity feature dict {name: Tensor[N_e, *]}
  X_R = optional relation feature dict {name: Tensor[N_r, *]}
  X_T = optional triple feature dict {name: Tensor[N_t, *]}
  W   = optional triple weight Tensor[N_t]
  τ   = optional timestamp Tensor[N_t]  (TemporalKnowledgeGraph)
  C   = optional confidence Tensor[N_t]
  M   = JSON-safe metadata dict

Core storage:
  triples: LongTensor[N_t, 3]
    col 0 = head entity IDs
    col 1 = relation IDs
    col 2 = tail entity IDs

Tensor features are stored in dicts and are NEVER silently flattened.
Every tensor must have its first dimension equal to the size of the
corresponding set (entities, relations, or triples).

Stability: Beta (v0.6.0+).
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

__all__ = [
    "KnowledgeGraph",
    "TemporalKnowledgeGraph",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _check_first_dim(name: str, tensor: torch.Tensor, expected: int) -> None:
    if tensor.dim() < 1:
        raise ValueError(f"Feature '{name}' must be at least 1-D; got scalar")
    if tensor.size(0) != expected:
        raise ValueError(
            f"Feature '{name}' has first dimension {tensor.size(0)}, "
            f"expected {expected}"
        )


def _move_dict(d: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if d is None:
        return None
    return {k: v.to(device=device) for k, v in d.items()}


def _detach_dict(d: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if d is None:
        return None
    return {k: v.detach() for k, v in d.items()}


def _clone_dict(d: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if d is None:
        return None
    return {k: v.clone() for k, v in d.items()}


def _summary_dict(d: Optional[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    if d is None:
        return {}
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in d.items()}


# ── KnowledgeGraph ────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """Tensor-aware directed multi-relational knowledge graph.

    Args:
        triples: ``LongTensor[N_t, 3]`` of (head, relation, tail) integer IDs.
        num_entities: Entity vocabulary size.  Inferred from max when None.
        num_relations: Relation vocabulary size.  Inferred from max when None.
        entity_features: Dict of entity feature tensors
            ``{name: Tensor[N_e, *]}``.  First dim must equal
            ``num_entities``.  Tensors are stored as-is — **never flattened**.
        relation_features: Dict of relation feature tensors
            ``{name: Tensor[N_r, *]}``.
        triple_features: Dict of triple/edge feature tensors
            ``{name: Tensor[N_t, *]}``.
        edge_weight: ``FloatTensor[N_t]`` optional triple weights.
        confidence: ``FloatTensor[N_t]`` optional triple confidence scores.
        entity_to_id: Dict mapping entity name string → integer ID.
            When provided, enables ``has_triple`` by name.
        relation_to_id: Dict mapping relation name string → integer ID.
        metadata: JSON-safe dict for provenance, dataset info, etc.

    Stability: Beta.
    """

    def __init__(
        self,
        triples: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        entity_features: Optional[Dict[str, torch.Tensor]] = None,
        relation_features: Optional[Dict[str, torch.Tensor]] = None,
        triple_features: Optional[Dict[str, torch.Tensor]] = None,
        edge_weight: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        entity_to_id: Optional[Dict[str, int]] = None,
        relation_to_id: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        # ── Multimodal extensions ──────────────────────────────────────────
        entity_types: Optional[torch.Tensor] = None,
        entity_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
        entity_type_to_id: Optional[Dict[str, int]] = None,
    ) -> None:
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError(
                f"triples must have shape [N_t, 3] where each row is "
                f"(head_id, relation_id, tail_id); got {tuple(triples.shape)}.\n"
                f"If you have separate heads, relations, tails tensors, use:\n"
                f"    KnowledgeGraph.from_hrt(heads, relations, tails, ...)\n"
                f"If you have a list of (h, r, t) tuples, use:\n"
                f"    KnowledgeGraph.from_triples([(h, r, t), ...])"
            )
        if triples.dtype != torch.long:
            triples = triples.to(torch.long)
        self.triples = triples

        # --- Entity count ---
        if triples.numel() > 0:
            _ne = int(max(int(triples[:, 0].max()), int(triples[:, 2].max())) + 1)
        else:
            _ne = 0
        self.num_entities: int = int(num_entities) if num_entities is not None else _ne

        # --- Relation count ---
        if triples.numel() > 0:
            _nr = int(triples[:, 1].max()) + 1
        else:
            _nr = 0
        self.num_relations: int = int(num_relations) if num_relations is not None else _nr

        N_t = int(triples.size(0))

        # --- Validate and store feature dicts ---
        self.entity_features: Dict[str, torch.Tensor] = {}
        for name, feat in (entity_features or {}).items():
            _check_first_dim(f"entity_features['{name}']", feat, self.num_entities)
            self.entity_features[name] = feat

        self.relation_features: Dict[str, torch.Tensor] = {}
        for name, feat in (relation_features or {}).items():
            _check_first_dim(f"relation_features['{name}']", feat, self.num_relations)
            self.relation_features[name] = feat

        self.triple_features: Dict[str, torch.Tensor] = {}
        for name, feat in (triple_features or {}).items():
            _check_first_dim(f"triple_features['{name}']", feat, N_t)
            self.triple_features[name] = feat

        # --- Optional scalar tensors ---
        if edge_weight is not None:
            if edge_weight.dim() != 1 or edge_weight.size(0) != N_t:
                raise ValueError(
                    f"edge_weight must be 1-D with length N_t={N_t}; "
                    f"got {tuple(edge_weight.shape)}"
                )
        self.edge_weight: Optional[torch.Tensor] = edge_weight

        if confidence is not None:
            if confidence.dim() != 1 or confidence.size(0) != N_t:
                raise ValueError(
                    f"confidence must be 1-D with length N_t={N_t}; "
                    f"got {tuple(confidence.shape)}"
                )
        self.confidence: Optional[torch.Tensor] = confidence

        # --- Entity types (optional, multimodal) ---
        if entity_types is not None:
            if entity_types.dim() != 1 or entity_types.size(0) != self.num_entities:
                raise ValueError(
                    f"entity_types must be 1-D LongTensor[N_e={self.num_entities}]; "
                    f"got {tuple(entity_types.shape)}"
                )
            entity_types = entity_types.to(torch.long)
        self.entity_types: Optional[torch.Tensor] = entity_types

        # --- Entity feature masks: BoolTensor[N_e] per modality ---
        self.entity_feature_masks: Dict[str, torch.Tensor] = {}
        for name, mask in (entity_feature_masks or {}).items():
            if mask.dim() != 1 or mask.size(0) != self.num_entities:
                raise ValueError(
                    f"entity_feature_masks['{name}'] must be 1-D BoolTensor[N_e={self.num_entities}]; "
                    f"got {tuple(mask.shape)}"
                )
            self.entity_feature_masks[name] = mask.to(torch.bool)

        # --- String entity-type mapping (optional) ---
        self.entity_type_to_id: Optional[Dict[str, int]] = entity_type_to_id
        self.id_to_entity_type: Optional[Dict[int, str]] = (
            {v: k for k, v in entity_type_to_id.items()} if entity_type_to_id else None
        )

        # --- ID mappings ---
        self.entity_to_id: Optional[Dict[str, int]] = entity_to_id
        self.relation_to_id: Optional[Dict[str, int]] = relation_to_id

        # --- Positive set for fast lookup ---
        self._positive_set: Set[Tuple[int, int, int]] = set()
        if triples.numel() > 0:
            for row in triples.tolist():
                self._positive_set.add((int(row[0]), int(row[1]), int(row[2])))

        self.metadata: Dict[str, Any] = dict(metadata or {})

    # ── Alternative constructors ────────────────────────────────────────────

    @classmethod
    def from_hrt(
        cls,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        **kwargs,
    ) -> "KnowledgeGraph":
        """Create a KnowledgeGraph from separate head, relation, and tail tensors.

        This is a convenience constructor for users who have three parallel
        1-D tensors rather than a combined ``[N_t, 3]`` triples matrix.

        Args:
            heads: ``LongTensor[N_t]`` of head entity IDs.
            relations: ``LongTensor[N_t]`` of relation IDs.
            tails: ``LongTensor[N_t]`` of tail entity IDs.
            num_entities: Vocabulary size for entities.  Inferred from max if None.
            num_relations: Vocabulary size for relations.  Inferred from max if None.
            **kwargs: Forwarded to :class:`KnowledgeGraph` (e.g.
                ``entity_features``, ``metadata``).

        Returns:
            :class:`KnowledgeGraph` with ``triples`` assembled as
            ``torch.stack([heads, relations, tails], dim=1)``.

        Example::

            import torch
            from tgraphx.kg import KnowledgeGraph

            heads     = torch.tensor([0, 1, 2])
            relations = torch.tensor([0, 0, 1])
            tails     = torch.tensor([1, 2, 0])
            kg = KnowledgeGraph.from_hrt(
                heads, relations, tails,
                num_entities=3, num_relations=2,
            )

        See also:
            :meth:`from_triples` accepts a list of ``(h, r, t)`` tuples or a
            ``[N_t, 3]`` tensor directly.
        """
        for name, t in [("heads", heads), ("relations", relations), ("tails", tails)]:
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, got {type(t).__name__}"
                )
            if t.dim() != 1:
                raise ValueError(
                    f"{name} must be a 1-D tensor; got shape {tuple(t.shape)}"
                )
        if not (heads.size(0) == relations.size(0) == tails.size(0)):
            raise ValueError(
                f"heads, relations, and tails must have the same length; "
                f"got {heads.size(0)}, {relations.size(0)}, {tails.size(0)}"
            )
        triples = torch.stack(
            [heads.to(torch.long), relations.to(torch.long), tails.to(torch.long)],
            dim=1,
        )
        return cls(triples, num_entities=num_entities, num_relations=num_relations, **kwargs)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def heads(self) -> torch.Tensor:
        return self.triples[:, 0]

    @property
    def relations(self) -> torch.Tensor:
        return self.triples[:, 1]

    @property
    def tails(self) -> torch.Tensor:
        return self.triples[:, 2]

    @property
    def num_triples(self) -> int:
        return int(self.triples.size(0))

    @property
    def device(self) -> torch.device:
        return self.triples.device

    def __len__(self) -> int:
        return self.num_triples

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={self.num_entities}, "
            f"relations={self.num_relations}, "
            f"triples={self.num_triples})"
        )

    # ── Factory methods ─────────────────────────────────────────────────────

    @classmethod
    def from_triples(
        cls,
        triples: Union[List[Tuple], torch.Tensor],
        entity_to_id: Optional[Dict[str, int]] = None,
        relation_to_id: Optional[Dict[str, int]] = None,
        **kwargs: Any,
    ) -> "KnowledgeGraph":
        """Build a KG from a list of (head, relation, tail) tuples or a tensor.

        When ``triples`` is a list of string tuples and ``entity_to_id`` and
        ``relation_to_id`` are provided, they are used for mapping; otherwise
        a deterministic sorted mapping is built.
        """
        if isinstance(triples, torch.Tensor):
            return cls(triples, entity_to_id=entity_to_id,
                       relation_to_id=relation_to_id, **kwargs)
        # List of tuples.  May be (str, str, str) or (int, int, int).
        heads_raw, rels_raw, tails_raw = zip(*triples) if triples else ([], [], [])
        if triples and isinstance(heads_raw[0], str):
            if entity_to_id is None:
                ents = sorted(set(heads_raw) | set(tails_raw))
                entity_to_id = {e: i for i, e in enumerate(ents)}
            if relation_to_id is None:
                rels = sorted(set(rels_raw))
                relation_to_id = {r: i for i, r in enumerate(rels)}
            h_ids = [entity_to_id[h] for h in heads_raw]
            r_ids = [relation_to_id[r] for r in rels_raw]
            t_ids = [entity_to_id[t] for t in tails_raw]
        else:
            h_ids = list(heads_raw)
            r_ids = list(rels_raw)
            t_ids = list(tails_raw)
        triples_t = torch.tensor(
            [[h, r, t] for h, r, t in zip(h_ids, r_ids, t_ids)],
            dtype=torch.long,
        ) if triples else torch.zeros((0, 3), dtype=torch.long)
        return cls(triples_t, entity_to_id=entity_to_id,
                   relation_to_id=relation_to_id, **kwargs)

    @classmethod
    def from_edge_index(
        cls,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        **kwargs: Any,
    ) -> "KnowledgeGraph":
        """Build KG from PyG-style edge_index and edge_type tensors."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if edge_type.dim() != 1 or edge_type.size(0) != edge_index.size(1):
            raise ValueError("edge_type must have shape [E] matching edge_index")
        triples = torch.stack([edge_index[0], edge_type, edge_index[1]], dim=1).long()
        return cls(triples, num_entities=num_entities, num_relations=num_relations,
                   **kwargs)

    # ── Lookups ─────────────────────────────────────────────────────────────

    def has_triple(self, h: int, r: int, t: int) -> bool:
        """Return True if (h, r, t) is in this graph."""
        return (int(h), int(r), int(t)) in self._positive_set

    def positive_triple_set(self) -> Set[Tuple[int, int, int]]:
        """Return a copy of the positive triple set."""
        return set(self._positive_set)

    # ── Derived views ───────────────────────────────────────────────────────

    def to_edge_index(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (edge_index, edge_type) PyG-compatible tensors.

        Returns:
            edge_index: ``LongTensor[2, N_t]``
            edge_type: ``LongTensor[N_t]``
        """
        ei = torch.stack([self.triples[:, 0], self.triples[:, 2]], dim=0)
        return ei, self.triples[:, 1]

    def add_inverse_relations(self) -> "KnowledgeGraph":
        """Return a new KG with inverse triples added.

        For each triple (h, r, t) the inverse (t, r+N_r, h) is appended.
        Inverse relation IDs are ``[N_r, 2*N_r)``.

        Returns:
            New KnowledgeGraph with 2*N_t triples and 2*N_r relations.
        """
        inv = torch.stack([
            self.triples[:, 2],
            self.triples[:, 1] + self.num_relations,
            self.triples[:, 0],
        ], dim=1)
        new_triples = torch.cat([self.triples, inv], dim=0)
        # Inverse triple features.
        new_tf: Dict[str, torch.Tensor] = {}
        for k, v in self.triple_features.items():
            new_tf[k] = torch.cat([v, v], dim=0)  # duplicate
        new_ew = None
        if self.edge_weight is not None:
            new_ew = torch.cat([self.edge_weight, self.edge_weight])
        new_conf = None
        if self.confidence is not None:
            new_conf = torch.cat([self.confidence, self.confidence])
        return KnowledgeGraph(
            new_triples,
            num_entities=self.num_entities,
            num_relations=self.num_relations * 2,
            entity_features=_clone_dict(self.entity_features) or {},
            relation_features=None,  # cannot simply double
            triple_features=new_tf or None,
            edge_weight=new_ew,
            confidence=new_conf,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            metadata={**self.metadata, "has_inverse_relations": True,
                      "original_num_relations": self.num_relations},
        )

    def train_valid_test_split(
        self,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 0,
    ) -> Tuple["KnowledgeGraph", "KnowledgeGraph", "KnowledgeGraph"]:
        """Random (non-chronological) triple split with no overlap.

        Returns three KnowledgeGraph objects whose positive sets are disjoint.
        """
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        T = self.num_triples
        perm = torch.randperm(T, generator=gen)
        n_train = int(round(train_ratio * T))
        n_valid = int(round(valid_ratio * T))
        idx_tr = perm[:n_train]
        idx_va = perm[n_train:n_train + n_valid]
        idx_te = perm[n_train + n_valid:]
        kw = dict(num_entities=self.num_entities, num_relations=self.num_relations,
                  entity_to_id=self.entity_to_id, relation_to_id=self.relation_to_id,
                  entity_types=self.entity_types.clone() if self.entity_types is not None else None,
                  entity_feature_masks=_clone_dict(self.entity_feature_masks) or None,
                  entity_type_to_id=dict(self.entity_type_to_id) if self.entity_type_to_id else None)
        def _sub(idx: torch.Tensor) -> "KnowledgeGraph":
            return KnowledgeGraph(
                self.triples[idx],
                entity_features=_clone_dict(self.entity_features) or None,
                relation_features=_clone_dict(self.relation_features) or None,
                triple_features={k: v[idx] for k, v in self.triple_features.items()} or None,
                edge_weight=self.edge_weight[idx] if self.edge_weight is not None else None,
                confidence=self.confidence[idx] if self.confidence is not None else None,
                metadata=dict(self.metadata),
                **kw,
            )
        return _sub(idx_tr), _sub(idx_va), _sub(idx_te)

    # ── Device / dtype movement ─────────────────────────────────────────────

    def to(self, device: Union[str, torch.device]) -> "KnowledgeGraph":
        """Move all tensors to ``device`` in-place and return self."""
        dev = torch.device(device)
        self.triples = self.triples.to(dev)
        self.entity_features = _move_dict(self.entity_features, dev) or {}
        self.relation_features = _move_dict(self.relation_features, dev) or {}
        self.triple_features = _move_dict(self.triple_features, dev) or {}
        self.entity_feature_masks = _move_dict(self.entity_feature_masks, dev) or {}
        if self.entity_types is not None:
            self.entity_types = self.entity_types.to(dev)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(dev)
        if self.confidence is not None:
            self.confidence = self.confidence.to(dev)
        return self

    def cpu(self) -> "KnowledgeGraph":
        return self.to("cpu")

    def cuda(self, device: Union[int, str, None] = None) -> "KnowledgeGraph":
        dev = f"cuda:{device}" if isinstance(device, int) else (device or "cuda")
        return self.to(dev)

    def clone(self) -> "KnowledgeGraph":
        """Return a deep copy on the same device."""
        return KnowledgeGraph(
            self.triples.clone(),
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            entity_features=_clone_dict(self.entity_features) or None,
            relation_features=_clone_dict(self.relation_features) or None,
            triple_features=_clone_dict(self.triple_features) or None,
            edge_weight=self.edge_weight.clone() if self.edge_weight is not None else None,
            confidence=self.confidence.clone() if self.confidence is not None else None,
            entity_to_id=dict(self.entity_to_id) if self.entity_to_id else None,
            relation_to_id=dict(self.relation_to_id) if self.relation_to_id else None,
            metadata=dict(self.metadata),
            entity_types=self.entity_types.clone() if self.entity_types is not None else None,
            entity_feature_masks=_clone_dict(self.entity_feature_masks) or None,
            entity_type_to_id=dict(self.entity_type_to_id) if self.entity_type_to_id else None,
        )

    def detach(self) -> "KnowledgeGraph":
        """Detach all tensors from any autograd graph; return self."""
        self.triples = self.triples.detach()
        self.entity_features = _detach_dict(self.entity_features) or {}
        self.relation_features = _detach_dict(self.relation_features) or {}
        self.triple_features = _detach_dict(self.triple_features) or {}
        self.entity_feature_masks = _detach_dict(self.entity_feature_masks) or {}
        if self.entity_types is not None:
            self.entity_types = self.entity_types.detach()
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.detach()
        if self.confidence is not None:
            self.confidence = self.confidence.detach()
        return self

    def detach_for_report(self) -> "KnowledgeGraph":
        """Return a CPU-detached clone suitable for JSON reporting."""
        kg = self.clone().cpu()
        kg.detach()
        return kg

    # ── Summaries ───────────────────────────────────────────────────────────

    def entity_type_counts(self) -> Dict[str, int]:
        """Return per-type entity counts (requires entity_types and entity_type_to_id)."""
        if self.entity_types is None:
            return {}
        counts: Dict[str, int] = {}
        if self.id_to_entity_type:
            for t_id in self.entity_types.tolist():
                name = self.id_to_entity_type.get(int(t_id), f"type_{t_id}")
                counts[name] = counts.get(name, 0) + 1
        else:
            for t_id in self.entity_types.tolist():
                name = f"type_{int(t_id)}"
                counts[name] = counts.get(name, 0) + 1
        return counts

    def summary(self) -> Dict[str, Any]:
        """JSON-serialisable summary for dashboard reporting."""
        out: Dict[str, Any] = {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_triples": self.num_triples,
            "directed": True,
            "has_entity_features": bool(self.entity_features),
            "has_relation_features": bool(self.relation_features),
            "has_triple_features": bool(self.triple_features),
            "has_edge_weight": self.edge_weight is not None,
            "has_confidence": self.confidence is not None,
            "has_entity_types": self.entity_types is not None,
            "has_entity_feature_masks": bool(self.entity_feature_masks),
            "entity_features": _summary_dict(self.entity_features),
            "relation_features": _summary_dict(self.relation_features),
            "triple_features": _summary_dict(self.triple_features),
            "entity_feature_masks": {
                k: {"coverage": int(v.sum().item()), "total": int(v.size(0))}
                for k, v in self.entity_feature_masks.items()
            },
            "entity_type_counts": self.entity_type_counts(),
        }
        if self.edge_weight is not None:
            w = self.edge_weight.float()
            out["edge_weight_stats"] = {
                "min": round(float(w.min()), 6),
                "max": round(float(w.max()), 6),
                "mean": round(float(w.mean()), 6),
            }
        if self.confidence is not None:
            c = self.confidence.float()
            out["confidence_stats"] = {
                "min": round(float(c.min()), 6),
                "max": round(float(c.max()), 6),
                "mean": round(float(c.mean()), 6),
            }
        out.update(self.metadata)
        return out

    def relation_summary(self) -> Dict[str, Any]:
        """Per-relation triple counts."""
        rel_counts: Dict[int, int] = {}
        for r in self.relations.tolist():
            rel_counts[int(r)] = rel_counts.get(int(r), 0) + 1
        return {
            "num_relations": self.num_relations,
            "max_relation_triples": max(rel_counts.values(), default=0),
            "min_relation_triples": min(rel_counts.values(), default=0),
            "mean_relation_triples": round(
                sum(rel_counts.values()) / max(1, len(rel_counts)), 2
            ),
            "relation_counts": {str(k): v for k, v in sorted(rel_counts.items())[:50]},
        }

    # ── IO ───────────────────────────────────────────────────────────────────

    def save_tsv(self, path: Union[str, Path]) -> None:
        """Save triples to a TSV file (no features, no tensor data).

        Format: ``head_id\\trelation_id\\ttail_id``
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"{int(h)}\t{int(r)}\t{int(t)}\n"
            for h, r, t in self.triples.tolist()
        ]
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.writelines(lines)
            os.replace(tmp, str(p))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load_tsv(cls, path: Union[str, Path], **kwargs: Any) -> "KnowledgeGraph":
        """Load triples from TSV file."""
        p = Path(path).expanduser()
        rows = []
        with p.open("r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    raise ValueError(f"Expected 3 tab-separated columns; got: {line!r}")
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        if rows:
            triples = torch.tensor(rows, dtype=torch.long)
        else:
            triples = torch.zeros((0, 3), dtype=torch.long)
        return cls(triples, **kwargs)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save KG structure (triples + ID mappings + metadata) as JSON.

        Only integer-typed data and the entity/relation ID maps are saved.
        Tensor features are not serialised (use save_pt for full save).
        """
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "triples": self.triples.cpu().tolist(),
            "entity_to_id": self.entity_to_id or {},
            "relation_to_id": self.relation_to_id or {},
            "metadata": self.metadata,
        }
        text = json.dumps(payload, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, str(p))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def load_json(cls, path: Union[str, Path], **kwargs: Any) -> "KnowledgeGraph":
        """Load KG from JSON file saved with ``save_json``."""
        p = Path(path).expanduser()
        payload = json.loads(p.read_text(encoding="utf-8"))
        triples = torch.tensor(payload["triples"], dtype=torch.long) if payload["triples"] else torch.zeros((0, 3), dtype=torch.long)
        return cls(
            triples,
            num_entities=payload.get("num_entities"),
            num_relations=payload.get("num_relations"),
            entity_to_id=payload.get("entity_to_id") or None,
            relation_to_id=payload.get("relation_to_id") or None,
            metadata=payload.get("metadata", {}),
            **kwargs,
        )


# ── TemporalKnowledgeGraph ────────────────────────────────────────────────────


class TemporalKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph with per-triple timestamps.

    Extends :class:`KnowledgeGraph` with a required ``timestamp``
    ``FloatTensor[N_t]`` recording the event time of each triple.

    The graph can be sorted chronologically and split without future leakage.

    Stability: Experimental.
    """

    def __init__(
        self,
        triples: torch.Tensor,
        timestamp: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        N_t = triples.size(0)
        if timestamp.dim() != 1 or timestamp.size(0) != N_t:
            raise ValueError(
                f"timestamp must be 1-D with length N_t={N_t}; "
                f"got {tuple(timestamp.shape)}"
            )
        super().__init__(
            triples, num_entities=num_entities, num_relations=num_relations, **kwargs
        )
        self.timestamp: torch.Tensor = timestamp.float()

    def to(self, device: Union[str, torch.device]) -> "TemporalKnowledgeGraph":
        super().to(device)
        self.timestamp = self.timestamp.to(device)
        return self  # type: ignore[return-value]

    def sort_by_time(self) -> "TemporalKnowledgeGraph":
        """Return a new graph sorted chronologically."""
        order = self.timestamp.argsort(stable=True)
        tf = {k: v[order] for k, v in self.triple_features.items()}
        ew = self.edge_weight[order] if self.edge_weight is not None else None
        conf = self.confidence[order] if self.confidence is not None else None
        return TemporalKnowledgeGraph(
            self.triples[order],
            self.timestamp[order],
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            entity_features=_clone_dict(self.entity_features) or None,
            relation_features=_clone_dict(self.relation_features) or None,
            triple_features=tf or None,
            edge_weight=ew,
            confidence=conf,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            metadata=dict(self.metadata),
        )

    def chronological_split(
        self,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple["TemporalKnowledgeGraph", "TemporalKnowledgeGraph", "TemporalKnowledgeGraph"]:
        """Strict chronological split ensuring no future leakage.

        All train events have timestamps <= all valid timestamps, and
        all valid events have timestamps <= all test timestamps.
        """
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        sorted_kg = self.sort_by_time()
        T = sorted_kg.num_triples
        n_tr = int(round(train_ratio * T))
        n_va = int(round(valid_ratio * T))
        kw = dict(num_entities=self.num_entities, num_relations=self.num_relations,
                  entity_to_id=self.entity_to_id, relation_to_id=self.relation_to_id)

        def _sub(idx: slice) -> "TemporalKnowledgeGraph":
            i = torch.arange(T)[idx]
            tf = {k: v[i] for k, v in sorted_kg.triple_features.items()}
            ew = sorted_kg.edge_weight[i] if sorted_kg.edge_weight is not None else None
            conf = sorted_kg.confidence[i] if sorted_kg.confidence is not None else None
            return TemporalKnowledgeGraph(
                sorted_kg.triples[i], sorted_kg.timestamp[i],
                entity_features=_clone_dict(sorted_kg.entity_features) or None,
                relation_features=_clone_dict(sorted_kg.relation_features) or None,
                triple_features=tf or None,
                edge_weight=ew, confidence=conf,
                metadata=dict(sorted_kg.metadata), **kw,
            )
        return _sub(slice(0, n_tr)), _sub(slice(n_tr, n_tr + n_va)), _sub(slice(n_tr + n_va, T))

    def summary(self) -> Dict[str, Any]:
        s = super().summary()
        t = self.timestamp
        s["temporal"] = {
            "has_timestamps": True,
            "time_min": round(float(t.min().item()), 4) if t.numel() else None,
            "time_max": round(float(t.max().item()), 4) if t.numel() else None,
            "num_unique_times": int(t.unique().size(0)),
        }
        return s
