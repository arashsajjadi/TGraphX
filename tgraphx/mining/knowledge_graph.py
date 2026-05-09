"""Knowledge graph representation learning foundations.

Provides triple-based KG data containers, negative sampling, filtered
ranking evaluation, and TransE / DistMult scoring models.

Stability: Experimental (v0.4.4+).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "KnowledgeGraph",
    "negative_triple_sampling",
    "filtered_ranking_metrics",
    "TransE",
    "DistMult",
    "train_kg_step",
]


class KnowledgeGraph:
    """Simple triple-based knowledge graph container.

    Stores (head, relation, tail) triples with integer-mapped entity and
    relation IDs.

    Args:
        triples: ``LongTensor[T, 3]`` of (head, relation, tail) triples.
        num_entities: Optional; inferred from ``triples.max() + 1`` when None.
        num_relations: Optional; inferred from relation column when None.

    Stability: Experimental.
    """

    def __init__(
        self,
        triples: torch.Tensor,
        num_entities: Optional[int] = None,
        num_relations: Optional[int] = None,
    ) -> None:
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("triples must have shape [T, 3] (head, relation, tail)")
        self.triples = triples.to(torch.long)
        self.num_entities = int(
            max(self.triples[:, 0].max().item(), self.triples[:, 2].max().item()) + 1
        ) if triples.numel() else (num_entities or 0)
        if num_entities is not None:
            self.num_entities = int(num_entities)
        self.num_relations = int(self.triples[:, 1].max().item() + 1) \
            if triples.numel() else (num_relations or 0)
        if num_relations is not None:
            self.num_relations = int(num_relations)
        # Build positive set for fast lookup.
        self._positive_set: set = set(
            (int(h), int(r), int(t))
            for h, r, t in self.triples.tolist()
        ) if triples.numel() else set()

    def __len__(self) -> int:
        return int(self.triples.size(0))

    @property
    def heads(self) -> torch.Tensor:
        return self.triples[:, 0]

    @property
    def relations(self) -> torch.Tensor:
        return self.triples[:, 1]

    @property
    def tails(self) -> torch.Tensor:
        return self.triples[:, 2]

    def is_positive(self, h: int, r: int, t: int) -> bool:
        return (int(h), int(r), int(t)) in self._positive_set

    def train_val_test_split(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 0,
    ) -> Tuple["KnowledgeGraph", "KnowledgeGraph", "KnowledgeGraph"]:
        """Chronological/random train/val/test split."""
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        T = len(self)
        perm = torch.randperm(T, generator=gen)
        n_train = int(ratios[0] * T)
        n_val = int(ratios[1] * T)
        train_idx = perm[:n_train]
        val_idx = perm[n_train: n_train + n_val]
        test_idx = perm[n_train + n_val:]
        kwargs = dict(num_entities=self.num_entities, num_relations=self.num_relations)
        return (
            KnowledgeGraph(self.triples[train_idx], **kwargs),
            KnowledgeGraph(self.triples[val_idx], **kwargs),
            KnowledgeGraph(self.triples[test_idx], **kwargs),
        )


def negative_triple_sampling(
    triples: torch.Tensor,
    num_entities: int,
    num_neg: int = 1,
    corrupt_head: bool = True,
    corrupt_tail: bool = True,
    seed: Optional[int] = None,
    positive_set: Optional[set] = None,
) -> torch.Tensor:
    """Sample negative triples by corrupting heads and/or tails.

    Args:
        triples: ``LongTensor[T, 3]`` positive (h, r, t) triples.
        num_entities: Entity vocabulary size.
        num_neg: Negative samples per positive triple.
        corrupt_head: Corrupt heads.
        corrupt_tail: Corrupt tails.
        seed: Optional RNG seed.
        positive_set: Known positives (to exclude as negatives if provided).

    Returns:
        ``LongTensor[T * num_neg, 3]`` negative triples.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    T = triples.size(0)
    neg_triples = triples.repeat_interleave(num_neg, dim=0).clone()
    # Choose corruption type per sample.
    if corrupt_head and corrupt_tail:
        # Randomly choose head or tail corruption.
        corrupt_which = torch.randint(2, (T * num_neg,), generator=gen)
        corrupt_h_mask = corrupt_which == 0
        corrupt_t_mask = ~corrupt_h_mask
    elif corrupt_head:
        corrupt_h_mask = torch.ones(T * num_neg, dtype=torch.bool)
        corrupt_t_mask = torch.zeros(T * num_neg, dtype=torch.bool)
    else:
        corrupt_h_mask = torch.zeros(T * num_neg, dtype=torch.bool)
        corrupt_t_mask = torch.ones(T * num_neg, dtype=torch.bool)
    # Corrupt heads.
    if corrupt_h_mask.any():
        n = int(corrupt_h_mask.sum())
        rand_ents = torch.randint(num_entities, (n,), generator=gen)
        neg_triples[corrupt_h_mask, 0] = rand_ents
    # Corrupt tails.
    if corrupt_t_mask.any():
        n = int(corrupt_t_mask.sum())
        rand_ents = torch.randint(num_entities, (n,), generator=gen)
        neg_triples[corrupt_t_mask, 2] = rand_ents
    return neg_triples


@torch.no_grad()
def filtered_ranking_metrics(
    model: nn.Module,
    test_triples: torch.Tensor,
    all_triples_set: set,
    num_entities: int,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Filtered MRR and Hits@1/3/10 for link prediction.

    For each test triple (h, r, t), ranks all entities as candidates
    for the tail position.  Candidates that are known positives
    (in ``all_triples_set``) are removed from the ranking (filtered).

    Args:
        model: KG scoring model with signature
            ``score(h, r, t) -> FloatTensor[B]`` (higher = more likely).
            Must accept ``(LongTensor[B], LongTensor[B], LongTensor[B])``.
        test_triples: ``LongTensor[T, 3]``.
        all_triples_set: Set of all known positive ``(h, r, t)`` tuples.
        num_entities: Entity count.
        batch_size: Entity batch size for scoring.

    Returns:
        Dict with ``mrr``, ``hits@1``, ``hits@3``, ``hits@10``.
    """
    model.eval()
    ranks = []
    for triple in test_triples.tolist():
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        # Score all tail candidates.
        all_ents = torch.arange(num_entities, dtype=torch.long)
        all_scores = []
        h_t = torch.full((num_entities,), h, dtype=torch.long)
        r_t = torch.full((num_entities,), r, dtype=torch.long)
        for start in range(0, num_entities, batch_size):
            end = min(start + batch_size, num_entities)
            batch_ents = all_ents[start:end]
            h_b = h_t[start:end]
            r_b = r_t[start:end]
            scores = model.score(h_b, r_b, batch_ents)
            all_scores.append(scores.detach().cpu())
        all_scores_t = torch.cat(all_scores)  # [E]
        # Filter known positives (except the test triple itself).
        for e in range(num_entities):
            if e != t and (h, r, e) in all_triples_set:
                all_scores_t[e] = -1e9
        target_score = float(all_scores_t[t].item())
        rank = int((all_scores_t > target_score).sum().item()) + 1
        ranks.append(rank)
    ranks_t = torch.tensor(ranks, dtype=torch.float)
    mrr = float((1.0 / ranks_t).mean().item())
    hits_1 = float((ranks_t <= 1).float().mean().item())
    hits_3 = float((ranks_t <= 3).float().mean().item())
    hits_10 = float((ranks_t <= 10).float().mean().item())
    return {"mrr": round(mrr, 4), "hits@1": round(hits_1, 4),
            "hits@3": round(hits_3, 4), "hits@10": round(hits_10, 4)}


# ── KG Scoring Models ─────────────────────────────────────────────────────────


class TransE(nn.Module):
    """TransE knowledge graph embedding model (Bordes et al., 2013).

    Score function: ``-||h + r - t||_p``  (higher = more likely).

    Args:
        num_entities: Entity count.
        num_relations: Relation count.
        embedding_dim: Embedding dimension.
        norm: L-norm order (1 or 2).
        margin: Margin for margin-based training loss.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        norm: int = 2,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.norm = int(norm)
        self.margin = float(margin)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.entity_emb.weight, -6 / math.sqrt(self.embedding_dim),
                          6 / math.sqrt(self.embedding_dim))
        nn.init.uniform_(self.relation_emb.weight, -6 / math.sqrt(self.embedding_dim),
                          6 / math.sqrt(self.embedding_dim))

    def score(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """TransE score: ``-||h + r - t||``.  Higher = more plausible.

        Args:
            heads: ``LongTensor[B]``.
            relations: ``LongTensor[B]``.
            tails: ``LongTensor[B]``.

        Returns:
            ``FloatTensor[B]``.
        """
        h = F.normalize(self.entity_emb(heads), p=2, dim=1)
        r = self.relation_emb(relations)
        t = F.normalize(self.entity_emb(tails), p=2, dim=1)
        return -torch.norm(h + r - t, p=self.norm, dim=1)

    def forward(
        self,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based pairwise ranking loss.

        Args:
            pos_triples: ``LongTensor[B, 3]``.
            neg_triples: ``LongTensor[B, 3]``.

        Returns:
            Scalar loss.
        """
        pos_score = self.score(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        return F.relu(self.margin + neg_score - pos_score).mean()


class DistMult(nn.Module):
    """DistMult knowledge graph embedding model (Yang et al., 2015).

    Score function: ``<h, r, t>`` (element-wise trilinear dot product).

    Args:
        num_entities: Entity count.
        num_relations: Relation count.
        embedding_dim: Embedding dimension.
        regularization: L2 regularization weight.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        regularization: float = 1e-3,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.regularization = float(regularization)
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """DistMult score: ``<h, r, t>`` element-wise product then sum.

        Returns:
            ``FloatTensor[B]``.
        """
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return (h * r * t).sum(dim=1)

    def forward(
        self,
        pos_triples: torch.Tensor,
        neg_triples: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss with L2 regularization.

        Args:
            pos_triples: ``LongTensor[B, 3]``.
            neg_triples: ``LongTensor[B, 3]``.

        Returns:
            Scalar loss.
        """
        pos_score = self.score(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        reg = self.regularization * (
            self.entity_emb.weight.norm(p=2) ** 2 +
            self.relation_emb.weight.norm(p=2) ** 2
        )
        return pos_loss + neg_loss + reg


def train_kg_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    pos_triples: torch.Tensor,
    neg_triples: torch.Tensor,
) -> float:
    """One training step for a KG scoring model.

    Args:
        model: TransE or DistMult (must implement forward(pos, neg)).
        optimizer: PyTorch optimizer.
        pos_triples: ``LongTensor[B, 3]``.
        neg_triples: ``LongTensor[B, 3]``.

    Returns:
        Float loss value.
    """
    model.train()
    optimizer.zero_grad()
    loss = model(pos_triples, neg_triples)
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())
