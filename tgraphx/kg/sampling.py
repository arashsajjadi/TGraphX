"""Negative sampling for knowledge graph training.

For a positive triple (h, r, t) a negative triple x' is produced by
corrupting either the head or the tail:

  Head corruption:   x' = (h', r, t),  h' ∈ E
  Tail corruption:   x' = (h, r, t'),  t' ∈ E

Filtered condition:  x' ∉ T_pos  (all known positives across train/valid/test)

All samplers:
- Accept a ``torch.Generator`` for deterministic, reproducible output.
- Accept batched input ``LongTensor[B, 3]``.
- Return ``LongTensor[B, K, 3]`` (K negatives per positive), which can
  be reshaped to ``[B*K, 3]`` if preferred.
- Operate on CPU.  Pass CUDA tensors via ``.to(device)`` after sampling.

Stability: Beta (Uniform, Bernoulli, Filtered) / Experimental (Typed, Hard, Temporal).
"""
from __future__ import annotations

import warnings
from typing import Optional, Set, Tuple, Union

import torch

__all__ = [
    "UniformNegativeSampler",
    "BernoulliNegativeSampler",
    "FilteredNegativeSampler",
    "TypedNegativeSampler",
]


# ── Base ──────────────────────────────────────────────────────────────────────


class _BaseNegativeSampler:
    """Abstract base for KG negative samplers."""

    def __init__(self, num_entities: int, num_negatives: int = 1) -> None:
        if num_entities < 2:
            raise ValueError(f"num_entities must be >= 2; got {num_entities}")
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be >= 1; got {num_negatives}")
        self.num_entities = int(num_entities)
        self.num_negatives = int(num_negatives)

    def sample(
        self,
        triples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample negative triples.

        Args:
            triples: ``LongTensor[B, 3]`` positive triples.
            generator: Optional ``torch.Generator`` for determinism.

        Returns:
            ``LongTensor[B, K, 3]`` where K = num_negatives.
        """
        raise NotImplementedError


# ── Uniform ───────────────────────────────────────────────────────────────────


class UniformNegativeSampler(_BaseNegativeSampler):
    """Uniform negative sampler.

    Each triple is corrupted by replacing head or tail uniformly at random.

    For batch (h, r, t):
      With probability ``corrupt_head_prob``:
        h' ~ Uniform(0, N_e - 1)
        return (h', r, t)
      Otherwise:
        t' ~ Uniform(0, N_e - 1)
        return (h, r, t')

    Args:
        num_entities: Entity vocabulary size N_e.
        num_negatives: K negatives per positive.
        corrupt_head_prob: Probability of corrupting the head (default 0.5).

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        corrupt_head_prob: float = 0.5,
    ) -> None:
        super().__init__(num_entities, num_negatives)
        if not (0.0 <= corrupt_head_prob <= 1.0):
            raise ValueError(
                f"corrupt_head_prob must be in [0, 1]; got {corrupt_head_prob}"
            )
        self.corrupt_head_prob = float(corrupt_head_prob)

    def sample(
        self,
        triples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``LongTensor[B, K, 3]``."""
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("triples must have shape [B, 3]")
        B, K, N = triples.size(0), self.num_negatives, self.num_entities
        # Expand: [B, K, 3]
        neg = triples.unsqueeze(1).expand(B, K, 3).clone()
        # Random entity replacements.
        rand_ents = torch.randint(N, (B, K), generator=generator)
        # Corruption mask: True = corrupt head.
        corrupt_h = torch.rand(B, K, generator=generator) < self.corrupt_head_prob
        neg[:, :, 0] = torch.where(corrupt_h, rand_ents, neg[:, :, 0])
        neg[:, :, 2] = torch.where(~corrupt_h, rand_ents, neg[:, :, 2])
        return neg


# ── Bernoulli ─────────────────────────────────────────────────────────────────


class BernoulliNegativeSampler(_BaseNegativeSampler):
    """Bernoulli negative sampler (Wang et al., 2014).

    For each relation r, estimate the average tails-per-head (tph) and
    heads-per-tail (hpt) from training triples, then set:

        p_corrupt_head(r) = tph_r / (tph_r + hpt_r)

    For one-to-many relations (large tph) it is more informative to
    corrupt the head; for many-to-one relations (large hpt) the tail.

    Args:
        num_entities: N_e.
        num_negatives: K.
        train_triples: ``LongTensor[T, 3]`` used to estimate tph/hpt.

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        train_triples: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(num_entities, num_negatives)
        # per_rel_prob_corrupt_head[r] = tph_r / (tph_r + hpt_r)
        # Default to 0.5 if no training triples provided.
        self._rel_prob: Optional[torch.Tensor] = None
        if train_triples is not None:
            self._rel_prob = self._estimate_bernoulli_probs(train_triples)

    @staticmethod
    def _estimate_bernoulli_probs(triples: torch.Tensor) -> torch.Tensor:
        """Estimate per-relation Bernoulli head-corruption probability."""
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("train_triples must be [T, 3]")
        N_r = int(triples[:, 1].max().item()) + 1
        tph = torch.zeros(N_r)  # total unique tails per head, summed over relations
        hpt = torch.zeros(N_r)  # total unique heads per tail, summed over relations

        for r in range(N_r):
            mask = triples[:, 1] == r
            if not mask.any():
                continue
            sub = triples[mask]
            heads = sub[:, 0]
            tails = sub[:, 2]
            # tph_r = mean(#unique tails per unique head)
            unique_heads = heads.unique()
            if unique_heads.numel() == 0:
                continue
            tails_per_head = []
            for h in unique_heads.tolist():
                tails_per_head.append(float(tails[heads == h].unique().numel()))
            tph_r = sum(tails_per_head) / len(tails_per_head)
            # hpt_r = mean(#unique heads per unique tail)
            unique_tails = tails.unique()
            heads_per_tail = []
            for t in unique_tails.tolist():
                heads_per_tail.append(float(heads[tails == t].unique().numel()))
            hpt_r = sum(heads_per_tail) / len(heads_per_tail)
            denom = tph_r + hpt_r
            tph[r] = tph_r / denom if denom > 0 else 0.5
            hpt[r] = 1.0 - tph[r]

        # p_corrupt_head[r] = tph[r] / (tph[r] + hpt[r])
        # tph already normalised.
        return tph

    def sample(
        self,
        triples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``LongTensor[B, K, 3]``."""
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("triples must have shape [B, 3]")
        B, K, N = triples.size(0), self.num_negatives, self.num_entities
        neg = triples.unsqueeze(1).expand(B, K, 3).clone()
        rand_ents = torch.randint(N, (B, K), generator=generator)

        if self._rel_prob is not None:
            # Per-triple corrupt head prob based on relation.
            rels = triples[:, 1]  # [B]
            # Clamp for safety (unseen relations default to 0.5).
            r_probs = torch.full((len(rels),), 0.5, dtype=torch.float)
            valid = rels < self._rel_prob.size(0)
            r_probs[valid] = self._rel_prob[rels[valid]]
            # [B, K] threshold mask.
            thresh = r_probs.unsqueeze(1).expand(B, K)
            corrupt_h = torch.rand(B, K, generator=generator) < thresh
        else:
            corrupt_h = torch.rand(B, K, generator=generator) < 0.5

        neg[:, :, 0] = torch.where(corrupt_h, rand_ents, neg[:, :, 0])
        neg[:, :, 2] = torch.where(~corrupt_h, rand_ents, neg[:, :, 2])
        return neg


# ── Filtered ──────────────────────────────────────────────────────────────────


class FilteredNegativeSampler(_BaseNegativeSampler):
    """Filtered negative sampler: rejects known positives.

    For each generated candidate negative triple, checks whether it exists
    in ``positive_set``.  If so, resamples up to ``max_attempts`` times.
    If the entity set is so dense that no negative can be found, a warning
    is emitted and the last candidate is returned (with the relation changed
    to a sentinel ``-1`` marker if ``fail_on_dense=True`` is set).

    Args:
        num_entities: N_e.
        num_negatives: K.
        positive_set: Set of all known positive ``(h, r, t)`` int tuples
            (train + valid + test).
        base_sampler: Inner sampler used for proposals.  When None, uses
            :class:`UniformNegativeSampler`.
        max_attempts: Maximum resampling attempts per negative slot.
        fail_on_dense: If True, raise ValueError when no valid negative
            is found.  Default: emit a warning and return the last sample.

    Stability: Beta.
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        positive_set: Optional[Set[Tuple[int, int, int]]] = None,
        base_sampler: Optional[_BaseNegativeSampler] = None,
        max_attempts: int = 100,
        fail_on_dense: bool = False,
    ) -> None:
        super().__init__(num_entities, num_negatives)
        self._positives: Set[Tuple[int, int, int]] = positive_set or set()
        self._base = base_sampler or UniformNegativeSampler(num_entities, 1)
        self._max_attempts = int(max_attempts)
        self._fail_on_dense = bool(fail_on_dense)

    @property
    def positive_set(self) -> Set[Tuple[int, int, int]]:
        return self._positives

    def sample(
        self,
        triples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return ``LongTensor[B, K, 3]`` with positives filtered out.

        For dense KGs where rejection is frequent this may be slow.
        """
        if triples.dim() != 2 or triples.size(1) != 3:
            raise ValueError("triples must have shape [B, 3]")
        B, K = triples.size(0), self.num_negatives
        result = torch.zeros(B, K, 3, dtype=torch.long)

        for i in range(B):
            triple = triples[i]
            for j in range(K):
                for _attempt in range(self._max_attempts):
                    cand = self._base.sample(triple.unsqueeze(0), generator=generator)
                    c = cand[0, 0]  # [3]
                    key = (int(c[0]), int(c[1]), int(c[2]))
                    if key not in self._positives:
                        result[i, j] = c
                        break
                else:
                    if self._fail_on_dense:
                        raise ValueError(
                            f"Could not find a valid negative for triple {triple.tolist()} "
                            f"after {self._max_attempts} attempts. "
                            f"The KG may be too dense for filtered sampling."
                        )
                    warnings.warn(
                        f"Could not find valid negative after {self._max_attempts} "
                        f"attempts; returning last candidate.",
                        RuntimeWarning, stacklevel=2,
                    )
                    result[i, j] = c  # last candidate regardless
        return result


# ── Typed ────────────────────────────────────────────────────────────────────


class TypedNegativeSampler(_BaseNegativeSampler):
    """Typed negative sampler respecting entity type constraints.

    For relation r, head corruptions are drawn from ``domains[r]`` and
    tail corruptions from ``ranges[r]``.  Falls back to uniform if
    the domain/range for a relation is not specified.

    Args:
        num_entities: N_e.
        num_negatives: K.
        entity_types: ``LongTensor[N_e]`` entity type IDs.
        domains: Dict mapping relation_id → set of allowed head entity IDs.
        ranges: Dict mapping relation_id → set of allowed tail entity IDs.
        corrupt_head_prob: Fallback if domain/range unavailable.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        entity_types: Optional[torch.Tensor] = None,
        domains: Optional[dict] = None,
        ranges: Optional[dict] = None,
        corrupt_head_prob: float = 0.5,
    ) -> None:
        super().__init__(num_entities, num_negatives)
        self._entity_types = entity_types
        self._domains = domains or {}
        self._ranges = ranges or {}
        self._corrupt_head_prob = corrupt_head_prob
        # Pre-build entity lists per type.
        self._type_entities: dict = {}
        if entity_types is not None:
            unique_types = entity_types.unique().tolist()
            for t in unique_types:
                self._type_entities[int(t)] = torch.where(entity_types == t)[0]

    def _sample_from(
        self,
        cands: torch.Tensor,
        n: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample n entities from candidate set."""
        idx = torch.randint(len(cands), (n,), generator=generator)
        return cands[idx]

    def sample(
        self,
        triples: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        B, K = triples.size(0), self.num_negatives
        result = torch.zeros(B, K, 3, dtype=torch.long)
        all_ents = torch.arange(self.num_entities)

        for i in range(B):
            h, r, t = int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2])
            dom = self._domains.get(r)
            rng = self._ranges.get(r)
            dom_t = (torch.tensor(list(dom)) if isinstance(dom, set) else
                     (dom if dom is not None else all_ents))
            rng_t = (torch.tensor(list(rng)) if isinstance(rng, set) else
                     (rng if rng is not None else all_ents))
            corrupt_h = torch.rand(K, generator=generator) < self._corrupt_head_prob
            heads_repl = self._sample_from(dom_t, int(corrupt_h.sum()), generator)
            tails_repl = self._sample_from(rng_t, int((~corrupt_h).sum()), generator)
            neg = triples[i].unsqueeze(0).expand(K, 3).clone()
            neg[corrupt_h, 0] = heads_repl.long()
            neg[~corrupt_h, 2] = tails_repl.long()
            result[i] = neg
        return result
