"""Temporal Knowledge Graph utilities.

Temporal KG triples: (h, r, t, τ)

Temporal negative sampling:
  Excludes future positives: for cutoff time τ_cut, a negative is valid
  if (h', r, t) or (h, r, t') does NOT exist at ANY time ≤ τ_cut.

Time-aware filtered evaluation:
  At evaluation time τ, only triples with timestamp ≤ τ are treated as
  known positives for filtering.  Future triples are not seen.

Stability: Experimental.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .data import TemporalKnowledgeGraph
from .evaluation import (
    _score_all_candidates,
    _compute_rank_avg_tie,
    _metrics_from_ranks,
    KGEvalResult,
)

__all__ = [
    "TemporalKGNegativeSampler",
    "evaluate_temporal_filtered_ranking",
]


class TemporalKGNegativeSampler:
    """Temporal negative sampler with no-future-leakage.

    For a positive temporal triple (h, r, t, τ), generates negatives
    by corrupting h or t.  In filtered mode, a negative is rejected if
    it exists in the positive set AT ANY TIME <= τ_cut.

    Args:
        num_entities: N_e.
        num_negatives: K.
        temporal_kg: Source temporal KG (used to build time-indexed pos set).
        corrupt_head_prob: Head-corruption probability.
        max_attempts: Rejection sampling attempts.

    Stability: Experimental.
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        temporal_kg: Optional[TemporalKnowledgeGraph] = None,
        corrupt_head_prob: float = 0.5,
        max_attempts: int = 100,
    ) -> None:
        self.num_entities = int(num_entities)
        self.num_negatives = int(num_negatives)
        self.corrupt_head_prob = float(corrupt_head_prob)
        self.max_attempts = int(max_attempts)
        self._temporal_pos: Optional[List[Tuple[int, int, int, float]]] = None
        if temporal_kg is not None:
            rows = temporal_kg.triples.tolist()
            times = temporal_kg.timestamp.tolist()
            self._temporal_pos = [(int(r[0]), int(r[1]), int(r[2]), float(t))
                                  for r, t in zip(rows, times)]

    def _is_positive_at_or_before(self, h: int, r: int, t: int, cutoff: float) -> bool:
        """Return True if (h, r, t) exists at any time <= cutoff."""
        if self._temporal_pos is None:
            return False
        for ph, pr, pt, pt_time in self._temporal_pos:
            if ph == h and pr == r and pt == t and pt_time <= cutoff:
                return True
        return False

    def sample(
        self,
        triples: torch.Tensor,
        timestamps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        filtered: bool = True,
    ) -> torch.Tensor:
        """Return ``LongTensor[B, K, 3]``.

        Args:
            triples: ``LongTensor[B, 3]`` positive triples.
            timestamps: ``FloatTensor[B]`` event timestamps.
            generator: Optional RNG.
            filtered: If True, reject future/known positives.
        """
        B, K, N = triples.size(0), self.num_negatives, self.num_entities
        result = torch.zeros(B, K, 3, dtype=torch.long)
        for i in range(B):
            h, r, t = int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2])
            tau = float(timestamps[i].item())
            for j in range(K):
                for _ in range(self.max_attempts):
                    corrupt_head = torch.rand(1, generator=generator).item() < self.corrupt_head_prob
                    ent = int(torch.randint(N, (1,), generator=generator).item())
                    if corrupt_head:
                        nh, nr, nt = ent, r, t
                    else:
                        nh, nr, nt = h, r, ent
                    if not filtered or not self._is_positive_at_or_before(nh, nr, nt, tau):
                        result[i, j] = torch.tensor([nh, nr, nt])
                        break
                else:
                    # Fallback: use last candidate.
                    warnings.warn(
                        f"TemporalNegativeSampler: no valid negative found for "
                        f"triple ({h},{r},{t},τ={tau}) after {self.max_attempts} attempts.",
                        RuntimeWarning, stacklevel=2,
                    )
                    result[i, j] = torch.tensor([nh, nr, nt])
        return result


@torch.no_grad()
def evaluate_temporal_filtered_ranking(
    model: nn.Module,
    test_kg: TemporalKnowledgeGraph,
    train_kg: TemporalKnowledgeGraph,
    num_entities: int,
    chunk_size: int = 50_000,
    hits_at: Tuple[int, ...] = (1, 3, 10),
    device: Union[str, torch.device] = "cpu",
) -> KGEvalResult:
    """Time-aware filtered ranking: at time τ, only triples with t' ≤ τ filtered.

    For each test triple (h, r, t, τ):
      - Candidate scores computed normally.
      - Filtered set: all train triples with timestamp ≤ τ AND same (h, r, ·) pattern.

    Stability: Experimental.
    """
    from typing import Union
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    hit_ks = list(hits_at)
    filt_tail_ranks: List[float] = []
    filt_head_ranks: List[float] = []

    # Build (h, r) → [(t, time)] mapping from training set.
    hr_to_tails_time: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    rt_to_heads_time: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for row, ts in zip(train_kg.triples.tolist(), train_kg.timestamp.tolist()):
        h, r, t = int(row[0]), int(row[1]), int(row[2])
        hr_to_tails_time.setdefault((h, r), []).append((t, float(ts)))
        rt_to_heads_time.setdefault((r, t), []).append((h, float(ts)))

    for triple, tau in zip(test_kg.triples.tolist(), test_kg.timestamp.tolist()):
        h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
        tau_f = float(tau)
        h_t = torch.tensor([h], dtype=torch.long)
        r_t = torch.tensor([r], dtype=torch.long)
        t_t = torch.tensor([t], dtype=torch.long)

        # Tail prediction.
        scores_tail = _score_all_candidates(
            model, h_t, r_t, t, predict_tail=True,
            num_entities=num_entities, chunk_size=chunk_size, device=dev,
        )
        # Filter: train positives with timestamp <= tau.
        pos_mask = torch.zeros(num_entities, dtype=torch.bool)
        for pt, pt_time in hr_to_tails_time.get((h, r), []):
            if pt != t and pt_time <= tau_f:
                pos_mask[pt] = True
        filt_tail_ranks.append(_compute_rank_avg_tie(scores_tail, t, pos_mask))

        # Head prediction.
        scores_head = _score_all_candidates(
            model, t_t, r_t, h, predict_tail=False,
            num_entities=num_entities, chunk_size=chunk_size, device=dev,
        )
        pos_mask_h = torch.zeros(num_entities, dtype=torch.bool)
        for ph, ph_time in rt_to_heads_time.get((r, t), []):
            if ph != h and ph_time <= tau_f:
                pos_mask_h[ph] = True
        filt_head_ranks.append(_compute_rank_avg_tie(scores_head, h, pos_mask_h))

    filt_mr_t, filt_mrr_t, filt_hits_t = _metrics_from_ranks(filt_tail_ranks, hit_ks)
    filt_mr_h, filt_mrr_h, filt_hits_h = _metrics_from_ranks(filt_head_ranks, hit_ks)
    return KGEvalResult(
        raw_mr_tail=filt_mr_t, raw_mrr_tail=filt_mrr_t, raw_hits_tail=filt_hits_t,
        filt_mr_tail=filt_mr_t, filt_mrr_tail=filt_mrr_t, filt_hits_tail=filt_hits_t,
        raw_mr_head=filt_mr_h, raw_mrr_head=filt_mrr_h, raw_hits_head=filt_hits_h,
        filt_mr_head=filt_mr_h, filt_mrr_head=filt_mrr_h, filt_hits_head=filt_hits_h,
    )
