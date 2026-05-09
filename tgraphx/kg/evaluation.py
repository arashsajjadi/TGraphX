"""Chunked head+tail filtered ranking for KG evaluation.

For each test triple (h, r, t), we rank all N_e candidate entities as
tail (or head) predictions.  The filtered rank removes all known positive
tails/heads from the ranking, keeping only the target entity.

Formal definitions:

  Tail prediction scores for (h, r, ·):
    s_e = f(h, r, e)  for all e ∈ E

  Raw tail rank:
    rank_raw(t) = 1 + |{e : s_e > s_t}|

  Filtered tail rank:
    rank_filt(t) = 1 + |{e : s_e > s_t  AND  (h, r, e) ∉ T_pos \\ {t}}|

  Similarly for head prediction:
    s_e = f(e, r, t)
    rank_raw(h)  = 1 + |{e : s_e > s_h}|
    rank_filt(h) = 1 + |{e : s_e > s_h  AND  (e, r, t) not in T_pos \\ {h}}|

Tie policy ("average"):
  Ties at score s_t contribute 0.5 * (count of ties) to the rank.
  This matches standard practice and avoids optimistic/pessimistic bias.

  rank = 1 + (strictly_higher) + 0.5 * (tied_not_target)

Metrics:
  MR     = mean(rank)
  MRR    = mean(1 / rank)
  Hits@K = mean(rank <= K)

Performance:
  Candidate scores are computed in chunks of size chunk_size to avoid
  allocating an [N_e] score vector all at once when N_e is very large.
  Memory budget: approximately chunk_size * embedding_dim * sizeof(float).

Stability: Beta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "KGEvaluator",
    "evaluate_filtered_ranking",
    "KGEvalResult",
]


@dataclass
class KGEvalResult:
    """Results of KG filtered ranking evaluation."""
    raw_mr_tail: float
    raw_mrr_tail: float
    raw_hits_tail: Dict[int, float]
    filt_mr_tail: float
    filt_mrr_tail: float
    filt_hits_tail: Dict[int, float]
    raw_mr_head: float
    raw_mrr_head: float
    raw_hits_head: Dict[int, float]
    filt_mr_head: float
    filt_mrr_head: float
    filt_hits_head: Dict[int, float]
    # Combined (head + tail averaged).
    filt_mr: float = 0.0
    filt_mrr: float = 0.0
    filt_hits: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k_set = set(self.filt_hits_head) | set(self.filt_hits_tail)
        self.filt_mr = (self.filt_mr_head + self.filt_mr_tail) / 2
        self.filt_mrr = (self.filt_mrr_head + self.filt_mrr_tail) / 2
        self.filt_hits = {
            k: (self.filt_hits_head.get(k, 0.0) + self.filt_hits_tail.get(k, 0.0)) / 2
            for k in k_set
        }

    def to_dict(self) -> Dict:
        def _r(v: float) -> float:
            return round(v, 4)
        return {
            "filtered": {
                "combined": {
                    "MR": _r(self.filt_mr), "MRR": _r(self.filt_mrr),
                    **{f"Hits@{k}": _r(v) for k, v in self.filt_hits.items()},
                },
                "tail": {
                    "MR": _r(self.filt_mr_tail), "MRR": _r(self.filt_mrr_tail),
                    **{f"Hits@{k}": _r(v) for k, v in self.filt_hits_tail.items()},
                },
                "head": {
                    "MR": _r(self.filt_mr_head), "MRR": _r(self.filt_mrr_head),
                    **{f"Hits@{k}": _r(v) for k, v in self.filt_hits_head.items()},
                },
            },
            "raw": {
                "tail": {
                    "MR": _r(self.raw_mr_tail), "MRR": _r(self.raw_mrr_tail),
                    **{f"Hits@{k}": _r(v) for k, v in self.raw_hits_tail.items()},
                },
                "head": {
                    "MR": _r(self.raw_mr_head), "MRR": _r(self.raw_mrr_head),
                    **{f"Hits@{k}": _r(v) for k, v in self.raw_hits_head.items()},
                },
            },
        }


# ── Core ranking logic ───────────────────────────────────────────────────────


@torch.no_grad()
def _score_all_candidates(
    model: nn.Module,
    fixed1: torch.Tensor,   # [1] head or tail (single entity)
    fixed2: torch.Tensor,   # [1] relation
    target_idx: int,        # entity index of the true answer
    predict_tail: bool,
    num_entities: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``FloatTensor[N_e]`` of all candidate scores.

    Scores are detached and returned on CPU.
    """
    scores = torch.empty(num_entities, dtype=torch.float)
    all_ents = torch.arange(num_entities, dtype=torch.long, device=device)
    for start in range(0, num_entities, chunk_size):
        end = min(start + chunk_size, num_entities)
        cands = all_ents[start:end]  # [chunk]
        B = cands.size(0)
        if predict_tail:
            h_b = fixed1.expand(B).to(device)
            r_b = fixed2.expand(B).to(device)
            t_b = cands
            s = model.score_triples(torch.stack([h_b, r_b, t_b], dim=1))
        else:
            h_b = cands
            r_b = fixed2.expand(B).to(device)
            t_b = fixed1.expand(B).to(device)
            s = model.score_triples(torch.stack([h_b, r_b, t_b], dim=1))
        scores[start:end] = s.detach().cpu()
    return scores


def _compute_rank_avg_tie(
    scores: torch.Tensor,
    target_idx: int,
    positive_mask: Optional[torch.Tensor] = None,
) -> float:
    """Average-tie rank for ``target_idx``.

    If ``positive_mask`` is provided, scores at those positions are set to
    ``-inf`` before ranking (filtered mode).  The target itself is excluded
    from the mask.

    Returns:
        float rank >= 1.
    """
    if positive_mask is not None:
        scores = scores.clone()
        scores[positive_mask] = float("-inf")
    target_score = float(scores[target_idx].item())
    strictly_higher = int((scores > target_score).sum().item())
    tied = int((scores == target_score).sum().item()) - 1  # exclude target
    return float(strictly_higher + 0.5 * tied + 1)


def _metrics_from_ranks(
    ranks: List[float], hit_ks: Sequence[int]
) -> Tuple[float, float, Dict[int, float]]:
    rt = torch.tensor(ranks, dtype=torch.float)
    mr = float(rt.mean().item())
    mrr = float((1.0 / rt).mean().item())
    hits = {k: float((rt <= k).float().mean().item()) for k in hit_ks}
    return mr, mrr, hits


# ── Public evaluator ─────────────────────────────────────────────────────────


def evaluate_filtered_ranking(
    model: nn.Module,
    test_triples: torch.Tensor,
    all_positive_set: Set[Tuple[int, int, int]],
    num_entities: int,
    filtered: bool = True,
    batch_size: int = 64,
    chunk_size: int = 50000,
    hits_at: Sequence[int] = (1, 3, 10),
    device: Union[str, torch.device] = "cpu",
) -> KGEvalResult:
    """Compute head + tail filtered ranking for all test triples.

    Args:
        model: KG scoring model implementing ``score_triples(triples) -> [B]``.
        test_triples: ``LongTensor[T, 3]``.
        all_positive_set: Set of ALL known positive ``(h, r, t)`` ints
            (train + valid + test).  Used for filtering.
        num_entities: N_e.
        filtered: If True, removes known positives from ranking.
        batch_size: Triples to process per outer iteration.
        chunk_size: Maximum entities scored at once (memory guard).
        hits_at: Sequence of K values for Hits@K.
        device: Target device.

    Returns:
        :class:`KGEvalResult` with raw + filtered head + tail metrics.

    Notes:
        - The target triple is NEVER removed from its own rank computation.
        - Memory usage: approximately chunk_size * 4 bytes per forward pass.
        - Autograd is disabled inside this function.
    """
    if test_triples.dim() != 2 or test_triples.size(1) != 3:
        raise ValueError("test_triples must have shape [T, 3]")
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    hit_ks = list(hits_at)

    raw_tail_ranks: List[float] = []
    filt_tail_ranks: List[float] = []
    raw_head_ranks: List[float] = []
    filt_head_ranks: List[float] = []

    # Pre-build per-(h,r) and per-(r,t) positive tail/head sets for filtering.
    hr_to_tails: Dict[Tuple[int, int], Set[int]] = {}
    rt_to_heads: Dict[Tuple[int, int], Set[int]] = {}
    for h, r, t in all_positive_set:
        hr_to_tails.setdefault((h, r), set()).add(t)
        rt_to_heads.setdefault((r, t), set()).add(h)

    with torch.no_grad():
        for start in range(0, test_triples.size(0), batch_size):
            end = min(start + batch_size, test_triples.size(0))
            batch = test_triples[start:end]
            for triple in batch.tolist():
                h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
                h_t = torch.tensor([h], dtype=torch.long)
                r_t = torch.tensor([r], dtype=torch.long)
                t_t = torch.tensor([t], dtype=torch.long)

                # ── Tail prediction ────────────────────────────────────────
                scores_tail = _score_all_candidates(
                    model, h_t, r_t, t, predict_tail=True,
                    num_entities=num_entities, chunk_size=chunk_size, device=dev,
                )
                raw_tail_ranks.append(_compute_rank_avg_tie(scores_tail, t))
                if filtered:
                    pos_tails = hr_to_tails.get((h, r), set())
                    pos_mask = torch.zeros(num_entities, dtype=torch.bool)
                    for pt in pos_tails:
                        if pt != t:
                            pos_mask[pt] = True
                    filt_tail_ranks.append(_compute_rank_avg_tie(scores_tail, t, pos_mask))
                else:
                    filt_tail_ranks.append(raw_tail_ranks[-1])

                # ── Head prediction ────────────────────────────────────────
                scores_head = _score_all_candidates(
                    model, t_t, r_t, h, predict_tail=False,
                    num_entities=num_entities, chunk_size=chunk_size, device=dev,
                )
                raw_head_ranks.append(_compute_rank_avg_tie(scores_head, h))
                if filtered:
                    pos_heads = rt_to_heads.get((r, t), set())
                    pos_mask_h = torch.zeros(num_entities, dtype=torch.bool)
                    for ph in pos_heads:
                        if ph != h:
                            pos_mask_h[ph] = True
                    filt_head_ranks.append(_compute_rank_avg_tie(scores_head, h, pos_mask_h))
                else:
                    filt_head_ranks.append(raw_head_ranks[-1])

    raw_mr_t, raw_mrr_t, raw_hits_t = _metrics_from_ranks(raw_tail_ranks, hit_ks)
    filt_mr_t, filt_mrr_t, filt_hits_t = _metrics_from_ranks(filt_tail_ranks, hit_ks)
    raw_mr_h, raw_mrr_h, raw_hits_h = _metrics_from_ranks(raw_head_ranks, hit_ks)
    filt_mr_h, filt_mrr_h, filt_hits_h = _metrics_from_ranks(filt_head_ranks, hit_ks)

    return KGEvalResult(
        raw_mr_tail=raw_mr_t, raw_mrr_tail=raw_mrr_t, raw_hits_tail=raw_hits_t,
        filt_mr_tail=filt_mr_t, filt_mrr_tail=filt_mrr_t, filt_hits_tail=filt_hits_t,
        raw_mr_head=raw_mr_h, raw_mrr_head=raw_mrr_h, raw_hits_head=raw_hits_h,
        filt_mr_head=filt_mr_h, filt_mrr_head=filt_mrr_h, filt_hits_head=filt_hits_h,
    )


# ── KGEvaluator class ─────────────────────────────────────────────────────────


class KGEvaluator:
    """Stateful evaluator that holds the positive set across splits.

    Args:
        train_triples: Train triples ``LongTensor[T_tr, 3]``.
        valid_triples: Validation triples ``LongTensor[T_va, 3]``.
        test_triples: Test triples ``LongTensor[T_te, 3]``.
        num_entities: Entity count.
        chunk_size: Chunked scoring size.
        hits_at: Tuple of K values.

    Stability: Beta.
    """

    def __init__(
        self,
        train_triples: torch.Tensor,
        valid_triples: torch.Tensor,
        test_triples: torch.Tensor,
        num_entities: int,
        chunk_size: int = 50_000,
        hits_at: Tuple[int, ...] = (1, 3, 10),
    ) -> None:
        self.num_entities = int(num_entities)
        self.chunk_size = int(chunk_size)
        self.hits_at = tuple(hits_at)
        # Build all-positive set from union of splits.
        self._all_pos: Set[Tuple[int, int, int]] = set()
        for split in [train_triples, valid_triples, test_triples]:
            if split is not None and split.numel() > 0:
                for row in split.tolist():
                    self._all_pos.add((int(row[0]), int(row[1]), int(row[2])))
        self.test_triples = test_triples
        self.valid_triples = valid_triples

    def evaluate(
        self,
        model: nn.Module,
        triples: Optional[torch.Tensor] = None,
        filtered: bool = True,
        batch_size: int = 64,
        device: Union[str, torch.device] = "cpu",
    ) -> KGEvalResult:
        """Evaluate ``model`` on ``triples`` (default: test set)."""
        eval_triples = triples if triples is not None else self.test_triples
        return evaluate_filtered_ranking(
            model=model,
            test_triples=eval_triples,
            all_positive_set=self._all_pos,
            num_entities=self.num_entities,
            filtered=filtered,
            batch_size=batch_size,
            chunk_size=self.chunk_size,
            hits_at=self.hits_at,
            device=device,
        )
