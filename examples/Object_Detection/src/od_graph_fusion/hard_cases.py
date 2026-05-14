"""Hard-case mining for the anchor router.

Selects the clusters whose existence makes the anchor-router task possible:
clusters where the anchor source is not the oracle source, or where union /
yolo provides the only TP, or where IoU disagreement is high. All mining
runs on TRAIN clusters only.

The HardCaseSampler emits batched cluster indices with a configurable mix
(default: 40% natural, 20% union-oracle, 20% yolo-oracle, 20% anchor-fail).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .source_router_v3 import NUM_SOURCES, SOURCE_SLOTS


HARD_CASE_TYPES = (
    "A_union_oracle_not_selected",
    "B_yolo_oracle_anchor_picks_rtdetr",
    "C_anchor_fails_alt_succeeds",
    "D_anchor_tp50_zero_alt_tp50_one",
    "E_high_conf_anchor_false_positive",
    "F_low_conf_alt_true_positive",
    "G_high_iou_disagreement",
    "H_aggregate_improves_localization",
)


@dataclass
class ClusterDescriptor:
    """Per-cluster bookkeeping used by hard-case mining.

    All scalars are CPU floats / ints — no tensors stored long-term."""
    graph_idx: int             # index into the train graphs list
    cluster_id: int            # cluster index within the graph
    anchor_slot: int
    oracle_slot: int           # best source slot by utility
    anchor_util: float
    oracle_util: float
    slot_avail: List[bool]
    slot_util: List[float]
    anchor_score: float        # detector confidence at anchor box
    iou_disagreement: float    # 1 - mean pairwise IoU among slot boxes
    hard_case_types: List[str] = field(default_factory=list)


def classify_hard_case(
    *,
    anchor_slot: int,
    oracle_slot: int,
    slot_avail: Sequence[bool],
    slot_util: Sequence[float],
    anchor_util: float,
    oracle_util: float,
    anchor_score: float,
    iou_disagreement: float,
    tp50_threshold: float = 0.5,
    fp_score_threshold: float = 0.7,
    low_conf_threshold: float = 0.3,
) -> List[str]:
    """Return the subset of HARD_CASE_TYPES that this cluster falls into.

    Utility values are AP50-style soft TPs in [0, ~1].
    """
    tags: List[str] = []
    union_slot = SOURCE_SLOTS.get("union", 4)
    yolo_slot = SOURCE_SLOTS.get("yolo_modern", 0)
    rtdetr_slot = SOURCE_SLOTS.get("rt_detr", 2)

    # A: union is oracle but anchor != union
    if oracle_slot == union_slot and anchor_slot != union_slot:
        tags.append("A_union_oracle_not_selected")
    # B: yolo is oracle, anchor is rt_detr
    if oracle_slot == yolo_slot and anchor_slot == rtdetr_slot:
        tags.append("B_yolo_oracle_anchor_picks_rtdetr")
    # C: oracle utility > anchor utility (any positive delta)
    if oracle_util > anchor_util + 0.01:
        tags.append("C_anchor_fails_alt_succeeds")
    # D: anchor below TP threshold, oracle above
    if anchor_util < tp50_threshold <= oracle_util:
        tags.append("D_anchor_tp50_zero_alt_tp50_one")
    # E: anchor has high confidence but oracle != anchor (false-positive-ish)
    if anchor_score >= fp_score_threshold and oracle_slot != anchor_slot and anchor_util < tp50_threshold:
        tags.append("E_high_conf_anchor_false_positive")
    # F: oracle wins with low confidence (we want to learn these too)
    if 0 <= oracle_slot < len(slot_util) and oracle_util >= tp50_threshold:
        # Approximate score-rank: low-conf-alt-TP only when oracle util high
        # but anchor was high-conf wrong. Reuse anchor_score as proxy.
        if anchor_score >= fp_score_threshold and oracle_util > anchor_util + 0.05:
            tags.append("F_low_conf_alt_true_positive")
    # G: high IoU disagreement
    if iou_disagreement >= 0.4:
        tags.append("G_high_iou_disagreement")
    # H: aggregate (union/wbf/nms/best_proposal) is oracle and improves over anchor
    if (oracle_slot >= union_slot and oracle_slot != anchor_slot
            and oracle_util > anchor_util + 0.02):
        tags.append("H_aggregate_improves_localization")
    return tags


def build_descriptors(
    train_clusters: Sequence[Dict[str, Any]],
    *,
    anchor_slot: int,
) -> List[ClusterDescriptor]:
    """Convert per-cluster dicts (from multi-seed runner) into typed descriptors."""
    out: List[ClusterDescriptor] = []
    for d in train_clusters:
        util = d["slot_utility"]
        avail = d["slot_avail"]
        if not bool(avail[anchor_slot]):
            continue
        anchor_u = float(util[anchor_slot].item())
        # Oracle slot among available
        u = util.clone()
        for s in range(u.shape[0]):
            if not bool(avail[s]):
                u[s] = float("-inf")
        oracle_slot = int(u.argmax().item())
        oracle_u = float(u[oracle_slot].item())
        anchor_score = float(d.get("anchor_score", 0.0))
        iou_dis = float(d.get("iou_disagreement", 0.0))
        tags = classify_hard_case(
            anchor_slot=anchor_slot,
            oracle_slot=oracle_slot,
            slot_avail=[bool(x) for x in avail.tolist()],
            slot_util=util.tolist(),
            anchor_util=anchor_u,
            oracle_util=oracle_u,
            anchor_score=anchor_score,
            iou_disagreement=iou_dis,
        )
        out.append(ClusterDescriptor(
            graph_idx=int(d["graph_idx"]),
            cluster_id=int(d["cluster_id"]),
            anchor_slot=anchor_slot,
            oracle_slot=oracle_slot,
            anchor_util=anchor_u,
            oracle_util=oracle_u,
            slot_avail=[bool(x) for x in avail.tolist()],
            slot_util=util.tolist(),
            anchor_score=anchor_score,
            iou_disagreement=iou_dis,
            hard_case_types=tags,
        ))
    return out


def hard_case_counts(
    descriptors: Sequence[ClusterDescriptor],
) -> Dict[str, int]:
    out = {t: 0 for t in HARD_CASE_TYPES}
    for d in descriptors:
        for t in d.hard_case_types:
            out[t] = out.get(t, 0) + 1
    out["_total_clusters"] = len(descriptors)
    out["_any_hard"] = sum(1 for d in descriptors if d.hard_case_types)
    return out


class HardCaseSampler:
    """Yields cluster-descriptor batches mixing natural and hard cases.

    Mix (per batch):
      40% natural
      20% A (union oracle not selected)
      20% B (yolo oracle but anchor picks rt_detr)
      20% C (anchor fails, alt succeeds)
    Falls back to natural if a category is empty.
    """

    def __init__(
        self,
        descriptors: Sequence[ClusterDescriptor],
        *,
        batch_size: int = 32,
        seed: int = 0,
        mix: Optional[Dict[str, float]] = None,
    ):
        self.descriptors = list(descriptors)
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)
        default_mix = {
            "natural": 0.40,
            "A_union_oracle_not_selected": 0.20,
            "B_yolo_oracle_anchor_picks_rtdetr": 0.20,
            "C_anchor_fails_alt_succeeds": 0.20,
        }
        self.mix = mix or default_mix
        self._buckets: Dict[str, List[int]] = {
            k: [] for k in self.mix.keys()
        }
        for i, d in enumerate(self.descriptors):
            self._buckets["natural"].append(i)
            for t in d.hard_case_types:
                if t in self._buckets:
                    self._buckets[t].append(i)

    def bucket_sizes(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._buckets.items()}

    def sample_batch(self) -> List[ClusterDescriptor]:
        out: List[ClusterDescriptor] = []
        for cat, frac in self.mix.items():
            n = max(0, int(round(self.batch_size * frac)))
            pool = self._buckets.get(cat, []) or self._buckets["natural"]
            if not pool:
                continue
            idx = torch.randint(0, len(pool), (n,), generator=self.gen).tolist()
            for i in idx:
                out.append(self.descriptors[pool[i]])
        # Pad with natural if rounding gave us fewer than batch_size
        while len(out) < self.batch_size and self._buckets["natural"]:
            i = int(torch.randint(0, len(self._buckets["natural"]), (1,), generator=self.gen).item())
            out.append(self.descriptors[self._buckets["natural"][i]])
        return out[: self.batch_size]
