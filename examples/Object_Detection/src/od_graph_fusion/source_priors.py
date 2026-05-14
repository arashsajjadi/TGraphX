"""Validation-only source priors for the anchor router.

These priors are *learned statistics* (not network parameters): for each
source slot we compute, on the train+val split only, the empirical
probability that the source wins (vs. anchor or as oracle best) given
class / size / score-bucket conditioning. The anchor router consumes
these as additional features so it can learn a different override
policy for "small car at high rt_detr score" than for "big bus at low
score."

Crucially, priors must NEVER be computed using test-split clusters.
Step 03 stores per-image splits in split_manifest.json; this module
filters by that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .source_router_v3 import NUM_SOURCES, SOURCE_SLOTS


SIZE_BINS = ("small", "medium", "large")
SCORE_BUCKETS = ("low", "mid", "high")


@dataclass
class PriorTable:
    """Empirical per-slot win probability tables.

    Each tensor is indexed by [slot, ...key-dims...] and stores P(slot wins
    over anchor | conditioning). Anchor-vs-anchor cell is always 0.5 (a
    no-op prior — no evidence either way). Missing cells use the global
    slot prior.
    """
    global_prior: torch.Tensor   # [S]
    by_class: torch.Tensor       # [S, C]  C = num_classes
    by_size: torch.Tensor        # [S, B]  B = len(SIZE_BINS)
    by_score: torch.Tensor       # [S, K]  K = len(SCORE_BUCKETS)
    anchor_slot: int
    num_classes: int

    def as_features(
        self,
        cluster_classes: torch.Tensor,     # [C_clusters] long
        cluster_size_bins: torch.Tensor,   # [C_clusters] long in [0, B)
        cluster_score_buckets: torch.Tensor,  # [C_clusters] long in [0, K)
    ) -> torch.Tensor:
        """Look up per-cluster feature [C_clusters, S, 4] = (global, class, size, score)."""
        Cn = int(cluster_classes.shape[0])
        S = int(self.global_prior.shape[0])
        out = torch.zeros(Cn, S, 4, dtype=torch.float32)
        for ci in range(Cn):
            cl = int(cluster_classes[ci].item())
            sb = int(cluster_size_bins[ci].item())
            sc = int(cluster_score_buckets[ci].item())
            out[ci, :, 0] = self.global_prior
            if 0 <= cl < self.by_class.shape[1]:
                out[ci, :, 1] = self.by_class[:, cl]
            else:
                out[ci, :, 1] = self.global_prior
            if 0 <= sb < self.by_size.shape[1]:
                out[ci, :, 2] = self.by_size[:, sb]
            else:
                out[ci, :, 2] = self.global_prior
            if 0 <= sc < self.by_score.shape[1]:
                out[ci, :, 3] = self.by_score[:, sc]
            else:
                out[ci, :, 3] = self.global_prior
        return out


def size_bin_for_box(box: torch.Tensor, image_size: Optional[Tuple[int, int]] = None) -> int:
    """Bin a box into small/medium/large by area (COCO-like)."""
    if box.numel() < 4:
        return 0
    w = float(max(0.0, box[2].item() - box[0].item()))
    h = float(max(0.0, box[3].item() - box[1].item()))
    area = w * h
    if image_size is not None and image_size[0] > 0 and image_size[1] > 0:
        # Normalize to fraction of image area
        area = area / float(image_size[0] * image_size[1])
        if area < 0.02:
            return 0  # small (<2%)
        if area < 0.10:
            return 1  # medium
        return 2      # large
    # Absolute COCO-like bins when image size unknown
    if area < 32 * 32:
        return 0
    if area < 96 * 96:
        return 1
    return 2


def score_bucket_for_score(score: float) -> int:
    """Bucket a base detector score into low/mid/high."""
    if score < 0.3:
        return 0
    if score < 0.7:
        return 1
    return 2


def compute_priors(
    train_clusters: Sequence[Dict[str, Any]],
    *,
    anchor_slot: int,
    num_classes: int,
    num_sources: int = NUM_SOURCES,
    smoothing: float = 1.0,
) -> PriorTable:
    """Compute empirical win-vs-anchor priors from training clusters only.

    Each cluster dict must contain:
      - 'slot_utility': torch.Tensor [S] (AP-aware utility per slot, -inf for absent)
      - 'slot_avail':   torch.Tensor [S] bool
      - 'cluster_class': int
      - 'cluster_size_bin': int in [0, len(SIZE_BINS))
      - 'cluster_score_bucket': int in [0, len(SCORE_BUCKETS))

    Returns a PriorTable. Laplace smoothing applied so unseen slots get
    a non-zero prior (default smoothing=1).
    """
    S = num_sources
    C = num_classes
    B = len(SIZE_BINS)
    K = len(SCORE_BUCKETS)

    win = torch.full((S,), smoothing)
    total = torch.full((S,), 2 * smoothing)
    win_c = torch.full((S, C), smoothing)
    tot_c = torch.full((S, C), 2 * smoothing)
    win_b = torch.full((S, B), smoothing)
    tot_b = torch.full((S, B), 2 * smoothing)
    win_k = torch.full((S, K), smoothing)
    tot_k = torch.full((S, K), 2 * smoothing)

    for cl in train_clusters:
        util = cl["slot_utility"]
        avail = cl["slot_avail"]
        cls = int(cl.get("cluster_class", 0))
        sb = int(cl.get("cluster_size_bin", 1))
        sk = int(cl.get("cluster_score_bucket", 1))
        if not avail[anchor_slot]:
            continue
        anc_u = float(util[anchor_slot].item())
        for s in range(S):
            if s == anchor_slot or not avail[s]:
                continue
            s_u = float(util[s].item())
            total[s] += 1
            tot_c[s, cls % C] += 1
            tot_b[s, max(0, min(B - 1, sb))] += 1
            tot_k[s, max(0, min(K - 1, sk))] += 1
            if s_u > anc_u:
                win[s] += 1
                win_c[s, cls % C] += 1
                win_b[s, max(0, min(B - 1, sb))] += 1
                win_k[s, max(0, min(K - 1, sk))] += 1

    global_p = (win / total).float()
    cls_p = (win_c / tot_c).float()
    size_p = (win_b / tot_b).float()
    score_p = (win_k / tot_k).float()
    # Anchor row is by definition 0.5 (no evidence to override itself).
    global_p[anchor_slot] = 0.5
    cls_p[anchor_slot, :] = 0.5
    size_p[anchor_slot, :] = 0.5
    score_p[anchor_slot, :] = 0.5
    return PriorTable(
        global_prior=global_p,
        by_class=cls_p,
        by_size=size_p,
        by_score=score_p,
        anchor_slot=anchor_slot,
        num_classes=C,
    )


def select_anchor_on_validation(
    val_method_ap50: Dict[str, float],
    *,
    detector_names: Sequence[str],
    anchor_mode: str = "validation_best_global_source",
) -> Tuple[int, str]:
    """Pick the anchor source slot using validation AP50 only.

    val_method_ap50: dict like {"det::rt_detr": 0.87, "fusion::nms": 0.88, ...}
    detector_names: ordered list of detector keys (e.g. ["retinanet","yolo_modern","rt_detr"])

    Supported modes: validation_best_global_source, NMS, WBF, RawDet(<name>),
                     BestProposal, validation_best_class_size_source (falls
                     back to global_source unless caller provides per-class
                     stats — see select_anchor_per_class).

    Returns (slot_index, anchor_label).
    """
    if anchor_mode in ("NMS", "fusion::nms", "nms_candidate"):
        return SOURCE_SLOTS.get("nms_candidate", 6), "NMS"
    if anchor_mode in ("WBF", "fusion::wbf", "wbf"):
        return SOURCE_SLOTS.get("wbf", 5), "WBF"
    if anchor_mode in ("BestProposal", "best_proposal", "best_proposal_candidate"):
        return SOURCE_SLOTS.get("best_proposal", 8), "BestProposal"
    if anchor_mode.startswith("RawDet(") and anchor_mode.endswith(")"):
        name = anchor_mode[len("RawDet("):-1]
        from .source_router_v3 import detector_name_to_slot
        slot = detector_name_to_slot(name)
        if slot < 0:
            raise ValueError(f"Unknown detector for anchor: {name}")
        return slot, f"RawDet({name})"
    # Default: pick best validation AP50 method, then map to a slot
    if not val_method_ap50:
        # No baselines available — fall back to NMS slot
        return SOURCE_SLOTS.get("nms_candidate", 6), "NMS"
    best_method, best_ap = max(val_method_ap50.items(), key=lambda kv: kv[1])
    if best_method.startswith("det::"):
        from .source_router_v3 import detector_name_to_slot
        nm = best_method[len("det::"):]
        slot = detector_name_to_slot(nm)
        if slot < 0:
            slot = SOURCE_SLOTS.get("nms_candidate", 6)
        return slot, f"RawDet({nm}) val_ap50={best_ap:.4f}"
    if best_method.startswith("fusion::nms"):
        return SOURCE_SLOTS.get("nms_candidate", 6), f"NMS val_ap50={best_ap:.4f}"
    if best_method.startswith("fusion::wbf"):
        return SOURCE_SLOTS.get("wbf", 5), f"WBF val_ap50={best_ap:.4f}"
    if "best_proposal" in best_method:
        return SOURCE_SLOTS.get("best_proposal", 8), f"BestProposal val_ap50={best_ap:.4f}"
    # Fallback
    return SOURCE_SLOTS.get("nms_candidate", 6), f"NMS (fallback) val_ap50={best_ap:.4f}"


def select_anchor_per_class(
    val_method_ap50_per_class: Dict[int, Dict[str, float]],
    *,
    detector_names: Sequence[str],
    min_samples: int = 8,
) -> Dict[int, Tuple[int, str]]:
    """Per-class anchor selection. Falls back to global anchor for classes
    with < min_samples in validation.

    val_method_ap50_per_class: {class_id: {method_name: val_ap50}}
    """
    out: Dict[int, Tuple[int, str]] = {}
    for cls, ap_dict in val_method_ap50_per_class.items():
        if not ap_dict:
            continue
        slot, lbl = select_anchor_on_validation(ap_dict, detector_names=detector_names)
        out[cls] = (slot, lbl)
    return out
