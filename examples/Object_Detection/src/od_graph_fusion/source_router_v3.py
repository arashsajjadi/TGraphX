"""TGraphXSourceRouterV3 — True source-slot routing.

Key differences from v2 (TGraphXSourceRouter):
- Explicit source slots [S]: YOLO=0, YOLOE=1, RTDETR=2, RetinaNet=3, Union=4, WBF=5, NMS=6(opt)
- Per-cluster source logits shape: [num_clusters, S]
- Source mask prevents absent-source selection
- Inference uses argmax(masked_source_logits) — same decision rule as training loss
- fuse_with_model returns a trace of chosen_node, chosen_source, scores
- source_selection_accuracy MUST be computed from the fuse trace

Override classifier (optional):
  base_source = NMS/best_proposal (the strong baseline)
  P(override): should I deviate from NMS?
  If yes, choose argmax(source_logits excl. base_source)
  This directly trains TGraphX to override NMS exactly when it helps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import ConvMessagePassing, Graph

# ── Universal source-slot definitions ────────────────────────────────────
# Every fusion method and raw detector is a real candidate source.
SOURCE_SLOTS: Dict[str, int] = {
    "yolo_modern":      0,   # YOLO / YOLO26 fallback
    "yolo_open_vocab":  1,   # YOLOE / YOLO-World / open-vocabulary
    "rt_detr":          2,   # RT-DETR / DETR-family transformer
    "retinanet":        3,   # RetinaNet anchor-based CNN
    "union":            4,   # Geometric union box
    "wbf":              5,   # Weighted Boxes Fusion
    "nms_candidate":    6,   # NMS highest-confidence candidate
    "soft_nms":         7,   # Soft-NMS decayed candidate
    "best_proposal":    8,   # Highest-calibrated-score proposal
    "calibrated_consensus": 9,  # Optional score-calibrated consensus
}
NUM_SOURCES = 10  # 0..9

# Detector / node name → source slot (comprehensive aliases)
_DETECTOR_TO_SLOT: Dict[str, int] = {
    # YOLO modern family (slot 0)
    "yolo_modern": 0, "yolo11n": 0, "yolo11s": 0, "yolo11m": 0, "yolo11x": 0,
    "yolo_modern_synthetic": 0,
    # Open-vocabulary YOLO family (slot 1)
    "yolo_open_vocab": 1, "yoloe": 1, "yoloworld": 1, "yolo_world": 1,
    "open_vocab": 1, "yoloe_11": 1, "yoloe-11": 1,
    "yolo_open_vocab_synthetic": 1,
    # RT-DETR / DETR family (slot 2)
    "rt_detr": 2, "rtdetr": 2, "rt-detr": 2, "detr": 2,
    "rt_detr_synthetic": 2,
    # RetinaNet family (slot 3)
    "retinanet": 3, "retina": 3, "retina_net": 3,
    "retinanet_synthetic": 3,
    # Fusion methods (slots 4–9)
    "union": 4, "consensus": 4,
    "wbf": 5, "weighted_boxes_fusion": 5, "cluster": 5,
    "nms_candidate": 6, "nms": 6,
    "soft_nms": 7, "soft_nms_candidate": 7,
    "best_proposal": 8, "best_proposal_candidate": 8,
    "calibrated_consensus": 9,
}


def detector_name_to_slot(name: str) -> int:
    """Map a detector name string to a source slot index, or -1 if unknown."""
    n = name.lower().replace(" ", "_").replace("-", "_")
    if n in _DETECTOR_TO_SLOT:
        return _DETECTOR_TO_SLOT[n]
    # Fuzzy matching
    for key, slot in _DETECTOR_TO_SLOT.items():
        if key in n or n in key:
            return slot
    return -1


@dataclass
class FuseTrace:
    """Per-cluster trace from deployed source selection."""
    cluster_id: int
    image_id: str
    chosen_node: int         # global node index in graph
    chosen_source_slot: int  # 0..S-1
    chosen_score: float
    base_score: float        # detector confidence at chosen node
    residual_score: float    # graph model contribution
    oracle_node: int = -1    # filled during evaluation
    oracle_source_slot: int = -1
    chosen_matches_oracle: bool = False


class SourceSlotAggregator(nn.Module):
    """Aggregate proposal nodes → per-source-slot embedding per cluster.

    Rule: for each (cluster, source_slot) pair, take the max-quality node
    embedding from that source. If no node exists for that slot in the
    cluster, the slot is masked.
    """

    def __init__(self, embed_dim: int, num_sources: int = NUM_SOURCES):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_sources = num_sources
        self.absent_emb = nn.Parameter(torch.zeros(embed_dim))  # learnable absent embedding

    def forward(
        self,
        node_emb: torch.Tensor,        # [N, embed_dim]
        cluster_of: torch.Tensor,       # [N] long
        node_source_slot: torch.Tensor, # [N] long (-1 for non-candidate)
        n_clusters: int,
        node_score: Optional[torch.Tensor] = None,  # [N] base confidence for top-score agg
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized scatter aggregation — no Python loop, no .item() sync.

        Returns:
            slot_emb:      [n_clusters, S, embed_dim]
            slot_mask:     [n_clusters, S] bool — True if slot has a node
            slot_node_idx: [n_clusters, S] long — EXACT node index per slot
        """
        device = node_emb.device
        N = node_emb.shape[0]
        S = self.num_sources
        D = self.embed_dim

        slot_emb    = self.absent_emb.unsqueeze(0).unsqueeze(0).expand(n_clusters, S, D).clone()
        slot_mask   = torch.zeros(n_clusters, S, dtype=torch.bool, device=device)
        slot_node_idx = torch.full((n_clusters, S), -1, dtype=torch.long, device=device)

        if N == 0:
            return slot_emb, slot_mask, slot_node_idx

        # Valid nodes: cluster in [0, n_clusters) and slot in [0, S)
        valid = (cluster_of >= 0) & (cluster_of < n_clusters) & \
                (node_source_slot >= 0) & (node_source_slot < S)
        if not valid.any():
            return slot_emb, slot_mask, slot_node_idx

        v_idx   = valid.nonzero(as_tuple=False).squeeze(-1)   # [V]
        v_c     = cluster_of[v_idx]                            # [V]
        v_s     = node_source_slot[v_idx]                      # [V]
        v_emb   = node_emb[v_idx]                              # [V, D]
        v_score = node_score[v_idx] if node_score is not None \
                  else torch.zeros(v_idx.shape[0], device=device)

        # Flat index into [n_clusters, S] grid
        flat_idx = v_c * S + v_s                               # [V]
        n_flat   = n_clusters * S

        # For each (cluster, slot) cell, keep the node with highest score.
        # Deterministic tie-breaking: encode (score, node_index) as a single float
        # key = score + node_idx * eps, so equal scores resolve to HIGHEST node index.
        # This is deterministic on both CPU and CUDA.
        eps_tiebreak = 1e-7 / max(1, N)  # small enough not to affect score ordering
        key = v_score.float() + v_idx.float() * eps_tiebreak

        best_key_flat = torch.full((n_flat,), float("-inf"), device=device)
        best_key_flat.scatter_reduce_(0, flat_idx, key, reduce="amax", include_self=True)

        # Winner is the node whose key matches the cell's best key (deterministic)
        wins = (key >= best_key_flat[flat_idx] - 1e-9)  # tiny tolerance for fp precision
        winner_node_flat = torch.full((n_flat,), -1, dtype=torch.long, device=device)
        winner_node_flat.scatter_(0, flat_idx[wins], v_idx[wins])

        # Cells that have at least one valid node
        has_node = winner_node_flat >= 0                       # [n_flat]
        has_node_2d = has_node.view(n_clusters, S)             # [C, S]
        slot_mask[:] = has_node_2d
        slot_node_idx[:] = winner_node_flat.view(n_clusters, S)

        # Scatter winner embeddings
        winner_cells = has_node.nonzero(as_tuple=False).squeeze(-1)  # cells with nodes
        if winner_cells.numel() > 0:
            w_ni = winner_node_flat[winner_cells]              # node indices
            w_c  = winner_cells // S
            w_s  = winner_cells %  S
            slot_emb[w_c, w_s] = node_emb[w_ni]

        return slot_emb, slot_mask, slot_node_idx


class TGraphXSourceRouterV3(nn.Module):
    """True source-slot router for best-source selection.

    Output: source_logits [num_clusters, S] with masked absent sources.
    Inference: chosen_source = argmax(masked_source_logits).
    Training loss: CE(source_logits, best_source_slot) + utility-KL + pairwise.
    """

    NUM_SOURCES = NUM_SOURCES

    def __init__(
        self,
        num_classes: int,
        num_detectors: int,
        crop_size: int,
        crop_channels: int = 16,
        hidden_dim: int = 64,
        metadata_dim: Optional[int] = None,
        edge_feat_dim: int = 14,
        num_message_passing: int = 2,
        num_sources: int = NUM_SOURCES,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_detectors = num_detectors
        self.crop_size = crop_size
        self.num_sources = num_sources

        from .models import CropCNN
        self.crop_encoder = CropCNN(in_channels=3, out_channels=crop_channels,
                                     crop_size=crop_size)
        sp = self.crop_encoder.out_spatial

        # Node encoding: edge-conditioned ConvMP layers
        from .source_router import EdgeConditionedMP
        self.ec_layers = nn.ModuleList([
            EdgeConditionedMP(in_shape=(crop_channels, sp, sp),
                               out_shape=(crop_channels, sp, sp),
                               edge_feat_dim=edge_feat_dim)
            for _ in range(num_message_passing)
        ])

        # Node embedding projection
        md_dim = metadata_dim if metadata_dim is not None else 8 + num_detectors + num_classes
        self.metadata_mlp = nn.Sequential(
            nn.Linear(md_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.node_proj = nn.Sequential(
            nn.Linear(crop_channels + hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        # Source aggregation
        self.slot_agg = SourceSlotAggregator(hidden_dim, num_sources)

        # Per-slot transformer: attend over slots
        self.slot_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)

        # Source routing head
        self.source_head = nn.Linear(hidden_dim, 1)  # score per slot

        # Override head: P(override base_source = NMS) — optional
        # override_head removed: V3 uses pure source_slot routing, not binary override

        # Auxiliary per-node quality head (for backward compat)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.calib_head = nn.Linear(hidden_dim, 1)

    def _build_node_source_slots(
        self,
        node_types: torch.Tensor,      # [N]
        cluster_of: torch.Tensor,       # [N]
        proposal_det_ids: torch.Tensor, # [N] (-1 for non-proposal)
        detector_names: List[str],
    ) -> torch.Tensor:
        """Map every node to a source slot index (forward-time, complete mapping).

        This runs INSIDE model.forward, so the slot_mask built from these slots
        is the authoritative visibility mask for SourceSlotAggregator.
        """
        from .graph_builder import NODE_TYPES
        N = node_types.shape[0]
        slots = torch.full((N,), -1, dtype=torch.long, device=node_types.device)
        for i in range(N):
            nt = int(node_types[i].item())
            if nt == NODE_TYPES["proposal"]:
                d = (int(proposal_det_ids[i].item())
                     if proposal_det_ids is not None and i < proposal_det_ids.shape[0]
                     else -1)
                if 0 <= d < len(detector_names):
                    slot = detector_name_to_slot(detector_names[d])
                    slots[i] = slot
            elif nt == NODE_TYPES["cluster"]:
                slots[i] = SOURCE_SLOTS["wbf"]
            elif nt == NODE_TYPES["consensus"]:
                slots[i] = SOURCE_SLOTS["union"]
            elif "nms_candidate" in NODE_TYPES and nt == NODE_TYPES["nms_candidate"]:
                slots[i] = SOURCE_SLOTS["nms_candidate"]
            elif "soft_nms_candidate" in NODE_TYPES and nt == NODE_TYPES["soft_nms_candidate"]:
                slots[i] = SOURCE_SLOTS["soft_nms"]
            elif "best_proposal_candidate" in NODE_TYPES and nt == NODE_TYPES["best_proposal_candidate"]:
                slots[i] = SOURCE_SLOTS["best_proposal"]
            elif "calibrated_consensus" in NODE_TYPES and nt == NODE_TYPES["calibrated_consensus"]:
                slots[i] = SOURCE_SLOTS["calibrated_consensus"]
            # context nodes and unknown types stay at -1 — not candidate sources
        return slots

    def forward(
        self,
        graph: Graph,
        detector_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        from .graph_builder import NODE_TYPES
        x = graph.node_features          # [N, 3, H, W]
        ei = graph.edge_index
        ea = graph.edge_features
        md = graph.metadata.get("node_metadata") if isinstance(graph.metadata, dict) else None
        node_types = graph.metadata.get("node_types") if isinstance(graph.metadata, dict) else None
        cluster_of = graph.metadata.get("cluster_of_raw")  # set during graph build if available
        device = x.device

        # Encode crops
        h = self.crop_encoder(x)
        for ec in self.ec_layers:
            ea_d = ea.to(device) if ea is not None else None
            h = F.relu(ec(h, ei, ea_d)) + h
        v = self.spatial_pool(h).squeeze(-1).squeeze(-1)  # [N, C_e]

        # Metadata
        if md is None:
            md = torch.zeros(x.shape[0], self.metadata_mlp[0].in_features, device=device, dtype=v.dtype)
        if md.device != device:
            md = md.to(device)
        m = self.metadata_mlp(md)

        # Node embedding
        node_emb = self.node_proj(torch.cat([v, m], dim=1))  # [N, hidden]

        # Per-node quality (auxiliary)
        quality = self.quality_head(node_emb).squeeze(-1)
        calib = self.calib_head(node_emb).squeeze(-1)

        # Source-slot routing
        source_logits = None
        source_mask = None
        slot_assignments = None

        if (node_types is not None and cluster_of is not None and
                isinstance(cluster_of, torch.Tensor) and cluster_of.numel() > 0):
            # Build source slot per node
            proposal_det_ids = graph.metadata.get("proposal_det_ids")
            slot_assignments = self._build_node_source_slots(
                node_types.to(device), cluster_of.to(device),
                (proposal_det_ids.to(device) if proposal_det_ids is not None else None),
                detector_names or [],
            )
            n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
            if n_clusters > 0:
                # Aggregate node embeddings into source slots (top-calibrated-score)
                node_score_t = graph.metadata.get("node_score") if isinstance(graph.metadata, dict) else None
                ns = node_score_t.to(device) if node_score_t is not None else None
                slot_emb, slot_mask, slot_node_idx = self.slot_agg(
                    node_emb, cluster_of.to(device), slot_assignments, n_clusters, ns
                )
                # slot_emb: [n_clusters, S, hidden]
                # Attention over slots to get context-aware embeddings
                attn_out, _ = self.slot_attn(slot_emb, slot_emb, slot_emb,
                                               key_padding_mask=~slot_mask)
                slot_emb2 = self.slot_norm(slot_emb + attn_out)  # [n_clusters, S, hidden]

                # Source logits
                source_logits = self.source_head(slot_emb2).squeeze(-1)  # [n_clusters, S]
                # Mask absent sources: set to -inf
                source_logits = source_logits.masked_fill(~slot_mask, float("-inf"))

                source_mask = slot_mask  # [n_clusters, S] bool
                source_logits_for_override = source_logits.clone()
                # Store slot_node_idx in output so fuse_v3 uses the EXACT aggregated node
                if isinstance(graph.metadata, dict):
                    graph.metadata["_slot_node_idx"] = slot_node_idx

                # Override logit: aggregate over available slots
                available = slot_emb2[slot_mask]  # [num_available, hidden]
                if available.shape[0] > 0:
                    pass  # no binary override in V3 — source_head handles routing

        return {
            "source_logits": source_logits,         # [C, S] or None
            "source_mask": source_mask,              # [C, S] or None
            "quality_logits": quality,               # [N] auxiliary
            "calib_residual": calib,                 # [N]
            "node_emb": node_emb,                    # [N, hidden]
            "slot_assignments": slot_assignments,     # [N] source slot per node
            # Back-compat
            "objectness_logits": quality,
            "class_logits": torch.zeros(x.shape[0], self.num_classes, device=device),
            "box_reg": torch.zeros(x.shape[0], 4, device=device),
        }


# ── Source-routing losses for V3 ──────────────────────────────────────────

def source_slot_loss(
    source_logits: torch.Tensor,   # [C, S]
    source_mask: torch.Tensor,      # [C, S] bool
    best_source_slot: torch.Tensor, # [C] long — slot index of oracle
    utility_per_slot: torch.Tensor, # [C, S] float — continuous utility
    *,
    beta_kl: float = 5.0,
    pairwise_weight: float = 0.5,
    regret_lambda: float = 1.0,
    baseline_slot: Optional[torch.Tensor] = None,
    source_class_weights: Optional[torch.Tensor] = None,  # [S] balance vector
    focal_gamma: float = 0.0,   # >0 adds focal-loss-style down-weighting
) -> Dict[str, torch.Tensor]:
    """Source-slot loss with class-balanced CE + optional focal weighting.

    source_class_weights: inverse-frequency weights per slot.
      Computed from oracle distribution and passed in to upweight rare sources.
    focal_gamma: gamma > 0 applies (1-p_t)^gamma focus on hard clusters.
    """
    device = source_logits.device
    C, S = source_logits.shape
    valid = (best_source_slot >= 0) & (best_source_slot < S)
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)

    if valid_idx.numel() == 0:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"total": z, "ce": z, "kl": z, "pairwise": z}

    per_cluster_losses = []

    for c in valid_idx.tolist():
        bs = int(best_source_slot[c].item())
        log_c = source_logits[c]
        util_c = utility_per_slot[c]
        mask_c = source_mask[c]

        avail = mask_c.nonzero(as_tuple=False).squeeze(-1)
        if avail.numel() == 0:
            continue
        local_log = log_c[avail]
        local_util = util_c[avail]

        local_best = (avail == bs).nonzero(as_tuple=False)
        if local_best.numel() == 0:
            continue
        local_best_idx = int(local_best[0].item())

        # Balanced CE: weight by inverse oracle frequency for this slot
        slot_weight = 1.0
        if source_class_weights is not None and bs < source_class_weights.shape[0]:
            slot_weight = float(source_class_weights[bs].item())
        ce_raw = F.cross_entropy(local_log.unsqueeze(0),
                                  torch.tensor([local_best_idx], device=device),
                                  reduction="none")
        # Focal: down-weight easy predictions, focus on hard ones
        if focal_gamma > 0:
            p_t = float(torch.softmax(local_log, dim=0)[local_best_idx].item())
            focal_w = (1.0 - p_t) ** focal_gamma
        else:
            focal_w = 1.0
        ce = slot_weight * focal_w * ce_raw.mean()

        # KL utility soft labels
        soft = F.softmax(beta_kl * local_util.float(), dim=0)
        kl = F.kl_div(F.log_softmax(local_log, dim=0), soft, reduction="sum")

        # Pairwise ranking (vectorized over local slots)
        Nc = avail.numel()
        pw = torch.tensor(0.0, device=device)
        if Nc > 1:
            n_pairs = 0
            for ai in range(Nc):
                for bi in range(ai + 1, Nc):
                    da = local_util[ai].item(); db = local_util[bi].item()
                    if abs(da - db) < 1e-6: continue
                    pw += F.softplus(local_log[bi] - local_log[ai]) if da > db \
                          else F.softplus(local_log[ai] - local_log[bi])
                    n_pairs += 1
            if n_pairs > 0: pw = pw / n_pairs

        # Per-cluster regret weight
        regret = 0.0
        if baseline_slot is not None:
            base = int(baseline_slot[c].item())
            base_util = float(util_c[base].item()) if 0 <= base < S and mask_c[base] else 0.0
            regret = max(0.0, float(util_c[bs].item()) - base_util)
        w_c = 1.0 + regret_lambda * regret

        per_cluster_losses.append(w_c * (ce + 0.5 * kl + pairwise_weight * pw))

    if not per_cluster_losses:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"total": z, "ce": z, "kl": z, "pairwise": z}

    total = torch.stack(per_cluster_losses).mean()
    return {"total": total, "n_valid": len(per_cluster_losses)}


def compute_source_class_weights(
    best_slots: List[torch.Tensor],
    num_sources: int = NUM_SOURCES,
    min_weight: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced CE.

    Args:
        best_slots: list of [C] tensors (oracle slot per cluster, from training set)
        Returns [S] weight tensor, higher for rare oracle sources.
    """
    counts = torch.zeros(num_sources, dtype=torch.float)
    for bs in best_slots:
        valid = bs[(bs >= 0) & (bs < num_sources)]
        for s in valid.tolist():
            counts[s] += 1.0
    total = counts.sum().item()
    if total == 0:
        return torch.ones(num_sources)
    freq = counts / total
    # w = 1/freq, normalized so mean = 1.0, clamped to [min_weight, max_weight]
    weights = torch.zeros(num_sources)
    for s in range(num_sources):
        if counts[s] > 0:
            weights[s] = (1.0 / freq[s]).clamp(min_weight, max_weight)
        else:
            weights[s] = max_weight  # never-seen oracle gets max weight (push to select)
    # Normalize to mean=1 over non-zero slots
    present = weights > 0
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def build_source_slot_labels(
    meta_list,
    gt_boxes_list,
    gt_labels_list,
    detector_names: List[str],
    class_agnostic: bool = True,
) -> List[Dict]:
    """Build per-cluster source slot utility and best-source labels.

    Returns list of dicts per cluster with:
      best_source_slot, utility_per_slot, baseline_slot
    """
    from .source_router import compute_source_utilities
    results = []
    for (graph, meta), gt_boxes, gt_labels in zip(meta_list, gt_boxes_list, gt_labels_list):
        if gt_boxes is None or gt_boxes.numel() == 0:
            continue
        node_box = graph.metadata.get("node_box")
        node_label = graph.metadata.get("node_label")
        node_score = graph.metadata.get("node_score")
        if node_box is None:
            continue
        util, best_src_node, cand_mask = compute_source_utilities(
            node_box, node_label, node_score, meta.cluster_of_node, meta.node_types,
            gt_boxes, gt_labels, class_agnostic=class_agnostic, iou_match=0.5,
        )
        slot_assignments = graph.metadata.get("slot_assignments")
        if slot_assignments is None:
            continue

        n_clusters = int(meta.cluster_of_node.max().item()) + 1 if meta.cluster_of_node.numel() > 0 else 0
        for c in range(n_clusters):
            in_c = (meta.cluster_of_node == c) & cand_mask
            if not in_c.any():
                continue
            idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)

            # -inf for absent slots; do NOT clamp before availability check
            raw_util = torch.full((NUM_SOURCES,), float('-inf'))
            for ni in idx_c.tolist():
                s = int(slot_assignments[ni].item()) if ni < slot_assignments.shape[0] else -1
                if 0 <= s < NUM_SOURCES:
                    raw_util[s] = max(raw_util[s].item(), util[ni].item())

            # Availability = slots that have at least one real node
            avail_mask = torch.isfinite(raw_util)
            if not avail_mask.any():
                continue
            avail_idx = avail_mask.nonzero(as_tuple=False).squeeze(-1)
            best_local = int(raw_util[avail_idx].argmax().item())
            best_slot_global = int(avail_idx[best_local].item())

            # util_per_slot: absent → 0 (for loss computation); availability tracked separately
            util_per_slot = raw_util.clone()
            util_per_slot[~avail_mask] = 0.0

            results.append({
                "cluster": c, "image_id": meta.image_id,
                "best_source_slot": best_slot_global,
                "utility_per_slot": util_per_slot,
                "source_mask": avail_mask,
                "baseline_slot": SOURCE_SLOTS.get("nms_candidate", 6),
            })
    return results


# ── Fuse with trace ────────────────────────────────────────────────────────

@torch.no_grad()
def fuse_v3(
    model: nn.Module,
    graph: Graph,
    meta,
    *,
    keep_threshold: float = 0.0,
    device: str = "cpu",
    detector_names: Optional[List[str]] = None,
    return_trace: bool = False,
    oracle_utils: Optional[torch.Tensor] = None,  # [N] for trace annotation
) -> Dict[str, Any]:
    """Unified source routing inference with optional trace.

    Decision rule: argmax(masked_source_logits) for V3 model;
    falls back to residual for legacy model.
    """
    from .graph_builder import DetectionGraphMeta, NODE_TYPES
    from .source_router import TGraphXSourceRouter

    model.eval()
    g = graph.to(device)
    is_v3 = isinstance(model, TGraphXSourceRouterV3)

    # Attach cluster_of_raw to graph metadata for V3 slot aggregation
    if is_v3 and isinstance(g.metadata, dict):
        g.metadata["cluster_of_raw"] = meta.cluster_of_node.to(device)
        # Map detector names → proposal slot assignments
        prop_det_ids = meta.proposal_detector_ids
        if prop_det_ids is not None:
            all_slots = torch.full((meta.node_types.shape[0],), -1, dtype=torch.long)
            for i in range(meta.num_proposals):
                d = int(prop_det_ids[i].item()) if i < prop_det_ids.shape[0] else -1
                if 0 <= d < len(detector_names or []):
                    all_slots[i] = detector_name_to_slot((detector_names or [])[d])
            nt = meta.node_types
            all_slots[nt == NODE_TYPES["cluster"]] = SOURCE_SLOTS["wbf"]
            all_slots[nt == NODE_TYPES["consensus"]] = SOURCE_SLOTS["union"]
            if NODE_TYPES.get("nms_candidate") is not None:
                all_slots[nt == NODE_TYPES["nms_candidate"]] = SOURCE_SLOTS.get("nms_candidate", 6)
            if NODE_TYPES.get("soft_nms_candidate") is not None:
                all_slots[nt == NODE_TYPES["soft_nms_candidate"]] = SOURCE_SLOTS.get("soft_nms", 7)
            if NODE_TYPES.get("best_proposal_candidate") is not None:
                all_slots[nt == NODE_TYPES["best_proposal_candidate"]] = SOURCE_SLOTS.get("best_proposal", 8)
            g.metadata["proposal_det_ids"] = prop_det_ids.to(device)
            g.metadata["slot_assignments"] = all_slots.to(device)

    out = model(g, detector_names=detector_names if is_v3 else None)

    node_box = graph.metadata.get("node_box")
    node_label = graph.metadata.get("node_label")
    node_score = graph.metadata.get("node_score")
    if node_box is None or node_label is None:
        r = {"boxes_xyxy": torch.zeros(0, 4), "scores": torch.zeros(0),
             "labels": torch.zeros(0, dtype=torch.long)}
        if return_trace:
            r["trace"] = []
        return r

    from .candidate_mask import candidate_node_mask
    cand_mask = candidate_node_mask(meta.node_types, NODE_TYPES)
    cluster_of = meta.cluster_of_node

    final_boxes: List[torch.Tensor] = []
    final_scores: List[float] = []
    final_labels: List[int] = []
    trace: List[FuseTrace] = []

    if is_v3 and out.get("source_logits") is not None:
        # V3: use source_logits for routing
        source_logits = out["source_logits"].cpu()  # [C, S]
        source_mask = out["source_mask"].cpu()       # [C, S]
        slot_assignments = out.get("slot_assignments")
        if slot_assignments is not None:
            slot_assignments = slot_assignments.cpu()

        for c in range(meta.num_clusters):
            if c >= source_logits.shape[0]:
                break
            sl = source_logits[c]     # [S]
            sm = source_mask[c]       # [S]
            if not sm.any():
                continue
            # Argmax over masked slots = the inference decision
            available_slots = sm.nonzero(as_tuple=False).squeeze(-1)
            best_local = int(sl[available_slots].argmax().item())
            chosen_slot = int(available_slots[best_local].item())
            chosen_score = float(sl[chosen_slot].item())

            # Use slot_node_idx for EXACT node — no mismatch possible
            slot_node_idx = g.metadata.get("_slot_node_idx")
            chosen_idx = None
            if slot_node_idx is not None and c < slot_node_idx.shape[0] and chosen_slot < slot_node_idx.shape[1]:
                ni = int(slot_node_idx[c, chosen_slot].item())
                if ni >= 0:
                    chosen_idx = ni
            if chosen_idx is None:
                # Fallback: search by slot assignment
                in_c = (cluster_of == c) & cand_mask
                if not in_c.any():
                    continue
                idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
                if slot_assignments is not None:
                    for ni in idx_c.tolist():
                        if ni < slot_assignments.shape[0] and int(slot_assignments[ni].item()) == chosen_slot:
                            chosen_idx = ni; break
                if chosen_idx is None:
                    chosen_idx = int(idx_c[0].item())

            # V3: never drop on raw logit — logits are unbounded and uncalibrated.
            # Dropping only happens in legacy path or when no valid node found.

            box = node_box[chosen_idx].clone()
            label = int(node_label[chosen_idx].item())
            base_s = float(node_score[chosen_idx].item()) if node_score is not None else 0.0
            quality_s = float(out["quality_logits"][chosen_idx].item())
            # Routing-calibrated score: sigmoid(source_logit) * max(base_s, 0.1).
            # sigmoid(source_logit) ranks clusters by routing confidence — clusters
            # where the model is certain about the chosen source rank higher, which
            # makes TP predictions outrank FP predictions from uncertain clusters.
            routing_prob = float(torch.sigmoid(torch.tensor(chosen_score)).item())
            final_score = max(routing_prob * max(base_s, 0.1), 1e-6)

            final_boxes.append(box)
            final_scores.append(final_score)
            final_labels.append(label)

            if return_trace:
                trace.append(FuseTrace(
                    cluster_id=c, image_id=meta.image_id,
                    chosen_node=chosen_idx, chosen_source_slot=chosen_slot,
                    chosen_score=chosen_score, base_score=base_s,
                    residual_score=quality_s - base_s,
                    oracle_node=int(oracle_utils.argmax().item()) if oracle_utils is not None else -1,
                ))
    else:
        # Legacy: per-node quality ranking (v2)
        obj_logits = out.get("quality_logits", out.get("objectness_logits"))
        resid = (torch.sigmoid(obj_logits) - 0.5).cpu()
        ns = node_score

        for c in range(meta.num_clusters):
            eligible = (cluster_of == c) & cand_mask
            if not eligible.any():
                continue
            eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
            if ns is not None:
                ranking = ns[eligible_idx] + 0.1 * resid[eligible_idx]
            else:
                ranking = resid[eligible_idx]
            best_local = int(ranking.argmax().item())
            chosen_idx = int(eligible_idx[best_local].item())
            chosen_score = float(ranking[best_local].item())

            if chosen_score < keep_threshold:
                continue

            box = node_box[chosen_idx].clone()
            label = int(node_label[chosen_idx].item())
            base_s = float(ns[chosen_idx].item()) if ns is not None else 0.0

            final_boxes.append(box)
            final_scores.append(max(base_s, 1e-6))
            final_labels.append(label)

            if return_trace:
                trace.append(FuseTrace(
                    cluster_id=c, image_id=meta.image_id,
                    chosen_node=chosen_idx, chosen_source_slot=-1,
                    chosen_score=chosen_score, base_score=base_s,
                    residual_score=float(resid[chosen_idx].item()),
                ))

    result: Dict[str, Any] = {
        "boxes_xyxy": torch.stack(final_boxes, 0) if final_boxes else torch.zeros(0, 4),
        "scores": torch.tensor(final_scores) if final_scores else torch.zeros(0),
        "labels": torch.tensor(final_labels, dtype=torch.long) if final_labels else torch.zeros(0, dtype=torch.long),
    }
    if return_trace:
        result["trace"] = trace
    return result
