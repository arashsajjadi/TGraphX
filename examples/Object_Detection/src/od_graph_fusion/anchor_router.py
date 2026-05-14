"""TGraphXAnchorRouter — guarded-improvement router.

Design contract (see reports/OPUS_FAILURE_AUDIT.md §11 for derivation):

  1. An *anchor source* is chosen once on the validation split. By default
     it is the single source (raw detector or classical fusion) with the
     highest validation AP50 on the same data the router sees.
  2. At inference, the router keeps the anchor by default. It overrides
     only when (a) the predicted per-source AP-aware gain over the anchor
     is positive, and (b) the predicted gain exceeds a calibrated
     threshold selected on validation.
  3. The training objective optimizes pairwise gain over the anchor, not
     free softmax over sources. False overrides are penalized 5–10× harder
     than missed overrides.
  4. Specialist heads exist for the suppressed sources (union, yolo_modern)
     and for the over-selected sources (rt_detr, retinanet). Each is a
     small binary head P(source beats anchor) conditioned on pairwise
     features the base v3 router never sees.
  5. Final detection scoring uses a TP50 head, not the routing logit. The
     TP50 head is temperature-scaled on validation before being deployed.

The class wraps `TGraphXSourceRouterV3`'s slot aggregator + transformer so
we re-use the heavy encoding work. New: priors, pair embeddings, delta /
specialist / TP50 heads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph

from .source_router_v3 import (
    NUM_SOURCES, SOURCE_SLOTS, detector_name_to_slot,
    SourceSlotAggregator, TGraphXSourceRouterV3,
)
from .pairwise_features import PAIRWISE_FEAT_DIM, SPECIALIST_EXTRA_DIM


# Slots we have explicit specialist heads for.
SPECIALIST_SLOTS: Dict[str, int] = {
    "union":       SOURCE_SLOTS["union"],
    "yolo_modern": SOURCE_SLOTS["yolo_modern"],
    "rt_detr":     SOURCE_SLOTS["rt_detr"],
    "retinanet":   SOURCE_SLOTS["retinanet"],
}


@dataclass
class AnchorRouterConfig:
    num_classes: int
    num_detectors: int
    crop_size: int
    anchor_slot: int
    crop_channels: int = 16
    hidden_dim: int = 64
    metadata_dim: Optional[int] = None
    edge_feat_dim: int = 14
    num_message_passing: int = 2
    num_sources: int = NUM_SOURCES
    pair_emb_dim: int = 16
    source_emb_dim: int = 16
    prior_feature_dim: int = 4   # global, class, size, score


class AnchorRouter(nn.Module):
    """Anchor-preserving router with delta-gain heads.

    Output keys per forward:
      source_logits:        [C, S] — masked logits for diagnostics / score modes
      source_mask:          [C, S] bool
      slot_node_idx:        [C, S] long — which graph node represents each slot
      keep_anchor_logit:    [C]   — BCE target = 1 if anchor is correct
      delta_ap50_hat:       [C, S] — predicted AP50 gain over anchor
      delta_iou_hat:        [C, S] — predicted IoU gain over anchor
      tp50_hat:             [C, S] — predicted P(TP@0.5) for each source
      tp75_hat:             [C, S]
      expected_iou_hat:     [C, S]
      specialist_logits:    dict slot_name → [C] specialist P(source beats anchor)
      anchor_slot:          int (echo)
      cluster_class:        [C] long (echo for tests/debug)
    """

    def __init__(self, cfg: AnchorRouterConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.num_detectors = cfg.num_detectors
        self.crop_size = cfg.crop_size
        self.anchor_slot = cfg.anchor_slot
        self.num_sources = cfg.num_sources

        # Reuse V3's encoder + slot aggregator + attention block.
        # Constructing V3 internally is cheap and keeps our forward simple.
        self._v3 = TGraphXSourceRouterV3(
            num_classes=cfg.num_classes,
            num_detectors=cfg.num_detectors,
            crop_size=cfg.crop_size,
            crop_channels=cfg.crop_channels,
            hidden_dim=cfg.hidden_dim,
            metadata_dim=cfg.metadata_dim,
            edge_feat_dim=cfg.edge_feat_dim,
            num_message_passing=cfg.num_message_passing,
            num_sources=cfg.num_sources,
        )

        H = cfg.hidden_dim
        S = cfg.num_sources

        # Learned source-id and source-pair embeddings.
        self.source_id_emb = nn.Embedding(S, cfg.source_emb_dim)
        self.source_pair_emb = nn.Embedding(S * S, cfg.pair_emb_dim)

        # Pairwise-feature projection (per slot vs anchor).
        self.pair_proj = nn.Linear(PAIRWISE_FEAT_DIM, cfg.pair_emb_dim)

        # Prior projection (4 prior dims per slot).
        self.prior_proj = nn.Linear(cfg.prior_feature_dim, cfg.pair_emb_dim)

        # Per-slot context = [slot_emb, anchor_slot_emb, source_id, pair_emb,
        #                     prior_emb, pair_features, source_pair_id].
        ctx_dim = (H              # slot_emb
                   + H            # anchor_slot_emb
                   + cfg.source_emb_dim
                   + cfg.pair_emb_dim   # pair features
                   + cfg.pair_emb_dim   # priors
                   + cfg.pair_emb_dim)  # learned source-pair embedding
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_dim, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
        )

        # Heads — all consume the per-slot context vector.
        self.delta_ap50_head = nn.Linear(H, 1)
        self.delta_iou_head = nn.Linear(H, 1)
        self.tp50_head = nn.Linear(H, 1)
        self.tp75_head = nn.Linear(H, 1)
        self.expected_iou_head = nn.Linear(H, 1)
        self.source_logit_head = nn.Linear(H, 1)
        # Keep-anchor head consumes the anchor-slot context only.
        self.keep_anchor_head = nn.Linear(H, 1)

        # Specialist heads — separate MLPs with specialist-extra features.
        spec_in = H + SPECIALIST_EXTRA_DIM
        self.specialist_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(spec_in, H), nn.GELU(),
                nn.Linear(H, 1),
            ) for name in SPECIALIST_SLOTS.keys()
        })

    # ── slot encoding via V3 ────────────────────────────────────────────
    def _encode(
        self,
        graph: Graph,
        detector_names: List[str],
    ) -> Dict[str, Any]:
        """Run V3 encoder; return slot embeddings + slot_node_idx + slot_mask.

        We do NOT re-use V3's source_logits — they go through a different
        head. We do re-use slot_emb (post-attention) because the encoder
        cost is dominant.
        """
        # Reach into V3 by calling forward but only consuming the parts we need.
        out = self._v3(graph, detector_names=detector_names)
        slot_node_idx = None
        if isinstance(graph.metadata, dict):
            slot_node_idx = graph.metadata.get("_slot_node_idx")
        return {
            "node_emb": out["node_emb"],
            "source_mask": out["source_mask"],
            "slot_node_idx": slot_node_idx,
            "slot_assignments": out["slot_assignments"],
            # The V3 source_logits live in `out["source_logits"]` and are
            # also returned in `forward` for back-compat with score modes.
            "v3_source_logits": out["source_logits"],
        }

    def _slot_embedding(
        self,
        graph: Graph,
        detector_names: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (slot_emb [C,S,H], slot_mask [C,S], slot_node_idx [C,S],
        v3_source_logits [C,S])."""
        # The cleanest way to get post-attention slot_emb is to re-run the
        # V3 internal aggregation directly (cheap because we already have
        # node embeddings).
        enc = self._encode(graph, detector_names)
        node_emb = enc["node_emb"]
        slot_assignments = enc["slot_assignments"]
        cluster_of = None
        if isinstance(graph.metadata, dict):
            cluster_of = graph.metadata.get("cluster_of_raw")
        if cluster_of is None:
            # Fall back: derive from V3's source_mask shape
            sm = enc["source_mask"]
            if sm is None:
                return (
                    torch.zeros(0, self.num_sources, self.cfg.hidden_dim, device=device),
                    torch.zeros(0, self.num_sources, dtype=torch.bool, device=device),
                    torch.full((0, self.num_sources), -1, dtype=torch.long, device=device),
                    enc["v3_source_logits"] if enc["v3_source_logits"] is not None else
                    torch.zeros(0, self.num_sources, device=device),
                )
            n_clusters = sm.shape[0]
        else:
            cluster_of = cluster_of.to(device)
            n_clusters = int(cluster_of.max().item()) + 1 if cluster_of.numel() > 0 else 0
        node_score_t = (graph.metadata.get("node_score").to(device)
                        if isinstance(graph.metadata, dict)
                        and graph.metadata.get("node_score") is not None else None)
        slot_emb, slot_mask, slot_node_idx = self._v3.slot_agg(
            node_emb, cluster_of, slot_assignments, n_clusters, node_score_t,
        )
        # Slot attention block (re-use V3's pretrained-from-scratch one).
        attn_out, _ = self._v3.slot_attn(slot_emb, slot_emb, slot_emb,
                                          key_padding_mask=~slot_mask)
        slot_emb2 = self._v3.slot_norm(slot_emb + attn_out)
        return slot_emb2, slot_mask, slot_node_idx, enc["v3_source_logits"]

    # ── forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        graph: Graph,
        detector_names: List[str],
        *,
        pairwise_feats: Optional[torch.Tensor] = None,     # [C, S, PAIRWISE_FEAT_DIM]
        priors_feats: Optional[torch.Tensor] = None,        # [C, S, prior_feature_dim]
        cluster_class: Optional[torch.Tensor] = None,       # [C] long
        specialist_extras: Optional[Dict[str, torch.Tensor]] = None,  # {name: [C, SPECIALIST_EXTRA_DIM]}
        anchor_slot_per_cluster: Optional[torch.Tensor] = None,        # [C] long
    ) -> Dict[str, Any]:
        device = graph.node_features.device
        slot_emb, slot_mask, slot_node_idx, v3_source_logits = self._slot_embedding(
            graph, detector_names, device,
        )
        C, S, H = slot_emb.shape
        if C == 0:
            empty_f = torch.zeros(0, S, device=device)
            return {
                "source_logits": v3_source_logits,
                "source_mask": slot_mask,
                "slot_node_idx": slot_node_idx,
                "keep_anchor_logit": torch.zeros(0, device=device),
                "delta_ap50_hat": empty_f, "delta_iou_hat": empty_f,
                "tp50_hat": empty_f, "tp75_hat": empty_f, "expected_iou_hat": empty_f,
                "specialist_logits": {k: torch.zeros(0, device=device) for k in SPECIALIST_SLOTS},
                "anchor_slot": self.anchor_slot,
                "cluster_class": torch.zeros(0, dtype=torch.long, device=device),
            }

        # Anchor slot per cluster (allow override; default = global anchor).
        if anchor_slot_per_cluster is None:
            anchor_slot_t = torch.full((C,), self.anchor_slot, dtype=torch.long, device=device)
        else:
            anchor_slot_t = anchor_slot_per_cluster.to(device)

        # Pair features / priors: zeros if not provided (test/diag).
        if pairwise_feats is None:
            pairwise_feats = torch.zeros(C, S, PAIRWISE_FEAT_DIM, device=device)
        else:
            pairwise_feats = pairwise_feats.to(device)
        if priors_feats is None:
            priors_feats = torch.zeros(C, S, self.cfg.prior_feature_dim, device=device)
        else:
            priors_feats = priors_feats.to(device)

        # Embeddings.
        slot_ids = torch.arange(S, device=device)
        source_id_e = self.source_id_emb(slot_ids).unsqueeze(0).expand(C, S, -1)  # [C, S, source_emb_dim]
        # Anchor slot embedding broadcast per cluster.
        anchor_slot_e = self.source_id_emb(anchor_slot_t)  # [C, source_emb_dim]
        # Source-pair embedding: index = anchor * S + s
        pair_ids = anchor_slot_t.unsqueeze(1) * S + slot_ids.unsqueeze(0)  # [C, S]
        pair_e = self.source_pair_emb(pair_ids)  # [C, S, pair_emb_dim]

        pair_feat_e = self.pair_proj(pairwise_feats)  # [C, S, pair_emb_dim]
        prior_e = self.prior_proj(priors_feats)        # [C, S, pair_emb_dim]

        # Anchor slot embedding from slot_emb (the actual representation of
        # whatever node sits in the anchor slot). Use slot_emb.gather.
        gather_idx = anchor_slot_t.view(C, 1, 1).expand(C, 1, H)
        anchor_slot_emb = slot_emb.gather(1, gather_idx).expand(C, S, H)  # [C, S, H]

        # Per-slot context.
        ctx = torch.cat([
            slot_emb,                              # [C, S, H]
            anchor_slot_emb,                       # [C, S, H]
            source_id_e,                           # [C, S, source_emb_dim]
            pair_feat_e,                            # [C, S, pair_emb_dim]
            prior_e,                                # [C, S, pair_emb_dim]
            pair_e,                                 # [C, S, pair_emb_dim]
        ], dim=-1)
        ctx_h = self.ctx_mlp(ctx)  # [C, S, H]

        # Heads.
        delta_ap50 = self.delta_ap50_head(ctx_h).squeeze(-1)
        delta_iou = self.delta_iou_head(ctx_h).squeeze(-1)
        tp50 = self.tp50_head(ctx_h).squeeze(-1)
        tp75 = self.tp75_head(ctx_h).squeeze(-1)
        e_iou = self.expected_iou_head(ctx_h).squeeze(-1)
        source_logits = self.source_logit_head(ctx_h).squeeze(-1)

        # Mask absent slots out of every per-slot output.
        neg_inf = float("-inf")
        delta_ap50 = delta_ap50.masked_fill(~slot_mask, 0.0)   # 0 gain for absent
        delta_iou = delta_iou.masked_fill(~slot_mask, 0.0)
        tp50 = tp50.masked_fill(~slot_mask, neg_inf)
        tp75 = tp75.masked_fill(~slot_mask, neg_inf)
        e_iou = e_iou.masked_fill(~slot_mask, 0.0)
        source_logits = source_logits.masked_fill(~slot_mask, neg_inf)

        # Keep-anchor logit (per cluster, uses anchor's context vector).
        anc_ctx = ctx_h.gather(1, anchor_slot_t.view(C, 1, 1).expand(C, 1, H)).squeeze(1)  # [C, H]
        keep_anchor_logit = self.keep_anchor_head(anc_ctx).squeeze(-1)  # [C]

        # Specialist heads.
        specialist_logits: Dict[str, torch.Tensor] = {}
        if specialist_extras is None:
            specialist_extras = {}
        for name, slot_idx in SPECIALIST_SLOTS.items():
            extra = specialist_extras.get(name)
            if extra is None:
                extra = torch.zeros(C, SPECIALIST_EXTRA_DIM, device=device)
            else:
                extra = extra.to(device)
            slot_ctx = ctx_h[:, slot_idx, :]  # [C, H]
            spec_in = torch.cat([slot_ctx, extra], dim=-1)
            mask_s = slot_mask[:, slot_idx].float()
            logit = self.specialist_heads[name](spec_in).squeeze(-1)
            # If slot absent, set to a large negative number so override gate ignores it.
            specialist_logits[name] = logit.masked_fill(mask_s < 0.5, -10.0)

        return {
            "source_logits": source_logits,
            "source_mask": slot_mask,
            "slot_node_idx": slot_node_idx,
            "keep_anchor_logit": keep_anchor_logit,
            "delta_ap50_hat": delta_ap50,
            "delta_iou_hat": delta_iou,
            "tp50_hat": tp50,
            "tp75_hat": tp75,
            "expected_iou_hat": e_iou,
            "specialist_logits": specialist_logits,
            "anchor_slot": int(self.anchor_slot),
            "cluster_class": cluster_class if cluster_class is not None
                              else torch.zeros(C, dtype=torch.long, device=device),
            "v3_source_logits": v3_source_logits,
        }

    # ── inference decision rule ─────────────────────────────────────────
    def decide(
        self,
        out: Dict[str, Any],
        *,
        override_threshold: float,
        anchor_slot_per_cluster: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Choose a slot per cluster following the keep/override rule.

        Returns (chosen_slot [C], chose_anchor [C] bool). The decision rule is:

          best_alt = argmax(delta_ap50_hat over available non-anchor slots)
          if delta_ap50_hat[best_alt] > override_threshold:
              chosen = best_alt
          else:
              chosen = anchor

        Slots whose specialist head predicts P < 0.5 are also rejected even
        if their delta head was positive — the specialist head is a *second*
        gate against false overrides.
        """
        device = out["delta_ap50_hat"].device
        delta = out["delta_ap50_hat"]
        mask = out["source_mask"]
        C, S = delta.shape
        if anchor_slot_per_cluster is None:
            anc = torch.full((C,), int(out["anchor_slot"]), dtype=torch.long, device=device)
        else:
            anc = anchor_slot_per_cluster.to(device)

        # Apply specialist gates (P < 0.5 → blocked).
        specialist = out.get("specialist_logits", {}) or {}
        spec_block = torch.zeros_like(mask)
        for name, slot_idx in SPECIALIST_SLOTS.items():
            if name not in specialist:
                continue
            prob = torch.sigmoid(specialist[name])
            spec_block[:, slot_idx] = prob < 0.5
        # Anchor cannot be blocked by specialist.
        spec_block.scatter_(1, anc.unsqueeze(1), False)

        # Mask out anchor slot AND absent slots AND specialist-blocked slots from "alts".
        alts_mask = mask.clone()
        alts_mask.scatter_(1, anc.unsqueeze(1), False)
        alts_mask = alts_mask & ~spec_block
        delta_alts = delta.masked_fill(~alts_mask, float("-inf"))
        best_delta, best_alt = delta_alts.max(dim=1)
        chose_alt = best_delta > override_threshold
        chosen = torch.where(chose_alt, best_alt, anc)
        chose_anchor = ~chose_alt
        return chosen, chose_anchor


def calibrate_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_steps: int = 100,
    lr: float = 0.05,
) -> float:
    """Single-temperature scaling on validation logits.

    Inputs are 1-D tensors of logits and {0,1} targets. Returns the temperature
    that minimizes BCE-with-logits / T. We never temperature-scale on test.
    """
    if logits.numel() == 0:
        return 1.0
    log_t = torch.log(torch.tensor(1.0, requires_grad=True))
    log_t = nn.Parameter(log_t.clone())
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=n_steps)
    targets = targets.float()

    def closure():
        opt.zero_grad()
        T = torch.exp(log_t)
        loss = F.binary_cross_entropy_with_logits(logits / T, targets)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return 1.0
    T = float(torch.exp(log_t).item())
    return max(0.05, min(20.0, T))
