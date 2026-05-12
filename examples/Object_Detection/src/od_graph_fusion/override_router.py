"""NMS-preserving graph override router.

Core idea:
  - NMS/best-proposal is the strong default expert.
  - TGraphX learns ONLY: "should I override NMS, and if so, to which source?"
  - If P(override) < threshold → keep NMS (safe default).
  - If P(override) ≥ threshold → choose argmax(source_logits excluding NMS).

This is much simpler than learning full source routing from scratch, and
directly addresses the two failures:
  - Residual mode: override=never → AP fine but never beats NMS.
  - V3 mode: trying to route all sources → near-random (too hard to learn).

Training labels:
  override_target = 1  if oracle_source != NMS_source
                     0  otherwise
  (ground truth used only during training, never at inference)

The model must:
  1. Keep NMS when NMS is oracle (override_target=0 → correct)
  2. Override to the right source when NMS is not oracle (override_target=1 → correct)

Success metrics:
  - override_precision  = TP_override / (TP_override + FP_override)
  - override_recall     = TP_override / (TP_override + FN_override)
  - successful_override_rate = overrides that improve IoU / all overrides
  - failed_override_rate = overrides that hurt IoU / all overrides
  - hard_case_gap_recovery  (most important)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgraphx import Graph
from .source_router_v3 import (
    TGraphXSourceRouterV3, SourceSlotAggregator, FuseTrace,
    SOURCE_SLOTS, NUM_SOURCES, detector_name_to_slot,
)
from .source_router import (
    compute_source_utilities, oracle_gap_recovery, EdgeConditionedMP,
)


# Source-slot including NMS
NMS_SLOT = SOURCE_SLOTS.get("nms_candidate", 6)   # already = 6
ALL_SLOTS = list(range(NUM_SOURCES))               # 0..6


class NMSOverrideRouter(nn.Module):
    """Override router: output override_logit + source_logits [C, S].

    Inference rule:
        if sigmoid(override_logit[c]) < threshold:
            chosen_slot = NMS_slot          # keep NMS
        else:
            chosen_slot = argmax(source_logits[c], mask=source_mask[c])

    Training:
        L_override: BCE(override_logit, override_target)
        L_source:   CE(source_logits, oracle_slot)   — hard cases only
        L_kl:       KL(softmax(beta*utility), softmax(source_logits))
        L_pair:     pairwise ranking over source utilities

        Per-cluster weight: w_c = 1 + lambda * regret_nms_c
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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_detectors = num_detectors

        from .models import CropCNN
        self.crop_encoder = CropCNN(3, crop_channels, crop_size)
        sp = self.crop_encoder.out_spatial

        self.ec_layers = nn.ModuleList([
            EdgeConditionedMP((crop_channels, sp, sp), (crop_channels, sp, sp), edge_feat_dim)
            for _ in range(num_message_passing)
        ])

        md_dim = metadata_dim if metadata_dim is not None else 8 + num_detectors + num_classes
        self.metadata_mlp = nn.Sequential(
            nn.Linear(md_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.node_proj = nn.Sequential(
            nn.Linear(crop_channels + hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.slot_agg = SourceSlotAggregator(hidden_dim, NUM_SOURCES)
        self.slot_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)

        # Cluster summary: aggregate all available source embeddings
        self.cluster_summary = nn.Linear(hidden_dim, hidden_dim)

        # Override head: per cluster, should we override NMS?
        self.override_head = nn.Linear(hidden_dim, 1)

        # Source head: which source to pick if we override?
        self.source_head = nn.Linear(hidden_dim, 1)  # per slot

        # Auxiliary quality head (per node)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def _encode_nodes(self, g: Graph) -> torch.Tensor:
        """Encode all nodes → node embeddings [N, hidden]."""
        x = g.node_features; ei = g.edge_index
        ea = g.edge_features; device = x.device
        h = self.crop_encoder(x)
        for ec in self.ec_layers:
            h = F.relu(ec(h, ei, ea.to(device) if ea is not None else None)) + h
        v = self.spatial_pool(h).squeeze(-1).squeeze(-1)  # [N, crop_channels]
        md = g.metadata.get("node_metadata") if isinstance(g.metadata, dict) else None
        if md is None:
            md = torch.zeros(x.shape[0], self.metadata_mlp[0].in_features, device=device)
        m = self.metadata_mlp(md.to(device))  # [N, hidden]
        return self.node_proj(torch.cat([v, m], dim=1))  # [N, hidden]

    def forward(
        self,
        g: Graph,
        detector_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        device = g.node_features.device
        node_emb = self._encode_nodes(g)  # [N, hidden]
        quality = self.quality_head(node_emb).squeeze(-1)

        node_types = g.metadata.get("node_types") if isinstance(g.metadata, dict) else None
        cluster_of = g.metadata.get("cluster_of_raw") if isinstance(g.metadata, dict) else None
        prop_det_ids = g.metadata.get("proposal_det_ids") if isinstance(g.metadata, dict) else None

        source_logits = None; source_mask = None; override_logits = None

        if node_types is not None and cluster_of is not None:
            from .graph_builder import NODE_TYPES
            N = node_emb.shape[0]
            slot_of = torch.full((N,), -1, dtype=torch.long, device=device)
            for i in range(N):
                nt = int(node_types.to(device)[i])
                if nt == NODE_TYPES["proposal"]:
                    d = int(prop_det_ids[i]) if prop_det_ids is not None and i < prop_det_ids.shape[0] else -1
                    if 0 <= d < len(detector_names or []):
                        slot_of[i] = detector_name_to_slot((detector_names or [])[d])
                elif nt == NODE_TYPES["cluster"]:
                    slot_of[i] = SOURCE_SLOTS["wbf"]
                elif nt == NODE_TYPES["consensus"]:
                    slot_of[i] = SOURCE_SLOTS["union"]
                elif nt == NODE_TYPES["proposal"] and prop_det_ids is None:
                    slot_of[i] = 0  # fallback

            n_clusters = int(cluster_of.to(device).max().item()) + 1 if cluster_of.numel() > 0 else 0
            if n_clusters > 0:
                slot_emb, slot_mask = self.slot_agg(node_emb, cluster_of.to(device),
                                                     slot_of, n_clusters)
                attn_out, _ = self.slot_attn(slot_emb, slot_emb, slot_emb,
                                              key_padding_mask=~slot_mask)
                slot_emb2 = self.slot_norm(slot_emb + attn_out)  # [C, S, hidden]

                # Source logits per slot
                source_logits = self.source_head(slot_emb2).squeeze(-1)  # [C, S]
                source_logits = source_logits.masked_fill(~slot_mask, float("-inf"))
                source_mask = slot_mask

                # Override logit: use summary of all available sources
                # Mask-mean over available slots
                masked_emb = slot_emb2 * slot_mask.unsqueeze(-1).float()  # [C, S, hidden]
                n_avail = slot_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                cluster_summary = self.cluster_summary(
                    masked_emb.sum(dim=1) / n_avail  # [C, hidden]
                )
                override_logits = self.override_head(cluster_summary).squeeze(-1)  # [C]

        return {
            "source_logits": source_logits,
            "source_mask": source_mask,
            "override_logits": override_logits,
            "quality_logits": quality,
            "objectness_logits": quality,  # back-compat
            "class_logits": torch.zeros(g.node_features.shape[0], self.num_classes, device=device),
            "box_reg": torch.zeros(g.node_features.shape[0], 4, device=device),
        }


# ── Override loss ──────────────────────────────────────────────────────────

def override_routing_loss(
    source_logits: torch.Tensor,    # [C, S]
    source_mask: torch.Tensor,      # [C, S] bool
    override_logits: torch.Tensor,  # [C]
    best_source_slot: torch.Tensor, # [C] oracle source slot
    nms_source_slot: torch.Tensor,  # [C] NMS source slot
    utility_per_slot: torch.Tensor, # [C, S]
    *,
    source_weight: float = 2.0,
    kl_weight: float = 0.5,
    pairwise_weight: float = 0.3,
    regret_lambda: float = 2.0,
    beta_kl: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Per-cluster override routing loss.

    override_target = 1 if oracle != NMS else 0
    w_c = 1 + lambda * regret_nms_c
    """
    device = source_logits.device
    C, S = source_logits.shape
    valid = (best_source_slot >= 0) & (best_source_slot < S)
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"total": z, "override": z, "source": z, "kl": z}

    per_cluster = []
    n_override, n_no_override = 0, 0

    for c in valid_idx.tolist():
        bs = int(best_source_slot[c]); ns = int(nms_source_slot[c])
        is_override = int(bs != ns)
        override_tgt = torch.tensor([float(is_override)], device=device)
        if is_override:
            n_override += 1
        else:
            n_no_override += 1

        # 1. Override BCE
        l_ov = F.binary_cross_entropy_with_logits(
            override_logits[c:c+1], override_tgt
        )

        # 2. Source CE (only for override cases, weighted more heavily)
        avail = source_mask[c].nonzero(as_tuple=False).squeeze(-1)
        if avail.numel() == 0:
            per_cluster.append(l_ov)
            continue
        local_bs = (avail == bs).nonzero(as_tuple=False)
        if local_bs.numel() == 0:
            per_cluster.append(l_ov)
            continue
        local_bs_idx = int(local_bs[0].item())
        log_c = source_logits[c, avail]
        l_src = F.cross_entropy(log_c.unsqueeze(0),
                                 torch.tensor([local_bs_idx], device=device))

        # 3. Utility KL
        util_c = utility_per_slot[c, avail]
        soft = F.softmax(beta_kl * util_c.float(), dim=0)
        l_kl = F.kl_div(F.log_softmax(log_c, dim=0), soft, reduction="sum")

        # 4. Pairwise ranking
        Nc = avail.numel()
        l_pw = torch.tensor(0.0, device=device)
        n_pairs = 0
        for ai in range(Nc):
            for bi in range(ai + 1, Nc):
                da = util_c[ai].item(); db = util_c[bi].item()
                if abs(da - db) < 1e-6:
                    continue
                sa = log_c[ai]; sb = log_c[bi]
                l_pw += F.softplus(sb - sa) if da > db else F.softplus(sa - sb)
                n_pairs += 1
        if n_pairs > 0:
            l_pw = l_pw / n_pairs

        # Per-cluster regret weight
        nms_util = float(utility_per_slot[c, ns].item()) if 0 <= ns < S else 0.0
        oracle_util = float(utility_per_slot[c, bs].item())
        regret_c = max(0.0, oracle_util - nms_util)
        w_c = 1.0 + regret_lambda * regret_c

        cluster_loss = l_ov + (source_weight * is_override * l_src
                                + kl_weight * l_kl + pairwise_weight * l_pw)
        per_cluster.append(w_c * cluster_loss)

    if not per_cluster:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"total": z, "n_override": 0, "n_no_override": 0}

    return {
        "total": torch.stack(per_cluster).mean(),
        "n_override": n_override,
        "n_no_override": n_no_override,
    }


# ── Sanity checks ──────────────────────────────────────────────────────────

def oracle_overfit_sanity(
    model: NMSOverrideRouter,
    graphs_and_metas: List[Tuple],
    gt_boxes_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    detector_names: List[str],
    device: str = "cpu",
    max_epochs: int = 200,
    target_source_acc: float = 0.95,
    class_agnostic: bool = True,
) -> Dict[str, Any]:
    """Stage A: can the model overfit oracle source labels on a tiny set?

    Returns dict with 'passed', 'final_source_acc', 'epochs', 'history'.
    """
    from .source_router_v3 import source_slot_loss, build_source_slot_labels

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    passed = False

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_clusters = 0

        for (g, meta), gt_b, gt_l in zip(graphs_and_metas, gt_boxes_list, gt_labels_list):
            if gt_b.numel() == 0:
                continue
            gg = g.to(device)
            # Attach metadata
            if isinstance(gg.metadata, dict):
                gg.metadata["cluster_of_raw"] = meta.cluster_of_node.to(device)
                gg.metadata["proposal_det_ids"] = gg.metadata.get("proposal_det_ids",
                    torch.full((meta.node_types.shape[0],), -1, dtype=torch.long))

            out = model(gg, detector_names=detector_names)
            if out.get("source_logits") is None:
                continue

            sl = out["source_logits"]   # [C, S]
            sm = out["source_mask"]     # [C, S]
            ol = out["override_logits"] # [C]

            # Build labels
            from .source_router import compute_source_utilities
            node_box = gg.metadata.get("node_box")
            node_label = gg.metadata.get("node_label")
            node_score = gg.metadata.get("node_score")
            slot_assigns = gg.metadata.get("slot_assignments", gg.metadata.get("proposal_det_ids"))
            if node_box is None:
                continue

            util, best_src_node, cand_mask = compute_source_utilities(
                node_box, node_label, node_score, meta.cluster_of_node.to(device),
                meta.node_types.to(device), gt_b.to(device), gt_l.to(device),
                class_agnostic=class_agnostic, iou_match=0.5,
            )

            # Map best_src_node → slot
            C, S = sl.shape
            best_slot = torch.full((C,), -1, dtype=torch.long, device=device)
            nms_slot = torch.full((C,), NUM_SOURCES - 1, dtype=torch.long, device=device)  # NMS/WBF default
            util_per_slot = torch.zeros(C, S, device=device)

            if slot_assigns is not None:
                slot_assigns_d = slot_assigns.to(device)
                cluster_of = meta.cluster_of_node.to(device)
                for c in range(C):
                    in_c = (cluster_of == c) & cand_mask
                    if not in_c.any():
                        continue
                    idx_c = in_c.nonzero(as_tuple=False).squeeze(-1)
                    for ni in idx_c.tolist():
                        s = int(slot_assigns_d[ni]) if ni < slot_assigns_d.shape[0] else -1
                        if 0 <= s < S:
                            u = float(util[ni].item())
                            if u > util_per_slot[c, s].item():
                                util_per_slot[c, s] = u
                    # Best slot = argmax utility
                    if util_per_slot[c].max() > 0:
                        best_slot[c] = int(util_per_slot[c].argmax().item())
                    # NMS slot = WBF (index 5) or highest base score
                    nms_slot[c] = SOURCE_SLOTS.get("wbf", 5)

            valid = best_slot >= 0
            if not valid.any():
                continue

            losses = override_routing_loss(
                sl[valid], sm[valid], ol[valid],
                best_slot[valid], nms_slot[valid],
                util_per_slot[valid],
            )
            loss = losses["total"]
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item())

            # Measure source accuracy on this batch
            with torch.no_grad():
                for ci, c in enumerate(valid.nonzero(as_tuple=False).squeeze(-1).tolist()):
                    avail = sm[c].nonzero(as_tuple=False).squeeze(-1)
                    if avail.numel() == 0:
                        continue
                    pred_slot = int(avail[sl[c, avail].argmax()].item())
                    oracle_slot = int(best_slot[c].item())
                    total_correct += int(pred_slot == oracle_slot)
                    total_clusters += 1

        source_acc = total_correct / max(1, total_clusters)
        history.append({"epoch": ep, "loss": total_loss, "source_acc": source_acc})

        if ep % 20 == 0 or ep <= 5:
            print(f"  [sanity A] ep={ep:3d} loss={total_loss:.4f} src_acc={source_acc:.3f}")

        if source_acc >= target_source_acc:
            passed = True
            print(f"  [sanity A] PASSED at epoch {ep} (source_acc={source_acc:.3f})")
            break

    return {
        "passed": passed, "final_source_acc": history[-1]["source_acc"],
        "epochs": len(history), "history": history,
        "n_clusters": total_clusters,
    }


def override_sanity_check(
    model: NMSOverrideRouter,
    hard_graphs: List[Tuple],
    device: str = "cpu",
    max_epochs: int = 100,
    target_precision: float = 0.85,
) -> Dict[str, Any]:
    """Stage B: can the model learn to override NMS on hard cases only?"""
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    passed = False

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_override_correct = 0; n_override_total = 0

        for g, meta, best_slot_t, nms_slot_t, util_t, sm_t in hard_graphs:
            gg = g.to(device)
            out = model(gg)
            if out.get("override_logits") is None:
                continue
            ol = out["override_logits"]; sl = out.get("source_logits"); sm = out.get("source_mask")
            if sl is None:
                continue
            losses = override_routing_loss(
                sl, sm, ol,
                best_slot_t.to(device), nms_slot_t.to(device), util_t.to(device),
            )
            loss = losses["total"]
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += float(loss.item())

            with torch.no_grad():
                preds = (ol.sigmoid() > 0.5).float()
                tgts = (best_slot_t != nms_slot_t).float()
                tp = ((preds == 1) & (tgts == 1)).sum().item()
                fp = ((preds == 1) & (tgts == 0)).sum().item()
                n_override_correct += tp
                n_override_total += tp + fp

        prec = n_override_correct / max(1, n_override_total)
        history.append({"epoch": ep, "loss": total_loss, "override_precision": prec})
        if ep % 10 == 0:
            print(f"  [sanity B] ep={ep:3d} loss={total_loss:.4f} override_prec={prec:.3f}")
        if prec >= target_precision and n_override_total > 0:
            passed = True
            print(f"  [sanity B] PASSED at epoch {ep}")
            break

    return {"passed": passed, "history": history,
            "final_override_precision": history[-1]["override_precision"]}


# ── Inference with override trace ──────────────────────────────────────────

@torch.no_grad()
def fuse_override(
    model: NMSOverrideRouter,
    graph: Graph,
    meta,
    *,
    override_threshold: float = 0.5,
    device: str = "cpu",
    detector_names: Optional[List[str]] = None,
    return_trace: bool = False,
) -> Dict[str, Any]:
    """NMS-guarded override inference.

    For each cluster:
      if sigmoid(override_logit) < threshold → use NMS source
      else → use argmax(source_logits)
    """
    from .graph_builder import NODE_TYPES
    model.eval()
    g = graph.to(device)

    # Attach metadata for V3 slot aggregation
    if isinstance(g.metadata, dict):
        g.metadata["cluster_of_raw"] = meta.cluster_of_node.to(device)

    out = model(g, detector_names=detector_names)

    node_box = graph.metadata.get("node_box")
    node_label = graph.metadata.get("node_label")
    node_score = graph.metadata.get("node_score")
    if node_box is None or node_label is None:
        return {"boxes_xyxy": torch.zeros(0, 4), "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long), "trace": []}

    cand_mask = (
        (meta.node_types == NODE_TYPES["proposal"])
        | (meta.node_types == NODE_TYPES["cluster"])
        | (meta.node_types == NODE_TYPES["consensus"])
    )
    cluster_of = meta.cluster_of_node
    source_logits = out.get("source_logits")
    source_mask = out.get("source_mask")
    override_logits = out.get("override_logits")

    final_boxes: List[torch.Tensor] = []
    final_scores: List[float] = []
    final_labels: List[int] = []
    traces = []

    slot_assigns = graph.metadata.get("slot_assignments", graph.metadata.get("proposal_det_ids"))

    for c in range(meta.num_clusters):
        eligible = (cluster_of == c) & cand_mask
        if not eligible.any():
            continue
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)

        # NMS source = highest base-score node
        ns = node_score
        nms_local = int(eligible_idx[(ns[eligible_idx] if ns is not None else torch.zeros_like(eligible_idx.float())).argmax().item()])
        nms_node = nms_local

        override_prob = 0.0
        chosen_node = nms_node
        chosen_mode = "nms"

        if (source_logits is not None and source_mask is not None
                and override_logits is not None and c < source_logits.shape[0]):
            override_prob = float(override_logits[c].sigmoid().item())
            if override_prob >= override_threshold:
                # Override: pick best source from source_logits
                sl = source_logits[c]; sm = source_mask[c]
                avail = sm.nonzero(as_tuple=False).squeeze(-1)
                if avail.numel() > 0:
                    best_slot = int(avail[sl[avail].argmax()].item())
                    # Find node in this cluster matching best_slot
                    if slot_assigns is not None:
                        for ni in eligible_idx.tolist():
                            s = int(slot_assigns[ni]) if ni < slot_assigns.shape[0] else -1
                            if s == best_slot:
                                chosen_node = ni
                                chosen_mode = "override"
                                break
                        else:
                            # Fallback to NMS if slot not found
                            chosen_node = nms_node
                            chosen_mode = "nms_fallback"

        chosen_score = float(node_score[chosen_node].item()) if node_score is not None else 1.0
        final_boxes.append(node_box[chosen_node].clone())
        final_scores.append(chosen_score)
        final_labels.append(int(node_label[chosen_node].item()))

        if return_trace:
            traces.append({
                "cluster": c, "image_id": meta.image_id,
                "chosen_node": chosen_node, "nms_node": nms_node,
                "override_prob": override_prob, "mode": chosen_mode,
                "threshold": override_threshold,
            })

    result = {
        "boxes_xyxy": torch.stack(final_boxes) if final_boxes else torch.zeros(0, 4),
        "scores": torch.tensor(final_scores) if final_scores else torch.zeros(0),
        "labels": torch.tensor(final_labels, dtype=torch.long) if final_labels else torch.zeros(0, dtype=torch.long),
    }
    if return_trace:
        result["trace"] = traces
    return result
