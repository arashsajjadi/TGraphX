"""Pairwise gain training for AnchorRouter.

L_delta:           SmoothL1(delta_ap50_hat[s], delta_ap50_true[s])
L_pairwise:        margin ranking loss over (s_i, s_j) by true delta
L_keep_override:   BCE(keep_anchor_logit, anchor_was_oracle)
L_specialist:      CE over candidate sources for positive-override clusters only
                   AND per-specialist BCE(P(source beats anchor))
L_tp50:            BCE(tp50_hat[s], tp50_true[s]) for available slots
L_false_override:  multiplier (5-10x) on per-cluster loss when the model
                   would override and the override would be wrong.

Total = L_delta + L_pairwise + 2*L_keep_override + 2*L_specialist + L_tp50
        with the false-override penalty applied per-cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnchorLossWeights:
    delta: float = 1.0
    pairwise: float = 1.0
    keep_override: float = 2.0
    specialist: float = 2.0
    tp50: float = 1.0
    false_override_penalty: float = 7.0   # tunable in [5, 10]
    pairwise_margin: float = 0.02         # AP50 margin for ranking


def _per_cluster_anchor_loss(
    *,
    delta_hat: torch.Tensor,        # [S]
    delta_true: torch.Tensor,       # [S]
    slot_mask: torch.Tensor,        # [S] bool
    anchor_slot: int,
    keep_anchor_logit: torch.Tensor,  # scalar
    keep_target: torch.Tensor,        # scalar 0/1
    tp50_hat: torch.Tensor,          # [S]
    tp50_true: torch.Tensor,         # [S]
    source_logits: torch.Tensor,     # [S]
    best_alt_slot: int,              # -1 if no positive alt
    specialist_logits: Dict[str, torch.Tensor],  # name → scalar
    specialist_true:   Dict[str, torch.Tensor],  # name → scalar 0/1
    weights: AnchorLossWeights,
    override_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Per-cluster losses. Returns dict with keys: delta, pairwise, keep,
    specialist, tp50, total."""
    device = delta_hat.device
    S = delta_hat.shape[0]
    avail = slot_mask.nonzero(as_tuple=False).squeeze(-1)
    if avail.numel() == 0:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"delta": z, "pairwise": z, "keep": z, "specialist": z, "tp50": z, "total": z}

    # ── delta regression on available non-anchor slots ──────────────────
    alts = avail[avail != anchor_slot]
    if alts.numel() > 0:
        L_delta = F.smooth_l1_loss(delta_hat[alts], delta_true[alts])
    else:
        L_delta = torch.tensor(0.0, device=device)

    # ── pairwise margin ranking on true delta ──────────────────────────
    L_pair = torch.tensor(0.0, device=device)
    if alts.numel() > 1:
        n_pairs = 0
        loss_acc = torch.tensor(0.0, device=device)
        for i in range(alts.numel()):
            for j in range(i + 1, alts.numel()):
                si = int(alts[i].item()); sj = int(alts[j].item())
                di = float(delta_true[si].item()); dj = float(delta_true[sj].item())
                if abs(di - dj) < 1e-6:
                    continue
                if di > dj:
                    target = torch.tensor(1.0, device=device)
                else:
                    target = torch.tensor(-1.0, device=device)
                margin = weights.pairwise_margin
                # margin ranking: max(0, margin - target * (logit_i - logit_j))
                diff = source_logits[si] - source_logits[sj]
                loss_acc = loss_acc + F.relu(margin - target * diff)
                n_pairs += 1
        if n_pairs > 0:
            L_pair = loss_acc / n_pairs

    # ── keep / override BCE ────────────────────────────────────────────
    L_keep = F.binary_cross_entropy_with_logits(
        keep_anchor_logit.unsqueeze(0), keep_target.unsqueeze(0).float(),
    )

    # ── specialist BCEs ────────────────────────────────────────────────
    L_spec = torch.tensor(0.0, device=device)
    if specialist_logits:
        spec_losses = []
        for name, lg in specialist_logits.items():
            tgt = specialist_true.get(name)
            if tgt is None:
                continue
            spec_losses.append(F.binary_cross_entropy_with_logits(
                lg.unsqueeze(0), tgt.unsqueeze(0).float()
            ))
        if spec_losses:
            L_spec = torch.stack(spec_losses).mean()
    # ── specialist *CE* over candidate sources for positive-override case
    if int(keep_target.item()) == 0 and best_alt_slot >= 0 and alts.numel() > 0:
        local_idx = (alts == best_alt_slot).nonzero(as_tuple=False)
        if local_idx.numel() > 0:
            local = int(local_idx[0].item())
            ce = F.cross_entropy(source_logits[alts].unsqueeze(0),
                                  torch.tensor([local], device=device))
            L_spec = L_spec + ce

    # ── TP50 BCE ───────────────────────────────────────────────────────
    L_tp = F.binary_cross_entropy_with_logits(tp50_hat[avail], tp50_true[avail])

    # ── per-cluster total with false-override penalty ──────────────────
    L_total = (weights.delta * L_delta
               + weights.pairwise * L_pair
               + weights.keep_override * L_keep
               + weights.specialist * L_spec
               + weights.tp50 * L_tp)

    # False-override penalty: if the model's argmax-over-alts is positive
    # by delta_hat but the true delta at that slot is non-positive, multiply
    # the per-cluster loss by `false_override_penalty`.
    if alts.numel() > 0:
        delta_hat_alts = delta_hat[alts]
        best_local = int(delta_hat_alts.argmax().item())
        best_alt_predicted = int(alts[best_local].item())
        pred_pos = float(delta_hat[best_alt_predicted].item()) > override_threshold
        true_at_pred = float(delta_true[best_alt_predicted].item())
        if pred_pos and true_at_pred <= 0.0:
            L_total = L_total * weights.false_override_penalty

    return {"delta": L_delta, "pairwise": L_pair, "keep": L_keep,
            "specialist": L_spec, "tp50": L_tp, "total": L_total}


def anchor_router_loss(
    out: Dict[str, Any],
    *,
    delta_true: torch.Tensor,       # [C, S]
    slot_avail: torch.Tensor,        # [C, S] bool
    tp50_true: torch.Tensor,         # [C, S]
    anchor_slot_per_cluster: torch.Tensor,   # [C] long
    best_alt_slot_per_cluster: torch.Tensor,  # [C] long (-1 if no positive alt)
    specialist_true: Dict[str, torch.Tensor],  # name → [C] 0/1
    weights: Optional[AnchorLossWeights] = None,
    override_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Batched anchor-router loss over all clusters."""
    weights = weights or AnchorLossWeights()
    device = out["delta_ap50_hat"].device
    delta_hat = out["delta_ap50_hat"]
    tp50_hat = out["tp50_hat"]
    src_log = out["source_logits"]
    keep_log = out["keep_anchor_logit"]
    spec_log = out.get("specialist_logits", {}) or {}
    C, S = delta_hat.shape
    if C == 0:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"delta": z, "pairwise": z, "keep": z, "specialist": z, "tp50": z, "total": z, "n_valid": 0}

    delta_true = delta_true.to(device).float()
    tp50_true = tp50_true.to(device).float()
    anc = anchor_slot_per_cluster.to(device).long()
    best_alt = best_alt_slot_per_cluster.to(device).long()
    avail = slot_avail.to(device).bool()
    # tp50_hat had -inf at absent slots; replace with 0 to avoid BCE issues.
    tp50_hat_safe = torch.where(avail, tp50_hat, torch.zeros_like(tp50_hat))

    per_cluster = []
    accum = {"delta": 0.0, "pairwise": 0.0, "keep": 0.0, "specialist": 0.0, "tp50": 0.0}
    for c in range(C):
        # keep_target = 1 if anchor is the oracle slot.
        anc_c = int(anc[c].item())
        if not bool(avail[c, anc_c].item()):
            continue
        if best_alt[c].item() < 0:
            keep_target = torch.tensor(1.0, device=device)
        else:
            anchor_util = float(delta_true[c, anc_c].item())   # 0 by construction
            best_util = float(delta_true[c, int(best_alt[c].item())].item())
            keep_target = torch.tensor(1.0 if best_util <= 0.0 else 0.0, device=device)

        spec_c_log = {name: spec_log[name][c] for name in spec_log}
        spec_c_true = {name: specialist_true[name][c]
                       for name in specialist_true if name in spec_log}
        losses = _per_cluster_anchor_loss(
            delta_hat=delta_hat[c],
            delta_true=delta_true[c],
            slot_mask=avail[c],
            anchor_slot=anc_c,
            keep_anchor_logit=keep_log[c],
            keep_target=keep_target,
            tp50_hat=tp50_hat_safe[c],
            tp50_true=tp50_true[c],
            source_logits=src_log[c],
            best_alt_slot=int(best_alt[c].item()),
            specialist_logits=spec_c_log,
            specialist_true=spec_c_true,
            weights=weights,
            override_threshold=override_threshold,
        )
        per_cluster.append(losses["total"])
        for k in accum:
            accum[k] += float(losses[k].item())

    if not per_cluster:
        z = torch.tensor(0.0, device=device, requires_grad=True)
        return {"delta": z, "pairwise": z, "keep": z, "specialist": z, "tp50": z, "total": z, "n_valid": 0}

    total = torch.stack(per_cluster).mean()
    n = len(per_cluster)
    return {
        "delta": torch.tensor(accum["delta"] / n, device=device),
        "pairwise": torch.tensor(accum["pairwise"] / n, device=device),
        "keep": torch.tensor(accum["keep"] / n, device=device),
        "specialist": torch.tensor(accum["specialist"] / n, device=device),
        "tp50": torch.tensor(accum["tp50"] / n, device=device),
        "total": total,
        "n_valid": n,
    }


# ── True-delta builder ────────────────────────────────────────────────


def build_anchor_targets(
    util_per_slot: torch.Tensor,    # [C, S]
    slot_avail: torch.Tensor,        # [C, S] bool
    anchor_slot: int,
    *,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """From per-slot oracle utilities, compute the training targets.

    Returns dict with:
      delta_true:      [C, S] (utility[s] - utility[anchor]); 0 at anchor, 0 where absent
      tp50_true:       [C, S] in [0, 1]: 1 if utility[s] >= 0.5, else 0; 0 where absent
      best_alt_slot:   [C] (-1 if no available alt or no positive-delta alt)
      anchor_slot_per_cluster: [C] (constant = anchor_slot for any cluster
                       whose anchor slot is available; -1 otherwise)
    """
    C, S = util_per_slot.shape
    anc = torch.full((C,), anchor_slot, dtype=torch.long)
    best_alt = torch.full((C,), -1, dtype=torch.long)
    delta_true = torch.zeros(C, S)
    tp50 = torch.zeros(C, S)
    for c in range(C):
        if not bool(slot_avail[c, anchor_slot].item()):
            anc[c] = -1
            continue
        anchor_u = float(util_per_slot[c, anchor_slot].item())
        best_delta = float("-inf"); best_s = -1
        for s in range(S):
            if not bool(slot_avail[c, s].item()):
                continue
            u = float(util_per_slot[c, s].item())
            d = u - anchor_u
            delta_true[c, s] = d
            tp50[c, s] = 1.0 if u >= 0.5 else 0.0
            if s != anchor_slot and d > best_delta:
                best_delta = d; best_s = s
        if best_s >= 0 and best_delta > margin:
            best_alt[c] = best_s
    return {
        "delta_true": delta_true,
        "tp50_true": tp50,
        "best_alt_slot": best_alt,
        "anchor_slot_per_cluster": anc,
    }


def specialist_targets(
    util_per_slot: torch.Tensor,
    slot_avail: torch.Tensor,
    anchor_slot: int,
    specialist_slot_map: Dict[str, int],
    *,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Per-cluster {0,1} target for each specialist: 1 if that source beats anchor."""
    C = util_per_slot.shape[0]
    out: Dict[str, torch.Tensor] = {}
    for name, s in specialist_slot_map.items():
        t = torch.zeros(C)
        for c in range(C):
            if not bool(slot_avail[c, anchor_slot].item()) or not bool(slot_avail[c, s].item()):
                continue
            anchor_u = float(util_per_slot[c, anchor_slot].item())
            s_u = float(util_per_slot[c, s].item())
            if s_u > anchor_u + margin:
                t[c] = 1.0
        out[name] = t
    return out
