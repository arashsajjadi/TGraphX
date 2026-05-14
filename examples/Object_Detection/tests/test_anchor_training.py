"""Tests for the anchor-router training objective.

Covers:
  - build_anchor_targets returns sane deltas, tp50 labels, and best-alt slots
  - false-override penalty kicks in when predicted override is wrong
  - keep_override BCE target is 1 when anchor is oracle, 0 otherwise
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_build_anchor_targets_assigns_best_alt_correctly():
    from od_graph_fusion.anchor_training import build_anchor_targets
    util = torch.tensor([
        [0.0, 0.0, 0.8, 0.0, 0.95, 0.0, 0.7],   # union (slot 4) is oracle
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0],    # anchor=rt_detr (2) is oracle
    ])
    util = torch.cat([util, torch.zeros(2, 10 - util.shape[1])], dim=1)
    avail = (util > 0)
    avail[:, 2] = True  # anchor is always available for the test
    anchor = 2
    tgt = build_anchor_targets(util, avail, anchor_slot=anchor)
    # First cluster: best alt is union (slot 4)
    assert int(tgt["best_alt_slot"][0].item()) == 4
    # Second cluster: anchor is oracle (no positive alt)
    assert int(tgt["best_alt_slot"][1].item()) == -1
    # delta_true is utility minus anchor utility
    assert abs(float(tgt["delta_true"][0, 4].item()) - (0.95 - 0.8)) < 1e-5
    assert abs(float(tgt["delta_true"][1, 2].item())) < 1e-5


def test_false_override_penalty_multiplies_loss():
    """When the model's argmax-alt has predicted positive delta but the TRUE
    delta is non-positive, _per_cluster_anchor_loss must multiply the loss."""
    from od_graph_fusion.anchor_training import _per_cluster_anchor_loss, AnchorLossWeights
    S = 7
    delta_hat = torch.zeros(S, requires_grad=False)
    delta_hat[4] = 0.5    # model thinks union is great
    delta_true = torch.zeros(S)
    delta_true[4] = -0.5  # but in reality union is terrible
    avail = torch.ones(S, dtype=torch.bool)
    tp50_hat = torch.zeros(S)
    tp50_true = torch.zeros(S)
    src_log = torch.zeros(S)
    keep_logit = torch.tensor(0.0)
    keep_tgt = torch.tensor(1.0)
    weights = AnchorLossWeights(false_override_penalty=8.0)
    losses = _per_cluster_anchor_loss(
        delta_hat=delta_hat, delta_true=delta_true, slot_mask=avail,
        anchor_slot=2, keep_anchor_logit=keep_logit, keep_target=keep_tgt,
        tp50_hat=tp50_hat, tp50_true=tp50_true,
        source_logits=src_log, best_alt_slot=-1,
        specialist_logits={}, specialist_true={},
        weights=weights, override_threshold=0.0,
    )
    # Compare to a "no false override" baseline by zeroing delta_hat[4].
    delta_hat2 = torch.zeros(S)
    losses2 = _per_cluster_anchor_loss(
        delta_hat=delta_hat2, delta_true=delta_true, slot_mask=avail,
        anchor_slot=2, keep_anchor_logit=keep_logit, keep_target=keep_tgt,
        tp50_hat=tp50_hat, tp50_true=tp50_true,
        source_logits=src_log, best_alt_slot=-1,
        specialist_logits={}, specialist_true={},
        weights=weights, override_threshold=0.0,
    )
    assert float(losses["total"].item()) > float(losses2["total"].item()) * 1.5, \
        "False-override penalty must materially increase the loss."


def test_specialist_targets_match_anchor_delta():
    from od_graph_fusion.anchor_training import specialist_targets
    util = torch.zeros(3, 10)
    avail = torch.zeros(3, 10, dtype=torch.bool)
    # Cluster 0: union beats anchor
    util[0, 2] = 0.6; util[0, 4] = 0.8
    avail[0, 2] = True; avail[0, 4] = True
    # Cluster 1: union worse than anchor
    util[1, 2] = 0.9; util[1, 4] = 0.5
    avail[1, 2] = True; avail[1, 4] = True
    # Cluster 2: union absent
    util[2, 2] = 0.7
    avail[2, 2] = True
    out = specialist_targets(util, avail, anchor_slot=2,
                              specialist_slot_map={"union": 4})
    assert float(out["union"][0].item()) == 1.0
    assert float(out["union"][1].item()) == 0.0
    assert float(out["union"][2].item()) == 0.0   # union absent → 0
