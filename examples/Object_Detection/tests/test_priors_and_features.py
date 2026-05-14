"""Tests for source priors, anchor selection, and pairwise features."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_select_anchor_picks_best_validation_method():
    from od_graph_fusion.source_priors import select_anchor_on_validation
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    val_ap = {
        "det::retinanet": 0.81,
        "det::rt_detr":   0.87,
        "fusion::nms":    0.85,
        "fusion::wbf":    0.86,
    }
    slot, label = select_anchor_on_validation(val_ap, detector_names=["retinanet","yolo_modern","rt_detr"])
    assert slot == SOURCE_SLOTS["rt_detr"]
    assert "rt_detr" in label


def test_compute_priors_returns_anchor_row_at_half():
    from od_graph_fusion.source_priors import compute_priors
    from od_graph_fusion.source_router_v3 import NUM_SOURCES, SOURCE_SLOTS
    anchor = SOURCE_SLOTS["rt_detr"]
    clusters = []
    for _ in range(20):
        u = torch.zeros(NUM_SOURCES)
        u[anchor] = 0.5
        u[SOURCE_SLOTS["union"]] = 0.9
        avail = torch.zeros(NUM_SOURCES, dtype=torch.bool)
        avail[anchor] = True
        avail[SOURCE_SLOTS["union"]] = True
        clusters.append({
            "slot_utility": u, "slot_avail": avail,
            "cluster_class": 0, "cluster_size_bin": 1, "cluster_score_bucket": 1,
        })
    p = compute_priors(clusters, anchor_slot=anchor, num_classes=1)
    # union should have very high prior (always beats anchor)
    assert float(p.global_prior[SOURCE_SLOTS["union"]].item()) > 0.8
    # anchor row is conventionally 0.5
    assert abs(float(p.global_prior[anchor].item()) - 0.5) < 1e-6


def test_pairwise_features_anchor_row_is_zero():
    from od_graph_fusion.pairwise_features import pairwise_features_for_cluster, PAIRWISE_FEAT_DIM
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    S = 10
    slot_node_idx = torch.full((1, S), -1, dtype=torch.long)
    avail = torch.zeros(1, S, dtype=torch.bool)
    # rt_detr is anchor, union is alt
    slot_node_idx[0, SOURCE_SLOTS["rt_detr"]] = 0
    slot_node_idx[0, SOURCE_SLOTS["union"]]   = 1
    avail[0, SOURCE_SLOTS["rt_detr"]] = True
    avail[0, SOURCE_SLOTS["union"]]   = True
    node_box = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [5.0, 5.0, 15.0, 15.0],
    ])
    node_score = torch.tensor([0.7, 0.4])
    node_label = torch.zeros(2, dtype=torch.long)
    feats = pairwise_features_for_cluster(
        0, slot_node_idx, avail, SOURCE_SLOTS["rt_detr"],
        node_box, node_score, node_label,
        n_proposals_in_cluster=2,
        detector_agreement_entropy=0.3, score_entropy=0.4,
        box_variance=0.2, proposal_max_iou=0.5,
    )
    assert feats.shape == (S, PAIRWISE_FEAT_DIM)
    # anchor row must be all zeros
    assert torch.all(feats[SOURCE_SLOTS["rt_detr"]] == 0)
    # union row should be non-trivial
    assert torch.any(feats[SOURCE_SLOTS["union"]] != 0)
