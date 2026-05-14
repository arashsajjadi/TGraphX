"""Tests for the new AnchorRouter model + training objective.

These exercise the model in isolation (no real detectors needed). The full
training-loop pathway is tested separately in test_anchor_training.py.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_anchor_router_constructs_and_runs():
    """The router builds with sensible defaults and produces all expected keys
    when given a minimal synthetic graph from the V3 builder."""
    from od_graph_fusion.anchor_router import AnchorRouter, AnchorRouterConfig
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    cfg = AnchorRouterConfig(
        num_classes=1, num_detectors=3, crop_size=32,
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        hidden_dim=24, crop_channels=8, num_message_passing=1,
    )
    model = AnchorRouter(cfg).eval()
    # Build a degenerate but valid graph: 3 proposal nodes (one per detector),
    # one cluster node, one consensus node. Edge index just connects them.
    from tgraphx import Graph
    from od_graph_fusion.graph_builder import NODE_TYPES
    N = 5
    node_types = torch.tensor([
        NODE_TYPES["proposal"], NODE_TYPES["proposal"], NODE_TYPES["proposal"],
        NODE_TYPES["cluster"], NODE_TYPES["consensus"],
    ], dtype=torch.long)
    pdet = torch.tensor([0, 1, 2, -1, -1], dtype=torch.long)
    cluster_of = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    node_box = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 52.0, 52.0],
        [8.0, 8.0, 48.0, 48.0],
        [10.0, 10.0, 50.0, 50.0],
        [9.0, 9.0, 51.0, 51.0],
    ])
    node_score = torch.tensor([0.5, 0.6, 0.7, 0.65, 0.55])
    node_label = torch.zeros(N, dtype=torch.long)
    crops = torch.randn(N, 3, 32, 32)
    ei = torch.tensor([[0, 1, 2, 3], [3, 3, 3, 4]], dtype=torch.long)
    ea = torch.zeros(ei.shape[1], 14)
    md = torch.zeros(N, 8 + 3 + 1)
    g = Graph(node_features=crops, edge_index=ei, edge_features=ea)
    g.metadata = {
        "node_metadata": md,
        "node_types": node_types,
        "cluster_of_raw": cluster_of,
        "proposal_det_ids": pdet,
        "node_box": node_box,
        "node_score": node_score,
        "node_label": node_label,
        "detector_names": ["retinanet", "yolo_modern", "rt_detr"],
    }
    out = model(g, detector_names=["retinanet", "yolo_modern", "rt_detr"])
    assert out["delta_ap50_hat"].shape[0] == 1
    assert out["source_mask"].shape == (1, cfg.num_sources)
    assert "keep_anchor_logit" in out
    assert "tp50_hat" in out
    for name in ("union", "yolo_modern", "rt_detr", "retinanet"):
        assert name in out["specialist_logits"]
        assert out["specialist_logits"][name].shape[0] == 1


def test_anchor_router_decide_prefers_anchor_when_deltas_small():
    """When all predicted deltas are at or below threshold, decide() must
    return the anchor — that's the guarded-improvement contract."""
    from od_graph_fusion.anchor_router import AnchorRouter, AnchorRouterConfig
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    cfg = AnchorRouterConfig(
        num_classes=1, num_detectors=3, crop_size=8,
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        hidden_dim=8, crop_channels=4, num_message_passing=1,
    )
    model = AnchorRouter(cfg).eval()
    S = cfg.num_sources
    mask = torch.ones(2, S, dtype=torch.bool)
    out = {
        "delta_ap50_hat": torch.zeros(2, S),
        "source_mask": mask,
        "anchor_slot": cfg.anchor_slot,
        "specialist_logits": {},
    }
    chosen, kept = model.decide(out, override_threshold=0.01)
    assert (chosen == cfg.anchor_slot).all()
    assert kept.all()


def test_anchor_router_decide_overrides_when_alt_is_high():
    """If one non-anchor slot has a clearly positive predicted delta, the
    router should switch to it — but only when the value exceeds threshold."""
    from od_graph_fusion.anchor_router import AnchorRouter, AnchorRouterConfig
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    cfg = AnchorRouterConfig(
        num_classes=1, num_detectors=3, crop_size=8,
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        hidden_dim=8, crop_channels=4, num_message_passing=1,
    )
    model = AnchorRouter(cfg).eval()
    S = cfg.num_sources
    mask = torch.ones(1, S, dtype=torch.bool)
    delta = torch.zeros(1, S)
    union_slot = SOURCE_SLOTS["union"]
    delta[0, union_slot] = 0.10   # clearly above threshold
    out = {
        "delta_ap50_hat": delta,
        "source_mask": mask,
        "anchor_slot": cfg.anchor_slot,
        # specialist for union has positive prob → not blocked
        "specialist_logits": {"union": torch.tensor([2.0])},
    }
    chosen, kept = model.decide(out, override_threshold=0.05)
    assert int(chosen.item()) == union_slot
    assert not bool(kept.item())


def test_anchor_router_specialist_gate_blocks_low_prob_override():
    """Even if delta is positive, if the specialist head says P(source>anchor)
    < 0.5 the override must be blocked. This is the second gate against
    false overrides."""
    from od_graph_fusion.anchor_router import AnchorRouter, AnchorRouterConfig
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    cfg = AnchorRouterConfig(
        num_classes=1, num_detectors=3, crop_size=8,
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        hidden_dim=8, crop_channels=4, num_message_passing=1,
    )
    model = AnchorRouter(cfg).eval()
    S = cfg.num_sources
    mask = torch.ones(1, S, dtype=torch.bool)
    delta = torch.zeros(1, S)
    delta[0, SOURCE_SLOTS["union"]] = 0.10
    out = {
        "delta_ap50_hat": delta,
        "source_mask": mask,
        "anchor_slot": cfg.anchor_slot,
        "specialist_logits": {"union": torch.tensor([-5.0])},   # P ≈ 0
    }
    chosen, kept = model.decide(out, override_threshold=0.05)
    assert int(chosen.item()) == cfg.anchor_slot
    assert bool(kept.item())
