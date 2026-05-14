"""Tests for hard-case classification, descriptor builder, and sampler."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def test_classify_hard_case_union_oracle_not_selected():
    from od_graph_fusion.hard_cases import classify_hard_case
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    tags = classify_hard_case(
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        oracle_slot=SOURCE_SLOTS["union"],
        slot_avail=[True] * 10,
        slot_util=[0.0] * 10,
        anchor_util=0.3, oracle_util=0.95,
        anchor_score=0.5, iou_disagreement=0.5,
    )
    assert "A_union_oracle_not_selected" in tags
    assert "C_anchor_fails_alt_succeeds" in tags
    assert "G_high_iou_disagreement" in tags
    assert "H_aggregate_improves_localization" in tags


def test_classify_hard_case_yolo_oracle_anchor_picks_rtdetr():
    from od_graph_fusion.hard_cases import classify_hard_case
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    tags = classify_hard_case(
        anchor_slot=SOURCE_SLOTS["rt_detr"],
        oracle_slot=SOURCE_SLOTS["yolo_modern"],
        slot_avail=[True] * 10,
        slot_util=[0.0] * 10,
        anchor_util=0.2, oracle_util=0.8,
        anchor_score=0.8, iou_disagreement=0.1,
    )
    assert "B_yolo_oracle_anchor_picks_rtdetr" in tags
    assert "D_anchor_tp50_zero_alt_tp50_one" in tags


def test_hard_case_sampler_yields_mix_close_to_target():
    """Sampler should respect the configured mix (within rounding)."""
    from od_graph_fusion.hard_cases import HardCaseSampler, ClusterDescriptor
    descs = []
    for i in range(40):
        tags = []
        if i < 5:
            tags = ["A_union_oracle_not_selected"]
        elif i < 10:
            tags = ["B_yolo_oracle_anchor_picks_rtdetr"]
        elif i < 15:
            tags = ["C_anchor_fails_alt_succeeds"]
        descs.append(ClusterDescriptor(
            graph_idx=0, cluster_id=i, anchor_slot=2,
            oracle_slot=4, anchor_util=0.3, oracle_util=0.9,
            slot_avail=[True] * 10, slot_util=[0.0] * 10,
            anchor_score=0.5, iou_disagreement=0.2,
            hard_case_types=tags,
        ))
    sampler = HardCaseSampler(descs, batch_size=20, seed=0)
    batch = sampler.sample_batch()
    assert len(batch) == 20
    sizes = sampler.bucket_sizes()
    assert sizes["natural"] == 40
    assert sizes["A_union_oracle_not_selected"] == 5
    assert sizes["B_yolo_oracle_anchor_picks_rtdetr"] == 5
    assert sizes["C_anchor_fails_alt_succeeds"] == 5
