"""
Tests that TGraphXSourceRouterV3.forward correctly maps ALL candidate node types
to source slots inside the model (not just patched post-hoc in fuse_v3).
"""
import torch
import pytest
from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES
from od_graph_fusion.detectors.registry import build_synthetic_detector
from od_graph_fusion.datasets import SYNTHETIC_CLASS_NAMES
from od_graph_fusion.source_router_v3 import (
    TGraphXSourceRouterV3, SOURCE_SLOTS, NUM_SOURCES
)
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata

DET_NAMES = ["yolo_modern", "yolo_open_vocab", "rt_detr", "retinanet"]
CLASS_NAMES = SYNTHETIC_CLASS_NAMES[:4]


def _make_graph(seed=0):
    dets = [build_synthetic_detector(n, "f", seed=seed+i, class_names=CLASS_NAMES, jitter=0.04+i*0.12)
            for i, n in enumerate(DET_NAMES)]
    [d.load() for d in dets]
    torch.manual_seed(seed)
    img = torch.rand(3, 32, 32)
    gt_b = torch.tensor([[3., 3., 22., 22.]])
    gt_l = torch.zeros(1, dtype=torch.long)
    res = [d.predict(img, "img0", gt_boxes=gt_b, gt_labels=gt_l) for d in dets]
    g, meta = build_detection_graph(
        img, "img0", (32, 32), res, DET_NAMES, CLASS_NAMES,
        gt_boxes=gt_b, gt_labels=gt_l,
        crop_size=8, max_proposals=6, include_context_node=False,
        include_consensus_nodes=True, is_training=False)
    _attach_slot_metadata(g, meta, DET_NAMES)
    return g, meta


def test_forward_maps_nms_candidate_slot():
    """nms_candidate nodes must appear in source_mask inside forward."""
    g, meta = _make_graph()
    assert NODE_TYPES.get("nms_candidate", -1) in meta.node_types.tolist(), (
        "Graph must contain nms_candidate nodes")
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]  # [C, S]
    nms_slot = SOURCE_SLOTS.get("nms_candidate", 6)
    assert src_mask.shape[1] == NUM_SOURCES
    # At least one cluster should have nms_candidate slot available
    assert src_mask[:, nms_slot].any(), (
        f"nms_candidate slot ({nms_slot}) is never available in source_mask. "
        "_build_node_source_slots is missing nms_candidate mapping."
    )


def test_forward_maps_soft_nms_slot():
    """soft_nms_candidate nodes must appear in source_mask inside forward."""
    g, meta = _make_graph(seed=1)
    assert NODE_TYPES.get("soft_nms_candidate", -1) in meta.node_types.tolist()
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]
    soft_slot = SOURCE_SLOTS.get("soft_nms", 7)
    assert src_mask[:, soft_slot].any(), (
        f"soft_nms slot ({soft_slot}) is never available. "
        "_build_node_source_slots is missing soft_nms_candidate mapping."
    )


def test_forward_maps_best_proposal_slot():
    """best_proposal_candidate nodes must appear in source_mask inside forward."""
    g, meta = _make_graph(seed=2)
    assert NODE_TYPES.get("best_proposal_candidate", -1) in meta.node_types.tolist()
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]
    bp_slot = SOURCE_SLOTS.get("best_proposal", 8)
    assert src_mask[:, bp_slot].any(), (
        f"best_proposal slot ({bp_slot}) is never available. "
        "_build_node_source_slots is missing best_proposal_candidate mapping."
    )


def test_forward_maps_all_detector_slots():
    """All 4 detector proposal slots (0-3) must be available per cluster."""
    g, meta = _make_graph(seed=0)
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]
    for det_name, expected_slot in [
        ("yolo_modern", 0), ("yolo_open_vocab", 1), ("rt_detr", 2), ("retinanet", 3)
    ]:
        assert src_mask[:, expected_slot].any(), (
            f"Detector slot {expected_slot} ({det_name}) not available in source_mask"
        )


def test_forward_wbf_union_slots():
    """WBF (cluster) and Union (consensus) must be available."""
    g, meta = _make_graph()
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]
    assert src_mask[:, SOURCE_SLOTS["wbf"]].any(), "WBF slot not available"
    assert src_mask[:, SOURCE_SLOTS["union"]].any(), "Union slot not available"


def test_all_9_source_slots_covered():
    """All 9 non-calibrated-consensus slots must be covered in at least one cluster."""
    g, meta = _make_graph(seed=3)
    model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                   crop_channels=4, hidden_dim=32, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=DET_NAMES)
    src_mask = out["source_mask"]  # [C, S]
    expected = {
        "yolo_modern": 0, "yolo_open_vocab": 1, "rt_detr": 2, "retinanet": 3,
        "union": 4, "wbf": 5, "nms_candidate": 6, "soft_nms": 7, "best_proposal": 8,
    }
    missing = []
    for name, slot in expected.items():
        if not src_mask[:, slot].any():
            missing.append(f"{name}(slot={slot})")
    assert not missing, (
        f"These source slots are never available in forward source_mask: {missing}. "
        "_build_node_source_slots is missing mappings."
    )
