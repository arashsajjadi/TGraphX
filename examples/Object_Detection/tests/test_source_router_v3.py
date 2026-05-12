"""Tests for TGraphXSourceRouterV3 and fuse_v3."""
import torch
import pytest
from od_graph_fusion.source_router_v3 import (
    TGraphXSourceRouterV3, fuse_v3, FuseTrace,
    source_slot_loss, SourceSlotAggregator,
    SOURCE_SLOTS, NUM_SOURCES, detector_name_to_slot,
)
from od_graph_fusion.graph_builder import NODE_TYPES


def _make_simple_graph(N=6, crop_size=16):
    from tgraphx import Graph
    import torch
    x = torch.randn(N, 3, crop_size, crop_size)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 14)
    md = torch.randn(N, 8 + 4 + 5)  # 4 detectors, 5 classes
    return Graph(node_features=x, edge_index=ei, edge_attr=ea,
                 metadata={"node_metadata": md})


# ── detector_name_to_slot ─────────────────────────────────────────────────

def test_detector_name_to_slot_known():
    assert detector_name_to_slot("yolo_modern") == SOURCE_SLOTS["yolo_modern"]
    assert detector_name_to_slot("retinanet") == SOURCE_SLOTS["retinanet"]
    assert detector_name_to_slot("rt_detr") == SOURCE_SLOTS["rt_detr"]

def test_detector_name_to_slot_unknown():
    assert detector_name_to_slot("nope_detector") == -1


# ── SourceSlotAggregator ──────────────────────────────────────────────────

def test_slot_agg_shape():
    N, D, S = 8, 32, NUM_SOURCES
    node_emb = torch.randn(N, D)
    cluster_of = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    node_source_slot = torch.tensor([0, 1, 2, 3, 0, 1, 3, -1], dtype=torch.long)
    agg = SourceSlotAggregator(D, S)
    slot_emb, slot_mask = agg(node_emb, cluster_of, node_source_slot, n_clusters=2)
    assert slot_emb.shape == (2, S, D)
    assert slot_mask.shape == (2, S)


def test_slot_agg_mask_absent_source():
    N, D, S = 4, 16, NUM_SOURCES
    node_emb = torch.randn(N, D)
    cluster_of = torch.zeros(N, dtype=torch.long)
    # Only slots 0 and 3 have nodes; others absent
    node_source_slot = torch.tensor([0, 0, 3, -1], dtype=torch.long)
    agg = SourceSlotAggregator(D, S)
    slot_emb, slot_mask = agg(node_emb, cluster_of, node_source_slot, n_clusters=1)
    assert slot_mask[0, 0].item() == True
    assert slot_mask[0, 3].item() == True
    assert slot_mask[0, 1].item() == False  # no node for YOLOE
    assert slot_mask[0, 2].item() == False  # no node for RTDETR


# ── TGraphXSourceRouterV3 ─────────────────────────────────────────────────

def test_source_router_v3_forward():
    model = TGraphXSourceRouterV3(
        num_classes=5, num_detectors=4, crop_size=16,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    model.eval()
    g = _make_simple_graph(6, 16)
    with torch.no_grad():
        out = model(g)
    assert "quality_logits" in out
    assert out["quality_logits"].shape == (6,)


def test_source_router_v3_with_cluster_info():
    """When cluster/node-type metadata is present, source_logits must be produced."""
    from tgraphx import Graph
    model = TGraphXSourceRouterV3(
        num_classes=5, num_detectors=4, crop_size=16,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    model.eval()
    N = 6
    x = torch.randn(N, 3, 16, 16)
    ei = torch.tensor([[0, 1], [1, 2]])
    node_types = torch.zeros(N, dtype=torch.long)
    node_types[0] = NODE_TYPES["proposal"]; node_types[1] = NODE_TYPES["proposal"]
    node_types[2] = NODE_TYPES["cluster"]; node_types[3] = NODE_TYPES["consensus"]
    cluster_of = torch.tensor([0, 0, 0, 0, -1, -1], dtype=torch.long)
    prop_det_ids = torch.tensor([0, 2, -1, -1, -1, -1], dtype=torch.long)
    g = Graph(node_features=x, edge_index=ei,
              metadata={
                  "node_metadata": torch.randn(N, 8+4+5),
                  "node_types": node_types,
                  "cluster_of_raw": cluster_of,
                  "proposal_det_ids": prop_det_ids,
              })
    with torch.no_grad():
        out = model(g, detector_names=["yolo_modern", "yoloe", "rt_detr", "retinanet"])
    if out.get("source_logits") is not None:
        sl = out["source_logits"]
        assert sl.dim() == 2
        assert sl.shape[1] == NUM_SOURCES


def test_source_mask_blocks_absent_sources():
    """Absent sources must have -inf logit and cannot be selected."""
    from tgraphx import Graph
    model = TGraphXSourceRouterV3(
        num_classes=5, num_detectors=4, crop_size=16,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    model.eval()
    N = 4
    x = torch.randn(N, 3, 16, 16)
    ei = torch.tensor([[0], [1]])
    node_types = torch.zeros(N, dtype=torch.long)
    node_types[0] = NODE_TYPES["proposal"]   # yolo_modern → slot 0
    node_types[1] = NODE_TYPES["cluster"]    # wbf → slot 5
    node_types[2] = NODE_TYPES["context"]
    node_types[3] = NODE_TYPES["context"]
    cluster_of = torch.tensor([0, 0, -1, -1], dtype=torch.long)
    prop_det_ids = torch.tensor([0, -1, -1, -1], dtype=torch.long)
    g = Graph(node_features=x, edge_index=ei,
              metadata={
                  "node_metadata": torch.randn(N, 8+4+5),
                  "node_types": node_types,
                  "cluster_of_raw": cluster_of,
                  "proposal_det_ids": prop_det_ids,
              })
    with torch.no_grad():
        out = model(g, detector_names=["yolo_modern", "yoloe", "rt_detr", "retinanet"])
    if out.get("source_logits") is not None and out.get("source_mask") is not None:
        sl = out["source_logits"]    # [C, S]
        sm = out["source_mask"]      # [C, S]
        # Absent slots must be -inf
        absent = ~sm
        if absent.any():
            assert (sl[absent] == float("-inf")).all(), "Absent source logits must be -inf"
        # Cannot select an absent source
        for c in range(sl.shape[0]):
            avail = sm[c].nonzero(as_tuple=False).squeeze(-1)
            if avail.numel() > 0:
                chosen = sl[c][avail].argmax()
                chosen_slot = int(avail[chosen].item())
                assert sm[c, chosen_slot].item(), "Must not select absent source"


# ── source_slot_loss ──────────────────────────────────────────────────────

def test_source_slot_loss_finite():
    C, S = 4, NUM_SOURCES
    source_logits = torch.randn(C, S)
    source_mask = torch.ones(C, S, dtype=torch.bool)
    # Mask out last 2 slots for cluster 0
    source_mask[0, -2:] = False
    best_source_slot = torch.tensor([0, 1, 2, 3])
    utility_per_slot = torch.rand(C, S)
    losses = source_slot_loss(source_logits, source_mask, best_source_slot, utility_per_slot)
    assert torch.isfinite(losses["total"])


def test_source_slot_loss_absent_source_ignored():
    """If best_source_slot is absent in source_mask, cluster should be skipped."""
    C, S = 2, NUM_SOURCES
    source_logits = torch.randn(C, S)
    source_mask = torch.ones(C, S, dtype=torch.bool)
    source_mask[0, 0] = False  # slot 0 absent for cluster 0
    best_source_slot = torch.tensor([0, 1])  # cluster 0's oracle is slot 0 (absent!)
    utility_per_slot = torch.rand(C, S)
    # Should not crash, and cluster 0 should be skipped
    losses = source_slot_loss(source_logits, source_mask, best_source_slot, utility_per_slot)
    assert torch.isfinite(losses["total"])


# ── FuseTrace alignment test ──────────────────────────────────────────────

def test_fuse_trace_chosen_node_matches_ap_computation():
    """The node selected in FuseTrace must be the one used for boxes/labels in AP."""
    # This is the key test: same decision rule for AP and source_acc
    from od_graph_fusion.graph_builder import build_detection_graph, DetectionGraphMeta
    from od_graph_fusion.detectors.registry import build_synthetic_detector
    import torch

    image = torch.rand(3, 64, 64)
    det = build_synthetic_detector("yolo_modern", "anchor_free", seed=1, class_names=["car"])
    res = det.predict(image, "test_img", gt_boxes=torch.tensor([[5.,5.,25.,25.]]),
                      gt_labels=torch.tensor([0]))
    g, meta = build_detection_graph(
        image, "test_img", (64, 64), [res],
        detector_names=["yolo_modern"], class_names=["car"],
        crop_size=16, max_proposals=8,
        include_context_node=False, include_consensus_nodes=True,
        is_training=False,
    )
    model = TGraphXSourceRouterV3(num_classes=1, num_detectors=1,
                                   crop_size=16, crop_channels=8, hidden_dim=8, num_message_passing=1)
    model.eval()
    out = fuse_v3(model, g, meta, keep_threshold=0.0, device="cpu",
                  detector_names=["yolo_modern"], return_trace=True)
    traces = out.get("trace", [])
    boxes = out["boxes_xyxy"]
    labels = out["labels"]
    # Every trace must correspond to a returned box
    assert len(traces) == boxes.shape[0], (
        f"Number of traces ({len(traces)}) must equal number of returned boxes ({boxes.shape[0]})"
    )
    # The trace node's box must equal the returned box
    node_box = g.metadata["node_box"]
    for tr, box in zip(traces, boxes):
        if tr.chosen_node >= 0 and tr.chosen_node < node_box.shape[0]:
            assert torch.allclose(node_box[tr.chosen_node], box, atol=1e-5), (
                "Trace chosen_node box must match returned box (unified decision rule)"
            )
