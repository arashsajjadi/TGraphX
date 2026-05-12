"""Tests for NMS as a real source node in the graph."""
import torch
from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES
from od_graph_fusion.detectors.registry import build_synthetic_detector
from od_graph_fusion.source_router_v3 import detector_name_to_slot, SOURCE_SLOTS


def _make_det(name, seed=1, n=3, class_names=None):
    if class_names is None: class_names = ["car"]
    d = build_synthetic_detector(name, "family", seed=seed, class_names=class_names)
    d.load(); return d


def test_nms_node_exists_in_graph():
    """Every cluster must have exactly one NMS node."""
    image = torch.rand(3, 64, 64)
    d0 = _make_det("yolo_modern"); d1 = _make_det("retinanet", seed=2)
    gt = torch.tensor([[5., 5., 25., 25.]]); gtl = torch.tensor([0])
    res = [d0.predict(image, "img", gt_boxes=gt, gt_labels=gtl),
           d1.predict(image, "img", gt_boxes=gt, gt_labels=gtl)]
    g, meta = build_detection_graph(
        image, "img", (64, 64), res, ["yolo_modern", "retinanet"], ["car"],
        crop_size=16, max_proposals=8, include_context_node=False, include_consensus_nodes=True,
        is_training=False,
    )
    nms_count = (meta.node_types == NODE_TYPES["nms_candidate"]).sum().item()
    assert nms_count == meta.num_clusters, (
        f"Expected {meta.num_clusters} NMS nodes, got {nms_count}"
    )


def test_nms_node_slot_assignment():
    """NMS nodes must map to SOURCE_SLOTS['nms_candidate'] (slot 6)."""
    image = torch.rand(3, 64, 64)
    d = _make_det("yolo_modern"); d.load()
    gt = torch.tensor([[5., 5., 25., 25.]]); gtl = torch.tensor([0])
    res = [d.predict(image, "img", gt_boxes=gt, gt_labels=gtl)]
    g, meta = build_detection_graph(
        image, "img", (64, 64), res, ["yolo_modern"], ["car"],
        crop_size=16, include_context_node=False, include_consensus_nodes=True, is_training=False,
    )
    slots = g.metadata.get("slot_assignments")
    nms_nodes = (meta.node_types == NODE_TYPES["nms_candidate"]).nonzero(as_tuple=False).squeeze(-1)
    if slots is not None and nms_nodes.numel() > 0:
        for ni in nms_nodes.tolist():
            s = int(slots[ni].item()) if ni < slots.shape[0] else -1
            assert s == SOURCE_SLOTS.get("nms_candidate", 6), (
                f"NMS node {ni} has slot {s}, expected {SOURCE_SLOTS.get('nms_candidate', 6)}"
            )


def test_nms_node_box_equals_highest_score_proposal():
    """NMS node box must equal the highest-confidence proposal in its cluster."""
    image = torch.rand(3, 64, 64)
    d = _make_det("yolo_modern", seed=1, n=3)
    gt = torch.tensor([[5., 5., 25., 25.]]); gtl = torch.tensor([0])
    res = [d.predict(image, "img", gt_boxes=gt, gt_labels=gtl)]
    g, meta = build_detection_graph(
        image, "img", (64, 64), res, ["yolo_modern"], ["car"],
        crop_size=16, include_context_node=False, include_consensus_nodes=False, is_training=False,
    )
    node_box = g.metadata["node_box"]
    node_score = g.metadata["node_score"]
    nms_nodes = (meta.node_types == NODE_TYPES["nms_candidate"]).nonzero(as_tuple=False).squeeze(-1)
    if nms_nodes.numel() == 0: return  # no clusters → skip

    cluster_of = meta.cluster_of_node
    for ni in nms_nodes.tolist():
        c = int(cluster_of[ni].item())
        cand_mask = ((meta.node_types == NODE_TYPES["proposal"]) & (cluster_of == c))
        if not cand_mask.any(): continue
        idx_c = cand_mask.nonzero(as_tuple=False).squeeze(-1)
        best_prop = int(idx_c[node_score[idx_c].argmax()].item())
        assert torch.allclose(node_box[ni], node_box[best_prop], atol=1e-5), (
            f"NMS node box {node_box[ni]} != highest-score proposal box {node_box[best_prop]}"
        )


def test_slot_aliases_all_resolve():
    aliases = {
        "yoloe": 1, "yolo_open_vocab": 1, "rtdetr": 2, "rt_detr": 2,
        "retinanet": 3, "retina": 3, "nms_candidate": 6, "nms": 6,
        "best_proposal": 6, "yolo_modern": 0,
    }
    for name, expected in aliases.items():
        got = detector_name_to_slot(name)
        assert got == expected, f"detector_name_to_slot({name!r}) = {got}, expected {expected}"
