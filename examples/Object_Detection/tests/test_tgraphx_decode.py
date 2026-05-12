"""Regression tests for selector-mode decode.

Verifies that TGraphX in selector mode is lower-bounded by the best
candidate node: when ``keep_threshold=0.0`` and ``apply_box_regression=False``,
the returned box must equal a box that was on some candidate node
(proposal / cluster / consensus), never a hallucinated box.
"""
import torch

from od_graph_fusion.detectors.base import DetectionResult
from od_graph_fusion.graph_builder import build_detection_graph
from od_graph_fusion.fusion import fuse_with_model
from od_graph_fusion.models import DetectionFusionModel


def _make_dr(image_id, name, boxes, scores, labels, label_ids, image_size=(128, 128)):
    return DetectionResult(
        image_id=image_id, model_name=name,
        boxes_xyxy=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        label_ids=torch.tensor(label_ids, dtype=torch.long),
        labels=labels, image_size=image_size,
    )


def test_selector_picks_existing_box_not_hallucinated():
    image = torch.rand(3, 128, 128)
    det = [
        _make_dr("img", "d0", [[10, 10, 50, 50]], [0.9], ["car"], [0]),
        _make_dr("img", "d1", [[12, 11, 52, 49]], [0.8], ["car"], [0]),
    ]
    g, meta = build_detection_graph(
        image, "img", (128, 128), det,
        detector_names=["d0", "d1"], class_names=["car"],
        crop_size=32, max_proposals=8, iou_cluster=0.5,
        include_context_node=False, include_consensus_nodes=True,
        is_training=False,
    )
    # Build a tiny random model
    model = DetectionFusionModel(
        num_classes=1, num_detectors=2, crop_size=32,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    out = fuse_with_model(model, g, meta, keep_threshold=0.0,
                          device="cpu", fusion_mode="selector",
                          apply_box_regression=False)
    # In selector mode, the returned box must equal one of the candidate node
    # boxes for the only cluster.
    node_box = g.metadata["node_box"]
    candidate_boxes = node_box.tolist()
    for fb in out["boxes_xyxy"].tolist():
        assert fb in candidate_boxes, (
            f"Box {fb} not found in candidate node boxes — TGraphX selector "
            "should not hallucinate boxes."
        )


def test_selector_threshold_zero_returns_candidates():
    image = torch.rand(3, 64, 64)
    det = [_make_dr("img", "d0", [[5, 5, 25, 25]], [0.9], ["car"], [0])]
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=["d0"], class_names=["car"],
        crop_size=16, include_context_node=False,
        include_consensus_nodes=True, is_training=False,
    )
    model = DetectionFusionModel(
        num_classes=1, num_detectors=1, crop_size=16,
        crop_channels=8, hidden_dim=8, num_message_passing=1,
    )
    out = fuse_with_model(model, g, meta, keep_threshold=0.0,
                          device="cpu", fusion_mode="selector")
    # threshold=0.0 must always return at least one box when there is a cluster
    assert out["boxes_xyxy"].shape[0] >= 1


def test_selector_lower_bound_against_best_proposal():
    """In a graph with one cluster containing two proposals and no GT,
    the selector must pick one of the existing proposal/cluster/consensus
    boxes — not an arbitrary refined output."""
    image = torch.rand(3, 64, 64)
    det = [
        _make_dr("img", "d0", [[5, 5, 25, 25]], [0.9], ["car"], [0]),
        _make_dr("img", "d1", [[6, 6, 26, 26]], [0.85], ["car"], [0]),
    ]
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=["d0", "d1"], class_names=["car"],
        crop_size=16, iou_cluster=0.5,
        include_context_node=False, include_consensus_nodes=True,
        is_training=False,
    )
    model = DetectionFusionModel(
        num_classes=1, num_detectors=2, crop_size=16,
        crop_channels=8, hidden_dim=8, num_message_passing=1,
    )
    out = fuse_with_model(model, g, meta, keep_threshold=0.0,
                          device="cpu", fusion_mode="selector",
                          apply_box_regression=False)
    # The chosen box must coincide (within numerical tolerance) with one
    # of the candidate node boxes.
    node_box = g.metadata["node_box"]
    for fb in out["boxes_xyxy"]:
        diffs = (node_box - fb.unsqueeze(0)).abs().sum(dim=1)
        assert diffs.min().item() < 1e-4, (
            "Selector returned a box that does not match any candidate node."
        )
