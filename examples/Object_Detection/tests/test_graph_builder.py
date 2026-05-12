import torch

from od_graph_fusion.detectors.base import DetectionResult
from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES


def _make_dr(image_id: str, name: str, boxes, scores, labels, label_ids,
              image_size=(128, 128)):
    return DetectionResult(
        image_id=image_id, model_name=name,
        boxes_xyxy=torch.tensor(boxes, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        label_ids=torch.tensor(label_ids, dtype=torch.long),
        labels=labels, image_size=image_size,
    )


def test_graph_basic_four_detectors():
    image = torch.rand(3, 128, 128)
    det = [
        _make_dr("img", "d0", [[10, 10, 50, 50]], [0.9], ["car"], [0]),
        _make_dr("img", "d1", [[12, 11, 52, 49]], [0.8], ["car"], [0]),
        _make_dr("img", "d2", [[80, 80, 120, 120]], [0.7], ["dog"], [1]),
        _make_dr("img", "d3", [[81, 79, 121, 121]], [0.6], ["dog"], [1]),
    ]
    g, meta = build_detection_graph(
        image, "img", (128, 128), det,
        detector_names=["d0", "d1", "d2", "d3"],
        class_names=["car", "dog"],
        crop_size=32, max_proposals=8, iou_cluster=0.5,
        include_context_node=True, include_consensus_nodes=True,
        is_training=False,
    )
    # 4 proposals + 2 clusters + 2 consensus + 2 NMS nodes + 1 context = 11 nodes
    assert g.num_nodes == 11
    assert meta.num_proposals == 4
    assert meta.num_clusters == 2
    assert meta.num_consensus == 2
    assert meta.has_context
    # Tensor crops preserve [3, 32, 32]
    assert g.node_features.shape == (11, 3, 32, 32)
    # Edges exist
    assert g.edge_index.shape[1] > 0


def test_graph_empty_detectors():
    image = torch.rand(3, 64, 64)
    det = []
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=[], class_names=["car"],
        crop_size=16, include_context_node=True,
    )
    assert g.num_nodes >= 1  # context node only
    assert meta.num_proposals == 0
    assert meta.num_clusters == 0


def test_graph_single_detector():
    image = torch.rand(3, 64, 64)
    det = [_make_dr("img", "d0", [[5, 5, 25, 25]], [0.9], ["car"], [0])]
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=["d0"], class_names=["car"],
        crop_size=16, include_context_node=False, include_consensus_nodes=False,
    )
    # 1 proposal + 1 cluster
    assert meta.num_proposals == 1
    assert meta.num_clusters == 1


def test_graph_training_target_when_gt_provided():
    image = torch.rand(3, 64, 64)
    det = [_make_dr("img", "d0", [[5, 5, 25, 25]], [0.9], ["car"], [0])]
    gt = torch.tensor([[5., 5., 25., 25.]])
    gt_l = torch.tensor([0])
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=["d0"], class_names=["car"],
        gt_boxes=gt, gt_labels=gt_l,
        crop_size=16, is_training=True,
    )
    assert meta.targets is not None
    # cluster node should be objectness=1.0
    cluster_idx = (meta.node_types == NODE_TYPES["cluster"]).nonzero().squeeze(-1)
    assert meta.targets["objectness"][cluster_idx].sum().item() > 0


def test_graph_test_split_has_no_targets():
    """Leakage policy: when is_training=False, targets should be None."""
    image = torch.rand(3, 64, 64)
    det = [_make_dr("img", "d0", [[5, 5, 25, 25]], [0.9], ["car"], [0])]
    gt = torch.tensor([[5., 5., 25., 25.]])
    g, meta = build_detection_graph(
        image, "img", (64, 64), det,
        detector_names=["d0"], class_names=["car"],
        gt_boxes=gt, gt_labels=torch.tensor([0]),
        crop_size=16, is_training=False,
    )
    assert meta.targets is None
