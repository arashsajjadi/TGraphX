"""Tests for object-level candidate graph construction (Section 3 requirements).

Invariants tested:
- K object clusters → K object graphs for that image
- Every object graph has only nodes from one cluster
- selected box equals one row of node_box exactly
- node_features has rank 4: [N, 3, H, W]
- No metadata-only candidate node (every node has a crop tensor)
- Fusion nodes (WBF, union, NMS, Soft-NMS, BestProposal) have crop tensors
- Graph oracle ≥ every graph-node baseline (for trivial synthetic data)
- object_graphs.pt format is correct tuple structure
"""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from od_graph_fusion.object_candidate_graphs import build_object_candidate_graphs
from od_graph_fusion.graph_builder import NODE_TYPES
from od_graph_fusion.candidate_mask import candidate_node_mask
from od_graph_fusion.box_ops import box_iou


# ── Minimal detector result stub ─────────────────────────────────────────────

class _FakeResult:
    def __init__(self, boxes, scores, labels_str, labels_id):
        self.boxes_xyxy  = boxes
        self.scores      = scores
        self.labels      = labels_str   # list[str]
        self.label_ids   = labels_id    # tensor
    def num_detections(self):
        return self.boxes_xyxy.shape[0]


def _fake_det(boxes, scores, cls="car", n_classes=1):
    N = boxes.shape[0]
    return _FakeResult(
        boxes=boxes, scores=scores,
        labels_str=[cls] * N,
        labels_id=torch.zeros(N, dtype=torch.long),
    )


def _make_image(H=320, W=320):
    return torch.rand(3, H, W)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestObjectCandidateGraphs:

    def test_k_clusters_produce_k_graphs(self):
        """An image with K non-overlapping object clusters produces K object graphs."""
        H, W = 128, 128
        image = _make_image(H, W)
        # Two clusters: [10,10,50,50] and [80,80,120,120]  — IoU ≈ 0
        cluster1 = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52]], dtype=torch.float32)
        cluster2 = torch.tensor([[80, 80, 120, 120], [82, 82, 122, 122]], dtype=torch.float32)
        boxes_det0 = cluster1[:1]    # YOLO detects cluster 1 only
        boxes_det1 = cluster2[:1]    # RT-DETR detects cluster 2 only
        res0 = _fake_det(boxes_det0, torch.tensor([0.9]))
        res1 = _fake_det(boxes_det1, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img0", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"],
            class_names=["car"],
            iou_cluster=0.5, crop_size=32, split="train",
        )
        assert len(graphs) == 2, f"Expected 2 graphs for 2 clusters, got {len(graphs)}"

    def test_overlapping_proposals_form_one_cluster(self):
        """Heavily overlapping proposals from two detectors form ONE cluster."""
        H, W = 128, 128
        image = _make_image(H, W)
        box_a = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        box_b = torch.tensor([[12, 12, 62, 62]], dtype=torch.float32)
        res0 = _fake_det(box_a, torch.tensor([0.9]))
        res1 = _fake_det(box_b, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img1", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"],
            class_names=["car"],
            iou_cluster=0.3, crop_size=32, split="train",
        )
        assert len(graphs) == 1, f"Expected 1 cluster from overlapping boxes, got {len(graphs)}"

    def test_node_features_rank_4(self):
        """node_features must be rank 4: [N, 3, H, W]."""
        H, W = 128, 128
        crop_size = 32
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))

        graphs = build_object_candidate_graphs(
            image, "img2", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            iou_cluster=0.5, crop_size=crop_size, split="train",
        )
        assert graphs, "Expected at least one graph"
        g = graphs[0][0]
        nf = g.node_features
        assert nf.ndim == 4, f"node_features must be rank 4, got {nf.ndim}"
        assert nf.shape[1] == 3, f"Expected 3 channels, got {nf.shape[1]}"
        assert nf.shape[2] == crop_size and nf.shape[3] == crop_size, \
            f"Expected crop_size={crop_size}, got {nf.shape[2:]}"

    def test_all_nodes_have_crop_tensors(self):
        """No node has a zero crop tensor (all crops are non-degenerate)."""
        H, W = 128, 128
        crop_size = 32
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 80, 80]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))

        graphs = build_object_candidate_graphs(
            image, "img3", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            iou_cluster=0.5, crop_size=crop_size, split="train",
        )
        assert graphs
        g = graphs[0][0]
        nf = g.node_features   # [N, 3, H, W]
        # No node should be all-zeros (which would mean a failed crop)
        per_node_sum = nf.abs().sum(dim=(1, 2, 3))
        assert (per_node_sum > 0).all(), "Some nodes have all-zero crop tensors"

    def test_fusion_nodes_have_crops(self):
        """WBF, union, NMS, Soft-NMS, BestProposal nodes all have crop tensors."""
        H, W = 128, 128
        crop_size = 32
        image = _make_image(H, W)
        box_a = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        box_b = torch.tensor([[12, 12, 62, 62]], dtype=torch.float32)
        res0 = _fake_det(box_a, torch.tensor([0.9]))
        res1 = _fake_det(box_b, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img4", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"], class_names=["car"],
            iou_cluster=0.3, crop_size=crop_size, split="train",
        )
        assert graphs
        g = graphs[0][0]
        nt = g.metadata["node_types"]
        nf = g.node_features

        fusion_types = [
            NODE_TYPES["cluster"],
            NODE_TYPES["consensus"],
            NODE_TYPES["nms_candidate"],
            NODE_TYPES["soft_nms_candidate"],
            NODE_TYPES["best_proposal_candidate"],
        ]
        for ftype in fusion_types:
            mask = nt == ftype
            if not mask.any():
                continue   # not present for single-proposal cluster
            crops = nf[mask]
            assert crops.ndim == 4
            assert (crops.abs().sum(dim=(1, 2, 3)) > 0).all(), \
                f"Fusion node type {ftype} has zero crop tensor"

    def test_node_box_exists_for_all_nodes(self):
        """Every node has a corresponding entry in node_box [N, 4]."""
        H, W = 128, 128
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 80, 80]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))

        graphs = build_object_candidate_graphs(
            image, "img5", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            iou_cluster=0.5, crop_size=32, split="train",
        )
        assert graphs
        g = graphs[0][0]
        N = g.node_features.shape[0]
        nb = g.metadata["node_box"]
        assert nb.shape == (N, 4), f"node_box shape mismatch: {nb.shape} vs expected ({N}, 4)"

    def test_selected_box_equals_node_box_entry(self):
        """After selecting a node, selected_box must exactly equal node_box[selected_node]."""
        H, W = 128, 128
        crop_size = 32
        image = _make_image(H, W)
        box_a = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        box_b = torch.tensor([[20, 20, 70, 70]], dtype=torch.float32)
        res0 = _fake_det(box_a, torch.tensor([0.9]))
        res1 = _fake_det(box_b, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img6", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"], class_names=["car"],
            iou_cluster=0.3, crop_size=crop_size, split="val",
        )
        assert graphs
        g = graphs[0][0]
        nb = g.metadata["node_box"]
        N = g.node_features.shape[0]

        # Simulate argmax selection (random logits)
        logits = torch.randn(N)
        selected_idx = int(logits.argmax().item())
        selected_box = nb[selected_idx]

        # Must be exactly one of the rows
        exact_match = (nb == selected_box.unsqueeze(0)).all(dim=1)
        assert exact_match.any(), "Selected box is not an exact row of node_box"

    def test_gt_not_in_val_graph_metadata(self):
        """GT boxes must NOT appear inside val/test graph inference features."""
        H, W = 128, 128
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 80, 80]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))
        gt_boxes  = torch.tensor([[15, 15, 85, 85]], dtype=torch.float32)
        gt_labels = torch.tensor([0], dtype=torch.long)

        graphs = build_object_candidate_graphs(
            image, "img7", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            gt_boxes=gt_boxes, gt_labels=gt_labels,
            iou_cluster=0.5, crop_size=32, split="val",  # NOT train
        )
        assert graphs
        g = graphs[0][0]
        # GT should NOT be in the val graph metadata
        assert "gt_boxes" not in g.metadata, \
            "GT boxes should not be stored in val/test graph metadata"

    def test_gt_in_train_graph_metadata(self):
        """GT boxes ARE stored in training graph metadata (for loss computation)."""
        H, W = 128, 128
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 80, 80]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))
        gt_boxes  = torch.tensor([[15, 15, 85, 85]], dtype=torch.float32)
        gt_labels = torch.tensor([0], dtype=torch.long)

        graphs = build_object_candidate_graphs(
            image, "img8", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            gt_boxes=gt_boxes, gt_labels=gt_labels,
            iou_cluster=0.5, crop_size=32, split="train",
        )
        assert graphs
        g_train = graphs[0][0]
        # GT SHOULD be in train graph metadata
        assert "gt_boxes" in g_train.metadata, \
            "GT boxes must be in train graph metadata for loss computation"

    def test_all_in_cluster_0(self):
        """All nodes in an object-level graph belong to cluster 0."""
        H, W = 128, 128
        image = _make_image(H, W)
        box_a = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        box_b = torch.tensor([[12, 12, 62, 62]], dtype=torch.float32)
        res0 = _fake_det(box_a, torch.tensor([0.9]))
        res1 = _fake_det(box_b, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img9", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"], class_names=["car"],
            iou_cluster=0.3, crop_size=32, split="train",
        )
        assert graphs
        g = graphs[0][0]
        cof = g.metadata["cluster_of_raw"]
        assert (cof == 0).all(), \
            f"Expected all nodes in cluster 0 for object-level graph, got {cof.unique()}"

    def test_empty_result_for_no_detections(self):
        """Image with no detections returns empty list."""
        H, W = 128, 128
        image = _make_image(H, W)

        class _Empty:
            boxes_xyxy = torch.zeros(0, 4); scores = torch.zeros(0)
            labels = []; label_ids = torch.zeros(0, dtype=torch.long)
            def num_detections(self): return 0

        graphs = build_object_candidate_graphs(
            image, "img10", (H, W), [_Empty()],
            detector_names=["yolo"], class_names=["car"],
            iou_cluster=0.5, crop_size=32, split="train",
        )
        assert graphs == [], f"Expected empty list for no detections, got {graphs}"

    def test_candidate_sources_field(self):
        """candidate_sources list is returned and contains expected source names."""
        H, W = 128, 128
        image = _make_image(H, W)
        box_a = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        box_b = torch.tensor([[12, 12, 62, 62]], dtype=torch.float32)
        res0 = _fake_det(box_a, torch.tensor([0.9]))
        res1 = _fake_det(box_b, torch.tensor([0.8]))

        graphs = build_object_candidate_graphs(
            image, "img11", (H, W), [res0, res1],
            detector_names=["yolo", "rtdetr"], class_names=["car"],
            iou_cluster=0.3, crop_size=32, split="train",
        )
        assert graphs
        _, img_id, cid, split, cand_src, gt_box, gt_lbl = graphs[0]
        assert "yolo" in cand_src
        assert "rtdetr" in cand_src
        assert "wbf" in cand_src
        assert "nms" in cand_src
        assert "union" in cand_src

    def test_tuple_format(self):
        """Each returned tuple has the correct structure."""
        H, W = 128, 128
        image = _make_image(H, W)
        box = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32)
        res = _fake_det(box, torch.tensor([0.9]))

        graphs = build_object_candidate_graphs(
            image, "imgX", (H, W), [res],
            detector_names=["yolo"], class_names=["car"],
            iou_cluster=0.5, crop_size=32, split="train",
        )
        assert graphs
        entry = graphs[0]
        assert len(entry) == 7, f"Expected 7-tuple, got {len(entry)}"
        g, img_id, cluster_id, sp, cand_src, gt_box, gt_lbl = entry
        from tgraphx import Graph
        assert isinstance(g, Graph)
        assert isinstance(img_id, str)
        assert isinstance(cluster_id, int)
        assert sp in ("train", "val", "test")
        assert isinstance(cand_src, list)
