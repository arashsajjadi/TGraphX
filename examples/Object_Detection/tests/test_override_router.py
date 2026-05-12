"""Tests for NMSOverrideRouter."""
import torch
import pytest
from od_graph_fusion.override_router import (
    NMSOverrideRouter, override_routing_loss, oracle_overfit_sanity,
)
from od_graph_fusion.source_router_v3 import NUM_SOURCES


def _simple_model():
    return NMSOverrideRouter(num_classes=5, num_detectors=4, crop_size=16,
                              crop_channels=8, hidden_dim=16, num_message_passing=1)


def test_override_router_forward():
    from tgraphx import Graph
    model = _simple_model()
    x = torch.randn(6, 3, 16, 16)
    ei = torch.tensor([[0, 1], [1, 2]])
    g = Graph(node_features=x, edge_index=ei)
    with torch.no_grad():
        out = model(g)
    assert "quality_logits" in out
    assert out["quality_logits"].shape == (6,)


def test_override_loss_override_target_correct():
    """override_target=1 when oracle != NMS, 0 otherwise."""
    C, S = 4, NUM_SOURCES
    source_logits = torch.randn(C, S)
    source_mask = torch.ones(C, S, dtype=torch.bool)
    override_logits = torch.randn(C)
    best_slot = torch.tensor([0, 1, 2, 2])   # cluster 0..2 different oracle
    nms_slot = torch.tensor([0, 0, 0, 2])    # nms is slot 0 for first 3, slot 2 for last
    util = torch.rand(C, S)
    losses = override_routing_loss(source_logits, source_mask, override_logits,
                                    best_slot, nms_slot, util)
    assert torch.isfinite(losses["total"])
    # Clusters 1,2 should be override cases; cluster 0,3 are no-override
    assert losses["n_override"] == 2
    assert losses["n_no_override"] == 2


def test_override_loss_absent_source_safe():
    """Loss must not crash when best_slot is -1 (cluster not matched)."""
    C, S = 3, NUM_SOURCES
    source_logits = torch.randn(C, S)
    source_mask = torch.ones(C, S, dtype=torch.bool)
    override_logits = torch.randn(C)
    best_slot = torch.tensor([-1, 1, 2])  # first cluster not matched
    nms_slot = torch.tensor([0, 0, 0])
    util = torch.rand(C, S)
    losses = override_routing_loss(source_logits, source_mask, override_logits,
                                    best_slot, nms_slot, util)
    assert torch.isfinite(losses["total"])


def test_override_loss_regret_weighting():
    """High-regret clusters must contribute more to total loss."""
    C, S = 2, NUM_SOURCES
    source_logits = torch.randn(C, S, requires_grad=True)
    source_mask = torch.ones(C, S, dtype=torch.bool)
    override_logits = torch.randn(C, requires_grad=True)
    best_slot = torch.tensor([0, 0])
    nms_slot = torch.tensor([1, 1])

    # Cluster 0: oracle util=0.9, NMS util=0.1 → high regret
    # Cluster 1: oracle util=0.6, NMS util=0.5 → low regret
    util = torch.zeros(C, S)
    util[0, 0] = 0.9; util[0, 1] = 0.1
    util[1, 0] = 0.6; util[1, 1] = 0.5

    losses = override_routing_loss(source_logits, source_mask, override_logits,
                                    best_slot, nms_slot, util, regret_lambda=5.0)
    assert torch.isfinite(losses["total"])
    # With lambda=0, should give same loss per cluster
    losses0 = override_routing_loss(source_logits.detach(), source_mask, override_logits.detach(),
                                     best_slot, nms_slot, util, regret_lambda=0.0)
    # With lambda=5, total loss should differ
    assert abs(losses["total"].item() - losses0["total"].item()) > 1e-6


def test_sanity_a_can_overfit_oracle():
    """Stage A: model must reach >80% source acc on tiny synthetic data."""
    from od_graph_fusion.detectors.registry import build_synthetic_detector
    from od_graph_fusion.graph_builder import build_detection_graph
    from od_graph_fusion.datasets import SYNTHETIC_CLASS_NAMES
    from od_graph_fusion.source_router_v3 import detector_name_to_slot, SOURCE_SLOTS
    from od_graph_fusion.graph_builder import NODE_TYPES
    import random

    torch.manual_seed(0); random.seed(0)
    DET_NAMES = ["yolo_modern", "yolo_open_vocab", "rt_detr", "retinanet"]
    CLASS_NAMES = SYNTHETIC_CLASS_NAMES[:4]
    # Make detectors with very different jitter so one is clearly better
    dets = [build_synthetic_detector(n, "f", seed=i, class_names=CLASS_NAMES, jitter=0.02 + i*0.12)
            for i, n in enumerate(DET_NAMES)]
    [d.load() for d in dets]
    rng = torch.Generator().manual_seed(0)
    graphs_metas, gt_boxes, gt_labels = [], [], []
    for img_i in range(15):
        img = torch.rand(3, 32, 32, generator=rng)
        gt_b = torch.tensor([[2., 2., 20., 20.]])
        gt_l = torch.zeros(1, dtype=torch.long)
        res = [d.predict(img, f"i{img_i}", gt_boxes=gt_b, gt_labels=gt_l) for d in dets]
        g, meta = build_detection_graph(img, f"i{img_i}", (32, 32), res, DET_NAMES, CLASS_NAMES,
            crop_size=8, max_proposals=6, include_context_node=False, include_consensus_nodes=True,
            is_training=True)
        N = meta.node_types.shape[0]
        slots = torch.full((N,), -1, dtype=torch.long)
        for i in range(meta.num_proposals):
            d = int(meta.proposal_detector_ids[i]) if i < meta.proposal_detector_ids.shape[0] else -1
            if 0 <= d < len(DET_NAMES): slots[i] = detector_name_to_slot(DET_NAMES[d])
        slots[meta.node_types == NODE_TYPES["cluster"]] = SOURCE_SLOTS["wbf"]
        slots[meta.node_types == NODE_TYPES["consensus"]] = SOURCE_SLOTS["union"]
        g.metadata.update({"slot_assignments": slots, "cluster_of_raw": meta.cluster_of_node,
                            "proposal_det_ids": meta.proposal_detector_ids})
        graphs_metas.append((g, meta)); gt_boxes.append(gt_b); gt_labels.append(gt_l)

    model = NMSOverrideRouter(num_classes=4, num_detectors=4, crop_size=8,
                               crop_channels=4, hidden_dim=16, num_message_passing=1)
    result = oracle_overfit_sanity(model, graphs_metas, gt_boxes, gt_labels, DET_NAMES,
                                   device="cpu", max_epochs=300, target_source_acc=0.80)
    assert result["passed"], (
        f"Stage A sanity failed: final src_acc={result['final_source_acc']:.4f} "
        "after 300 epochs. Architecture or loss is broken."
    )
    assert result["final_source_acc"] >= 0.75
