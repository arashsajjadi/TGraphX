"""
Tests verifying correctness of V3 training path:
1. best_source labels are source SLOTS, not node indices
2. Absent sources stay masked after utility construction
3. source_slot_loss trains source_head and slot_attn (gradients non-zero)
4. baseline_slot respects the chosen baseline_source parameter
5. IoU utility is continuous — no IoU threshold before ranking
"""
import torch
import pytest
from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES
from od_graph_fusion.detectors.registry import build_synthetic_detector
from od_graph_fusion.datasets import SYNTHETIC_CLASS_NAMES
from od_graph_fusion.source_router_v3 import (
    TGraphXSourceRouterV3, source_slot_loss, SOURCE_SLOTS, NUM_SOURCES
)
from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels


DET_NAMES = ["yolo_modern", "yolo_open_vocab", "rt_detr", "retinanet"]
CLASS_NAMES = SYNTHETIC_CLASS_NAMES[:4]


def _make_graph(seed=0, jitter_base=0.04):
    dets = [build_synthetic_detector(n, "f", seed=seed+i, class_names=CLASS_NAMES,
                                     jitter=jitter_base+i*0.10)
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
        include_consensus_nodes=True, is_training=True)
    _attach_slot_metadata(g, meta, DET_NAMES)
    return g, meta, gt_b, gt_l


class TestBestSourceLabels:
    def test_best_source_is_slot_not_node_index(self):
        """Oracle label must be a source SLOT index (0..S-1), not a node index."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=True)
        assert result is not None
        _, best_slot, _, _, _ = result
        # All valid best_slots must be in [0, NUM_SOURCES)
        valid = best_slot[best_slot >= 0]
        assert valid.max().item() < NUM_SOURCES, (
            f"best_slot max={valid.max().item()} but NUM_SOURCES={NUM_SOURCES}. "
            "Labels must be SLOT indices, not node indices."
        )

    def test_oracle_slot_for_known_offset(self):
        """Create graph where oracle node is at large node index; verify slot < NUM_SOURCES."""
        g, meta, gt_b, gt_l = _make_graph(seed=7)
        result = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=True)
        assert result is not None
        _, best_slot, _, _, _ = result
        n_nodes = meta.node_types.shape[0]
        for c in range(best_slot.shape[0]):
            bs = int(best_slot[c].item())
            if bs >= 0:
                assert bs < NUM_SOURCES, (
                    f"Cluster {c}: best_slot={bs} >= NUM_SOURCES={NUM_SOURCES}. "
                    f"Graph has {n_nodes} nodes. Labels must be slot indices."
                )
                assert bs != n_nodes - 1, "best_slot must not equal last node index"


class TestAbsentSourceMask:
    def test_absent_slots_truly_masked(self):
        """Slots with no nodes in a cluster must be absent in slot_avail."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=True)
        assert result is not None
        _, _, _, util_per_slot, slot_avail = result
        slots = g.metadata["slot_assignments"]
        cluster_of = meta.cluster_of_node

        # Check manually which slots exist per cluster
        for c in range(meta.num_clusters):
            in_c = (cluster_of == c)
            present_slots = set()
            for ni in in_c.nonzero().squeeze(-1).tolist():
                if ni < slots.shape[0]:
                    s = int(slots[ni].item())
                    if s >= 0:
                        present_slots.add(s)
            for s in range(NUM_SOURCES):
                if s not in present_slots:
                    assert not slot_avail[c, s].item(), (
                        f"Cluster {c} slot {s}: not present but slot_avail=True. "
                        "Absent sources must be masked."
                    )

    def test_absent_slot_cannot_be_oracle(self):
        """If only slots 0 and 3 exist, best_slot must be one of {0, 3}."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=True)
        assert result is not None
        _, best_slot, _, _, slot_avail = result
        for c in range(best_slot.shape[0]):
            bs = int(best_slot[c].item())
            if bs >= 0:
                assert slot_avail[c, bs].item(), (
                    f"Cluster {c}: oracle slot {bs} is marked absent. "
                    "Oracle must come from an available slot."
                )

    def test_iou_utility_continuous_no_threshold(self):
        """Utility must be assigned even if IoU < 0.5; argmax over continuous values."""
        # Use very high jitter so most IoUs < 0.5
        g, meta, gt_b, gt_l = _make_graph(seed=0, jitter_base=0.45)
        result = _build_util_and_labels(g, meta, gt_b, gt_l, class_agnostic=True)
        assert result is not None
        util, best_slot, _, util_per_slot, slot_avail = result
        # Even with all IoU < 0.5, there must be valid best_slots
        n_valid = (best_slot >= 0).sum().item()
        n_clusters = meta.num_clusters
        assert n_valid > 0, (
            f"With continuous utility, at least some clusters must have valid best_slot. "
            f"Got {n_valid}/{n_clusters}. IoU thresholding is corrupting labels."
        )
        # Verify best_slot is argmax of available utility
        for c in range(best_slot.shape[0]):
            bs = int(best_slot[c].item())
            if bs < 0: continue
            avail = slot_avail[c]
            if not avail.any(): continue
            avail_idx = avail.nonzero().squeeze(-1)
            best_util = float(util_per_slot[c, avail_idx].max().item())
            chosen_util = float(util_per_slot[c, bs].item())
            assert abs(chosen_util - best_util) < 1e-5, (
                f"Cluster {c}: best_slot={bs} utility={chosen_util:.4f} but "
                f"max available utility={best_util:.4f}. Oracle must be argmax."
            )


class TestBaselineSlot:
    def test_baseline_nms_by_default(self):
        """Default baseline_source='nms_candidate' → bl_slot == SOURCE_SLOTS['nms_candidate']."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l,
                                        baseline_source="nms_candidate")
        assert result is not None
        _, _, bl_slot, _, slot_avail = result
        expected = SOURCE_SLOTS.get("nms_candidate", 6)
        for c in range(bl_slot.shape[0]):
            if slot_avail[c, expected]:
                assert int(bl_slot[c].item()) == expected, (
                    f"Cluster {c}: baseline_slot={bl_slot[c]} but expected {expected}."
                )

    def test_baseline_best_proposal(self):
        """baseline_source='best_proposal' → bl_slot == SOURCE_SLOTS['best_proposal']."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l,
                                        baseline_source="best_proposal")
        assert result is not None
        _, _, bl_slot, _, slot_avail = result
        expected = SOURCE_SLOTS.get("best_proposal", 8)
        for c in range(bl_slot.shape[0]):
            if slot_avail[c, expected]:
                assert int(bl_slot[c].item()) == expected

    def test_no_wbf_hardcode(self):
        """Default bl_slot must NOT always equal SOURCE_SLOTS['wbf']."""
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l)
        assert result is not None
        _, _, bl_slot, _, _ = result
        wbf_slot = SOURCE_SLOTS.get("wbf", 5)
        nms_slot_val = SOURCE_SLOTS.get("nms_candidate", 6)
        # Default is nms_candidate (6), should NOT equal WBF (5)
        if bl_slot.numel() > 0:
            assert not (bl_slot == wbf_slot).all().item(), (
                "bl_slot is all WBF — WBF hardcode still present."
            )


class TestGradients:
    def test_source_head_gets_gradients(self):
        """source_slot_loss must back-propagate into source_head.weight."""
        torch.manual_seed(0)
        model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4,
                                       crop_size=8, crop_channels=4,
                                       hidden_dim=32, num_message_passing=1)
        g, meta, gt_b, gt_l = _make_graph()
        result = _build_util_and_labels(g, meta, gt_b, gt_l)
        assert result is not None
        _, best_slot, bl_slot, util_per_slot, slot_avail = result
        if (best_slot < 0).all():
            pytest.skip("No valid clusters in this graph")

        model.train()
        out = model(g, detector_names=DET_NAMES)
        src_logits = out.get("source_logits")
        src_mask = out.get("source_mask")
        assert src_logits is not None, "V3 model must output source_logits"

        # Use source_slot_loss
        valid = best_slot >= 0
        if not valid.any():
            pytest.skip("No valid clusters")
        loss_dict = source_slot_loss(
            src_logits[valid], src_mask[valid],
            best_slot[valid], util_per_slot[valid],
            baseline_slot=bl_slot[valid],
        )
        loss = loss_dict["total"]
        loss.backward()

        # source_head must have non-zero gradients
        source_head_grad = model.source_head.weight.grad
        assert source_head_grad is not None, "source_head.weight.grad is None"
        assert source_head_grad.norm().item() > 1e-8, (
            f"source_head.weight.grad norm={source_head_grad.norm().item():.2e} ≈ 0. "
            "source_slot_loss is not training the source router."
        )

    def test_slot_attn_gets_gradients(self):
        """slot_attn must also receive gradients."""
        torch.manual_seed(1)
        model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4,
                                       crop_size=8, crop_channels=4,
                                       hidden_dim=32, num_message_passing=1)
        g, meta, gt_b, gt_l = _make_graph(seed=3)
        result = _build_util_and_labels(g, meta, gt_b, gt_l)
        assert result is not None
        _, best_slot, bl_slot, util_per_slot, slot_avail = result
        if (best_slot < 0).all():
            pytest.skip("No valid clusters")

        model.train()
        out = model(g, detector_names=DET_NAMES)
        src_logits = out["source_logits"]
        src_mask = out["source_mask"]
        valid = best_slot >= 0
        if not valid.any(): pytest.skip("No valid clusters")

        loss = source_slot_loss(src_logits[valid], src_mask[valid],
                                best_slot[valid], util_per_slot[valid])["total"]
        loss.backward()

        # Check any slot_attn parameter
        for name, p in model.slot_attn.named_parameters():
            if p.grad is not None and p.grad.norm().item() > 1e-10:
                return  # At least one parameter has gradient
        pytest.fail("slot_attn has no gradients from source_slot_loss")

    def test_source_logits_not_quality_logits_primary(self):
        """V3 forward must return source_logits; quality_logits is auxiliary only."""
        model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4,
                                       crop_size=8, crop_channels=4,
                                       hidden_dim=32, num_message_passing=1)
        g, meta, gt_b, gt_l = _make_graph()
        with torch.no_grad():
            out = model(g, detector_names=DET_NAMES)
        assert "source_logits" in out, "V3 must output source_logits"
        assert "source_mask" in out, "V3 must output source_mask"
        sl = out["source_logits"]
        sm = out["source_mask"]
        assert sl.ndim == 2 and sl.shape[1] == NUM_SOURCES
        assert sm.shape == sl.shape and sm.dtype == torch.bool
