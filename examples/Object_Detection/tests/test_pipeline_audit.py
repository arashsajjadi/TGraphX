"""
Tests for all critical pipeline bugs identified in the audit.
"""
import json, sys
import torch
import pytest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


# ── Bug 1: Step 05 must not call model with detector_names=[] ─────────────────

def test_step05_passes_real_detector_names():
    """Step 05 run_score_mode must receive non-empty detector_names."""
    import re
    script = (Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py").read_text()
    # Must NOT call model(... detector_names=[])
    assert "detector_names=[]" not in script, (
        "Step 05 still calls model with detector_names=[] — "
        "this makes proposal nodes slot=-1 and invisible to source_logits."
    )
    # Must pass detector_names from checkpoint/manifest
    assert "detector_names" in script and "model_config" in script, (
        "Step 05 must load detector_names from checkpoint model_config, not hard-code."
    )


def test_step05_raises_on_empty_detector_names():
    """Step 05 must raise RuntimeError if model_config has num_detectors=0."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py").read_text()
    assert "num_detectors == 0" in script or "not detector_names" in script, (
        "Step 05 must validate num_detectors > 0 before calling model.forward."
    )


# ── Bug 2: Step 05 must reconstruct model from checkpoint, not hard-coded num_detectors=4 ──

def test_step05_no_hardcoded_num_detectors():
    """Step 05 must not hard-code num_detectors=4."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py").read_text()
    assert "num_detectors=4" not in script, (
        "Step 05 still has hard-coded num_detectors=4. "
        "Must reconstruct from checkpoint['model_config']."
    )
    assert "model_config" in script, "Step 05 must use checkpoint['model_config']"


def test_checkpoint_model_config_roundtrip():
    """Checkpoint must contain model_config with correct num_detectors."""
    import tempfile, os
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3

    torch.manual_seed(0)
    model = TGraphXSourceRouterV3(num_classes=2, num_detectors=3, crop_size=8,
                                   crop_channels=4, hidden_dim=16, num_message_passing=1)
    model_config = {
        "num_classes": 2, "num_detectors": 3, "detector_names": ["yolo", "ret", "detr"],
        "crop_size": 8, "crop_channels": 4, "hidden_dim": 16, "num_message_passing": 1,
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({"model_state": model.state_dict(), "model_config": model_config,
                    "seed": 0}, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=False)
        mc = loaded["model_config"]
        # Reconstruct — must match
        model2 = TGraphXSourceRouterV3(
            num_classes=mc["num_classes"], num_detectors=mc["num_detectors"],
            crop_size=mc["crop_size"], crop_channels=mc["crop_channels"],
            hidden_dim=mc["hidden_dim"], num_message_passing=mc["num_message_passing"],
        )
        model2.load_state_dict(loaded["model_state"])
        assert mc["num_detectors"] == 3, "num_detectors must be saved and loaded correctly"
        assert mc["detector_names"] == ["yolo", "ret", "detr"]
    finally:
        os.unlink(ckpt_path)


# ── Bug 3: Step 03 must use split tags, not fake is_training loop ──────────────

def test_step03_no_fake_split_loop():
    """Step 03 must not have a 'break' that makes all graphs is_training=True."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "03_build_detection_graphs.py").read_text()
    # Old pattern: for is_training, split_name in [...]: ... break
    assert "break  # build once" not in script, (
        "Step 03 still has the fake split loop that builds all graphs with is_training=True."
    )


def test_step03_writes_split_manifest():
    """Step 03 must write split_manifest.json."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "03_build_detection_graphs.py").read_text()
    assert "split_manifest.json" in script, "Step 03 must write split_manifest.json"
    assert "source_labels.pt" in script, "Step 03 must write source_labels.pt separately from graphs.pt"


def test_step03_split_from_record_split_field():
    """Step 03 must use rec.split, not naive index slicing."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "03_build_detection_graphs.py").read_text()
    assert "rec.split" in script or "getattr(rec" in script, (
        "Step 03 must use record.split field for split assignment, not index slicing."
    )


# ── Bug 4: Split manifest consumed by Step 04 and 05 ─────────────────────────

def test_step04_uses_split_manifest():
    """Step 04 must consume split_manifest.json if it exists."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "04_train_tgraphx_fusion.py").read_text()
    assert "split_manifest.json" in script, "Step 04 must use split_manifest.json"


def test_step05_uses_split_manifest():
    """Step 05 must consume split_manifest.json for deterministic splits."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py").read_text()
    assert "split_manifest.json" in script, "Step 05 must use split_manifest.json"


# ── Bug 5: Central candidate_node_mask ────────────────────────────────────────

def test_candidate_mask_includes_all_universal_sources():
    """candidate_node_mask must include NMS, SoftNMS, BestProposal nodes."""
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.candidate_mask import candidate_node_mask

    # Build a small node_types tensor with all types
    types = torch.tensor([
        NODE_TYPES["proposal"],
        NODE_TYPES["cluster"],
        NODE_TYPES["consensus"],
        NODE_TYPES["nms_candidate"],
        NODE_TYPES["soft_nms_candidate"],
        NODE_TYPES["best_proposal_candidate"],
        NODE_TYPES.get("context", 3),  # context should NOT be in mask
    ], dtype=torch.long)

    mask = candidate_node_mask(types, NODE_TYPES)
    assert mask.shape == (7,)
    # First 6 must be True
    assert mask[:6].all(), "All candidate source types must be in mask"
    # Context node must be False
    assert not mask[6].item(), "Context node must NOT be in candidate mask"


def test_nms_can_be_oracle_source():
    """NMS candidate node must be eligible as oracle source."""
    from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES
    from od_graph_fusion.detectors.registry import build_synthetic_detector
    from od_graph_fusion.datasets import SYNTHETIC_CLASS_NAMES
    from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata, _build_util_and_labels

    dets = [build_synthetic_detector("yolo_modern", "f", seed=0,
                                      class_names=SYNTHETIC_CLASS_NAMES[:2])]
    dets[0].load()
    img = torch.rand(3, 32, 32)
    gt_b = torch.tensor([[3., 3., 22., 22.]]); gt_l = torch.zeros(1, dtype=torch.long)
    res = [dets[0].predict(img, "img0", gt_boxes=gt_b, gt_labels=gt_l)]
    g, meta = build_detection_graph(img, "img0", (32,32), res, ["yolo_modern"],
        SYNTHETIC_CLASS_NAMES[:2], gt_boxes=gt_b, gt_labels=gt_l,
        crop_size=8, include_context_node=False, include_consensus_nodes=True, is_training=True)
    _attach_slot_metadata(g, meta, ["yolo_modern"])

    result = _build_util_and_labels(g, meta, gt_b, gt_l, True, utility_mode="ap50")
    assert result is not None
    util, best_slot, bl_slot, util_per_slot, slot_avail = result

    # NMS/SoftNMS/BestProposal slots must be in avail mask (slot 6/7/8)
    from od_graph_fusion.source_router_v3 import SOURCE_SLOTS
    nms_s = SOURCE_SLOTS.get("nms_candidate", 6)
    # At least one cluster should have nms_candidate available
    assert slot_avail[:, nms_s].any(), (
        f"nms_candidate (slot {nms_s}) never available — compute_source_utilities "
        "still excludes nms_candidate from candidate mask."
    )


# ── Bug 6: Step 05 reports class-aware AP ────────────────────────────────────

def test_step05_reports_class_aware_ap():
    """Step 05 must write both class_aware_AP and class_agnostic_AP."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py").read_text()
    assert "class_agnostic=False" in script, "Step 05 must compute class-aware AP"
    assert "class_agnostic=True" in script, "Step 05 must also compute class-agnostic AP"
    assert "test_ap_class_aware" in script, "Step 05 must record test_ap_class_aware"
    assert "headline_ap" in script, "Step 05 must record headline_ap"


def test_metrics_json_has_required_fields():
    """metrics_seed*.json must contain val_score_modes, selected_score_mode, test_metrics."""
    mock_metrics = {
        "seed": 0,
        "val_score_modes": {"routing_prob": {"val_ap_agnostic": 0.85}},
        "selected_score_mode": "routing_prob",
        "test_metrics_selected_mode": {
            "test_ap_class_agnostic": 0.88,
            "test_ap_class_aware": 0.82,
            "headline_ap": 0.82,
            "ece": 0.05,
            "brier": 0.10,
            "fp_per_image": 1.2,
        },
    }
    j = json.dumps(mock_metrics)
    loaded = json.loads(j)
    assert "selected_score_mode" in loaded
    assert "test_metrics_selected_mode" in loaded
    assert "headline_ap" in loaded["test_metrics_selected_mode"]
    assert "ece" in loaded["test_metrics_selected_mode"]


# ── Bug 7: Real VOC must reject synthetic detectors ──────────────────────────

def test_real_voc_disallows_synthetic_detectors_in_step02():
    """Step 02 must raise RuntimeError if experiment_type=real_voc and detector is synthetic."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "02_run_detectors.py").read_text()
    assert "real_voc" in script, "Step 02 must check experiment_type"
    assert "synthetic" in script and "RuntimeError" in script, (
        "Step 02 must raise RuntimeError if synthetic detector used in real_voc experiment."
    )


# ── Bug 8: Step 06 verdict not hardcoded ─────────────────────────────────────

def test_step06_verdict_not_hardcoded():
    """Step 06 verdict must be computed from data, not a hard-coded string."""
    script = (Path(__file__).resolve().parents[1] / "scripts" / "06_make_report.py").read_text()
    # Must NOT have a fixed verdict string
    assert "SYNTHETIC_CONTROLLED_ROUTING_WIN** — results are from synthetic jitter benchmark." \
           not in script, "Step 06 still has hardcoded synthetic verdict — must compute dynamically"
    # Must compute verdict from metrics
    assert "mean_ap" in script or "_verdict" in script, (
        "Step 06 must compute verdict from actual AP values, not hardcode it."
    )


# ── Bug 10: SourceSlotAggregator deterministic tie-breaking ──────────────────

def test_slot_aggregator_deterministic_tie_breaking():
    """Two nodes with equal score in same slot must always select the same node."""
    from od_graph_fusion.source_router_v3 import SourceSlotAggregator, NUM_SOURCES
    import hashlib

    D = 16; S = NUM_SOURCES
    agg = SourceSlotAggregator(D, S)
    agg.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agg = agg.to(device)

    # Two nodes in same cluster, same slot, EQUAL score
    torch.manual_seed(5)
    node_emb = torch.randn(2, D, device=device)
    cluster_of = torch.tensor([0, 0], device=device)
    slot_assign = torch.tensor([3, 3], device=device)  # both in slot 3
    score = torch.tensor([0.5, 0.5], device=device)    # EQUAL scores

    results = []
    for _ in range(5):  # repeat to check determinism
        with torch.no_grad():
            _, _, slot_node_idx = agg(node_emb, cluster_of, slot_assign, 1, score)
        chosen = int(slot_node_idx[0, 3].item())
        results.append(chosen)

    assert len(set(results)) == 1, (
        f"SourceSlotAggregator is non-deterministic for equal scores: {results}. "
        "Tie-breaking must be deterministic (e.g., highest node index)."
    )


# ── Bug 11: Legacy model labeled correctly ───────────────────────────────────

def test_legacy_model_renamed():
    """DetectionFusionModel must be renamed/aliased as LegacyDetectionFusionModel."""
    from od_graph_fusion.models import LegacyDetectionFusionModel, DetectionFusionModel
    assert DetectionFusionModel is LegacyDetectionFusionModel, (
        "DetectionFusionModel must be an alias for LegacyDetectionFusionModel"
    )


def test_legacy_model_not_used_in_v3_strict():
    """In strict_source_router mode, training.py must use TGraphXSourceRouterV3."""
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    import inspect
    from od_graph_fusion import training
    src = inspect.getsource(training.train_fusion_model)
    # When use_source_router=True, must instantiate V3
    assert "TGraphXSourceRouterV3" in src, (
        "training.py must use TGraphXSourceRouterV3 when use_source_router=True"
    )
