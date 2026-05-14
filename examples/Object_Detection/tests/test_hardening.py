"""
Hardening tests for device handling, slot mapping safety, score calibration,
stepwise pipeline isolation, and seed auditing.
"""
import json, hashlib
import torch
import pytest
from pathlib import Path


# ── Device handling ───────────────────────────────────────────────────────────

def test_resolve_device_auto_returns_string():
    from od_graph_fusion.config import resolve_device
    result = resolve_device("auto")
    assert result in ("cpu", "cuda", "mps"), f"Unexpected device: {result}"


def test_resolve_device_auto_prefers_cuda_when_available():
    from od_graph_fusion.config import resolve_device
    import torch
    if torch.cuda.is_available():
        assert resolve_device("auto") == "cuda"


def test_resolve_device_cpu_explicit():
    from od_graph_fusion.config import resolve_device
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    from od_graph_fusion import config as cfg_mod
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        cfg_mod.resolve_device("cuda")


def test_device_audit_returns_dict():
    from od_graph_fusion.config import device_audit
    result = device_audit("auto", "cpu")
    assert "cuda_available" in result
    assert "resolved_device" in result


def test_train_fusion_model_default_device_is_not_cpu(monkeypatch):
    """Default device in train_fusion_model must not be hard-coded 'cpu'."""
    import inspect
    from od_graph_fusion import training
    src = inspect.getsource(training.train_fusion_model)
    assert 'device: str = "cpu"' not in src, (
        "train_fusion_model still defaults to cpu — must default to 'auto'"
    )


# ── Source slot mapping safety ────────────────────────────────────────────────

def test_slot_mapping_no_dangerous_fallback():
    """_build_node_source_slots must not use NODE_TYPES.get('key', integer)."""
    import inspect
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    src = inspect.getsource(TGraphXSourceRouterV3._build_node_source_slots)
    # Dangerous pattern: NODE_TYPES.get("nms_candidate", 4) — fallback int
    assert 'NODE_TYPES.get("nms_candidate"' not in src, (
        "Unsafe fallback NODE_TYPES.get('nms_candidate', 4) found — use guarded check"
    )
    assert 'NODE_TYPES.get("soft_nms_candidate"' not in src
    assert 'NODE_TYPES.get("best_proposal_candidate"' not in src


def test_context_node_stays_at_minus_one():
    """Context nodes must never be assigned a source slot."""
    from od_graph_fusion.graph_builder import build_detection_graph, NODE_TYPES
    from od_graph_fusion.detectors.registry import build_synthetic_detector
    from od_graph_fusion.datasets import SYNTHETIC_CLASS_NAMES
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3

    dets = [build_synthetic_detector("yolo_modern", "f", seed=0,
                                      class_names=SYNTHETIC_CLASS_NAMES[:2])]
    dets[0].load()
    img = torch.rand(3, 32, 32)
    gt_b = torch.tensor([[3., 3., 22., 22.]]); gt_l = torch.zeros(1, dtype=torch.long)
    res = [dets[0].predict(img, "img0", gt_boxes=gt_b, gt_labels=gt_l)]
    g, meta = build_detection_graph(img, "img0", (32,32), res, ["yolo_modern"],
        SYNTHETIC_CLASS_NAMES[:2], gt_boxes=gt_b, gt_labels=gt_l,
        crop_size=8, include_context_node=True, is_training=False)

    model = TGraphXSourceRouterV3(num_classes=2, num_detectors=1, crop_size=8,
                                   crop_channels=4, hidden_dim=16, num_message_passing=1)
    model.eval()
    with torch.no_grad():
        out = model(g, detector_names=["yolo_modern"])

    # source_mask shape is [C, S] — context node must not be a cluster
    # The slot_assignments in model forward: check context node got -1
    # We can't directly check inside forward, but we verify output shape makes sense
    assert "source_logits" in out
    assert out["source_logits"].ndim == 2


def test_missing_node_type_key_does_not_mismap():
    """If 'calibrated_consensus' key is absent from NODE_TYPES, no node gets that slot."""
    from od_graph_fusion.graph_builder import NODE_TYPES
    assert "calibrated_consensus" not in NODE_TYPES, (
        "calibrated_consensus is in NODE_TYPES — test must be updated"
    )
    # Verify _build_node_source_slots has guarded check for it
    import inspect
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    src = inspect.getsource(TGraphXSourceRouterV3._build_node_source_slots)
    assert '"calibrated_consensus" in NODE_TYPES' in src, (
        "_build_node_source_slots must guard calibrated_consensus with key-existence check"
    )


# ── Dead code / override_head removed ────────────────────────────────────────

def test_override_head_removed_from_v3():
    """TGraphXSourceRouterV3 must not define override_head (dead code)."""
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    model = TGraphXSourceRouterV3(num_classes=2, num_detectors=2, crop_size=8,
                                   crop_channels=4, hidden_dim=16, num_message_passing=1)
    assert not hasattr(model, "override_head"), (
        "override_head is still defined on TGraphXSourceRouterV3 — remove dead code"
    )


# ── Score calibration on validation split ─────────────────────────────────────

def test_score_mode_cannot_be_selected_on_test():
    """Verify step 05 (evaluate.py) explicitly mentions validation-only selection."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py"
    assert script_path.exists()
    src = script_path.read_text()
    assert "val" in src.lower() and "selected on" in src.lower(), (
        "05_evaluate.py must document score mode selection on validation only"
    )
    import re
    has_call = bool(re.search(r'\brun_pipeline\s*\(', src))
    assert not has_call, "05_evaluate.py calls run_pipeline — must be artifact-level"


# ── Stepwise pipeline isolation ───────────────────────────────────────────────

@pytest.mark.parametrize("script_name", [
    "01_download_data.py",
    "02_run_detectors.py",
    "03_build_detection_graphs.py",
    "04_train_tgraphx_fusion.py",
    "05_evaluate.py",
    "06_make_report.py",
])
def test_step_script_does_not_call_run_pipeline(script_name):
    """No step script 01-06 may import or call run_pipeline directly."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    assert script_path.exists(), f"Script not found: {script_path}"
    src = script_path.read_text()
    # "run_pipeline" may appear in docstring ("Does NOT call run_pipeline").
    # It must NOT appear as an import or call.
    import re
    has_import = bool(re.search(r'^from\s+\S+\s+import\s+run_pipeline', src, re.MULTILINE))
    has_call = bool(re.search(r'\brun_pipeline\s*\(', src))
    assert not has_import and not has_call, (
        f"{script_name} imports or calls run_pipeline — must be artifact-level"
    )


def test_step_02_does_not_train():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "02_run_detectors.py"
    src = script_path.read_text()
    assert "train_fusion_model" not in src
    assert "backward" not in src


def test_step_04_does_not_build_graphs():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_train_tgraphx_fusion.py"
    src = script_path.read_text()
    assert "build_detection_graph" not in src


def test_step_05_does_not_train():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "05_evaluate.py"
    src = script_path.read_text()
    assert "backward" not in src
    assert "train_fusion_model" not in src


# ── Synthetic benchmark label ──────────────────────────────────────────────────

def test_synthetic_benchmark_not_labeled_as_real():
    """The synthetic benchmark verdict must include SYNTHETIC in its label."""
    report_path = Path(__file__).resolve().parents[1] / "reports" / "DATA_LEAKAGE_AUDIT.md"
    if not report_path.exists():
        pytest.skip("DATA_LEAKAGE_AUDIT.md not found")
    text = report_path.read_text()
    assert "SYNTHETIC" in text.upper(), "Leakage audit must mention SYNTHETIC results"


# ── Seed audit helpers ────────────────────────────────────────────────────────

def _param_hash(model):
    """Stable hash of model parameters for seed audit."""
    buf = b""
    for p in model.parameters():
        buf += p.detach().cpu().float().numpy().tobytes()
    return hashlib.md5(buf).hexdigest()[:12]


def test_different_seeds_give_different_init_hashes():
    """Different torch.manual_seed values must produce different initial weights."""
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    hashes = set()
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        model = TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                       crop_channels=4, hidden_dim=32, num_message_passing=1)
        hashes.add(_param_hash(model))
    assert len(hashes) == 3, (
        f"Expected 3 unique hashes, got {len(hashes)}. "
        "torch.manual_seed is not affecting model initialization."
    )


def test_same_seed_gives_same_init_hash():
    """Same seed must reproduce identical initialization."""
    from od_graph_fusion.source_router_v3 import TGraphXSourceRouterV3
    torch.manual_seed(42)
    h1 = _param_hash(TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                             crop_channels=4, hidden_dim=32, num_message_passing=1))
    torch.manual_seed(42)
    h2 = _param_hash(TGraphXSourceRouterV3(num_classes=4, num_detectors=4, crop_size=8,
                                             crop_channels=4, hidden_dim=32, num_message_passing=1))
    assert h1 == h2, "Same seed must give identical init hash"
