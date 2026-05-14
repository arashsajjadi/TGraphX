"""train_fusion_model must REFUSE to run V3 with empty detector_names.

Background: empty detector_names made proposal nodes invisible to the
SourceSlotAggregator in v8/v9, which silently trained the wrong objective.
The audit (reports/CRITICAL_CODE_PATH_AUDIT.md §2) flagged this; Part 8
mandates a hard guard.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def _dummy_graph_meta():
    """Build a single train graph + meta with the minimum metadata train_fusion_model needs."""
    from tgraphx import Graph
    from od_graph_fusion.graph_builder import NODE_TYPES, DetectionGraphMeta
    N = 3
    node_types = torch.tensor([
        NODE_TYPES["proposal"], NODE_TYPES["proposal"], NODE_TYPES["cluster"],
    ], dtype=torch.long)
    pdet = torch.tensor([0, 1, -1], dtype=torch.long)
    cluster_of = torch.tensor([0, 0, 0], dtype=torch.long)
    crops = torch.randn(N, 3, 32, 32)
    ei = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    ea = torch.zeros(ei.shape[1], 14)
    md = torch.zeros(N, 8 + 2 + 1)
    g = Graph(node_features=crops, edge_index=ei, edge_features=ea)
    g.metadata = {
        "node_metadata": md, "node_types": node_types,
        "cluster_of_raw": cluster_of, "proposal_det_ids": pdet,
        "node_box": torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [0., 0., 11., 11.]]),
        "node_score": torch.tensor([0.5, 0.6, 0.55]),
        "node_label": torch.zeros(N, dtype=torch.long),
        "detector_names": [],
    }
    meta = DetectionGraphMeta(
        image_id="x", image_size=(64, 64),
        num_proposals=2, num_clusters=1, num_consensus=0, has_context=False,
        detector_names=[], class_names=["car"],
        node_types=node_types,
        node_to_proposal_index=torch.tensor([0, 1, -1], dtype=torch.long),
        cluster_of_node=cluster_of,
        cluster_boxes=torch.tensor([[0., 0., 10., 10.]]),
        cluster_labels=torch.tensor([0], dtype=torch.long),
        proposal_boxes=torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.]]),
        proposal_scores=torch.tensor([0.5, 0.6]),
        proposal_labels=torch.zeros(2, dtype=torch.long),
        proposal_detector_ids=torch.tensor([0, 1], dtype=torch.long),
        cluster_score=torch.tensor([0.55]),
        targets={"objectness": torch.tensor([1., 1., 1.]),
                  "class": torch.tensor([0, 0, 0], dtype=torch.long),
                  "box_reg": torch.zeros(3, 4),
                  "iou": torch.tensor([0.6, 0.7, 0.65]),
                  "candidate_mask": torch.tensor([True, True, True])},
    )
    return (g, meta)


def test_training_v3_raises_if_detector_names_empty():
    from od_graph_fusion.training import train_fusion_model
    gm = _dummy_graph_meta()
    with pytest.raises(RuntimeError, match="detector_names is required"):
        train_fusion_model(
            train_graphs=[gm], val_graphs=[gm],
            num_classes=1, num_detectors=2, crop_size=32,
            hidden_dim=16, num_message_passing=1,
            epochs=1, device="cpu",
            use_source_router=True,
            detector_names=[],   # ← the bug we are guarding against
        )


def test_training_v3_raises_if_detector_names_none():
    from od_graph_fusion.training import train_fusion_model
    gm = _dummy_graph_meta()
    with pytest.raises(RuntimeError, match="detector_names is required"):
        train_fusion_model(
            train_graphs=[gm], val_graphs=[gm],
            num_classes=1, num_detectors=2, crop_size=32,
            hidden_dim=16, num_message_passing=1,
            epochs=1, device="cpu",
            use_source_router=True,
            detector_names=None,
        )


def test_step04_passes_detector_names_to_training():
    """Audit: Step 04 must pass detector_names through to train_fusion_model."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "04_train_tgraphx_fusion.py").read_text()
    assert "detector_names=detector_names" in src, (
        "Step 04 must explicitly forward detector_names to train_fusion_model."
    )
    assert "utility_mode=utility_mode" in src, (
        "Step 04 must forward utility_mode."
    )
    assert "class_agnostic=class_agnostic" in src, (
        "Step 04 must forward class_agnostic."
    )
