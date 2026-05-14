"""Tests that every candidate node type carries a real crop tensor.

Required by the paper-faithful formulation addendum.
"""
from __future__ import annotations
from pathlib import Path
import sys, pytest, torch

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))


def _load_graphs(run_dir="runs/real_voc_car_v2"):
    p = Path(run_dir) / "graphs.pt"
    if not p.exists():
        pytest.skip(f"graphs.pt not found at {p}")
    return torch.load(p, weights_only=False)


def test_all_candidate_nodes_have_crop_tensors():
    """node_features must be non-empty for every candidate node."""
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.candidate_mask import candidate_node_mask
    graphs = _load_graphs()
    for entry in graphs[:30]:
        g = entry[0]
        nf = g.node_features
        assert nf is not None and nf.numel() > 0, "node_features must not be empty"
        nt = g.metadata.get("node_types") if isinstance(g.metadata, dict) else None
        if nt is None: continue
        cand = candidate_node_mask(nt, NODE_TYPES)
        assert nf[cand].numel() > 0, "candidate nodes must have crop tensors"


def test_fusion_node_types_exist():
    """WBF/union/NMS node types must be present after graph build."""
    from od_graph_fusion.graph_builder import NODE_TYPES
    graphs = _load_graphs()
    found = {k: 0 for k in ["cluster", "consensus", "nms_candidate",
                              "soft_nms_candidate", "best_proposal_candidate"]}
    for entry in graphs[:20]:
        nt = entry[0].metadata.get("node_types") if isinstance(entry[0].metadata, dict) else None
        if nt is None: continue
        for k, v in NODE_TYPES.items():
            if k in found and (nt == v).any():
                found[k] += 1
    for k, cnt in found.items():
        assert cnt > 0, f"No graphs contain node type '{k}' in first 20 images"


def test_crop_tensor_shape_consistent():
    """All crop tensors must have shape [N, 3, H, W] with H=W."""
    graphs = _load_graphs()
    sizes = set()
    for entry in graphs[:20]:
        nf = entry[0].node_features
        if nf is None or nf.numel() == 0: continue
        assert nf.dim() == 4, f"Expected [N,3,H,W] but got {tuple(nf.shape)}"
        assert nf.shape[1] == 3, f"Expected 3 channels but got {nf.shape[1]}"
        assert nf.shape[2] == nf.shape[3], "Crop must be square"
        sizes.add(nf.shape[2])
    # All graphs should use the same crop size
    assert len(sizes) == 1, f"Mixed crop sizes in one run: {sizes}"


def test_selected_box_must_equal_a_graph_node_box():
    """After inference, selected_box must exactly match one of graph.node_box entries.

    This is a contract test: TGraphX selects, it does not generate/hallucinate.
    Uses the existing 64px run to verify the selector output is a real node box.
    """
    from od_graph_fusion.candidate_node_selector import (
        CandidateSelectorConfig, TGraphXCandidateNodeSelector, select_per_cluster,
    )
    from od_graph_fusion.candidate_mask import candidate_node_mask
    from od_graph_fusion.graph_builder import NODE_TYPES
    from od_graph_fusion.multi_seed_v2 import _attach_slot_metadata

    graphs = _load_graphs()
    entry = graphs[0]
    g, meta = entry[0], entry[1]
    import json
    manifest = json.loads((Path("runs/real_voc_car_v2") / "split_manifest.json").read_text())
    det_names = manifest["detector_names"]
    _attach_slot_metadata(g, meta, det_names)

    md = g.metadata.get("node_metadata")
    ea = g.edge_features
    cfg = CandidateSelectorConfig(
        num_classes=1, num_detectors=len(det_names), crop_size=64,
        hidden_dim=16, crop_channels=4, num_message_passing=1,
        metadata_dim=md.shape[1] if md is not None else None,
        edge_feat_dim=ea.shape[1] if ea is not None and ea.numel() > 0 else 14,
    )
    model = TGraphXCandidateNodeSelector(cfg).eval()
    with torch.no_grad():
        out = model(g, detector_names=det_names)
    cand = candidate_node_mask(meta.node_types, NODE_TYPES)
    cluster_of = meta.cluster_of_node
    nb = g.metadata["node_box"]
    nl = g.metadata.get("node_label", torch.zeros(nb.shape[0], dtype=torch.long))
    picked = select_per_cluster(out, cluster_of=cluster_of, cand_mask=cand,
                                  node_box=nb, node_label=nl, score_head="p_tp50")
    if picked["boxes_xyxy"].numel() == 0:
        return  # no clusters in this graph
    # Each picked box must exactly appear in node_box
    node_boxes = nb.cpu()
    for b in picked["boxes_xyxy"]:
        matches = ((node_boxes - b.unsqueeze(0)).abs().max(dim=1)[0] < 1e-4)
        assert matches.any(), f"Selected box {b.tolist()} does not match any node_box in graph"
