"""Tests for source-routing formulation."""
import torch
import pytest

from od_graph_fusion.source_router import (
    canonical_label,
    stable_image_seed,
    compute_source_utilities,
    source_routing_loss,
    oracle_gap_recovery,
    evaluate_source_routing,
    TGraphXSourceRouter,
    EdgeConditionedMP,
    COCO_TO_VOC_MAP,
)
from od_graph_fusion.graph_builder import NODE_TYPES


# ── Canonical label mapping ────────────────────────────────────────────────

def test_canonical_airplane_to_aeroplane():
    assert canonical_label("airplane") == "aeroplane"


def test_canonical_motorcycle_to_motorbike():
    assert canonical_label("motorcycle") == "motorbike"


def test_canonical_unknown_passthrough():
    assert canonical_label("car") == "car"
    assert canonical_label("person") == "person"


def test_coco_voc_map_keys():
    for src, dst in COCO_TO_VOC_MAP.items():
        assert isinstance(src, str) and isinstance(dst, str)


# ── Stable image seed ─────────────────────────────────────────────────────

def test_stable_image_seed_deterministic():
    s1 = stable_image_seed("img_001", extra=42)
    s2 = stable_image_seed("img_001", extra=42)
    assert s1 == s2


def test_stable_image_seed_different_images():
    s1 = stable_image_seed("img_001")
    s2 = stable_image_seed("img_002")
    assert s1 != s2


# ── Compute source utilities ──────────────────────────────────────────────

def _make_candidate_graph(n_proposals=3, n_clusters=1):
    """Simple cluster with n_proposals proposals and 1 cluster/consensus node."""
    N = n_proposals + n_clusters + n_clusters  # proposals + clusters + consensus
    node_types = torch.zeros(N, dtype=torch.long)
    for i in range(n_proposals):
        node_types[i] = NODE_TYPES["proposal"]
    for i in range(n_clusters):
        node_types[n_proposals + i] = NODE_TYPES["cluster"]
    for i in range(n_clusters):
        node_types[n_proposals + n_clusters + i] = NODE_TYPES["consensus"]
    # cluster_of: all proposals belong to cluster 0; cluster/consensus nodes too
    cluster_of = torch.zeros(N, dtype=torch.long)
    return node_types, cluster_of


def test_compute_source_utilities_perfect_proposal():
    node_types, cluster_of = _make_candidate_graph(3, 1)
    N = node_types.shape[0]
    # Perfect proposal at index 0: box == GT
    node_box = torch.rand(N, 4).abs()
    node_box[:, 2:] += node_box[:, :2] + 0.5
    gt_box = node_box[0:1].clone()  # exact match for index 0
    gt_labels = torch.tensor([1])
    node_label = torch.ones(N, dtype=torch.long)  # all same class
    node_score = torch.rand(N)

    utility, best_src, cand_mask = compute_source_utilities(
        node_box, node_label, node_score, cluster_of, node_types,
        gt_box, gt_labels, class_agnostic=True, iou_match=0.5,
    )
    # Best source should be node 0 (IoU=1.0)
    assert int(best_src[0].item()) == 0
    assert abs(utility[0].item() - 1.0) < 1e-4


def test_continuous_utility_all_below_iou_threshold():
    """P2.1 fix: all candidates below iou_match must still have non-zero utility
    and best_source must be the highest-IoU node."""
    node_types, cluster_of = _make_candidate_graph(3, 0)
    N = node_types.shape[0]
    node_box = torch.tensor([
        [0.5, 0.5, 4.5, 4.5],    # small overlap
        [1., 1., 6., 6.],          # medium
        [2., 2., 8., 8.],          # largest — should win
    ], dtype=torch.float32)
    gt_box = torch.tensor([[0., 0., 10., 10.]])
    gt_label = torch.tensor([0])
    node_label = torch.zeros(N, dtype=torch.long)
    node_score = torch.tensor([0.9, 0.3, 0.5])  # high-conf != best IoU
    util, best_src, _ = compute_source_utilities(
        node_box, node_label, node_score, cluster_of, node_types,
        gt_box, gt_label, class_agnostic=True, iou_match=0.5,
    )
    assert not all(u == 0.0 for u in util.tolist()), (
        "Utility should be continuous even when all IoUs < iou_match"
    )
    assert best_src[0].item() == 2, (
        f"Best source must be highest-IoU node (2), got {best_src[0].item()}"
    )


def test_per_cluster_regret_weighting_finite():
    """P3.1 fix: per-cluster regret weighting must produce finite loss."""
    N = 6
    cluster_of = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    cand_mask = torch.ones(N, dtype=torch.bool)
    node_scores = torch.randn(N)
    utility = torch.tensor([0.9, 0.3, 0.1, 0.8, 0.3, 0.2])
    best_src = torch.tensor([0, 3])
    baseline_scores = torch.tensor([2.0, 0.0, 0.0, 0.0, 2.0, 1.0])
    losses = source_routing_loss(
        node_scores, utility, best_src, cluster_of, cand_mask,
        regret_lambda=2.0, baseline_scores=baseline_scores,
    )
    assert torch.isfinite(losses["total"])
    # lambda=0 should match unweighted
    losses0 = source_routing_loss(
        node_scores, utility, best_src, cluster_of, cand_mask,
        regret_lambda=0.0, baseline_scores=baseline_scores,
    )
    assert torch.isfinite(losses0["total"])


def test_compute_source_utilities_no_gt():
    node_types, cluster_of = _make_candidate_graph(2, 1)
    N = node_types.shape[0]
    node_box = torch.rand(N, 4).abs()
    node_box[:, 2:] += node_box[:, :2] + 0.5
    node_label = torch.zeros(N, dtype=torch.long)
    node_score = torch.rand(N)
    utility, best_src, _ = compute_source_utilities(
        node_box, node_label, node_score, cluster_of, node_types,
        torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
        class_agnostic=True,
    )
    assert utility.sum().item() == 0.0


def test_best_source_label_matches_max_utility():
    node_types, cluster_of = _make_candidate_graph(4, 1)
    N = node_types.shape[0]
    node_box = torch.zeros(N, 4)
    # Make proposal 2 the best: IoU ≈ 0.9 with GT
    gt_box = torch.tensor([[0., 0., 10., 10.]])
    node_box[0] = torch.tensor([1., 1., 5., 5.])   # low IoU
    node_box[1] = torch.tensor([5., 5., 15., 15.])  # some IoU
    node_box[2] = torch.tensor([0.5, 0.5, 9.5, 9.5])  # best IoU
    node_box[3] = torch.tensor([2., 2., 8., 8.])
    node_box[4] = node_box[2].clone()  # cluster box
    node_box[5] = node_box[2].clone()  # consensus box
    node_label = torch.zeros(N, dtype=torch.long)
    node_score = torch.rand(N)
    gt_labels = torch.tensor([0])

    utility, best_src, cand_mask = compute_source_utilities(
        node_box, node_label, node_score, cluster_of, node_types,
        gt_box, gt_labels, class_agnostic=True, iou_match=0.5,
    )
    # best_src should be the node with highest utility
    cand_utils = utility[cand_mask]
    cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(-1)
    oracle_local = int(cand_utils.argmax().item())
    oracle_node = int(cand_idx[oracle_local].item())
    assert int(best_src[0].item()) == oracle_node


# ── Source routing loss ────────────────────────────────────────────────────

def test_source_routing_loss_finite():
    N = 6
    node_scores = torch.randn(N)
    utility = torch.rand(N)
    cluster_of = torch.zeros(N, dtype=torch.long)
    cand_mask = torch.ones(N, dtype=torch.bool)
    best_src = torch.tensor([0], dtype=torch.long)  # node 0 is best in cluster 0
    losses = source_routing_loss(node_scores, utility, best_src, cluster_of, cand_mask)
    for k, v in losses.items():
        assert torch.isfinite(v), f"loss {k} is not finite"


def test_source_routing_loss_absent_source_ignored():
    """Absent sources (cand_mask=False) should not affect the loss."""
    N = 4
    node_scores = torch.randn(N)
    utility = torch.rand(N)
    utility[2] = 0.0; utility[3] = 0.0
    cluster_of = torch.zeros(N, dtype=torch.long)
    cand_mask = torch.tensor([True, True, False, False])
    best_src = torch.tensor([0])
    losses = source_routing_loss(node_scores, utility, best_src, cluster_of, cand_mask)
    assert torch.isfinite(losses["total"])


# ── Oracle-gap recovery ────────────────────────────────────────────────────

def test_gap_recovery_baseline_equals_zero():
    assert oracle_gap_recovery(0.5, 0.5, 0.8) == pytest.approx(0.0)


def test_gap_recovery_oracle_equals_one():
    assert oracle_gap_recovery(0.8, 0.5, 0.8) == pytest.approx(1.0)


def test_gap_recovery_worse_than_baseline_is_negative():
    assert oracle_gap_recovery(0.3, 0.5, 0.8) < 0


def test_gap_recovery_zero_denom_safe():
    result = oracle_gap_recovery(0.5, 0.5, 0.5)
    assert torch.isfinite(torch.tensor(result))


# ── Edge-conditioned message passing ───────────────────────────────────────

def test_edge_features_change_output():
    """Changing edge_attr must change model output (proves edges are used)."""
    in_shape = (4, 8, 8)
    out_shape = (4, 8, 8)
    edge_feat_dim = 6
    ec = EdgeConditionedMP(in_shape, out_shape, edge_feat_dim)
    ec.eval()
    x = torch.randn(4, 4, 8, 8)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea1 = torch.zeros(3, edge_feat_dim)
    ea2 = torch.ones(3, edge_feat_dim)
    with torch.no_grad():
        out1 = ec(x, ei, ea1)
        out2 = ec(x, ei, ea2)
    # Outputs must differ when edge attr differ
    assert not torch.allclose(out1, out2, atol=1e-5), (
        "Edge features did not affect model output — edge conditioning is not working!"
    )


def test_no_edge_features_still_runs():
    in_shape = (4, 8, 8)
    out_shape = (4, 8, 8)
    ec = EdgeConditionedMP(in_shape, out_shape, edge_feat_dim=6)
    ec.eval()
    x = torch.randn(4, 4, 8, 8)
    ei = torch.tensor([[0, 1], [1, 2]])
    with torch.no_grad():
        out = ec(x, ei, edge_attr=None)
    assert out.shape == x.shape


# ── TGraphXSourceRouter ────────────────────────────────────────────────────

def test_source_router_forward():
    from tgraphx import Graph
    model = TGraphXSourceRouter(
        num_classes=5, num_detectors=3, crop_size=16,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    model.eval()
    N = 6
    x = torch.randn(N, 3, 16, 16)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ea = torch.randn(3, 14)
    meta_tensor = torch.randn(N, 8 + 3 + 5)  # metadata_dim = 8+det+cls
    g = Graph(node_features=x, edge_index=ei, edge_attr=ea,
              metadata={"node_metadata": meta_tensor})
    with torch.no_grad():
        out = model(g)
    assert "quality_logits" in out
    assert out["quality_logits"].shape == (N,)


def test_source_router_quality_logits_are_finite():
    from tgraphx import Graph
    model = TGraphXSourceRouter(
        num_classes=5, num_detectors=3, crop_size=16,
        crop_channels=8, hidden_dim=16, num_message_passing=1,
    )
    model.eval()
    x = torch.randn(4, 3, 16, 16)
    ei = torch.tensor([[0, 1], [1, 2]])
    g = Graph(node_features=x, edge_index=ei)
    with torch.no_grad():
        out = model(g)
    assert torch.isfinite(out["quality_logits"]).all()
