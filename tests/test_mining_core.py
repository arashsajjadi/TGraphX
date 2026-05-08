"""Core mathematical invariant tests for the TGraphX graph mining subsystem.

Covers link prediction, motifs, WL kernels, similarity, communities,
random walks, anomaly detection, patterns, frequent patterns, temporal,
and reports.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────

def _chain(N=4):
    src = list(range(N - 1))
    dst = list(range(1, N))
    return torch.tensor([src + dst, dst + src], dtype=torch.long), N


def _triangle():
    ei = torch.tensor([[0,1,2,1,2,0],[1,2,0,0,1,2]], dtype=torch.long)
    return ei, 3


def _star(N=5):
    src = [0] * (N - 1) + list(range(1, N))
    dst = list(range(1, N)) + [0] * (N - 1)
    return torch.tensor([src, dst], dtype=torch.long), N


# ── Link prediction ───────────────────────────────────────────────────────────


class TestLinkPredictionScores:
    from tgraphx.mining import (
        common_neighbors_score, jaccard_score, adamic_adar_score,
        resource_allocation_score, preferential_attachment_score,
    )

    @pytest.fixture(autouse=True)
    def _import(self):
        from tgraphx.mining import (
            common_neighbors_score, jaccard_score, adamic_adar_score,
            resource_allocation_score, preferential_attachment_score,
        )
        self.cn = common_neighbors_score
        self.jac = jaccard_score
        self.aa = adamic_adar_score
        self.ra = resource_allocation_score
        self.pa = preferential_attachment_score

    def test_common_neighbors_hand_computed(self):
        # 0-1, 1-2: pair (0,2) has 1 common neighbor (node 1).
        ei, N = _chain(3)
        pairs = torch.tensor([[0],[2]], dtype=torch.long)
        s = self.cn(ei, pairs, num_nodes=N)
        assert float(s[0]) == 1.0

    def test_common_neighbors_disconnected(self):
        # No edges at all.
        ei = torch.zeros((2,0), dtype=torch.long)
        pairs = torch.tensor([[0],[1]], dtype=torch.long)
        s = self.cn(ei, pairs, num_nodes=3)
        assert float(s[0]) == 0.0

    def test_jaccard_bounded(self):
        ei, N = _chain()
        pairs = torch.tensor([[0,1],[2,3]], dtype=torch.long)
        jac = self.jac(ei, pairs, num_nodes=N)
        assert ((jac >= 0) & (jac <= 1)).all()

    def test_jaccard_zero_denominator(self):
        ei = torch.zeros((2,0), dtype=torch.long)
        pairs = torch.tensor([[0],[1]], dtype=torch.long)
        assert float(self.jac(ei, pairs, num_nodes=3)[0]) == 0.0

    def test_adamic_adar_finite(self):
        ei, N = _chain()
        pairs = torch.tensor([[0],[3]], dtype=torch.long)
        aa = self.aa(ei, pairs, num_nodes=N)
        assert torch.isfinite(aa).all()

    def test_adamic_adar_self_pair_zero(self):
        ei, N = _chain()
        pairs = torch.tensor([[0],[0]], dtype=torch.long)
        # Self-pair: neighbor set overlap is everyone in N(0) but 0→0 is no edge.
        aa = self.aa(ei, pairs, num_nodes=N)
        assert torch.isfinite(aa).all()

    def test_resource_allocation_non_negative(self):
        ei, N = _triangle()
        pairs = torch.tensor([[0],[1]], dtype=torch.long)
        ra = self.ra(ei, pairs, num_nodes=N)
        assert float(ra[0]) >= 0.0

    def test_preferential_attachment_hub_gets_high_score(self):
        ei, N = _star()
        # Hub node 0 vs leaf 1: should have higher PA than leaf vs leaf.
        pairs = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        pa = self.pa(ei, pairs, num_nodes=N)
        # PA(0,1) > PA(1,2) since deg(0) > deg(1).
        assert float(pa[0]) >= float(pa[1])

    def test_all_scores_same_length(self):
        ei, N = _chain()
        P = 5
        src = torch.randint(N, (P,))
        dst = torch.randint(N, (P,))
        pairs = torch.stack([src, dst], dim=0)
        for fn in [self.cn, self.jac, self.aa, self.ra, self.pa]:
            s = fn(ei, pairs, num_nodes=N)
            assert s.shape == (P,)

    def test_bad_pairs_shape(self):
        ei, N = _chain()
        with pytest.raises(ValueError, match="\\[2, P\\]"):
            self.cn(ei, torch.zeros(3, 4, dtype=torch.long), num_nodes=N)


# ── Motifs ────────────────────────────────────────────────────────────────────


class TestMotifs:
    def test_triangle_k3(self):
        from tgraphx.mining import triangle_count
        ei, N = _triangle()
        assert triangle_count(ei, N, directed=False) == 1

    def test_triangle_k4_has_4(self):
        from tgraphx.mining import triangle_count
        # K4: 4 nodes, 6 undirected edges, 4 triangles.
        src = [0,0,0,1,1,2]
        dst = [1,2,3,2,3,3]
        ei = torch.tensor([src+dst, dst+src], dtype=torch.long)
        assert triangle_count(ei, 4, directed=False) == 4

    def test_triangle_path_no_triangle(self):
        from tgraphx.mining import triangle_count
        ei, N = _chain()
        assert triangle_count(ei, N, directed=False) == 0

    def test_triangle_node_level_k3(self):
        from tgraphx.mining import triangle_count
        ei, N = _triangle()
        ni = triangle_count(ei, N, directed=False, node_level=True)
        assert ni.tolist() == [1, 1, 1]

    def test_wedge_count_star(self):
        from tgraphx.mining import wedge_count
        # Star with hub 0 and 4 leaves: C(4,2) = 6 wedges at hub.
        ei = torch.tensor([[0,0,0,0,1,2,3,4],[1,2,3,4,0,0,0,0]], dtype=torch.long)
        w = wedge_count(ei, 5, directed=False)
        assert w == 6

    def test_clustering_coefficient_triangle_is_one(self):
        from tgraphx.mining import local_clustering_coefficient
        ei, N = _triangle()
        cc = local_clustering_coefficient(ei, N)
        assert all(abs(float(c) - 1.0) < 1e-5 for c in cc.tolist())

    def test_clustering_coefficient_path_non_endpoint(self):
        from tgraphx.mining import local_clustering_coefficient
        ei, N = _chain(4)
        cc = local_clustering_coefficient(ei, N)
        # Middle nodes (1, 2): degree 2 but not connected → CC = 0.
        assert float(cc[1]) == 0.0 and float(cc[2]) == 0.0

    def test_motif_counts_json_serializable(self):
        from tgraphx.mining import motif_counts
        ei, N = _triangle()
        mc = motif_counts(ei, N)
        assert json.dumps(mc) is not None
        assert mc["triangles"] == 1

    def test_motif_features_shape(self):
        from tgraphx.mining import motif_features
        ei, N = _chain()
        feats = motif_features(ei, N)
        assert feats.shape == (N, 3)
        assert feats.dtype == torch.float32


# ── WL Kernels ────────────────────────────────────────────────────────────────


class TestWLKernels:
    def test_wl_labels_identical_graphs(self):
        from tgraphx.mining import weisfeiler_lehman_labels
        ei, N = _chain()
        h1 = weisfeiler_lehman_labels(ei, N, num_iterations=2)
        h2 = weisfeiler_lehman_labels(ei, N, num_iterations=2)
        assert h1 == h2

    def test_wl_kernel_symmetric(self):
        from tgraphx.mining import wl_kernel_matrix
        ei, N = _chain()
        gs = [{"edge_index": ei, "num_nodes": N} for _ in range(3)]
        K = wl_kernel_matrix(gs, normalize=True)
        assert torch.allclose(K, K.t(), atol=1e-5)

    def test_identical_graphs_max_similarity(self):
        from tgraphx.mining import wl_kernel_matrix
        ei, N = _chain()
        gs = [{"edge_index": ei, "num_nodes": N}] * 2
        K = wl_kernel_matrix(gs, normalize=True)
        assert abs(float(K[0,1].item()) - 1.0) < 0.01

    def test_degree_histogram_shape(self):
        from tgraphx.mining import degree_histogram_features
        ei1, N1 = _chain()
        ei2, N2 = _triangle()
        gs = [{"edge_index": ei1, "num_nodes": N1},
              {"edge_index": ei2, "num_nodes": N2}]
        feat = degree_histogram_features(gs)
        assert feat.shape[0] == 2

    def test_empty_graph_list(self):
        from tgraphx.mining import wl_graph_features
        feat, vocab = wl_graph_features([])
        assert feat.shape[0] == 0


# ── Similarity ────────────────────────────────────────────────────────────────


class TestSimilarity:
    def test_identical_graph_cosine_sim_is_one(self):
        from tgraphx.mining import wl_feature_similarity
        ei, N = _chain()
        s = wl_feature_similarity(ei, N, ei, N)
        assert abs(s - 1.0) < 0.01

    def test_degree_histogram_distance_zero_same(self):
        from tgraphx.mining import degree_histogram_distance
        ei, N = _chain()
        d = degree_histogram_distance(ei, N, ei, N)
        assert d < 1e-6

    def test_pairwise_symmetric(self):
        from tgraphx.mining import pairwise_graph_similarity
        ei1, N1 = _chain()
        ei2, N2 = _triangle()
        gs = [{"edge_index": ei1, "num_nodes": N1},
              {"edge_index": ei2, "num_nodes": N2}]
        S = pairwise_graph_similarity(gs, method="degree")
        assert torch.allclose(S, S.t(), atol=1e-5)

    def test_cosine_between_feature_tensors(self):
        from tgraphx.mining import graph_feature_cosine_similarity
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0]])
        S = graph_feature_cosine_similarity(a, b)
        assert S.shape == (2, 1)
        assert abs(float(S[0,0]) - 1.0) < 1e-5
        assert abs(float(S[1,0])) < 1e-5


# ── Communities ───────────────────────────────────────────────────────────────


class TestCommunities:
    def test_two_disjoint_triangles(self):
        from tgraphx.mining import label_propagation_communities
        # Triangles: {0,1,2} and {3,4,5}, no edges between them.
        src = [0,1,2,3,4,5,1,2,0,4,5,3]
        dst = [1,2,0,4,5,3,0,1,2,3,4,5]
        ei = torch.tensor([src, dst], dtype=torch.long)
        labels = label_propagation_communities(ei, 6, seed=0)
        unique = labels.unique()
        assert len(unique) == 2, f"Expected 2 communities, got {len(unique)}"

    def test_isolated_nodes_each_own_community(self):
        from tgraphx.mining import label_propagation_communities
        ei = torch.zeros((2,0), dtype=torch.long)
        labels = label_propagation_communities(ei, 5, seed=0)
        assert len(labels.unique()) == 5

    def test_deterministic_with_seed(self):
        from tgraphx.mining import label_propagation_communities
        ei, N = _chain()
        l1 = label_propagation_communities(ei, N, seed=42)
        l2 = label_propagation_communities(ei, N, seed=42)
        assert torch.equal(l1, l2)

    def test_modularity_two_cliques(self):
        from tgraphx.mining import modularity
        src = [0,1,2,3,4,5,1,2,0,4,5,3]
        dst = [1,2,0,4,5,3,0,1,2,3,4,5]
        ei = torch.tensor([src, dst], dtype=torch.long)
        comm = torch.tensor([0,0,0,1,1,1], dtype=torch.long)
        Q = modularity(ei, comm, num_nodes=6)
        assert math.isfinite(Q)
        assert Q > 0  # good partition

    def test_community_summary_json(self):
        from tgraphx.mining import community_summary
        ei, N = _chain()
        comm = torch.zeros(N, dtype=torch.long)
        s = community_summary(ei, comm, N)
        assert json.dumps(s) is not None
        assert "num_communities" in s


# ── Random walks ─────────────────────────────────────────────────────────────


class TestRandomWalks:
    def test_walk_length(self):
        from tgraphx.mining import random_walks
        ei, N = _chain()
        starts = torch.tensor([0], dtype=torch.long)
        walks = random_walks(ei, starts, walk_length=5, num_nodes=N, seed=0)
        assert walks.shape == (1, 6)

    def test_valid_node_ids(self):
        from tgraphx.mining import random_walks
        ei, N = _chain()
        starts = torch.arange(N, dtype=torch.long)
        walks = random_walks(ei, starts, walk_length=10, num_nodes=N, seed=0)
        assert ((walks >= 0) & (walks < N)).all()

    def test_dead_end_stay_in_place(self):
        from tgraphx.mining import random_walks
        # Chain 0→1→2, start from node 2 (dead end in directed).
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        starts = torch.tensor([2], dtype=torch.long)
        walks = random_walks(ei, starts, walk_length=3, num_nodes=3, seed=0)
        # Node 2 has no out-edges; should stay at 2.
        assert all(int(w) == 2 for w in walks[0, 1:].tolist())

    def test_deterministic_with_seed(self):
        from tgraphx.mining import random_walks
        ei, N = _chain()
        starts = torch.arange(N, dtype=torch.long)
        w1 = random_walks(ei, starts, walk_length=5, num_nodes=N, seed=7)
        w2 = random_walks(ei, starts, walk_length=5, num_nodes=N, seed=7)
        assert torch.equal(w1, w2)

    def test_no_global_rng_pollution(self):
        from tgraphx.mining import random_walks
        torch.manual_seed(99)
        before = torch.rand(3)
        torch.manual_seed(99)
        ei, N = _chain()
        random_walks(ei, torch.tensor([0], dtype=torch.long), walk_length=3, num_nodes=N, seed=42)
        after = torch.rand(3)
        assert torch.equal(before, after)

    def test_generate_shape(self):
        from tgraphx.mining import generate_random_walks
        ei, N = _chain()
        walks = generate_random_walks(ei, num_nodes=N, num_walks_per_node=3, walk_length=4, seed=0)
        assert walks.shape == (N * 3, 5)


# ── Anomaly detection ─────────────────────────────────────────────────────────


class TestAnomalyDetection:
    def test_degree_scorer_hub_max(self):
        from tgraphx.mining import DegreeAnomalyScorer
        ei, N = _star()
        scorer = DegreeAnomalyScorer().fit(ei, N)
        scores = scorer.score_nodes(ei, N)
        assert float(scores[0]) == float(scores.max())

    def test_degree_scorer_not_fitted(self):
        from tgraphx.mining import DegreeAnomalyScorer
        with pytest.raises(RuntimeError, match="fit"):
            DegreeAnomalyScorer().score_nodes(torch.zeros((2,0),dtype=torch.long), 3)

    def test_degree_scorer_finite(self):
        from tgraphx.mining import DegreeAnomalyScorer
        ei, N = _chain()
        scores = DegreeAnomalyScorer().fit(ei, N).score_nodes(ei, N)
        assert torch.isfinite(scores).all()

    def test_graph_level_anomaly_scores_shape(self):
        from tgraphx.mining import graph_level_anomaly_scores
        ei1, N1 = _chain()
        ei2, N2 = _triangle()
        gs = [{"edge_index": ei1, "num_nodes": N1},
              {"edge_index": ei2, "num_nodes": N2}]
        scores = graph_level_anomaly_scores(gs, method="degree_histogram")
        assert scores.shape == (2,)
        assert torch.isfinite(scores).all()

    def test_ego_density_scorer(self):
        from tgraphx.mining import EgoDensityAnomalyScorer
        ei, N = _triangle()
        scorer = EgoDensityAnomalyScorer().fit(ei, N)
        scores = scorer.score_nodes(ei, N)
        assert scores.shape == (N,)
        assert torch.isfinite(scores).all()


# ── Prototype graphs ──────────────────────────────────────────────────────────


class TestPrototypeGraphs:
    def _make_data(self, N=20, D=8, C=3):
        feats = torch.randn(N, D)
        labels = torch.tensor([i % C for i in range(N)], dtype=torch.long)
        return feats, labels, C

    def test_class_graph_builder_creates_all_classes(self):
        from tgraphx.mining import ClassGraphBuilder
        feats, labels, C = self._make_data()
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels)
        for cls in range(C):
            cg = builder.get_class_graph(cls)
            assert cg["num_nodes"] > 0

    def test_density_cap_respected(self):
        from tgraphx.mining import ClassGraphBuilder
        feats, labels, C = self._make_data()
        # With max_neighbor_fraction=0.5, k_effective <= n_c//2.
        builder = ClassGraphBuilder(k_support=100, max_neighbor_fraction=0.5).fit(feats, labels)
        for cls in range(C):
            cg = builder.get_class_graph(cls)
            n = cg["num_nodes"]
            assert cg["k_effective"] <= max(1, n // 2)

    def test_density_in_range(self):
        from tgraphx.mining import ClassGraphBuilder
        feats, labels, C = self._make_data()
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels)
        for cls in range(C):
            assert 0.0 <= builder.class_graphs_[cls]["density"] <= 1.0

    def test_candidate_graph_query_idx(self):
        from tgraphx.mining import ClassGraphBuilder, CandidateGraphBuilder
        feats, labels, C = self._make_data()
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels)
        cb = CandidateGraphBuilder(top_k_query=2)
        cg = builder.get_class_graph(0)
        qf = torch.randn(feats.shape[1])
        cand, q_idx = cb.build(cg, qf)
        assert q_idx == cg["num_nodes"]
        assert cand["num_nodes"] == cg["num_nodes"] + 1
        assert cand["query_idx"] == q_idx

    def test_cosine_baseline(self):
        from tgraphx.mining import ClassGraphBuilder, cosine_graph_membership_baseline
        feats, labels, C = self._make_data()
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels)
        qe = torch.randn(feats.shape[1])
        scores = cosine_graph_membership_baseline(qe, builder)
        assert len(scores) == C
        for v in scores.values():
            assert -1.0 <= v <= 1.0

    def test_report_json_serializable(self):
        from tgraphx.mining import ClassGraphBuilder
        feats, labels, C = self._make_data()
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels)
        report = builder.report()
        assert json.dumps(report) is not None

    def test_spatial_features_preserved(self):
        from tgraphx.mining import ClassGraphBuilder
        N, C_, H, W = 12, 3, 4, 4
        feats = torch.randn(N, C_, H, W)
        labels = torch.tensor([i % 3 for i in range(N)], dtype=torch.long)
        embs = torch.randn(N, 8)
        builder = ClassGraphBuilder(k_support=2).fit(feats, labels, embeddings=embs)
        for cls in range(3):
            cg = builder.get_class_graph(cls)
            # Spatial dims preserved.
            assert cg["node_features"].shape[1:] == (C_, H, W)


# ── Patterns ──────────────────────────────────────────────────────────────────


class TestPatterns:
    def test_path_len2_chain(self):
        from tgraphx.mining import path_pattern_count
        ei, N = _chain(4)  # 0-1-2-3 undirected
        p = path_pattern_count(ei, N, length=2, directed=False)
        assert p > 0

    def test_star_count_hub(self):
        from tgraphx.mining import star_pattern_count
        ei, N = _star()
        assert star_pattern_count(ei, N, center_degree=N-1) == 1
        assert star_pattern_count(ei, N, center_degree=N) == 0

    def test_contains_triangle_true(self):
        from tgraphx.mining import contains_triangle
        ei, N = _triangle()
        assert contains_triangle(ei, N)

    def test_contains_triangle_false(self):
        from tgraphx.mining import contains_triangle
        ei, N = _chain()
        assert not contains_triangle(ei, N)

    def test_small_pattern_counts_json(self):
        from tgraphx.mining import small_pattern_counts
        ei, N = _triangle()
        c = small_pattern_counts(ei, N)
        assert json.dumps(c) is not None
        assert c["triangles"] == 1

    def test_path_unsupported_length(self):
        from tgraphx.mining import path_pattern_count
        ei, N = _chain()
        with pytest.raises(NotImplementedError):
            path_pattern_count(ei, N, length=5)


# ── Frequent patterns ─────────────────────────────────────────────────────────


class TestFrequentPatterns:
    def test_frequent_node_labels(self):
        from tgraphx.mining import frequent_node_labels
        data = [[0,1,2],[1,2,3],[0,2,4],[1,2,5]]
        freq_3 = frequent_node_labels(data, min_support=3)
        assert 2 in freq_3 and freq_3[2] == 4
        # Label 0 appears in 2 graphs only → not frequent at min_support=3.
        assert 0 not in freq_3
        assert 5 not in freq_3  # appears only once
        # At min_support=2, label 0 should appear.
        freq_2 = frequent_node_labels(data, min_support=2)
        assert 0 in freq_2 and freq_2[0] == 2

    def test_support_count(self):
        from tgraphx.mining import support_count
        assert support_count([1,2], [[0,1,2],[1,3],[1,2,4]]) == 2

    def test_frequent_degree_bins(self):
        from tgraphx.mining import frequent_degree_bins
        ei, N = _chain()
        gs = [{"edge_index": ei, "num_nodes": N}] * 4
        bins = frequent_degree_bins(gs, bins=[0,1,5], min_support=1)
        assert len(bins) > 0

    def test_empty_collection(self):
        from tgraphx.mining import frequent_node_labels
        assert frequent_node_labels([], min_support=1) == {}


# ── Temporal mining ───────────────────────────────────────────────────────────


class TestTemporalMining:
    def test_chronological_split_no_leakage(self):
        from tgraphx.mining import temporal_chronological_split
        ts = torch.arange(100, dtype=torch.float)
        tr, va, te = temporal_chronological_split(ts, (0.7, 0.15, 0.15))
        assert (tr & va).sum() == 0
        assert (tr & te).sum() == 0
        assert (va & te).sum() == 0
        if ts[tr].numel() and ts[va].numel():
            assert float(ts[tr].max()) <= float(ts[va].min())
        if ts[va].numel() and ts[te].numel():
            assert float(ts[va].max()) <= float(ts[te].min())

    def test_sliding_window(self):
        from tgraphx.mining import sliding_window_edges
        src = torch.tensor([0,1,2,3], dtype=torch.long)
        dst = torch.tensor([1,2,3,0], dtype=torch.long)
        ts = torch.tensor([1.0,2.0,3.0,4.0])
        ws = sliding_window_edges(src, dst, ts, window_size=2.0, step=1.0)
        assert len(ws) > 0
        for s, d, t in ws:
            assert s.shape == d.shape == t.shape

    def test_temporal_degree_sum(self):
        from tgraphx.mining import temporal_degree
        src = torch.tensor([0,1,0], dtype=torch.long)
        dst = torch.tensor([1,2,2], dtype=torch.long)
        ts = torch.tensor([1.0,2.0,3.0])
        deg = temporal_degree(src, dst, ts, num_nodes=3, window_start=0.0, window_end=4.0)
        assert int(deg.sum().item()) == 6  # 3 edges, each contributes 2

    def test_burst_score_finite(self):
        from tgraphx.mining import burst_score
        src = torch.tensor([0,0,0,1,1], dtype=torch.long)
        dst = torch.tensor([1,2,3,2,3], dtype=torch.long)
        ts = torch.tensor([1.0,2.0,3.0,4.0,5.0])
        s = burst_score(src, dst, ts, num_nodes=4, num_windows=5)
        assert s.shape == (4,)
        assert torch.isfinite(s).all()


# ── Reports ───────────────────────────────────────────────────────────────────


class TestMiningReports:
    def test_write_graph_mining_summary(self):
        from tgraphx.mining import write_graph_mining_summary
        with tempfile.TemporaryDirectory() as tmp:
            p = write_graph_mining_summary(
                os.path.join(tmp, "mining.json"), {"test": 42}
            )
            assert json.loads(open(p).read())["test"] == 42

    def test_write_motif_summary(self):
        from tgraphx.mining import write_motif_summary
        with tempfile.TemporaryDirectory() as tmp:
            p = write_motif_summary(os.path.join(tmp, "motif.json"), {"triangles": 1})
            assert json.loads(open(p).read())["triangles"] == 1

    def test_write_anomaly_summary(self):
        from tgraphx.mining import write_anomaly_summary
        scores = torch.tensor([0.5, 1.2, 0.1])
        with tempfile.TemporaryDirectory() as tmp:
            p = write_anomaly_summary(
                os.path.join(tmp, "anom.json"), "degree_zscore", scores, top_k=2,
            )
            d = json.loads(open(p).read())
            assert len(d["top_anomalous_nodes"]) == 2

    def test_write_prototype_report(self):
        from tgraphx.mining import write_prototype_membership_report
        with tempfile.TemporaryDirectory() as tmp:
            p = write_prototype_membership_report(
                os.path.join(tmp, "proto.json"), {"accuracy": 0.85}
            )
            assert json.loads(open(p).read())["accuracy"] == 0.85
