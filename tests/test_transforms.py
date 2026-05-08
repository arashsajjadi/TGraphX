"""Transform tests (v0.2.9)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import Graph
from tgraphx.transforms import (
    AddAdjacencyBias,
    AddConstantFeatures,
    AddDegreeEncoding,
    AddDegreeFeatures,
    AddLaplacianEigenvectors,
    AddSelfLoops,
    BuildGridGraph,
    CoalesceEdges,
    Compose,
    DropEdges,
    FeatureNoise,
    FixedSplit,
    LambdaTransform,
    NodeFeatureMask,
    NormalizeEdgeFeatures,
    NormalizeFeatures,
    PatchifyImage,
    RandomApply,
    RandomGraphSplit,
    RandomLinkSplit,
    RandomNodeSplit,
    RemoveSelfLoops,
    StandardizeFeatures,
    ToUndirected,
)


def _g(N=5, D=4, E=6, edge_weight=False, seed=0):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=rng)
    src = torch.randint(0, N, (E,), generator=rng)
    dst = torch.randint(0, N, (E,), generator=rng)
    ew = torch.rand(E, generator=rng) if edge_weight else None
    return Graph(x, torch.stack([src, dst]).long(), edge_weight=ew)


# ── compose ──────────────────────────────────────────────────────────────────


class TestCompose:
    def test_compose_runs_in_order(self):
        order = []

        def make(name):
            def fn(g):
                order.append(name)
                return g
            return LambdaTransform(fn)

        Compose([make("a"), make("b"), make("c")])(_g())
        assert order == ["a", "b", "c"]

    def test_random_apply_p_zero(self):
        called = {"n": 0}

        def fn(g):
            called["n"] += 1
            return g

        RandomApply(LambdaTransform(fn), p=0.0, seed=0)(_g())
        assert called["n"] == 0

    def test_random_apply_p_one(self):
        called = {"n": 0}

        def fn(g):
            called["n"] += 1
            return g

        RandomApply(LambdaTransform(fn), p=1.0, seed=0)(_g())
        assert called["n"] == 1


# ── structure ────────────────────────────────────────────────────────────────


class TestStructure:
    def test_add_self_loops_idempotent(self):
        g = _g()
        out1 = AddSelfLoops()(g)
        out2 = AddSelfLoops()(out1)
        assert (out1.edge_index[0] == out1.edge_index[1]).sum() == g.num_nodes
        assert (out2.edge_index[0] == out2.edge_index[1]).sum() == g.num_nodes

    def test_remove_self_loops(self):
        g = _g(edge_weight=True)
        ws = AddSelfLoops()(g)
        out = RemoveSelfLoops()(ws)
        assert (out.edge_index[0] == out.edge_index[1]).sum() == 0
        if out.edge_weight is not None:
            assert out.edge_weight.numel() == out.num_edges

    def test_to_undirected_has_reverse(self):
        g = Graph(torch.randn(4, 2), torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        out = ToUndirected()(g)
        # Each forward edge has a reverse counterpart.
        forward = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        present = set(zip(out.edge_index[0].tolist(), out.edge_index[1].tolist()))
        for s, d in forward:
            assert (d, s) in present

    def test_drop_edges_deterministic(self):
        g = _g(edge_weight=True)
        a = DropEdges(p=0.3, seed=7)(g)
        b = DropEdges(p=0.3, seed=7)(g)
        assert torch.equal(a.edge_index, b.edge_index)


# ── features ─────────────────────────────────────────────────────────────────


class TestFeatures:
    def test_normalize_features_l1_rows_sum(self):
        x = torch.tensor([[2.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
        g = Graph(x, torch.zeros((2, 0), dtype=torch.long))
        out = NormalizeFeatures(ord=1)(g)
        # Row 0: sum should be 1
        assert torch.isclose(out.node_features[0].abs().sum(), torch.tensor(1.0))
        # Row 1: zero stays zero
        assert torch.equal(out.node_features[1], torch.zeros(2))

    def test_standardize_features(self):
        torch.manual_seed(0)
        g = _g(N=20, D=4)
        out = StandardizeFeatures()(g)
        # Per-feature mean ≈ 0, std ≈ 1 on output.
        assert torch.allclose(out.node_features.mean(dim=0), torch.zeros(4), atol=1e-5)
        assert torch.allclose(out.node_features.std(dim=0), torch.ones(4), atol=1e-5)

    def test_add_degree_features_matches_count(self):
        g = Graph(
            torch.zeros(4, 1),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        )
        out = AddDegreeFeatures("both", normalize=False)(g)
        # in-degree, out-degree appended
        assert out.node_features.shape == (4, 1 + 2)
        # node 1: in=1, out=1; node 0: in=0, out=1
        assert torch.equal(out.node_features[:, 1], torch.tensor([0.0, 1.0, 1.0, 1.0]))
        assert torch.equal(out.node_features[:, 2], torch.tensor([1.0, 1.0, 1.0, 0.0]))

    def test_add_constant_features(self):
        g = _g()
        out = AddConstantFeatures(value=1.0, num_features=2)(g)
        assert out.node_features.shape == (g.num_nodes, g.node_features.shape[1] + 2)

    def test_feature_noise_deterministic(self):
        g = _g()
        a = FeatureNoise(0.1, seed=3)(g)
        b = FeatureNoise(0.1, seed=3)(g)
        assert torch.equal(a.node_features, b.node_features)

    def test_node_feature_mask_zeros_some_entries(self):
        torch.manual_seed(0)
        g = _g(N=200, D=8)
        out = NodeFeatureMask(p=0.5, seed=0)(g)
        # Roughly half the entries should be zero.
        zero_frac = (out.node_features == 0).float().mean().item()
        assert 0.3 < zero_frac < 0.7


# ── splits ───────────────────────────────────────────────────────────────────


class TestSplits:
    def test_random_node_split_disjoint(self):
        g = _g(N=20)
        out = RandomNodeSplit(0.6, 0.2, seed=0)(g)
        m = out.metadata["masks"]
        assert (m["train_mask"] | m["val_mask"] | m["test_mask"]).all()
        assert not (m["train_mask"] & m["val_mask"]).any()
        assert not (m["val_mask"] & m["test_mask"]).any()
        assert not (m["train_mask"] & m["test_mask"]).any()

    def test_fixed_split_from_indices(self):
        g = _g(N=10)
        out = FixedSplit(
            train=torch.tensor([0, 1, 2]),
            val=torch.tensor([3, 4]),
            test=torch.tensor([5, 6, 7, 8, 9]),
        )(g)
        m = out.metadata["masks"]
        assert m["train_mask"].sum() == 3
        assert m["val_mask"].sum() == 2
        assert m["test_mask"].sum() == 5

    def test_random_link_split(self):
        g = _g(E=20)
        out = RandomLinkSplit(0.7, 0.15, seed=0)(g)
        m = out.metadata["edge_masks"]
        assert (m["train_mask"] | m["val_mask"] | m["test_mask"]).all()

    def test_random_graph_split_returns_indices(self):
        s = RandomGraphSplit(0.6, 0.2, seed=0)
        train, val, test = s(20)
        assert len(train) + len(val) + len(test) == 20


# ── positional ───────────────────────────────────────────────────────────────


class TestPositional:
    def test_degree_encoding_appends(self):
        g = _g(N=6, D=4)
        out = AddDegreeEncoding(dim=4, direction="both")(g)
        assert out.node_features.shape == (6, 4 + 8)  # 2*dim for "both"

    def test_laplacian_finite(self):
        g = _g(N=6, D=4)
        out = AddLaplacianEigenvectors(dim=2)(g)
        assert torch.isfinite(out.node_features).all()

    def test_laplacian_max_nodes_guard(self):
        g = _g(N=6, D=4)
        with pytest.raises(ValueError, match="N >"):
            AddLaplacianEigenvectors(dim=2, max_nodes=3)(g)

    def test_adjacency_bias_metadata(self):
        g = _g(N=6, D=4)
        out = AddAdjacencyBias(neg_inf=True)(g)
        bias = out.metadata["edge_bias_dense"]
        assert bias.shape == (6, 6)


# ── patch ────────────────────────────────────────────────────────────────────


class TestPatch:
    def test_patchify_image_then_grid(self):
        img = torch.randn(1, 3, 8, 8)
        g = Graph(img, torch.zeros((2, 0), dtype=torch.long))
        out = Compose([PatchifyImage(patch_size=4), BuildGridGraph()])(g)
        assert out.node_features.shape == (4, 3, 4, 4)
        assert out.metadata["grid_shape"] == (2, 2)
        assert out.edge_index.size(1) > 0


# ── compose ──────────────────────────────────────────────────────────────────


class TestComposeApi:
    def test_metadata_preserved(self):
        g = Graph(torch.randn(3, 4),
                  torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                  metadata={"keep_me": 1})
        out = Compose([NormalizeFeatures(), AddSelfLoops()])(g)
        assert out.metadata["keep_me"] == 1
