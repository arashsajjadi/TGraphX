"""HeteroConv + readouts + models tests (v0.2.5)."""
from __future__ import annotations

import pytest
import torch

from tgraphx.core.hetero_graph import HeteroGraph
from tgraphx.core.hetero_batch import HeteroGraphBatch
from tgraphx.layers import LinearMessagePassing
from tgraphx.layers.hetero import HeteroConv
from tgraphx.layers.hetero_readout import (
    hetero_concat_pool,
    hetero_max_pool,
    hetero_mean_pool,
    hetero_sum_pool,
)
from tgraphx.models.hetero_models import HeteroGraphClassifier, HeteroNodeClassifier


def _hetero_pair(n_p=5, n_a=3, n_e=4, seed=0):
    torch.manual_seed(seed)
    return HeteroGraph(
        node_stores={"paper": torch.randn(n_p, 8), "author": torch.randn(n_a, 8)},
        edge_stores={
            ("author", "writes", "paper"): torch.stack(
                [torch.randint(0, n_a, (n_e,)),
                 torch.randint(0, n_p, (n_e,))],
                dim=0,
            ).long(),
        },
    )


# ── HeteroConv ────────────────────────────────────────────────────────────────

class TestHeteroConv:
    def test_vector_forward(self):
        g = _hetero_pair()
        conv = HeteroConv({
            ("author", "writes", "paper"): LinearMessagePassing((8,), (8,)),
        }, aggr="sum")
        out = conv(g.x_dict, g.edge_index_dict)
        assert out["paper"].shape == (5, 8)
        # author has no relation writing into it — pass-through
        assert torch.equal(out["author"], g.node_features("author"))

    def test_two_relations_into_same_dest(self):
        torch.manual_seed(0)
        x = {"a": torch.randn(4, 8), "b": torch.randn(3, 8)}
        ei_dict = {
            ("a", "r1", "b"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            ("a", "r2", "b"): torch.tensor([[2, 3], [1, 2]], dtype=torch.long),
        }
        conv = HeteroConv({
            ("a", "r1", "b"): LinearMessagePassing((8,), (8,)),
            ("a", "r2", "b"): LinearMessagePassing((8,), (8,)),
        }, aggr="sum")
        out = conv(x, ei_dict)
        assert out["b"].shape == (3, 8)

    def test_aggr_modes(self):
        x = {"a": torch.randn(3, 4), "b": torch.randn(2, 4)}
        ei_dict = {
            ("a", "r1", "b"): torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
            ("a", "r2", "b"): torch.tensor([[1, 2], [1, 0]], dtype=torch.long),
        }
        for aggr in ("sum", "mean", "max"):
            conv = HeteroConv({
                ("a", "r1", "b"): LinearMessagePassing((4,), (4,)),
                ("a", "r2", "b"): LinearMessagePassing((4,), (4,)),
            }, aggr=aggr)
            out = conv(x, ei_dict)
            assert out["b"].shape == (2, 4)

    def test_edge_weight_path(self):
        g = _hetero_pair()
        conv = HeteroConv({
            ("author", "writes", "paper"): LinearMessagePassing((8,), (8,)),
        })
        ew = {("author", "writes", "paper"): torch.rand(g.num_edges(("author", "writes", "paper")))}
        out = conv(g.x_dict, g.edge_index_dict, edge_weight_dict=ew)
        assert out["paper"].shape == (5, 8)

    def test_backward(self):
        g = _hetero_pair()
        x_a = g.node_features("author").clone().requires_grad_(True)
        conv = HeteroConv({
            ("author", "writes", "paper"): LinearMessagePassing((8,), (8,)),
        })
        out = conv({"paper": g.node_features("paper"), "author": x_a},
                   g.edge_index_dict)
        out["paper"].sum().backward()
        assert x_a.grad is not None and torch.isfinite(x_a.grad).all()

    def test_invalid_aggr(self):
        with pytest.raises(ValueError, match="aggr"):
            HeteroConv({("a", "r", "b"): LinearMessagePassing((4,), (4,))}, aggr="bad")

    def test_missing_node_type_in_x_dict(self):
        g = _hetero_pair()
        conv = HeteroConv({
            ("author", "writes", "paper"): LinearMessagePassing((8,), (8,)),
        })
        partial = {"paper": g.node_features("paper")}  # missing 'author'
        with pytest.raises(KeyError, match="author"):
            conv(partial, g.edge_index_dict)


# ── Readouts ──────────────────────────────────────────────────────────────────

class TestHeteroReadouts:
    def test_per_type_mean_pool_single(self):
        x_dict = {"a": torch.randn(5, 4), "b": torch.randn(3, 4)}
        out = hetero_mean_pool(x_dict)
        assert out["a"].shape == (1, 4)
        assert out["b"].shape == (1, 4)

    def test_per_type_pool_with_batch(self):
        x_dict = {"a": torch.randn(6, 4)}
        batch = {"a": torch.tensor([0, 0, 0, 1, 1, 1])}
        out = hetero_mean_pool(x_dict, batch_dict=batch)
        assert out["a"].shape == (2, 4)

    def test_sum_max_pool(self):
        x_dict = {"a": torch.randn(4, 3)}
        batch = {"a": torch.tensor([0, 0, 1, 1])}
        sums = hetero_sum_pool(x_dict, batch_dict=batch)
        maxs = hetero_max_pool(x_dict, batch_dict=batch)
        assert sums["a"].shape == (2, 3)
        assert maxs["a"].shape == (2, 3)

    def test_concat_pool_stable_order(self):
        x_dict = {"b": torch.ones(3, 2), "a": torch.zeros(3, 2)}
        batch = {"b": torch.tensor([0, 0, 1]), "a": torch.tensor([0, 1, 1])}
        out = hetero_concat_pool(x_dict, batch_dict=batch)  # default sorted
        # Sorted: 'a' first, 'b' second.  a row 0 = [0, 0]; b row 0 = mean of [1,1] for batch 0
        assert out.shape == (2, 4)
        assert out[0, :2].tolist() == [0.0, 0.0]  # 'a' batch 0


# ── Models ────────────────────────────────────────────────────────────────────

class TestHeteroModels:
    def test_graph_classifier_smoke(self):
        b = HeteroGraphBatch([_hetero_pair(seed=0), _hetero_pair(seed=1)])
        clf = HeteroGraphClassifier(
            node_in_dims={"paper": 8, "author": 8},
            edge_types=[("author", "writes", "paper")],
            hidden_dim=16, num_layers=2, num_classes=3,
        )
        logits = clf(b.x_dict, b.edge_index_dict, batch_dict=b.batch_dict)
        assert logits.shape == (2, 3)

    def test_graph_classifier_backward(self):
        b = HeteroGraphBatch([_hetero_pair(seed=0), _hetero_pair(seed=1)])
        clf = HeteroGraphClassifier(
            node_in_dims={"paper": 8, "author": 8},
            edge_types=[("author", "writes", "paper")],
            hidden_dim=16, num_layers=1, num_classes=3,
        )
        logits = clf(b.x_dict, b.edge_index_dict, batch_dict=b.batch_dict)
        logits.sum().backward()
        for p in clf.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_node_classifier_smoke(self):
        b = HeteroGraphBatch([_hetero_pair(seed=0), _hetero_pair(seed=1)])
        nc = HeteroNodeClassifier(
            node_in_dims={"paper": 8, "author": 8},
            edge_types=[("author", "writes", "paper")],
            hidden_dim=16, num_layers=2, num_classes=4,
            target_type="paper",
        )
        out = nc(b.x_dict, b.edge_index_dict)
        assert out.shape == (b.num_nodes_dict["paper"], 4)

    def test_classifier_invalid_target_raises(self):
        with pytest.raises(ValueError, match="target_type"):
            HeteroNodeClassifier(
                node_in_dims={"paper": 8},
                edge_types=[("paper", "cites", "paper")],
                hidden_dim=8, num_layers=1, num_classes=2,
                target_type="venue",
            )
