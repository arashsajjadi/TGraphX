"""HeteroGraphBatch tests (v0.2.5)."""
from __future__ import annotations

import pytest
import torch

from tgraphx.core.hetero_graph import HeteroGraph
from tgraphx.core.hetero_batch import HeteroGraphBatch


def _g(n_paper, n_author, n_edges, seed=0, with_ew=False, with_ef=False, with_nl=False, gl=None):
    torch.manual_seed(seed)
    nodes = {
        "paper": torch.randn(n_paper, 8),
        "author": torch.randn(n_author, 4),
    }
    src = torch.randint(0, n_author, (n_edges,))
    dst = torch.randint(0, n_paper, (n_edges,))
    edge_index = torch.stack([src, dst], dim=0).long()
    edges = {("author", "writes", "paper"): edge_index}
    ew = {("author", "writes", "paper"): torch.rand(n_edges)} if with_ew else None
    ef = {("author", "writes", "paper"): torch.randn(n_edges, 3)} if with_ef else None
    nl = {"paper": torch.randint(0, 3, (n_paper,))} if with_nl else None
    return HeteroGraph(
        node_stores=nodes, edge_stores=edges,
        edge_weight_stores=ew, edge_feature_stores=ef,
        node_label_stores=nl, graph_label=gl,
    )


class TestBasic:
    def test_batch_two_graphs(self):
        b = HeteroGraphBatch([_g(5, 3, 4, 0), _g(4, 2, 3, 1)])
        assert b.num_graphs == 2
        assert b.num_nodes_dict == {"paper": 9, "author": 5}
        assert b.num_edges_dict == {("author", "writes", "paper"): 7}

    def test_batch_dict_shape(self):
        b = HeteroGraphBatch([_g(5, 3, 4, 0), _g(4, 2, 3, 1)])
        assert b.batch_dict["paper"].shape == (9,)
        assert b.batch_dict["paper"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
        assert b.batch_dict["author"].shape == (5,)
        assert b.batch_dict["author"].tolist() == [0, 0, 0, 1, 1]

    def test_edge_index_offsets(self):
        b = HeteroGraphBatch([_g(5, 3, 4, 0), _g(4, 2, 3, 1)])
        ei = b.edge_index(("author", "writes", "paper"))
        # First 4 edges: src in [0,3), dst in [0,5)
        # Next 3: src in [3,5), dst in [5,9)
        assert (ei[0, :4] < 3).all()
        assert (ei[1, :4] < 5).all()
        assert (ei[0, 4:] >= 3).all()
        assert (ei[0, 4:] < 5).all()
        assert (ei[1, 4:] >= 5).all()
        assert (ei[1, 4:] < 9).all()


class TestAttributes:
    def test_edge_weight_concat(self):
        b = HeteroGraphBatch([
            _g(5, 3, 4, 0, with_ew=True),
            _g(4, 2, 3, 1, with_ew=True),
        ])
        ew = b.edge_weight(("author", "writes", "paper"))
        assert ew.shape == (7,)

    def test_edge_features_concat(self):
        b = HeteroGraphBatch([
            _g(5, 3, 4, 0, with_ef=True),
            _g(4, 2, 3, 1, with_ef=True),
        ])
        ef = b.edge_features(("author", "writes", "paper"))
        assert ef.shape == (7, 3)

    def test_node_labels_concat(self):
        b = HeteroGraphBatch([
            _g(5, 3, 4, 0, with_nl=True),
            _g(4, 2, 3, 1, with_nl=True),
        ])
        nl = b.node_labels("paper")
        assert nl.shape == (9,)

    def test_graph_labels_stack(self):
        b = HeteroGraphBatch([
            _g(5, 3, 4, 0, gl=torch.tensor(1)),
            _g(4, 2, 3, 1, gl=torch.tensor(0)),
        ])
        assert b.graph_labels.tolist() == [1, 0]

    def test_metadata_preserved(self):
        g1 = _g(5, 3, 4, 0)
        g1.metadata = {"name": "g1"}
        g2 = _g(4, 2, 3, 1)
        g2.metadata = {"name": "g2"}
        b = HeteroGraphBatch([g1, g2])
        assert [m["name"] for m in b.metadata] == ["g1", "g2"]


class TestErrors:
    def test_inconsistent_node_types_raises(self):
        g1 = HeteroGraph(
            node_stores={"a": torch.randn(3, 4)},
            edge_stores={("a", "rel", "a"): torch.tensor([[0], [1]], dtype=torch.long)},
        )
        g2 = HeteroGraph(
            node_stores={"b": torch.randn(3, 4)},
            edge_stores={("b", "rel", "b"): torch.tensor([[0], [1]], dtype=torch.long)},
        )
        with pytest.raises(ValueError, match="node-type schema"):
            HeteroGraphBatch([g1, g2])

    def test_inconsistent_node_feature_shape_raises(self):
        g1 = HeteroGraph(
            node_stores={"a": torch.randn(3, 4)},
            edge_stores={("a", "rel", "a"): torch.tensor([[0], [1]], dtype=torch.long)},
        )
        g2 = HeteroGraph(
            node_stores={"a": torch.randn(3, 8)},  # different feature dim
            edge_stores={("a", "rel", "a"): torch.tensor([[0], [1]], dtype=torch.long)},
        )
        with pytest.raises(ValueError, match="feature shape"):
            HeteroGraphBatch([g1, g2])

    def test_partial_edge_weight_raises(self):
        g1 = _g(5, 3, 4, 0, with_ew=True)
        g2 = _g(4, 2, 3, 1, with_ew=False)
        with pytest.raises(ValueError, match="edge_weight"):
            HeteroGraphBatch([g1, g2])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            HeteroGraphBatch([])

    def test_non_hetero_graph_raises(self):
        with pytest.raises(TypeError):
            HeteroGraphBatch([object()])  # type: ignore[list-item]


class TestDevice:
    def test_to_cpu(self):
        b = HeteroGraphBatch([_g(5, 3, 4, 0), _g(4, 2, 3, 1)])
        b2 = b.to("cpu")
        assert b2.device.type == "cpu"

    def test_no_regression_to_graphbatch(self):
        from tgraphx import GraphBatch, Graph
        g1 = Graph(torch.randn(3, 4), None)
        g2 = Graph(torch.randn(2, 4), None)
        b = GraphBatch([g1, g2])
        assert b.num_nodes == 5
