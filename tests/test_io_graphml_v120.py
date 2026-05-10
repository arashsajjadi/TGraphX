"""GraphML round-trip tests (v1.2)."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from tgraphx import Graph
from tgraphx.io import read_graphml, write_graphml


class TestGraphMLBasicRoundTrip:
    def test_directed_structure_round_trip(self, tmp_path):
        x = torch.zeros(4, 1)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "g.graphml"
        write_graphml(g, path)

        g2 = read_graphml(path)
        assert g2.num_nodes == 4
        assert g2.num_edges == 3
        assert torch.equal(g2.edge_index, ei)
        assert g2.metadata["graphml_directed"] is True

    def test_undirected_round_trip(self, tmp_path):
        # Symmetric edges → graph.is_undirected() returns True.
        x = torch.zeros(3, 1)
        ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "u.graphml"
        write_graphml(g, path)
        g2 = read_graphml(path)
        # The marker for undirected default in the file should round-trip.
        assert g2.metadata["graphml_directed"] is False

    def test_empty_graph(self, tmp_path):
        x = torch.zeros(2, 1)
        g = Graph(node_features=x)
        path = tmp_path / "e.graphml"
        write_graphml(g, path)
        g2 = read_graphml(path)
        assert g2.num_nodes == 2
        assert g2.num_edges == 0


class TestGraphMLEdgeWeight:
    def test_weight_round_trip(self, tmp_path):
        x = torch.zeros(3, 1)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        w = torch.tensor([0.5, 1.5])
        g = Graph(node_features=x, edge_index=ei, edge_weight=w)
        path = tmp_path / "w.graphml"
        write_graphml(g, path)
        g2 = read_graphml(path)
        assert g2.edge_weight is not None
        assert torch.allclose(g2.edge_weight, w, atol=1e-6)


class TestGraphMLLabels:
    def test_node_labels_round_trip(self, tmp_path):
        x = torch.zeros(3, 1)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        y = torch.tensor([2, 0, 1], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei, y=y)
        path = tmp_path / "y.graphml"
        write_graphml(g, path)
        g2 = read_graphml(path)
        assert g2.node_labels is not None
        # Labels round-trip; integer dtype preserved when all values are integer.
        assert g2.node_labels.dtype == torch.long
        assert torch.equal(g2.node_labels, y)

    def test_edge_labels_round_trip(self, tmp_path):
        x = torch.zeros(3, 1)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        el = torch.tensor([1, 0], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei, edge_labels=el)
        path = tmp_path / "el.graphml"
        write_graphml(g, path)
        g2 = read_graphml(path)
        assert g2.edge_labels is not None
        assert torch.equal(g2.edge_labels, el)


class TestGraphMLTensorFeatures:
    def test_1d_node_features_round_trip(self, tmp_path):
        x = torch.tensor([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "x.graphml"
        write_graphml(g, path, include_tensor_features=True)
        g2 = read_graphml(path)
        assert torch.allclose(g2.node_features, x, atol=1e-6)

    def test_3d_node_features_rejected(self, tmp_path):
        # [N, C, H, W] — clearly unsafe to flatten through GraphML.
        x = torch.randn(2, 4, 3, 3)
        g = Graph(node_features=x)
        path = tmp_path / "bad.graphml"
        with pytest.raises(ValueError, match="multi-dimensional tensor"):
            write_graphml(g, path, include_tensor_features=True)

    def test_3d_node_features_omitted_by_default(self, tmp_path):
        # Default include_tensor_features=False — no error, just no features written.
        x = torch.randn(2, 4, 3, 3)
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "ok.graphml"
        write_graphml(g, path)  # no include_tensor_features
        g2 = read_graphml(path)
        # Round-trip default: zero-filled [N, 1] node features.
        assert g2.num_nodes == 2
        assert g2.num_edges == 2


class TestGraphMLPaths:
    def test_pathlib_path(self, tmp_path):
        path = tmp_path / "p.graphml"
        x = torch.zeros(2, 1)
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        result = write_graphml(g, path)
        assert isinstance(result, Path)
        assert result.exists()

    def test_string_path(self, tmp_path):
        path = str(tmp_path / "s.graphml")
        x = torch.zeros(2, 1)
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        write_graphml(g, path)
        g2 = read_graphml(path)
        assert g2.num_nodes == 2


class TestGraphMLValidation:
    def test_read_invalid_root_raises(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<not_graphml/>", encoding="utf-8")
        with pytest.raises(ValueError, match="root tag"):
            read_graphml(path)

    def test_read_no_graph_raises(self, tmp_path):
        path = tmp_path / "empty.xml"
        path.write_text(
            '<?xml version="1.0"?>'
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"></graphml>',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="no <graph>"):
            read_graphml(path)

    def test_xml_is_well_formed(self, tmp_path):
        x = torch.zeros(3, 1)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g = Graph(node_features=x, edge_index=ei)
        path = tmp_path / "f.graphml"
        write_graphml(g, path)
        # Parsing the file as XML must succeed.
        ET.parse(path)
