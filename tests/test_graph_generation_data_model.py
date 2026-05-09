"""Tests for graph generation data model.

Tests GeneratedGraph, GraphEditState, GraphGenerationBatch.
"""
import pytest
import torch

from tgraphx.generation.data_model import (
    GeneratedGraph,
    GraphEditState,
    GraphGenerationBatch,
    graph_to_generation_state,
    generation_state_to_graph,
    validate_generated_graph,
    graph_generation_summary,
)


def _simple_graph(n=4, e=3) -> GeneratedGraph:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_features = torch.randn(n, 8)
    return GeneratedGraph(
        edge_index=edge_index,
        num_nodes=n,
        directed=False,
        node_features=node_features,
    )


# ── GeneratedGraph ───────────────────────────────────────────────────────────

class TestGeneratedGraph:

    def test_vector_node_features_preserved_through_clone(self):
        g = _simple_graph()
        g2 = g.clone()
        assert torch.allclose(g.node_features, g2.node_features)
        # Mutating clone does not affect original
        g2.node_features[0, 0] = 99.0
        assert not torch.allclose(g.node_features, g2.node_features)

    def test_vector_node_features_preserved_through_to_device(self):
        g = _simple_graph()
        g2 = g.to(torch.device("cpu"))
        assert torch.allclose(g.node_features, g2.node_features)

    def test_image_node_features_preserved(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_features = torch.randn(3, 2, 8, 8)  # [N, C, H, W]
        g = GeneratedGraph(
            edge_index=edge_index,
            num_nodes=3,
            node_features=node_features,
        )
        g2 = g.clone()
        assert g2.node_features.shape == (3, 2, 8, 8)
        assert torch.allclose(g.node_features, g2.node_features)

    def test_edge_features_preserved(self):
        g = _simple_graph()
        g.edge_features = torch.randn(3, 4)
        g2 = g.clone()
        assert torch.allclose(g.edge_features, g2.edge_features)

    def test_directed_flag_preserved(self):
        g = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=2,
            directed=True,
        )
        g2 = g.clone()
        assert g2.directed is True

    def test_action_mask_shape_valid(self):
        g = _simple_graph()
        g.action_mask = torch.ones(10, dtype=torch.bool)
        assert g.action_mask.shape == (10,)

    def test_to_device_moves_all_tensors(self):
        g = _simple_graph()
        g.edge_weight = torch.ones(3)
        g2 = g.to(torch.device("cpu"))
        assert g2.edge_index.device == torch.device("cpu")
        assert g2.node_features.device == torch.device("cpu")
        assert g2.edge_weight.device == torch.device("cpu")

    def test_detach_for_report_no_raw_tensors(self):
        import json

        g = _simple_graph()
        g.metadata = {"some_tensor": torch.randn(3), "label": "test"}
        report = g.detach_for_report()

        # JSON-serialize to detect non-JSON-safe types
        text = json.dumps(report, default=str)
        parsed = json.loads(text)

        # Recursively check no Tensor
        def _check_no_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return False
            if isinstance(obj, dict):
                return all(_check_no_tensor(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return all(_check_no_tensor(v) for v in obj)
            return True

        assert _check_no_tensor(parsed), "Report contains non-JSON-safe types"

    def test_invalid_graph_raises_value_error(self):
        # edge_index references node ID >= num_nodes
        edge_index = torch.tensor([[0], [5]], dtype=torch.long)
        g = GeneratedGraph(edge_index=edge_index, num_nodes=3)
        with pytest.raises(ValueError, match="node ID"):
            g.validate()

    def test_validate_catches_edge_id_out_of_range(self):
        edge_index = torch.tensor([[0, 10], [1, 2]], dtype=torch.long)
        g = GeneratedGraph(edge_index=edge_index, num_nodes=5)
        with pytest.raises(ValueError):
            g.validate()

    def test_validate_negative_node_id(self):
        edge_index = torch.tensor([[-1, 0], [0, 1]], dtype=torch.long)
        g = GeneratedGraph(edge_index=edge_index, num_nodes=3)
        with pytest.raises(ValueError):
            g.validate()

    def test_validate_feature_shape_mismatch(self):
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        g = GeneratedGraph(
            edge_index=edge_index,
            num_nodes=3,
            node_features=torch.randn(5, 4),  # wrong N
        )
        with pytest.raises(ValueError):
            g.validate()

    def test_generation_state_to_graph_roundtrip(self):
        g = _simple_graph()
        state = GraphEditState(graph=g)
        from tgraphx.core.graph import Graph
        graph_obj = generation_state_to_graph(state)
        assert isinstance(graph_obj, Graph)
        assert torch.allclose(graph_obj.edge_index, g.edge_index)


# ── GraphGenerationBatch ──────────────────────────────────────────────────────

class TestGraphGenerationBatch:

    def test_batch_offsets_correct(self):
        g1 = GeneratedGraph(
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            num_nodes=3,
        )
        g2 = GeneratedGraph(
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            num_nodes=2,
        )
        batch = GraphGenerationBatch.from_graphs([g1, g2])
        # g2 edges should be offset by 3
        assert batch.edge_index_batch[0, 2].item() == 3  # 0 + 3
        assert batch.edge_index_batch[1, 2].item() == 4  # 1 + 3

    def test_batch_vector_correct(self):
        g1 = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=3,
        )
        g2 = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=2,
        )
        batch = GraphGenerationBatch.from_graphs([g1, g2])
        assert batch.batch_vector.tolist() == [0, 0, 0, 1, 1]

    def test_batch_node_features_stacked(self):
        g1 = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=3,
            node_features=torch.randn(3, 8),
        )
        g2 = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            num_nodes=2,
            node_features=torch.randn(2, 8),
        )
        batch = GraphGenerationBatch.from_graphs([g1, g2])
        assert batch.node_features_batch is not None
        assert batch.node_features_batch.shape == (5, 8)

    def test_graph_generation_summary_no_tensors(self):
        g = _simple_graph()
        summary = graph_generation_summary(g)
        import json
        # Should serialize without error
        json.dumps(summary, default=str)
        assert "num_nodes" in summary
        assert "density" in summary
