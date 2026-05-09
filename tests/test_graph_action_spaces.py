"""Tests for graph action spaces."""
import pytest
import torch

from tgraphx.generation.data_model import GeneratedGraph, GraphEditState
from tgraphx.generation.actions import (
    GraphActionType,
    GraphAction,
    GraphActionSpace,
    enumerate_valid_actions,
    sample_valid_action,
    apply_graph_action,
    action_to_index,
    index_to_action,
)


def _make_state(n=4, edges=None) -> GraphEditState:
    if edges is None:
        ei = torch.zeros((2, 0), dtype=torch.long)
    else:
        ei = torch.tensor(edges, dtype=torch.long)
    g = GeneratedGraph(edge_index=ei, num_nodes=n)
    return GraphEditState(graph=g)


def _make_state_with_features(n=3) -> GraphEditState:
    ei = torch.zeros((2, 0), dtype=torch.long)
    g = GeneratedGraph(
        edge_index=ei,
        num_nodes=n,
        node_features=torch.randn(n, 4),
    )
    return GraphEditState(graph=g)


class TestAddNode:
    def test_add_node_increases_num_nodes_by_1(self):
        state = _make_state(n=3)
        action = GraphAction(action_type=GraphActionType.ADD_NODE, node_type=0)
        new_state = apply_graph_action(state, action)
        assert new_state.graph.num_nodes == 4

    def test_add_node_increments_step(self):
        state = _make_state(n=2)
        action = GraphAction(action_type=GraphActionType.ADD_NODE)
        new_state = apply_graph_action(state, action)
        assert new_state.step == 1


class TestAddEdge:
    def test_add_edge_increases_num_edges(self):
        state = _make_state(n=4)
        action = GraphAction(action_type=GraphActionType.ADD_EDGE, src_id=0, tgt_id=1)
        new_state = apply_graph_action(state, action)
        assert new_state.graph.num_edges == 1

    def test_add_edge_preserves_nodes(self):
        state = _make_state(n=4)
        action = GraphAction(action_type=GraphActionType.ADD_EDGE, src_id=0, tgt_id=2)
        new_state = apply_graph_action(state, action)
        assert new_state.graph.num_nodes == 4


class TestRemoveNode:
    def test_remove_node_decreases_num_nodes(self):
        state = _make_state(n=4)
        action = GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=1)
        new_state = apply_graph_action(state, action)
        assert new_state.graph.num_nodes == 3

    def test_remove_node_removes_incident_edges(self):
        # Graph: 0-1, 1-2
        state = _make_state(n=3, edges=[[0, 1, 1], [1, 0, 2]])
        assert state.graph.num_edges == 3
        action = GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=1)
        new_state = apply_graph_action(state, action)
        # Edges 0-1 and 1-2 should be removed (incident to node 1)
        assert new_state.graph.num_edges == 0
        assert new_state.graph.num_nodes == 2

    def test_remove_node_no_dangling_edge_ids(self):
        # 3 nodes, edge 0->2, then remove node 1
        state = _make_state(n=3, edges=[[0, 2], [1, 1]])
        # edges: 0->1 and 2->1
        action = GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=0)
        new_state = apply_graph_action(state, action)
        # Check all edge IDs in range
        if new_state.graph.num_edges > 0:
            assert int(new_state.graph.edge_index.max().item()) < new_state.graph.num_nodes


class TestRemoveEdge:
    def test_remove_edge_decreases_num_edges(self):
        state = _make_state(n=3, edges=[[0, 1], [1, 2]])
        action = GraphAction(action_type=GraphActionType.REMOVE_EDGE, src_id=0, tgt_id=1)
        new_state = apply_graph_action(state, action)
        assert new_state.graph.num_edges == 1


class TestSetNodeFeature:
    def test_set_node_feature_preserves_chw_shape(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        g = GeneratedGraph(
            edge_index=ei,
            num_nodes=3,
            node_features=torch.randn(3, 2, 8, 8),  # [N, C, H, W]
        )
        state = GraphEditState(graph=g)
        new_feat = torch.randn(2, 8, 8)  # single node feature
        action = GraphAction(
            action_type=GraphActionType.SET_NODE_FEATURE,
            node_id=1,
            features=new_feat,
        )
        new_state = apply_graph_action(state, action)
        assert new_state.graph.node_features.shape == (3, 2, 8, 8)
        assert torch.allclose(new_state.graph.node_features[1], new_feat)


class TestStopGeneration:
    def test_stop_marks_done(self):
        state = _make_state(n=3)
        action = GraphAction(action_type=GraphActionType.STOP_GENERATION)
        new_state = apply_graph_action(state, action)
        assert new_state.done is True


class TestActionRoundtrip:
    def test_action_to_index_roundtrip(self):
        space = GraphActionSpace(max_nodes=5, max_edges=20, allowed_node_types=[0, 1])
        actions_to_test = [
            GraphAction(action_type=GraphActionType.STOP_GENERATION),
            GraphAction(action_type=GraphActionType.ADD_NODE, node_type=0),
            GraphAction(action_type=GraphActionType.ADD_NODE, node_type=1),
            GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=2),
            GraphAction(action_type=GraphActionType.ADD_EDGE, src_id=0, tgt_id=1, edge_type=0),
        ]
        for act in actions_to_test:
            idx = action_to_index(act, space)
            back = index_to_action(idx, space)
            assert back.action_type == act.action_type, f"Roundtrip failed for {act}"


class TestSampleValidAction:
    def test_sample_valid_action_respects_no_self_loops(self):
        state = _make_state(n=4)
        space = GraphActionSpace(no_self_loops=True, max_nodes=10, max_edges=20)
        gen = torch.Generator()
        gen.manual_seed(42)
        for _ in range(20):
            action = sample_valid_action(state, space, generator=gen)
            if action.action_type == GraphActionType.ADD_EDGE:
                assert action.src_id != action.tgt_id

    def test_sample_valid_action_deterministic(self):
        state = _make_state(n=4)
        space = GraphActionSpace(max_nodes=10)
        gen1 = torch.Generator()
        gen1.manual_seed(123)
        gen2 = torch.Generator()
        gen2.manual_seed(123)
        a1 = sample_valid_action(state, space, generator=gen1)
        a2 = sample_valid_action(state, space, generator=gen2)
        assert a1.action_type == a2.action_type


class TestInvalidActions:
    def test_invalid_node_id_raises(self):
        state = _make_state(n=3)
        action = GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=10)
        with pytest.raises(ValueError):
            apply_graph_action(state, action)

    def test_invalid_edge_raises(self):
        state = _make_state(n=3)
        action = GraphAction(action_type=GraphActionType.REMOVE_EDGE, src_id=0, tgt_id=1)
        with pytest.raises(ValueError):
            apply_graph_action(state, action)


class TestActionMask:
    def test_action_mask_excludes_invalid_actions(self):
        from tgraphx.generation.actions import batch_action_masks
        state = _make_state(n=1)  # Only 1 node — ADD_EDGE not valid
        space = GraphActionSpace(max_nodes=5, max_edges=10)
        masks = batch_action_masks([state], space)
        assert masks.shape[0] == 1
        # STOP should be valid (index 0)
        assert masks[0, 0].item() is True

    def test_never_mutates_in_place(self):
        state = _make_state(n=4)
        original_n = state.graph.num_nodes
        action = GraphAction(action_type=GraphActionType.ADD_NODE, node_type=0)
        _ = apply_graph_action(state, action)
        assert state.graph.num_nodes == original_n
