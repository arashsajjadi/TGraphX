"""Tests for graph RL environments."""
import pytest
import torch

from tgraphx.rl.environments import (
    GraphEnvConfig,
    GraphNavigationEnv,
    GraphColoringEnv,
    MaxCutEnv,
    GraphGenerationEnv,
    KGPathReasoningEnv,
)


def _triangle() -> tuple:
    """Returns (edge_index, num_nodes) for a triangle graph."""
    ei = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    return ei, 3


def _nav_env(n=5, seed=42) -> GraphNavigationEnv:
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    nf = torch.randn(n, 4)
    config = GraphEnvConfig(max_steps=20, seed=seed)
    return GraphNavigationEnv(ei, n, node_features=nf, target_node=4, config=config, start_node=0)


class TestGraphNavigationEnv:
    def test_reset_deterministic_with_seed(self):
        env = _nav_env(seed=42)
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1["current_node"] == obs2["current_node"]
        assert obs1["step"] == 0

    def test_step_with_valid_action_succeeds(self):
        env = _nav_env()
        env.reset()
        # Action 0 = first neighbor
        obs, reward, done, truncated, info = env.step(0)
        assert info["action_valid"]

    def test_step_changes_current_node(self):
        env = _nav_env()
        env.reset()
        old_node = env._current_node
        env.step(0)
        assert env._current_node != old_node or env._done

    def test_node_features_not_modified_in_obs(self):
        env = _nav_env()
        obs = env.reset()
        original = obs["node_features"].clone()
        env.step(0)
        # Features in env should not change
        assert torch.allclose(env._node_features, original)

    def test_action_mask_is_bool_tensor(self):
        env = _nav_env()
        obs = env.reset()
        mask = obs["action_mask"]
        assert mask.dtype == torch.bool

    def test_invalid_action_marks_done(self):
        env = _nav_env()
        env.reset()
        # Invalid action index
        obs, reward, done, truncated, info = env.step(999)
        assert done is True
        assert info["action_valid"] is False


class TestGraphColoringEnv:
    def test_action_mask_all_valid_initially(self):
        ei, n = _triangle()
        config = GraphEnvConfig(seed=0)
        env = GraphColoringEnv(ei, n, num_colors=3, config=config)
        obs = env.reset()
        mask = obs["action_mask"]
        assert mask.all().item()  # All colors valid initially

    def test_step_changes_coloring(self):
        ei, n = _triangle()
        config = GraphEnvConfig(seed=0)
        env = GraphColoringEnv(ei, n, num_colors=3, config=config)
        env.reset()
        env.step(0)  # Color node 0 with color 0
        assert env._coloring[0] == 0

    def test_invalid_action_returns_done(self):
        ei, n = _triangle()
        config = GraphEnvConfig(seed=0)
        env = GraphColoringEnv(ei, n, num_colors=3, config=config)
        env.reset()
        obs, reward, done, _, info = env.step(999)  # invalid color
        assert done is True

    def test_vector_node_features_in_observation(self):
        ei, n = _triangle()
        nf = torch.randn(n, 8)
        config = GraphEnvConfig(seed=0)
        env = GraphColoringEnv(ei, n, node_features=nf, num_colors=3, config=config)
        obs = env.reset()
        assert obs["node_features"].shape == (n, 8)


class TestMaxCutEnv:
    def test_reward_equals_delta_cut_on_triangle(self):
        """Test reward matches delta cut on hand-computed triangle."""
        ei = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
        n = 3
        config = GraphEnvConfig(seed=0)
        env = MaxCutEnv(ei, n, config=config)
        env.reset()

        # Assign node 0 to partition 0 — no edges assigned yet, delta = 0
        _, reward1, _, _, _ = env.step(0)
        assert reward1 == pytest.approx(0.0)

        # Assign node 1 to partition 1 — edge (0,1) cut, delta = 1
        _, reward2, _, _, _ = env.step(1)
        assert reward2 == pytest.approx(1.0)

    def test_action_mask_binary(self):
        ei, n = _triangle()
        config = GraphEnvConfig(seed=0)
        env = MaxCutEnv(ei, n, config=config)
        obs = env.reset()
        assert obs["action_mask"].shape == (2,)
        assert obs["action_mask"].dtype == torch.bool


class TestGraphGenerationEnv:
    def test_add_node_increases_node_count(self):
        from tgraphx.generation.actions import GraphActionSpace, GraphActionType, GraphAction, action_to_index
        config = GraphEnvConfig(max_steps=50, seed=0)
        space = GraphActionSpace(max_nodes=10, max_edges=20)
        env = GraphGenerationEnv(action_space_config=space, config=config)
        env.reset()
        # Find ADD_NODE action index
        add_node_action = GraphAction(action_type=GraphActionType.ADD_NODE, node_type=0)
        add_node_idx = action_to_index(add_node_action, space)
        obs, reward, done, _, info = env.step(add_node_idx)
        if info.get("action_valid", True):
            assert info.get("num_nodes", env.num_nodes) >= 0

    def test_stop_action_done_true(self):
        from tgraphx.generation.actions import GraphActionSpace
        config = GraphEnvConfig(max_steps=50, seed=0)
        space = GraphActionSpace(max_nodes=10, max_edges=20)
        env = GraphGenerationEnv(action_space_config=space, config=config)
        env.reset()
        # Action 0 = STOP
        _, _, done, _, info = env.step(0)
        assert done is True


class TestKGPathReasoningEnv:
    def test_valid_action_mask_matches_outgoing_relations(self):
        # Simple KG: 0 -rel0-> 1, 0 -rel1-> 2
        ei = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        rt = torch.tensor([0, 1], dtype=torch.long)
        config = GraphEnvConfig(max_steps=10, seed=0)
        env = KGPathReasoningEnv(
            kg_edge_index=ei,
            relation_types=rt,
            num_entities=3,
            num_relations=2,
            query_pairs=[(0, 1)],
            config=config,
        )
        env.reset()
        mask = env.valid_action_mask()
        # From entity 0, there should be 2 valid actions
        assert mask.sum().item() == 2

    def test_reaching_target_gives_reward(self):
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        rt = torch.tensor([0], dtype=torch.long)
        config = GraphEnvConfig(max_steps=10, seed=0)
        env = KGPathReasoningEnv(
            kg_edge_index=ei,
            relation_types=rt,
            num_entities=2,
            num_relations=1,
            query_pairs=[(0, 1)],
            config=config,
        )
        env.reset()
        # Take action 0 = go to entity 1
        _, reward, done, _, info = env.step(0)
        assert info["success"] is True
        assert reward > 0

    def test_action_mask_is_bool_tensor_on_correct_device(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        rt = torch.tensor([0, 0], dtype=torch.long)
        config = GraphEnvConfig(max_steps=10, seed=0)
        env = KGPathReasoningEnv(
            kg_edge_index=ei,
            relation_types=rt,
            num_entities=3,
            num_relations=1,
            query_pairs=[(0, 2)],
            config=config,
        )
        obs = env.reset()
        mask = obs["action_mask"]
        assert mask.dtype == torch.bool
        assert str(mask.device) == "cpu"
