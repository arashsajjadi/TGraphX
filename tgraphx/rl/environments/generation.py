"""Graph generation environment for RL.

The agent builds a graph by choosing graph edit actions.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph, GraphEditState
from tgraphx.generation.actions import (
    GraphActionSpace, GraphActionType, GraphAction,
    enumerate_valid_actions, apply_graph_action, action_to_index,
)
from .base import GraphEnv, GraphEnvConfig

__all__ = ["GraphGenerationEnv"]


class GraphGenerationEnv(GraphEnv):
    """Build a graph by selecting graph edit actions.

    State: current GeneratedGraph being built.
    Actions: ADD_NODE, ADD_EDGE, STOP (from action_space_config).
    Reward: validity_bonus + target_score + diversity_bonus + size_penalty.
    Done: STOP action or max_steps.

    Args:
        target_properties: Dict with optional 'target_density', 'connected' etc.
        action_space_config: GraphActionSpace for valid actions.
        config: GraphEnvConfig.
        target_score_fn: Optional callable (GeneratedGraph -> float) for target scoring.
    """

    def __init__(
        self,
        target_properties: Optional[Dict[str, Any]] = None,
        action_space_config: Optional[GraphActionSpace] = None,
        config: Optional[GraphEnvConfig] = None,
        target_score_fn: Optional[Callable[[GeneratedGraph], float]] = None,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._target_properties = target_properties or {}
        self._space = action_space_config or GraphActionSpace(
            max_nodes=self.config.max_nodes,
            max_edges=self.config.max_edges,
        )
        self._target_score_fn = target_score_fn
        self._state: Optional[GraphEditState] = None
        self._done: bool = False
        self._step_count: int = 0

    def _make_initial_state(self) -> GraphEditState:
        g = GeneratedGraph(
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
            num_nodes=0,
            directed=self.config.directed,
        )
        return GraphEditState(graph=g)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        self._state = self._make_initial_state()
        self._done = False
        self._step_count = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            self.reset()

        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        from tgraphx.generation.actions import index_to_action
        act = index_to_action(action, self._space)

        # Check validity
        valid_acts = enumerate_valid_actions(self._state, self._space)
        valid_indices = {action_to_index(a, self._space) for a in valid_acts}
        if action not in valid_indices:
            self._done = True
            reward = -1.0 * self.config.reward_scale
            return self.observe(), reward, True, False, {"action_valid": False, "error": "invalid action"}

        try:
            new_state = apply_graph_action(self._state, act)
        except ValueError as e:
            self._done = True
            return self.observe(), -1.0 * self.config.reward_scale, True, False, {
                "action_valid": False, "error": str(e)
            }

        self._state = new_state
        self._step_count += 1

        # Compute reward
        reward = self._compute_reward(act, new_state)

        stop = (act.action_type == GraphActionType.STOP_GENERATION)
        truncated = self._step_count >= self.config.max_steps
        self._done = stop or truncated or new_state.done

        obs = self.observe()
        info = {
            "action_valid": True,
            "stop": stop,
            "num_nodes": new_state.graph.num_nodes,
            "num_edges": new_state.graph.num_edges,
        }
        return obs, reward, self._done, truncated, info

    def _compute_reward(self, action: GraphAction, state: GraphEditState) -> float:
        g = state.graph
        reward = 0.0

        # Size penalty (small negative for each step)
        reward -= 0.01

        # Target score
        if self._target_score_fn is not None and g.num_nodes > 0:
            reward += self._target_score_fn(g) * 0.1

        # Target density bonus
        if "target_density" in self._target_properties and g.num_nodes > 1:
            n = g.num_nodes
            actual = g.num_edges / (n * (n - 1))
            target = self._target_properties["target_density"]
            reward += max(0, 1.0 - abs(actual - target)) * 0.5

        # Stop bonus if valid
        if action.action_type == GraphActionType.STOP_GENERATION and g.num_nodes >= 2:
            reward += 1.0

        return reward * self.config.reward_scale

    def observe(self) -> Dict[str, Any]:
        if self._state is None:
            self._state = self._make_initial_state()

        g = self._state.graph
        n = g.num_nodes

        node_feat = g.node_features.clone() if g.node_features is not None else \
            torch.zeros(max(n, 1), 1, device=self.device)

        # Graph stats embedding: [num_nodes, num_edges, density]
        density = g.num_edges / max(n * (n - 1), 1)
        graph_stats = torch.tensor(
            [float(n), float(g.num_edges), density],
            dtype=torch.float, device=self.device,
        )

        return {
            "edge_index": g.edge_index.clone(),
            "node_features": node_feat,
            "edge_features": g.edge_features.clone() if g.edge_features is not None else None,
            "action_mask": self.valid_action_mask(),
            "graph_stats_embedding": graph_stats,
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        if self._state is None:
            return torch.zeros(self._space.total_discrete_actions(), dtype=torch.bool, device=self.device)
        total = self._space.total_discrete_actions()
        mask = torch.zeros(total, dtype=torch.bool, device=self.device)
        valid_acts = enumerate_valid_actions(self._state, self._space)
        for act in valid_acts:
            idx = action_to_index(act, self._space)
            if 0 <= idx < total:
                mask[idx] = True
        return mask

    def state_to_graph(self) -> GeneratedGraph:
        if self._state is None:
            return GeneratedGraph(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=0,
            )
        return self._state.graph.clone()

    @property
    def num_nodes(self) -> int:
        if self._state is None:
            return 0
        return self._state.graph.num_nodes

    @property
    def num_edges(self) -> int:
        if self._state is None:
            return 0
        return self._state.graph.num_edges

    @property
    def action_space(self) -> int:
        return self._space.total_discrete_actions()
