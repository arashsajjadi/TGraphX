"""Graph navigation environment for RL.

The agent moves along edges to reach a target node.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig

__all__ = ["GraphNavigationEnv"]


class GraphNavigationEnv(GraphEnv):
    """Navigate a fixed graph to reach a target node.

    State: current node ID.
    Action: choose which neighbor to move to (discrete, masked by valid neighbors).
    Reward: +reward_reach if target reached, step_penalty per step.
    Done: reach target or max_steps.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional FloatTensor [N, F].
        edge_features: Optional FloatTensor [E, Fe].
        target_node: Target node ID.
        config: GraphEnvConfig.
        reward_reach: Reward for reaching the target.
        step_penalty: Penalty per step (negative).
        start_node: Starting node (default 0).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        target_node: int = 1,
        config: Optional[GraphEnvConfig] = None,
        reward_reach: float = 10.0,
        step_penalty: float = -0.1,
        start_node: int = 0,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._edge_index = edge_index.to(self.device)
        self._num_nodes = num_nodes
        self._node_features = node_features.to(self.device) if node_features is not None else None
        self._edge_features = edge_features.to(self.device) if edge_features is not None else None
        self._target_node = target_node
        self._reward_reach = reward_reach
        self._step_penalty = step_penalty
        self._start_node = start_node

        # Build adjacency list
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                self._adj[s].append(d)
                if not self.config.directed:
                    self._adj[d].append(s)

        self._current_node: int = start_node
        self._done: bool = False
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        self._current_node = self._start_node
        self._done = False
        self._step_count = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        neighbors = self._adj[self._current_node]
        if action < 0 or action >= len(neighbors):
            # Invalid action — mark done
            self._done = True
            return self.observe(), self._step_penalty * self.config.reward_scale, True, False, {
                "action_valid": False, "error": f"Invalid action {action}; {len(neighbors)} valid"
            }

        # Move to selected neighbor
        self._current_node = neighbors[action]
        self._step_count += 1

        reached = self._current_node == self._target_node
        truncated = self._step_count >= self.config.max_steps

        if reached:
            reward = self._reward_reach * self.config.reward_scale
            self._done = True
        else:
            reward = self._step_penalty * self.config.reward_scale
            self._done = truncated

        obs = self.observe()
        info = {
            "action_valid": True,
            "success": reached,
            "current_node": self._current_node,
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        mask = self.valid_action_mask()
        node_feat = self._node_features.clone() if self._node_features is not None else \
            torch.zeros(self._num_nodes, 1, device=self.device)

        return {
            "edge_index": self._edge_index.clone(),
            "node_features": node_feat,
            "edge_features": self._edge_features.clone() if self._edge_features is not None else None,
            "current_node": self._current_node,
            "target_node": self._target_node,
            "action_mask": mask,
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        neighbors = self._adj[self._current_node]
        max_actions = max(max(len(v) for v in self._adj.values()), 1) if self._adj else 1
        mask = torch.zeros(max_actions, dtype=torch.bool, device=self.device)
        mask[:len(neighbors)] = True
        return mask

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._edge_index.clone(),
            num_nodes=self._num_nodes,
            directed=self.config.directed,
            node_features=self._node_features.clone() if self._node_features is not None else None,
        )

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return int(self._edge_index.shape[1])

    @property
    def action_space(self) -> int:
        return max(max(len(v) for v in self._adj.values()), 1) if self._adj else 1
