"""Graph coloring environment for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig

__all__ = ["GraphColoringEnv"]


class GraphColoringEnv(GraphEnv):
    """Assign colors to nodes one at a time.

    State: current coloring assignment (per-node color or -1 if uncolored).
    Action: assign a color (0..num_colors-1) to the next uncolored node.
    Reward: -conflict_count at each step, +big_bonus if valid coloring complete.
    Done: all nodes colored or max_steps.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional FloatTensor [N, F].
        num_colors: Number of available colors.
        config: GraphEnvConfig.
        completion_bonus: Bonus for valid complete coloring.
        conflict_penalty: Penalty per conflict per step.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: Optional[torch.Tensor] = None,
        num_colors: int = 3,
        config: Optional[GraphEnvConfig] = None,
        completion_bonus: float = 20.0,
        conflict_penalty: float = 1.0,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._edge_index = edge_index.to(self.device)
        self._num_nodes = num_nodes
        self._node_features = node_features.to(self.device) if node_features is not None else None
        self._num_colors = num_colors
        self._completion_bonus = completion_bonus
        self._conflict_penalty = conflict_penalty

        # Build adjacency
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                self._adj[s].append(d)
                self._adj[d].append(s)

        self._coloring: List[int] = [-1] * num_nodes
        self._current_node: int = 0
        self._done: bool = False
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        self._coloring = [-1] * self._num_nodes
        self._current_node = 0
        self._done = False
        self._step_count = 0
        return self.observe()

    def _count_conflicts(self) -> int:
        count = 0
        for s, d in zip(self._edge_index[0].tolist(), self._edge_index[1].tolist()):
            if self._coloring[s] != -1 and self._coloring[d] != -1:
                if self._coloring[s] == self._coloring[d]:
                    count += 1
        return count // 2  # undirected

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        if action < 0 or action >= self._num_colors:
            self._done = True
            return self.observe(), -self._conflict_penalty * self.config.reward_scale, True, False, {
                "action_valid": False, "error": f"Invalid action {action}"
            }

        # Color current node
        self._coloring[self._current_node] = action
        self._current_node += 1
        self._step_count += 1

        # Count conflicts
        conflicts = self._count_conflicts()
        reward = -self._conflict_penalty * conflicts * self.config.reward_scale

        all_colored = self._current_node >= self._num_nodes
        truncated = self._step_count >= self.config.max_steps

        if all_colored and conflicts == 0:
            reward += self._completion_bonus * self.config.reward_scale
            self._done = True
        elif all_colored or truncated:
            self._done = True

        obs = self.observe()
        info = {
            "action_valid": True,
            "conflicts": conflicts,
            "success": all_colored and conflicts == 0,
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        coloring_t = torch.tensor(self._coloring, dtype=torch.long, device=self.device)
        node_feat = self._node_features.clone() if self._node_features is not None else \
            torch.zeros(self._num_nodes, 1, device=self.device)
        return {
            "edge_index": self._edge_index.clone(),
            "node_features": node_feat,
            "coloring": coloring_t,
            "current_node": self._current_node,
            "action_mask": self.valid_action_mask(),
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        # All colors are valid (no preemptive constraint filtering)
        return torch.ones(self._num_colors, dtype=torch.bool, device=self.device)

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._edge_index.clone(),
            num_nodes=self._num_nodes,
            directed=False,
            node_features=self._node_features.clone() if self._node_features is not None else None,
            metadata={"coloring": list(self._coloring)},
        )

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return int(self._edge_index.shape[1])

    @property
    def action_space(self) -> int:
        return self._num_colors
