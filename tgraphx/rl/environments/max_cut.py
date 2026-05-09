"""Max-Cut environment for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig

__all__ = ["MaxCutEnv"]


class MaxCutEnv(GraphEnv):
    """Assign nodes to partition 0 or 1 to maximize the cut.

    State: current partition assignment (per-node, -1 = unassigned).
    Action: assign next unassigned node to partition 0 or 1.
    Reward: delta cut value after each assignment (dense reward).
    Done: all nodes assigned.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional FloatTensor [N, F].
        config: GraphEnvConfig.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: Optional[torch.Tensor] = None,
        config: Optional[GraphEnvConfig] = None,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._edge_index = edge_index.to(self.device)
        self._num_nodes = num_nodes
        self._node_features = node_features.to(self.device) if node_features is not None else None

        # Build adjacency
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                self._adj[s].append(d)
                self._adj[d].append(s)

        self._assignment: List[int] = [-1] * num_nodes
        self._current_node: int = 0
        self._done: bool = False
        self._step_count: int = 0

    def _cut_value(self) -> int:
        cut = 0
        for s, d in zip(self._edge_index[0].tolist(), self._edge_index[1].tolist()):
            if self._assignment[s] != -1 and self._assignment[d] != -1:
                if self._assignment[s] != self._assignment[d]:
                    cut += 1
        return cut // 2  # undirected

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        self._assignment = [-1] * self._num_nodes
        self._current_node = 0
        self._done = False
        self._step_count = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        if action not in (0, 1):
            self._done = True
            return self.observe(), -1.0 * self.config.reward_scale, True, False, {
                "action_valid": False, "error": f"action must be 0 or 1, got {action}"
            }

        # Assign current node
        old_cut = self._cut_value()
        self._assignment[self._current_node] = action
        new_cut = self._cut_value()
        delta_cut = new_cut - old_cut

        self._current_node += 1
        self._step_count += 1

        all_assigned = self._current_node >= self._num_nodes
        truncated = self._step_count >= self.config.max_steps
        self._done = all_assigned or truncated

        reward = float(delta_cut) * self.config.reward_scale
        obs = self.observe()
        info = {
            "action_valid": True,
            "cut_value": new_cut,
            "delta_cut": delta_cut,
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        assignment_t = torch.tensor(self._assignment, dtype=torch.long, device=self.device)
        node_feat = self._node_features.clone() if self._node_features is not None else \
            torch.zeros(self._num_nodes, 1, device=self.device)
        return {
            "edge_index": self._edge_index.clone(),
            "node_features": node_feat,
            "assignment": assignment_t,
            "current_node": self._current_node,
            "action_mask": self.valid_action_mask(),
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        if self._current_node >= self._num_nodes:
            return torch.zeros(2, dtype=torch.bool, device=self.device)
        return torch.ones(2, dtype=torch.bool, device=self.device)

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._edge_index.clone(),
            num_nodes=self._num_nodes,
            directed=False,
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
        return 2
