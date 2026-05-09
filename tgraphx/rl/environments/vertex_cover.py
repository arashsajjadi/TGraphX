"""Vertex cover environment for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig

__all__ = ["VertexCoverEnv"]


class VertexCoverEnv(GraphEnv):
    """Find a minimum vertex cover.

    State: current cover set (bool per node).
    Action: select any uncovered node to add to the cover.
    Reward: -1 per node added, +edge_coverage_bonus for each newly covered edge.
    Done: all edges covered or max_steps.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional FloatTensor [N, F].
        config: GraphEnvConfig.
        edge_coverage_bonus: Bonus per newly covered edge.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: Optional[torch.Tensor] = None,
        config: Optional[GraphEnvConfig] = None,
        edge_coverage_bonus: float = 0.5,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._edge_index = edge_index.to(self.device)
        self._num_nodes = num_nodes
        self._node_features = node_features.to(self.device) if node_features is not None else None
        self._edge_coverage_bonus = edge_coverage_bonus

        # Build edge set for coverage check
        self._edges: List[Tuple[int, int]] = []
        if edge_index.numel() > 0:
            seen: Set[tuple] = set()
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                key = (min(s, d), max(s, d))
                if key not in seen:
                    seen.add(key)
                    self._edges.append(key)

        self._cover: List[bool] = [False] * num_nodes
        self._done: bool = False
        self._step_count: int = 0

    def _covered_edges(self) -> int:
        return sum(
            1 for s, d in self._edges
            if self._cover[s] or self._cover[d]
        )

    def _all_covered(self) -> bool:
        return self._covered_edges() == len(self._edges)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        self._cover = [False] * self._num_nodes
        self._done = False
        self._step_count = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        if action < 0 or action >= self._num_nodes:
            self._done = True
            return self.observe(), -1.0 * self.config.reward_scale, True, False, {
                "action_valid": False, "error": f"Invalid action {action}"
            }

        old_covered = self._covered_edges()
        self._cover[action] = True
        new_covered = self._covered_edges()
        newly_covered = new_covered - old_covered

        self._step_count += 1
        reward = (-1.0 + self._edge_coverage_bonus * newly_covered) * self.config.reward_scale

        truncated = self._step_count >= self.config.max_steps
        self._done = self._all_covered() or truncated

        obs = self.observe()
        info = {
            "action_valid": True,
            "covered_edges": new_covered,
            "cover_size": sum(self._cover),
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        cover_t = torch.tensor(self._cover, dtype=torch.bool, device=self.device)
        node_feat = self._node_features.clone() if self._node_features is not None else \
            torch.zeros(self._num_nodes, 1, device=self.device)
        return {
            "edge_index": self._edge_index.clone(),
            "node_features": node_feat,
            "cover": cover_t,
            "action_mask": self.valid_action_mask(),
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        mask = torch.ones(self._num_nodes, dtype=torch.bool, device=self.device)
        for i, c in enumerate(self._cover):
            if c:
                mask[i] = False
        return mask

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
        return self._num_nodes
