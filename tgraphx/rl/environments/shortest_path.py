"""Shortest path environment.

Agent navigates graph trying to find shortest path to target.
Oracle (BFS) provides optimal path for regret calculation.

Stability: Experimental (v0.7.0+)
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch

from .navigation import GraphNavigationEnv
from .base import GraphEnvConfig

__all__ = ["ShortestPathEnv"]


def _bfs_shortest_path(
    adj: Dict[int, List[int]],
    start: int,
    target: int,
    num_nodes: int,
) -> Optional[List[int]]:
    """Compute BFS shortest path from start to target.

    Args:
        adj: Adjacency list.
        start: Source node.
        target: Target node.
        num_nodes: Total number of nodes.

    Returns:
        List of node IDs on shortest path (inclusive), or None if unreachable.
    """
    if start == target:
        return [start]

    visited = {start: None}  # node -> parent
    queue: deque = deque([start])

    while queue:
        node = queue.popleft()
        for nb in adj.get(node, []):
            if nb not in visited:
                visited[nb] = node
                if nb == target:
                    # Reconstruct path
                    path = [target]
                    cur = target
                    while visited[cur] is not None:
                        cur = visited[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(nb)

    return None  # unreachable


class ShortestPathEnv(GraphNavigationEnv):
    """Like GraphNavigationEnv but reward shaped toward optimal path.

    Oracle BFS computes the shortest path at reset time.

    Reward per step:
        +1.0 bonus if reach target in optimal_length steps.
        -(1/optimal_length) if moving along the shortest path.
        -0.2 penalty if deviating from shortest path.

    Extra info keys: 'optimal_length', 'path_length', 'regret'.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: Optional FloatTensor [N, F].
        edge_features: Optional FloatTensor [E, Fe].
        target_node: Target node ID.
        config: GraphEnvConfig.
        start_node: Starting node.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        target_node: int = 1,
        config: Optional[GraphEnvConfig] = None,
        start_node: int = 0,
    ) -> None:
        super().__init__(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_features=node_features,
            edge_features=edge_features,
            target_node=target_node,
            config=config,
            reward_reach=1.0,
            step_penalty=-0.2,
            start_node=start_node,
        )

        self._optimal_path: Optional[List[int]] = None
        self._optimal_length: int = 0
        self._path_length: int = 0  # steps taken so far
        self._compute_oracle()

    def _compute_oracle(self) -> None:
        """Compute BFS oracle at construction time."""
        path = _bfs_shortest_path(
            self._adj, self._start_node, self._target_node, self._num_nodes
        )
        self._optimal_path = path
        if path is not None:
            # Length = number of steps = len(path) - 1
            self._optimal_length = max(len(path) - 1, 1)
        else:
            self._optimal_length = self._num_nodes  # fallback

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset and recompute oracle."""
        obs = super().reset(seed=seed)
        self._compute_oracle()
        self._path_length = 0
        return obs

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step with shaped reward.

        Returns:
            (obs, reward, done, truncated, info) with extra info keys.
        """
        prev_node = self._current_node
        obs, _, done, truncated, info = super().step(action)
        self._path_length += 1

        reached = info.get("success", False)
        on_path = (
            self._optimal_path is not None
            and prev_node in self._optimal_path
            and self._current_node in self._optimal_path
        )

        if reached:
            if self._path_length <= self._optimal_length:
                # Reached in optimal or fewer steps
                reward = 1.0
            else:
                # Reached but not optimally
                reward = 0.5
        else:
            if on_path:
                # Moving along optimal path
                reward = -(1.0 / max(self._optimal_length, 1))
            else:
                reward = -0.2

        reward *= self.config.reward_scale

        regret = max(0, self._path_length - self._optimal_length) if reached else self._optimal_length

        info["optimal_length"] = self._optimal_length
        info["path_length"] = self._path_length
        info["regret"] = regret

        return obs, reward, done, truncated, info
