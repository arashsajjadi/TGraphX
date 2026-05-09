"""Continuous-action graph environments for DDPG/TD3/SAC.

These environments expose a continuous action_dim-dimensional action space.
The action vector is an embedding that a decoder maps to graph operations.

Stability: Experimental (v0.7.0+)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig
from .navigation import GraphNavigationEnv

__all__ = [
    "ContinuousGraphActionSpace",
    "ContinuousNavigationEnv",
    "ContinuousGraphEditEnv",
]


class ContinuousGraphActionSpace:
    """Continuous action space: box [action_low, action_high]^action_dim.

    Args:
        action_dim: Dimension of action vector.
        action_low: Lower bound.
        action_high: Upper bound.
    """

    def __init__(
        self,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample uniformly from [action_low, action_high]^action_dim.

        Args:
            generator: Optional RNG.

        Returns:
            FloatTensor [action_dim].
        """
        u = torch.rand(self.action_dim, generator=generator)
        return u * (self.action_high - self.action_low) + self.action_low

    def clip(self, action: torch.Tensor) -> torch.Tensor:
        """Clip action to valid range.

        Args:
            action: FloatTensor.

        Returns:
            Clipped FloatTensor.
        """
        return action.clamp(self.action_low, self.action_high)


class ContinuousNavigationEnv:
    """Navigation with continuous action embedding decoded to neighbor selection.

    Same graph as GraphNavigationEnv but action is a float vector decoded to
    cosine-similarity ranking of neighbor embeddings.

    The continuous action vector is compared via cosine similarity to
    learned positional embeddings of neighbors. The neighbor with highest
    cosine similarity to the action is selected.

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        node_features: FloatTensor [N, F] (required).
        action_dim: Action embedding dimension.
        target_node: Target node ID.
        config: GraphEnvConfig.
        reward_reach: Reward for reaching target.
        step_penalty: Step penalty.
        start_node: Starting node.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_features: torch.Tensor,
        action_dim: int = 8,
        target_node: int = 1,
        config: Optional[GraphEnvConfig] = None,
        reward_reach: float = 10.0,
        step_penalty: float = -0.1,
        start_node: int = 0,
    ) -> None:
        self.config = config or GraphEnvConfig()
        self.device = torch.device(self.config.device)

        self._edge_index = edge_index.to(self.device)
        self._num_nodes = num_nodes
        self._node_features = node_features.to(self.device)
        self._action_dim = action_dim
        self._target_node = target_node
        self._reward_reach = reward_reach
        self._step_penalty = step_penalty
        self._start_node = start_node

        self.action_space = ContinuousGraphActionSpace(action_dim, -1.0, 1.0)

        # Build adjacency list
        self._adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        if edge_index.numel() > 0:
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                self._adj[s].append(d)
                if not self.config.directed:
                    self._adj[d].append(s)

        # Node feature projection to action_dim for cosine decoding
        feat_dim = node_features.shape[1]
        self._proj = torch.randn(feat_dim, action_dim, device=self.device) * 0.1
        self._proj = self._proj / (self._proj.norm(dim=0, keepdim=True) + 1e-8)

        self._current_node: int = start_node
        self._done: bool = False
        self._step_count: int = 0
        self._rng: Optional[torch.Generator] = None

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset environment.

        Returns:
            Observation dict with node_features, edge_index, action_space_bounds.
        """
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        self._current_node = self._start_node
        self._done = False
        self._step_count = 0
        return self.observe()

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step with continuous action.

        The action vector is decoded via cosine similarity to neighbor embeddings.

        Args:
            action: FloatTensor [action_dim].

        Returns:
            (obs, reward, done, truncated, info).
        """
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        neighbors = self._adj.get(self._current_node, [])

        if len(neighbors) == 0:
            self._done = True
            return self.observe(), self._step_penalty, True, False, {
                "action_valid": False, "error": "No neighbors"
            }

        # Decode: project action to node space, pick neighbor with max cosine sim
        action_t = torch.as_tensor(action, dtype=torch.float, device=self.device)
        if action_t.dim() == 0:
            action_t = action_t.unsqueeze(0)
        action_t = action_t.flatten()[:self._action_dim]
        if action_t.shape[0] < self._action_dim:
            action_t = torch.cat([action_t, torch.zeros(self._action_dim - action_t.shape[0], device=self.device)])

        # Embed each neighbor
        neighbor_feats = self._node_features[neighbors]  # [K, F]
        neighbor_embs = neighbor_feats @ self._proj       # [K, action_dim]
        neighbor_embs = neighbor_embs / (neighbor_embs.norm(dim=1, keepdim=True) + 1e-8)
        action_norm = action_t / (action_t.norm() + 1e-8)

        similarities = neighbor_embs @ action_norm  # [K]
        best_neighbor_idx = int(similarities.argmax().item())
        self._current_node = neighbors[best_neighbor_idx]
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
        """Return current observation."""
        return {
            "edge_index": self._edge_index.clone(),
            "node_features": self._node_features.clone(),
            "current_node": self._current_node,
            "target_node": self._target_node,
            "action_space_bounds": {
                "low": self.action_space.action_low,
                "high": self.action_space.action_high,
                "dim": self.action_space.action_dim,
            },
            "step": self._step_count,
            "done": self._done,
        }

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._edge_index.clone(),
            num_nodes=self._num_nodes,
            directed=self.config.directed,
            node_features=self._node_features.clone(),
        )


class ContinuousGraphEditEnv:
    """Graph editing with continuous action decoded to (add_node_prob, add_edge_prob, remove_prob, feature_delta).

    The action vector is split into segments controlling different edit operations.
    - action[:1] (sigmoid) -> probability to add a node
    - action[1:2] (sigmoid) -> probability to add an edge
    - action[2:3] (sigmoid) -> probability to remove an edge
    - action[3:] -> feature delta for random node

    Args:
        initial_graph: GeneratedGraph.
        action_dim: Action vector dimension (min 4).
        max_nodes: Maximum allowed nodes.
        max_edges: Maximum allowed edges.
        reward_fn: callable(graph: GeneratedGraph) -> float.
        config: GraphEnvConfig.
    """

    def __init__(
        self,
        initial_graph: GeneratedGraph,
        action_dim: int = 8,
        max_nodes: int = 20,
        max_edges: int = 100,
        reward_fn: Optional[Callable[[GeneratedGraph], float]] = None,
        config: Optional[GraphEnvConfig] = None,
    ) -> None:
        self.config = config or GraphEnvConfig()
        self.device = torch.device(self.config.device)
        self._initial_graph = initial_graph
        self._action_dim = action_dim
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._reward_fn = reward_fn or self._default_reward

        self.action_space = ContinuousGraphActionSpace(action_dim, -1.0, 1.0)

        self._graph: GeneratedGraph = initial_graph
        self._step_count: int = 0
        self._done: bool = False
        self._rng: Optional[torch.Generator] = None

    def _default_reward(self, graph: GeneratedGraph) -> float:
        """Default reward: fraction of reachable pairs (connectivity)."""
        n = graph.num_nodes
        if n <= 1:
            return 1.0
        e = int(graph.edge_index.shape[1])
        if e == 0:
            return 0.0
        max_e = n * (n - 1)
        return min(e / max_e, 1.0)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset to initial graph.

        Returns:
            Observation dict.
        """
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        self._graph = GeneratedGraph(
            edge_index=self._initial_graph.edge_index.clone(),
            num_nodes=self._initial_graph.num_nodes,
            node_features=self._initial_graph.node_features.clone() if self._initial_graph.node_features is not None else None,
        )
        self._step_count = 0
        self._done = False
        return self.observe()

    def decode_action(self, action: torch.Tensor) -> None:
        """Apply graph edit based on decoded action probabilities (in-place on self._graph).

        Args:
            action: FloatTensor [action_dim].
        """
        a = torch.sigmoid(action)
        add_node_prob = float(a[0].item()) if len(a) > 0 else 0.0
        add_edge_prob = float(a[1].item()) if len(a) > 1 else 0.0
        remove_prob = float(a[2].item()) if len(a) > 2 else 0.0

        n = self._graph.num_nodes
        ei = self._graph.edge_index

        # Add node
        if add_node_prob > 0.5 and n < self._max_nodes:
            n += 1
            nf = self._graph.node_features
            if nf is not None:
                feat_dim = nf.shape[1]
                new_feat = torch.randn(1, feat_dim, generator=self._rng)
                nf = torch.cat([nf, new_feat], dim=0)
            self._graph = GeneratedGraph(
                edge_index=ei, num_nodes=n, node_features=nf
            )

        n = self._graph.num_nodes
        ei = self._graph.edge_index
        num_edges = int(ei.shape[1])

        # Add edge
        if add_edge_prob > 0.5 and num_edges < self._max_edges and n >= 2:
            src = int(torch.randint(n, (1,), generator=self._rng).item())
            dst = int(torch.randint(n, (1,), generator=self._rng).item())
            if src != dst:
                new_edge = torch.tensor([[src], [dst]], dtype=torch.long)
                ei = torch.cat([ei, new_edge], dim=1)
                self._graph = GeneratedGraph(
                    edge_index=ei, num_nodes=n, node_features=self._graph.node_features
                )

        # Remove edge
        if remove_prob > 0.5 and num_edges > 0:
            ei = self._graph.edge_index
            num_edges = int(ei.shape[1])
            if num_edges > 0:
                rm_idx = int(torch.randint(num_edges, (1,), generator=self._rng).item())
                keep = [i for i in range(num_edges) if i != rm_idx]
                if keep:
                    ei = ei[:, keep]
                else:
                    ei = torch.zeros((2, 0), dtype=torch.long)
                self._graph = GeneratedGraph(
                    edge_index=ei, num_nodes=n, node_features=self._graph.node_features
                )

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Take a step with continuous action.

        Args:
            action: FloatTensor [action_dim].

        Returns:
            (obs, reward, done, truncated, info).
        """
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        action_t = torch.as_tensor(action, dtype=torch.float)
        self.decode_action(action_t)

        reward = self._reward_fn(self._graph)
        self._step_count += 1
        truncated = self._step_count >= self.config.max_steps
        self._done = truncated

        obs = self.observe()
        info = {
            "action_valid": True,
            "num_nodes": self._graph.num_nodes,
            "num_edges": int(self._graph.edge_index.shape[1]),
            "reward": reward,
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        """Return current observation."""
        nf = self._graph.node_features
        if nf is None:
            nf = torch.zeros(self._graph.num_nodes, 1, device=self.device)
        return {
            "edge_index": self._graph.edge_index.clone(),
            "node_features": nf.clone(),
            "num_nodes": self._graph.num_nodes,
            "num_edges": int(self._graph.edge_index.shape[1]),
            "action_space_bounds": {
                "low": self.action_space.action_low,
                "high": self.action_space.action_high,
                "dim": self.action_space.action_dim,
            },
            "step": self._step_count,
            "done": self._done,
        }

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._graph.edge_index.clone(),
            num_nodes=self._graph.num_nodes,
            node_features=self._graph.node_features.clone() if self._graph.node_features is not None else None,
        )
