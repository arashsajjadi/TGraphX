"""Knowledge graph path reasoning environment for RL.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from tgraphx.generation.data_model import GeneratedGraph
from .base import GraphEnv, GraphEnvConfig

__all__ = ["KGPathReasoningEnv"]


class KGPathReasoningEnv(GraphEnv):
    """Navigate a knowledge graph to answer queries.

    The agent starts at a head entity and must reach the tail entity.

    State: current entity in the KG traversal.
    Action: choose an outgoing (relation, entity) pair.
    Reward: +reward_reach if reach target, step_penalty per step.
    Done: reach target or max_steps.

    Args:
        kg_edge_index: LongTensor [2, E] — (head, tail) entity pairs.
        relation_types: LongTensor [E] — relation type per edge.
        num_entities: Total number of entities.
        num_relations: Total number of relation types.
        query_pairs: List of (head, tail) query pairs to cycle through.
        config: GraphEnvConfig.
        reward_reach: Reward for reaching target.
        step_penalty: Penalty per step.
    """

    def __init__(
        self,
        kg_edge_index: torch.Tensor,
        relation_types: torch.Tensor,
        num_entities: int,
        num_relations: int,
        query_pairs: List[Tuple[int, int]],
        config: Optional[GraphEnvConfig] = None,
        reward_reach: float = 1.0,
        step_penalty: float = -0.05,
    ) -> None:
        self.config = config or GraphEnvConfig()
        super().__init__(self.config)

        self._kg_edge_index = kg_edge_index.to(self.device)
        self._relation_types = relation_types.to(self.device)
        self._num_entities = num_entities
        self._num_relations = num_relations
        self._query_pairs = query_pairs
        self._reward_reach = reward_reach
        self._step_penalty = step_penalty

        # Build out-edge adjacency: entity -> list of (relation, target_entity)
        self._out_adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_entities)}
        if kg_edge_index.numel() > 0:
            for i, (h, t) in enumerate(zip(kg_edge_index[0].tolist(), kg_edge_index[1].tolist())):
                r = int(relation_types[i].item())
                self._out_adj[h].append((r, t))

        self._max_actions = max(len(v) for v in self._out_adj.values()) if num_entities > 0 else 1

        self._current_entity: int = 0
        self._target_entity: int = 0
        self._query_idx: int = 0
        self._done: bool = False
        self._step_count: int = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(seed)
        elif self.config.seed is not None:
            self._rng = torch.Generator()
            self._rng.manual_seed(self.config.seed)

        if self._query_pairs:
            head, tail = self._query_pairs[self._query_idx % len(self._query_pairs)]
            self._current_entity = head
            self._target_entity = tail
            self._query_idx += 1
        else:
            self._current_entity = 0
            self._target_entity = 0

        self._done = False
        self._step_count = 0
        return self.observe()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._done:
            return self.observe(), 0.0, True, False, {"action_valid": False}

        out_edges = self._out_adj[self._current_entity]
        if action < 0 or action >= len(out_edges):
            self._done = True
            return self.observe(), self._step_penalty * self.config.reward_scale, True, False, {
                "action_valid": False,
                "error": f"Invalid action {action}; {len(out_edges)} valid",
            }

        rel, next_entity = out_edges[action]
        self._current_entity = next_entity
        self._step_count += 1

        reached = self._current_entity == self._target_entity
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
            "relation_taken": rel,
            "current_entity": self._current_entity,
            "success": reached,
        }
        return obs, reward, self._done, truncated, info

    def observe(self) -> Dict[str, Any]:
        out_edges = self._out_adj[self._current_entity]
        mask = self.valid_action_mask()
        return {
            "edge_index": self._kg_edge_index.clone(),
            "node_features": torch.zeros(self._num_entities, 1, device=self.device),
            "relation_types": self._relation_types.clone(),
            "current_entity": self._current_entity,
            "target_entity": self._target_entity,
            "action_mask": mask,
            "step": self._step_count,
            "done": self._done,
        }

    def valid_action_mask(self) -> torch.Tensor:
        out_edges = self._out_adj[self._current_entity]
        mask = torch.zeros(self._max_actions, dtype=torch.bool, device=self.device)
        mask[:len(out_edges)] = True
        return mask

    def state_to_graph(self) -> GeneratedGraph:
        return GeneratedGraph(
            edge_index=self._kg_edge_index.clone(),
            num_nodes=self._num_entities,
            directed=True,
        )

    @property
    def num_nodes(self) -> int:
        return self._num_entities

    @property
    def num_edges(self) -> int:
        return int(self._kg_edge_index.shape[1])

    @property
    def action_space(self) -> int:
        return self._max_actions
