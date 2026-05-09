"""Data model for graph generation.

Core data structures used throughout the generation subsystem.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


__all__ = [
    "GeneratedGraph",
    "GraphEditState",
    "GraphGenerationTrajectory",
    "GraphGenerationBatch",
    "graph_to_generation_state",
    "generation_state_to_graph",
    "validate_generated_graph",
    "graph_generation_summary",
]


def _tensor_to_shape(t: Optional[torch.Tensor]) -> Optional[list]:
    if t is None:
        return None
    return list(t.shape)


def _no_raw_tensors(d: Any) -> bool:
    """Recursively verify no Tensor values exist in d."""
    if isinstance(d, torch.Tensor):
        return False
    if isinstance(d, dict):
        return all(_no_raw_tensors(v) for v in d.values())
    if isinstance(d, (list, tuple)):
        return all(_no_raw_tensors(v) for v in d)
    return True


@dataclass
class GeneratedGraph:
    """A generated graph with full tensor support.

    Supports node features of shapes:
        - [N, F]          — vector features
        - [N, C, H, W]    — image features (require ImageNodeEncoder for GNNs)
        - [N, C, D, H, W] — volumetric features (require VolumeNodeEncoder)

    Args:
        edge_index: LongTensor [2, E].
        num_nodes: Number of nodes.
        directed: Whether the graph is directed.
        node_features: Optional feature tensor.
        edge_features: Optional edge feature tensor [E, *].
        edge_weight: Optional edge weight tensor [E].
        graph_features: Optional graph-level feature tensor.
        node_types: Optional LongTensor [N] node type IDs.
        edge_types: Optional LongTensor [E] edge type IDs.
        timestamps: Optional FloatTensor [E] edge timestamps.
        valid_node_mask: Optional BoolTensor [N].
        valid_edge_mask: Optional BoolTensor [E].
        action_mask: Optional BoolTensor for action space.
        metadata: Arbitrary metadata dict (must not contain raw tensors for serialization).
    """

    edge_index: torch.Tensor
    num_nodes: int
    directed: bool = False
    node_features: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None
    edge_weight: Optional[torch.Tensor] = None
    graph_features: Optional[torch.Tensor] = None
    node_types: Optional[torch.Tensor] = None
    edge_types: Optional[torch.Tensor] = None
    timestamps: Optional[torch.Tensor] = None
    valid_node_mask: Optional[torch.Tensor] = None
    valid_edge_mask: Optional[torch.Tensor] = None
    action_mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate internal consistency. Raises ValueError with a clear message."""
        if not isinstance(self.edge_index, torch.Tensor):
            raise ValueError("edge_index must be a torch.Tensor")
        if self.edge_index.dim() != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must have shape [2, E] but got {list(self.edge_index.shape)}"
            )
        if self.edge_index.dtype != torch.long:
            raise ValueError(
                f"edge_index must be dtype torch.long but got {self.edge_index.dtype}"
            )
        num_edges = self.edge_index.shape[1]
        if num_edges > 0:
            max_id = int(self.edge_index.max().item())
            if max_id >= self.num_nodes:
                raise ValueError(
                    f"edge_index contains node ID {max_id} but num_nodes={self.num_nodes}"
                )
            if int(self.edge_index.min().item()) < 0:
                raise ValueError("edge_index contains negative node IDs")

        if self.node_features is not None:
            if self.node_features.shape[0] != self.num_nodes:
                raise ValueError(
                    f"node_features.shape[0]={self.node_features.shape[0]} "
                    f"!= num_nodes={self.num_nodes}"
                )
        if self.edge_features is not None:
            if self.edge_features.shape[0] != num_edges:
                raise ValueError(
                    f"edge_features.shape[0]={self.edge_features.shape[0]} "
                    f"!= num_edges={num_edges}"
                )
        if self.edge_weight is not None:
            if self.edge_weight.shape[0] != num_edges:
                raise ValueError(
                    f"edge_weight.shape[0]={self.edge_weight.shape[0]} "
                    f"!= num_edges={num_edges}"
                )
        if self.node_types is not None:
            if self.node_types.shape[0] != self.num_nodes:
                raise ValueError(
                    f"node_types.shape[0]={self.node_types.shape[0]} "
                    f"!= num_nodes={self.num_nodes}"
                )
        if self.edge_types is not None:
            if self.edge_types.shape[0] != num_edges:
                raise ValueError(
                    f"edge_types.shape[0]={self.edge_types.shape[0]} "
                    f"!= num_edges={num_edges}"
                )
        if self.valid_node_mask is not None:
            if self.valid_node_mask.shape[0] != self.num_nodes:
                raise ValueError(
                    f"valid_node_mask.shape[0]={self.valid_node_mask.shape[0]} "
                    f"!= num_nodes={self.num_nodes}"
                )
        if self.valid_edge_mask is not None:
            if self.valid_edge_mask.shape[0] != num_edges:
                raise ValueError(
                    f"valid_edge_mask.shape[0]={self.valid_edge_mask.shape[0]} "
                    f"!= num_edges={num_edges}"
                )

    def to(self, device: torch.device) -> "GeneratedGraph":
        """Move all tensors to device. Returns a new GeneratedGraph."""
        def _mv(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device) if t is not None else None

        return GeneratedGraph(
            edge_index=self.edge_index.to(device),
            num_nodes=self.num_nodes,
            directed=self.directed,
            node_features=_mv(self.node_features),
            edge_features=_mv(self.edge_features),
            edge_weight=_mv(self.edge_weight),
            graph_features=_mv(self.graph_features),
            node_types=_mv(self.node_types),
            edge_types=_mv(self.edge_types),
            timestamps=_mv(self.timestamps),
            valid_node_mask=_mv(self.valid_node_mask),
            valid_edge_mask=_mv(self.valid_edge_mask),
            action_mask=_mv(self.action_mask),
            metadata=copy.deepcopy(self.metadata),
        )

    def clone(self) -> "GeneratedGraph":
        """Deep clone of this graph."""
        def _cl(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.clone() if t is not None else None

        return GeneratedGraph(
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            directed=self.directed,
            node_features=_cl(self.node_features),
            edge_features=_cl(self.edge_features),
            edge_weight=_cl(self.edge_weight),
            graph_features=_cl(self.graph_features),
            node_types=_cl(self.node_types),
            edge_types=_cl(self.edge_types),
            timestamps=_cl(self.timestamps),
            valid_node_mask=_cl(self.valid_node_mask),
            valid_edge_mask=_cl(self.valid_edge_mask),
            action_mask=_cl(self.action_mask),
            metadata=copy.deepcopy(self.metadata),
        )

    def detach_for_report(self) -> Dict[str, Any]:
        """Return a JSON-safe dict with shapes only — NO raw tensors.

        All tensor values are replaced with their shape as a list.
        """
        result: Dict[str, Any] = {
            "num_nodes": self.num_nodes,
            "num_edges": int(self.edge_index.shape[1]),
            "directed": self.directed,
            "edge_index_shape": _tensor_to_shape(self.edge_index),
            "node_features_shape": _tensor_to_shape(self.node_features),
            "edge_features_shape": _tensor_to_shape(self.edge_features),
            "edge_weight_shape": _tensor_to_shape(self.edge_weight),
            "graph_features_shape": _tensor_to_shape(self.graph_features),
            "node_types_shape": _tensor_to_shape(self.node_types),
            "edge_types_shape": _tensor_to_shape(self.edge_types),
            "timestamps_shape": _tensor_to_shape(self.timestamps),
            "valid_node_mask_shape": _tensor_to_shape(self.valid_node_mask),
            "valid_edge_mask_shape": _tensor_to_shape(self.valid_edge_mask),
            "action_mask_shape": _tensor_to_shape(self.action_mask),
        }
        # Include non-tensor metadata values only
        safe_meta: Dict[str, Any] = {}
        for k, v in self.metadata.items():
            if isinstance(v, torch.Tensor):
                safe_meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                safe_meta[k] = v
        result["metadata"] = safe_meta
        return result

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def device(self) -> torch.device:
        return self.edge_index.device


@dataclass
class GraphEditState:
    """Current graph being built during autoregressive generation.

    Tracks the full trajectory of editing steps.

    Args:
        graph: Current GeneratedGraph.
        step: Current step count.
        trajectory: List of (state_summary, action_repr, reward) triples.
        constraints: Optional constraint dict.
        done: Whether generation has been stopped.
    """

    graph: GeneratedGraph
    step: int = 0
    trajectory: List[Tuple[Dict, Any, float]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    done: bool = False

    def clone(self) -> "GraphEditState":
        return GraphEditState(
            graph=self.graph.clone(),
            step=self.step,
            trajectory=list(self.trajectory),
            constraints=copy.deepcopy(self.constraints),
            done=self.done,
        )


@dataclass
class GraphGenerationTrajectory:
    """Sequence of (state, action, reward) triples from a generation episode.

    Args:
        states: List of GraphEditState snapshots.
        actions: List of action representations.
        rewards: List of rewards.
        total_return: Cumulative sum of rewards.
    """

    states: List[GraphEditState] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        return sum(self.rewards)

    @property
    def length(self) -> int:
        return len(self.rewards)


@dataclass
class GraphGenerationBatch:
    """Batch of GeneratedGraphs with proper node-ID offsets.

    Edges from different graphs are offset so they can be used in batch GNN.

    Args:
        graphs: List of GeneratedGraph.
        edge_index_batch: Combined edge_index with offsets [2, sum_E].
        batch_vector: LongTensor [sum_N] mapping each node to graph index.
        node_features_batch: Optional stacked or concatenated node features.
    """

    graphs: List[GeneratedGraph]
    edge_index_batch: torch.Tensor
    batch_vector: torch.Tensor
    node_features_batch: Optional[torch.Tensor] = None

    @classmethod
    def from_graphs(cls, graphs: List[GeneratedGraph]) -> "GraphGenerationBatch":
        """Build a batched representation from a list of GeneratedGraphs."""
        if not graphs:
            ei = torch.zeros((2, 0), dtype=torch.long)
            bv = torch.zeros(0, dtype=torch.long)
            return cls(graphs=[], edge_index_batch=ei, batch_vector=bv)

        edge_indices = []
        batch_parts = []
        offset = 0

        for i, g in enumerate(graphs):
            edge_indices.append(g.edge_index + offset)
            batch_parts.append(torch.full((g.num_nodes,), i, dtype=torch.long))
            offset += g.num_nodes

        edge_index_batch = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)
        batch_vector = torch.cat(batch_parts) if batch_parts else torch.zeros(0, dtype=torch.long)

        # Stack node features if compatible
        node_features_batch: Optional[torch.Tensor] = None
        feats = [g.node_features for g in graphs]
        if all(f is not None for f in feats):
            all_feats: List[torch.Tensor] = [f for f in feats if f is not None]  # type: ignore
            if len(set(f.shape[1:] for f in all_feats)) == 1:
                node_features_batch = torch.cat(all_feats, dim=0)

        return cls(
            graphs=graphs,
            edge_index_batch=edge_index_batch,
            batch_vector=batch_vector,
            node_features_batch=node_features_batch,
        )

    def to(self, device: torch.device) -> "GraphGenerationBatch":
        graphs = [g.to(device) for g in self.graphs]
        return GraphGenerationBatch(
            graphs=graphs,
            edge_index_batch=self.edge_index_batch.to(device),
            batch_vector=self.batch_vector.to(device),
            node_features_batch=self.node_features_batch.to(device) if self.node_features_batch is not None else None,
        )


# ── Conversion utilities ──────────────────────────────────────────────────────


def graph_to_generation_state(graph: Any) -> GraphEditState:
    """Convert a tgraphx.Graph to a GraphEditState.

    Args:
        graph: A tgraphx.core.graph.Graph instance.

    Returns:
        GraphEditState wrapping the graph's data.
    """
    nf = getattr(graph, "node_features", None)
    ef = getattr(graph, "edge_features", None)
    ew = getattr(graph, "edge_weight", None)
    ei = graph.edge_index
    nn = graph.num_nodes if hasattr(graph, "num_nodes") else int(ei.max().item()) + 1 if ei.numel() > 0 else 0
    meta = dict(getattr(graph, "metadata", {}) or {})

    gen_graph = GeneratedGraph(
        edge_index=ei.clone(),
        num_nodes=nn,
        directed=True,  # conservatively treat as directed
        node_features=nf.clone() if nf is not None else None,
        edge_features=ef.clone() if ef is not None else None,
        edge_weight=ew.clone() if ew is not None else None,
        metadata=meta,
    )
    return GraphEditState(graph=gen_graph)


def generation_state_to_graph(state: GraphEditState) -> Any:
    """Convert a GraphEditState back to a tgraphx.Graph.

    Args:
        state: GraphEditState to convert.

    Returns:
        tgraphx.core.graph.Graph instance.
    """
    from tgraphx.core.graph import Graph

    g = state.graph
    return Graph(
        node_features=g.node_features,
        edge_index=g.edge_index,
        edge_features=g.edge_features,
        edge_weight=g.edge_weight,
        metadata=dict(g.metadata),
    )


def validate_generated_graph(
    graph: GeneratedGraph,
    constraints: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Validate a GeneratedGraph against constraints.

    Args:
        graph: Graph to validate.
        constraints: Dict with optional keys:
            - max_nodes (int)
            - max_edges (int)
            - no_self_loops (bool)
            - connected (bool) — weakly connected
            - acyclic (bool)
            - min_nodes (int)

    Returns:
        (valid, violations) where violations is a list of human-readable strings.
    """
    violations: List[str] = []

    try:
        graph.validate()
    except ValueError as e:
        violations.append(f"Structure error: {e}")
        return False, violations

    n = graph.num_nodes
    e = graph.num_edges

    if "max_nodes" in constraints and n > constraints["max_nodes"]:
        violations.append(f"num_nodes={n} > max_nodes={constraints['max_nodes']}")
    if "min_nodes" in constraints and n < constraints["min_nodes"]:
        violations.append(f"num_nodes={n} < min_nodes={constraints['min_nodes']}")
    if "max_edges" in constraints and e > constraints["max_edges"]:
        violations.append(f"num_edges={e} > max_edges={constraints['max_edges']}")

    if constraints.get("no_self_loops", False) and e > 0:
        src, dst = graph.edge_index[0], graph.edge_index[1]
        if (src == dst).any():
            violations.append("Graph contains self-loops but no_self_loops=True")

    if constraints.get("connected", False) and n > 0 and e > 0:
        # BFS reachability from node 0
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        src_arr = graph.edge_index[0].tolist()
        dst_arr = graph.edge_index[1].tolist()
        for s, d in zip(src_arr, dst_arr):
            adj[s].append(d)
            adj[d].append(s)
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) < n:
            violations.append(
                f"Graph is disconnected: only {len(visited)}/{n} nodes reachable"
            )

    if constraints.get("acyclic", False) and n > 0:
        # DFS cycle detection for directed graphs
        adj_d: Dict[int, List[int]] = {i: [] for i in range(n)}
        for s, d in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()):
            adj_d[s].append(d)
        color = [0] * n  # 0=white, 1=gray, 2=black
        has_cycle = False

        def dfs(v: int) -> None:
            nonlocal has_cycle
            if has_cycle:
                return
            color[v] = 1
            for u in adj_d[v]:
                if color[u] == 1:
                    has_cycle = True
                    return
                if color[u] == 0:
                    dfs(u)
            color[v] = 2

        for start in range(n):
            if color[start] == 0:
                dfs(start)

        if has_cycle:
            violations.append("Graph contains cycle but acyclic=True")

    return len(violations) == 0, violations


def graph_generation_summary(graph: GeneratedGraph) -> Dict[str, Any]:
    """Return a JSON-safe summary dict for a GeneratedGraph.

    No raw tensors — all tensors are replaced with shape summaries.
    """
    summary = graph.detach_for_report()
    # Add extra stats
    if graph.num_edges > 0 and graph.num_nodes > 0:
        # Degree statistics
        degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
        degrees.scatter_add_(
            0,
            graph.edge_index[1],
            torch.ones(graph.num_edges, dtype=torch.long),
        )
        summary["degree_mean"] = float(degrees.float().mean().item())
        summary["degree_max"] = int(degrees.max().item())
        summary["degree_min"] = int(degrees.min().item())
    else:
        summary["degree_mean"] = 0.0
        summary["degree_max"] = 0
        summary["degree_min"] = 0

    density = (
        graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1))
        if graph.num_nodes > 1
        else 0.0
    )
    summary["density"] = density
    return summary
