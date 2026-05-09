"""Graph action space for the generation MDP.

Mathematical contract
---------------------
The graph generation process is formalized as a finite-horizon Markov Decision
Process:

    M = (S, A, P, R, gamma)

where:
    S — set of all possible GraphEditStates (partial graphs)
    A — set of GraphActions (add/remove node/edge, set feature, stop)
    P(s' | s, a) — deterministic transition (for structural actions)
    R(s, a) — scalar reward (defined externally by environment)
    gamma — discount factor in [0, 1]

The MDP terminates when:
    - A STOP_GENERATION action is taken, or
    - The maximum step count is reached.

Action masks enforce hard constraints (no self-loops, acyclicity, etc.)
before any action is sampled or applied.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .data_model import GeneratedGraph, GraphEditState

__all__ = [
    "GraphActionType",
    "GraphAction",
    "GraphActionSpace",
    "enumerate_valid_actions",
    "sample_valid_action",
    "apply_graph_action",
    "batch_action_masks",
    "action_to_index",
    "index_to_action",
]


class GraphActionType(enum.Enum):
    """Enumeration of all valid graph editing actions."""

    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    REMOVE_NODE = "remove_node"
    REMOVE_EDGE = "remove_edge"
    SET_NODE_FEATURE = "set_node_feature"
    SET_EDGE_FEATURE = "set_edge_feature"
    STOP_GENERATION = "stop_generation"


@dataclass
class GraphAction:
    """A single graph editing action.

    Args:
        action_type: Type of the action.
        node_id: Target node ID (for node-level actions).
        src_id: Source node ID (for edge actions).
        tgt_id: Target node ID (for edge actions).
        node_type: Integer node type (for typed graphs).
        edge_type: Integer edge type (for typed graphs).
        features: Optional Tensor — preserved as-is (shape is application-specific).
        edge_weight: Optional float edge weight.
        metadata: Arbitrary metadata.
    """

    action_type: GraphActionType
    node_id: Optional[int] = None
    src_id: Optional[int] = None
    tgt_id: Optional[int] = None
    node_type: Optional[int] = None
    edge_type: Optional[int] = None
    features: Optional[torch.Tensor] = None
    edge_weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphActionSpace:
    """Defines which actions are valid in the current state.

    Args:
        max_nodes: Maximum allowed nodes.
        max_edges: Maximum allowed edges.
        allowed_node_types: List of allowed node type integers.
        allowed_edge_types: List of allowed edge type integers.
        no_self_loops: Disallow self-loop edges.
        connected_required: New graph must remain (weakly) connected.
        acyclic_required: New graph must remain acyclic (DAG).
        node_feature_dim: Dimension for ADD_NODE features.
        edge_feature_dim: Dimension for ADD_EDGE features.
    """

    max_nodes: int = 100
    max_edges: int = 1000
    allowed_node_types: List[int] = field(default_factory=lambda: [0])
    allowed_edge_types: List[int] = field(default_factory=lambda: [0])
    no_self_loops: bool = True
    connected_required: bool = False
    acyclic_required: bool = False
    node_feature_dim: Optional[int] = None
    edge_feature_dim: Optional[int] = None

    def action_types(self) -> List[GraphActionType]:
        """Return the ordered list of action types supported by this space."""
        return list(GraphActionType)

    def total_discrete_actions(self) -> int:
        """Return the total number of discrete action slots in the index mapping.

        Layout:
            [0]                   — STOP_GENERATION
            [1..max_nodes]        — ADD_NODE (one per node type slot)
            [max_nodes+1 .. 2*max_nodes+1] — REMOVE_NODE
            [above .. above + max_edges] — ADD_EDGE slots
            [above .. above + max_edges] — REMOVE_EDGE slots
        This is an upper-bound, not all slots are valid at once.
        """
        n_types = max(1, len(self.allowed_node_types))
        e_types = max(1, len(self.allowed_edge_types))
        return (
            1  # STOP
            + n_types  # ADD_NODE per type
            + self.max_nodes  # REMOVE_NODE per node slot
            + self.max_nodes * self.max_nodes * e_types  # ADD_EDGE (src, dst, type)
            + self.max_edges  # REMOVE_EDGE per edge slot
        )


def enumerate_valid_actions(
    state: GraphEditState,
    space: GraphActionSpace,
) -> List[GraphAction]:
    """List all valid GraphActions given the current state and action space.

    Args:
        state: Current GraphEditState.
        space: GraphActionSpace defining constraints.

    Returns:
        List of valid GraphActions (always includes STOP_GENERATION).
    """
    g = state.graph
    n = g.num_nodes
    num_edges = g.num_edges
    actions: List[GraphAction] = []

    # STOP is always valid
    actions.append(GraphAction(action_type=GraphActionType.STOP_GENERATION))

    # ADD_NODE — allowed if below max_nodes
    if n < space.max_nodes:
        for nt in space.allowed_node_types:
            actions.append(GraphAction(action_type=GraphActionType.ADD_NODE, node_type=nt))

    # REMOVE_NODE — any node (if graph has nodes)
    if n > 0:
        for nid in range(n):
            actions.append(GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=nid))

    # ADD_EDGE — valid pairs not already present
    if num_edges < space.max_edges and n >= 2:
        existing_edges: set = set()
        if num_edges > 0:
            src_list = g.edge_index[0].tolist()
            dst_list = g.edge_index[1].tolist()
            existing_edges = set(zip(src_list, dst_list))

        for et in space.allowed_edge_types:
            for src in range(n):
                for dst in range(n):
                    if space.no_self_loops and src == dst:
                        continue
                    if (src, dst) in existing_edges:
                        continue
                    actions.append(
                        GraphAction(
                            action_type=GraphActionType.ADD_EDGE,
                            src_id=src,
                            tgt_id=dst,
                            edge_type=et,
                        )
                    )

    # REMOVE_EDGE — any existing edge
    if num_edges > 0:
        src_list = g.edge_index[0].tolist()
        dst_list = g.edge_index[1].tolist()
        for eid in range(num_edges):
            actions.append(
                GraphAction(
                    action_type=GraphActionType.REMOVE_EDGE,
                    src_id=src_list[eid],
                    tgt_id=dst_list[eid],
                    metadata={"edge_id": eid},
                )
            )

    return actions


def sample_valid_action(
    state: GraphEditState,
    space: GraphActionSpace,
    generator: Optional[torch.Generator] = None,
) -> GraphAction:
    """Sample one valid action from the action space.

    Deterministic with a fixed generator.

    Args:
        state: Current GraphEditState.
        space: GraphActionSpace.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        A randomly sampled valid GraphAction.
    """
    valid = enumerate_valid_actions(state, space)
    n = len(valid)
    idx = int(torch.randint(n, (1,), generator=generator).item())
    return valid[idx]


def _apply_stop(state: GraphEditState, action: GraphAction) -> GraphEditState:
    new_state = state.clone()
    new_state.done = True
    new_state.step = state.step + 1
    return new_state


def _apply_add_node(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    new_state = state.clone()
    new_g = new_state.graph
    new_g.num_nodes = g.num_nodes + 1

    if g.node_features is not None:
        feat_shape = list(g.node_features.shape[1:])
        if action.features is not None:
            if list(action.features.shape) != feat_shape:
                raise ValueError(
                    f"ADD_NODE feature shape {list(action.features.shape)} "
                    f"!= existing node feature shape {feat_shape}"
                )
            new_feat = action.features.unsqueeze(0)
        else:
            new_feat = torch.zeros(
                1, *feat_shape,
                dtype=g.node_features.dtype,
                device=g.node_features.device,
            )
        new_g.node_features = torch.cat([g.node_features, new_feat], dim=0)

    if g.node_types is not None:
        nt = action.node_type if action.node_type is not None else 0
        new_type = torch.tensor([nt], dtype=torch.long, device=g.node_types.device)
        new_g.node_types = torch.cat([g.node_types, new_type], dim=0)

    new_state.step = state.step + 1
    return new_state


def _remap_node_ids(t: torch.Tensor, removed: int) -> torch.Tensor:
    return torch.where(t > removed, t - 1, t)


def _apply_remove_node(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    nid = action.node_id
    if nid is None:
        raise ValueError("REMOVE_NODE action requires node_id")
    if nid < 0 or nid >= g.num_nodes:
        raise ValueError(f"REMOVE_NODE: node_id={nid} out of range [0, {g.num_nodes})")

    new_state = state.clone()
    new_g = new_state.graph
    n = g.num_nodes

    if g.num_edges > 0:
        src_arr, dst_arr = g.edge_index[0], g.edge_index[1]
        keep_mask = (src_arr != nid) & (dst_arr != nid)
        new_src = _remap_node_ids(src_arr[keep_mask], nid)
        new_dst = _remap_node_ids(dst_arr[keep_mask], nid)
        new_g.edge_index = torch.stack([new_src, new_dst], dim=0)
        if g.edge_features is not None:
            new_g.edge_features = g.edge_features[keep_mask]
        if g.edge_weight is not None:
            new_g.edge_weight = g.edge_weight[keep_mask]
        if g.edge_types is not None:
            new_g.edge_types = g.edge_types[keep_mask]
        if g.timestamps is not None:
            new_g.timestamps = g.timestamps[keep_mask]
    else:
        new_g.edge_index = torch.zeros((2, 0), dtype=torch.long, device=g.device)

    new_g.num_nodes = n - 1
    idx_keep = torch.cat([
        torch.arange(nid, device=g.device),
        torch.arange(nid + 1, n, device=g.device),
    ])
    if g.node_features is not None:
        new_g.node_features = g.node_features[idx_keep]
    if g.node_types is not None:
        new_g.node_types = g.node_types[idx_keep]

    new_state.step = state.step + 1
    return new_state


def _apply_add_edge(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    src, tgt = action.src_id, action.tgt_id
    if src is None or tgt is None:
        raise ValueError("ADD_EDGE action requires src_id and tgt_id")
    if src < 0 or src >= g.num_nodes:
        raise ValueError(f"ADD_EDGE: src_id={src} out of range [0, {g.num_nodes})")
    if tgt < 0 or tgt >= g.num_nodes:
        raise ValueError(f"ADD_EDGE: tgt_id={tgt} out of range [0, {g.num_nodes})")

    new_state = state.clone()
    new_g = new_state.graph
    new_edge = torch.tensor([[src], [tgt]], dtype=torch.long, device=g.device)
    new_g.edge_index = torch.cat([g.edge_index, new_edge], dim=1)

    if g.edge_features is not None:
        feat_shape = list(g.edge_features.shape[1:])
        new_ef = (
            action.features.unsqueeze(0)
            if action.features is not None
            else torch.zeros(1, *feat_shape, dtype=g.edge_features.dtype, device=g.device)
        )
        new_g.edge_features = torch.cat([g.edge_features, new_ef], dim=0)

    if g.edge_weight is not None:
        w = action.edge_weight if action.edge_weight is not None else 1.0
        new_ew = torch.tensor([w], dtype=g.edge_weight.dtype, device=g.device)
        new_g.edge_weight = torch.cat([g.edge_weight, new_ew], dim=0)

    if g.edge_types is not None:
        et = action.edge_type if action.edge_type is not None else 0
        new_et = torch.tensor([et], dtype=torch.long, device=g.device)
        new_g.edge_types = torch.cat([g.edge_types, new_et], dim=0)

    new_state.step = state.step + 1
    return new_state


def _apply_remove_edge(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    src, tgt = action.src_id, action.tgt_id
    if src is None or tgt is None:
        raise ValueError("REMOVE_EDGE action requires src_id and tgt_id")
    if g.num_edges == 0:
        raise ValueError("REMOVE_EDGE: no edges to remove")

    src_arr, dst_arr = g.edge_index[0], g.edge_index[1]
    match = (src_arr == src) & (dst_arr == tgt)
    if not match.any():
        raise ValueError(f"REMOVE_EDGE: edge ({src}, {tgt}) not found in edge_index")

    first_idx = int(match.nonzero(as_tuple=False)[0].item())
    keep_mask = torch.ones(g.num_edges, dtype=torch.bool, device=g.device)
    keep_mask[first_idx] = False

    new_state = state.clone()
    new_g = new_state.graph
    new_g.edge_index = g.edge_index[:, keep_mask]
    if g.edge_features is not None:
        new_g.edge_features = g.edge_features[keep_mask]
    if g.edge_weight is not None:
        new_g.edge_weight = g.edge_weight[keep_mask]
    if g.edge_types is not None:
        new_g.edge_types = g.edge_types[keep_mask]

    new_state.step = state.step + 1
    return new_state


def _apply_set_node_feature(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    nid = action.node_id
    if nid is None:
        raise ValueError("SET_NODE_FEATURE requires node_id")
    if nid < 0 or nid >= g.num_nodes:
        raise ValueError(f"SET_NODE_FEATURE: node_id={nid} out of range")
    if action.features is None:
        raise ValueError("SET_NODE_FEATURE requires features")

    new_state = state.clone()
    new_g = new_state.graph
    if new_g.node_features is None:
        raise ValueError(
            "SET_NODE_FEATURE: graph has no node_features tensor yet. "
            "Initialize node_features first."
        )
    if list(action.features.shape) != list(new_g.node_features.shape[1:]):
        raise ValueError(
            f"SET_NODE_FEATURE: feature shape {list(action.features.shape)} "
            f"!= existing node feature shape {list(new_g.node_features.shape[1:])}"
        )
    new_g.node_features = new_g.node_features.clone()
    new_g.node_features[nid] = action.features.to(new_g.node_features.device)
    new_state.step = state.step + 1
    return new_state


def _apply_set_edge_feature(state: GraphEditState, action: GraphAction) -> GraphEditState:
    g = state.graph
    new_state = state.clone()
    new_g = new_state.graph
    if new_g.edge_features is None:
        raise ValueError("SET_EDGE_FEATURE: graph has no edge_features tensor yet.")

    eid_meta = action.metadata.get("edge_id")
    if eid_meta is not None:
        eid = int(eid_meta)
    elif action.src_id is not None and action.tgt_id is not None:
        src_arr, dst_arr = g.edge_index[0], g.edge_index[1]
        match = (src_arr == action.src_id) & (dst_arr == action.tgt_id)
        if not match.any():
            raise ValueError(
                f"SET_EDGE_FEATURE: edge ({action.src_id}, {action.tgt_id}) not found"
            )
        eid = int(match.nonzero(as_tuple=False)[0].item())
    else:
        raise ValueError("SET_EDGE_FEATURE requires edge_id in metadata or src_id+tgt_id")

    if action.features is None:
        raise ValueError("SET_EDGE_FEATURE requires features")

    new_g.edge_features = new_g.edge_features.clone()
    new_g.edge_features[eid] = action.features.to(new_g.edge_features.device)
    new_state.step = state.step + 1
    return new_state


_ACTION_HANDLERS = {
    GraphActionType.STOP_GENERATION: _apply_stop,
    GraphActionType.ADD_NODE: _apply_add_node,
    GraphActionType.REMOVE_NODE: _apply_remove_node,
    GraphActionType.ADD_EDGE: _apply_add_edge,
    GraphActionType.REMOVE_EDGE: _apply_remove_edge,
    GraphActionType.SET_NODE_FEATURE: _apply_set_node_feature,
    GraphActionType.SET_EDGE_FEATURE: _apply_set_edge_feature,
}


def apply_graph_action(
    state: GraphEditState,
    action: GraphAction,
) -> GraphEditState:
    """Apply a GraphAction to a GraphEditState.

    Returns a NEW GraphEditState — never mutates in place.

    Args:
        state: Current state.
        action: Action to apply.

    Returns:
        New GraphEditState after applying the action.

    Raises:
        ValueError: If the action is invalid for the current state.
    """
    handler = _ACTION_HANDLERS.get(action.action_type)
    if handler is None:
        raise ValueError(f"Unknown action type: {action.action_type}")
    return handler(state, action)


def batch_action_masks(
    states: List[GraphEditState],
    space: GraphActionSpace,
) -> torch.Tensor:
    """Compute action masks for a batch of states.

    Args:
        states: List of B GraphEditStates.
        space: GraphActionSpace.

    Returns:
        BoolTensor [B, max_actions] where True = action is valid.
    """
    total = space.total_discrete_actions()
    B = len(states)
    masks = torch.zeros(B, total, dtype=torch.bool)

    for i, state in enumerate(states):
        valid_actions = enumerate_valid_actions(state, space)
        for act in valid_actions:
            idx = action_to_index(act, space)
            if 0 <= idx < total:
                masks[i, idx] = True

    return masks


def action_to_index(action: GraphAction, space: GraphActionSpace) -> int:
    """Bijection from GraphAction to flat integer index.

    Index layout:
        0                      — STOP
        1..n_types             — ADD_NODE (by node_type)
        above..above+max_nodes — REMOVE_NODE (by node_id)
        above..above+max_nodes*max_nodes*n_e_types — ADD_EDGE
        above..above+max_edges — REMOVE_EDGE

    Args:
        action: GraphAction to encode.
        space: GraphActionSpace defining the layout.

    Returns:
        Integer index.
    """
    n_types = max(1, len(space.allowed_node_types))
    e_types = max(1, len(space.allowed_edge_types))
    at = action.action_type

    if at == GraphActionType.STOP_GENERATION:
        return 0

    offset = 1

    if at == GraphActionType.ADD_NODE:
        nt = action.node_type if action.node_type is not None else 0
        nt_idx = space.allowed_node_types.index(nt) if nt in space.allowed_node_types else 0
        return offset + nt_idx

    offset += n_types

    if at == GraphActionType.REMOVE_NODE:
        nid = action.node_id if action.node_id is not None else 0
        return offset + nid

    offset += space.max_nodes

    if at == GraphActionType.ADD_EDGE:
        src = action.src_id if action.src_id is not None else 0
        tgt = action.tgt_id if action.tgt_id is not None else 0
        et = action.edge_type if action.edge_type is not None else 0
        et_idx = space.allowed_edge_types.index(et) if et in space.allowed_edge_types else 0
        return offset + src * space.max_nodes * e_types + tgt * e_types + et_idx

    offset += space.max_nodes * space.max_nodes * e_types

    if at == GraphActionType.REMOVE_EDGE:
        eid = action.metadata.get("edge_id", 0) if action.metadata else 0
        return offset + int(eid)

    return 0


def index_to_action(idx: int, space: GraphActionSpace) -> GraphAction:
    """Inverse of action_to_index.

    Args:
        idx: Integer index.
        space: GraphActionSpace.

    Returns:
        GraphAction corresponding to idx.
    """
    n_types = max(1, len(space.allowed_node_types))
    e_types = max(1, len(space.allowed_edge_types))

    if idx == 0:
        return GraphAction(action_type=GraphActionType.STOP_GENERATION)

    idx -= 1
    if idx < n_types:
        nt = space.allowed_node_types[idx] if idx < len(space.allowed_node_types) else idx
        return GraphAction(action_type=GraphActionType.ADD_NODE, node_type=nt)

    idx -= n_types
    if idx < space.max_nodes:
        return GraphAction(action_type=GraphActionType.REMOVE_NODE, node_id=idx)

    idx -= space.max_nodes
    total_add_edge = space.max_nodes * space.max_nodes * e_types
    if idx < total_add_edge:
        src = idx // (space.max_nodes * e_types)
        rem = idx % (space.max_nodes * e_types)
        tgt = rem // e_types
        et_idx = rem % e_types
        et = space.allowed_edge_types[et_idx] if et_idx < len(space.allowed_edge_types) else et_idx
        return GraphAction(action_type=GraphActionType.ADD_EDGE, src_id=src, tgt_id=tgt, edge_type=et)

    idx -= total_add_edge
    return GraphAction(
        action_type=GraphActionType.REMOVE_EDGE,
        metadata={"edge_id": idx},
    )
