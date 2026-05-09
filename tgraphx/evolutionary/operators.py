"""Mutation and crossover operators for graph evolution.

All operators are PURE (never mutate in place).

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .genome import GraphGenome

__all__ = [
    "mutate_add_node",
    "mutate_remove_node",
    "mutate_add_edge",
    "mutate_remove_edge",
    "mutate_rewire_edge",
    "mutate_node_feature",
    "mutate_edge_feature",
    "mutate_node_type",
    "mutate_edge_type",
    "apply_mutation",
    "edge_set_crossover",
    "node_induced_crossover",
    "feature_crossover",
]


def mutate_add_node(
    genome: GraphGenome,
    generator: Optional[torch.Generator] = None,
    node_type: int = 0,
    feature_dim: Optional[int] = None,
) -> GraphGenome:
    """Add a new node (and optionally a random feature).

    Args:
        genome: Source genome.
        generator: Optional RNG.
        node_type: Type of the new node.
        feature_dim: Feature dimension. If None, uses existing node_features dim.

    Returns:
        New genome with one more node.
    """
    g = genome.clone()
    n = g.num_nodes
    g.num_nodes = n + 1

    if g.node_features is not None:
        fdim = feature_dim if feature_dim is not None else g.node_features.shape[1]
        # Preserve full feature shape
        feat_shape = list(g.node_features.shape[1:])
        new_feat = torch.randn(1, *feat_shape, generator=generator,
                               dtype=g.node_features.dtype, device=g.node_features.device)
        g.node_features = torch.cat([g.node_features, new_feat], dim=0)
    elif feature_dim is not None:
        g.node_features = torch.randn(n + 1, feature_dim, generator=generator)

    if g.node_types is not None:
        new_type = torch.tensor([node_type], dtype=torch.long, device=g.node_types.device)
        g.node_types = torch.cat([g.node_types, new_type], dim=0)

    return g


def mutate_remove_node(
    genome: GraphGenome,
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Remove a random node and all its incident edges.

    After removal, all edge IDs > removed_node are decremented by 1.

    Args:
        genome: Source genome.
        generator: Optional RNG.

    Returns:
        New genome with one fewer node. Edge IDs are remapped.

    Raises:
        ValueError: If genome has 0 nodes.
    """
    if genome.num_nodes == 0:
        raise ValueError("mutate_remove_node: genome has 0 nodes")

    g = genome.clone()
    n = g.num_nodes
    nid = int(torch.randint(n, (1,), generator=generator).item())

    # Remove incident edges
    if g.num_edges > 0:
        src_arr = g.edge_index[0]
        dst_arr = g.edge_index[1]
        keep_mask = (src_arr != nid) & (dst_arr != nid)

        kept_src = src_arr[keep_mask]
        kept_dst = dst_arr[keep_mask]

        def remap(t: torch.Tensor, removed: int) -> torch.Tensor:
            return torch.where(t > removed, t - 1, t)

        g.edge_index = torch.stack([remap(kept_src, nid), remap(kept_dst, nid)], dim=0)

        if g.edge_features is not None:
            g.edge_features = g.edge_features[keep_mask]
        if g.edge_types is not None:
            g.edge_types = g.edge_types[keep_mask]
    else:
        g.edge_index = torch.zeros((2, 0), dtype=torch.long, device=g.device)

    # Remove node features
    idx_keep = torch.cat([
        torch.arange(nid, device=g.device),
        torch.arange(nid + 1, n, device=g.device),
    ]) if n > 1 else torch.zeros(0, dtype=torch.long, device=g.device)

    if g.node_features is not None and n > 0:
        g.node_features = g.node_features[idx_keep] if len(idx_keep) > 0 else torch.zeros(0, *list(g.node_features.shape[1:]), dtype=g.node_features.dtype, device=g.device)

    if g.node_types is not None and n > 0:
        g.node_types = g.node_types[idx_keep] if len(idx_keep) > 0 else torch.zeros(0, dtype=torch.long, device=g.device)

    g.num_nodes = n - 1
    return g


def mutate_add_edge(
    genome: GraphGenome,
    generator: Optional[torch.Generator] = None,
    edge_type: int = 0,
) -> GraphGenome:
    """Add a random valid edge (no self-loop).

    Args:
        genome: Source genome.
        generator: Optional RNG.
        edge_type: Type of the new edge.

    Returns:
        New genome with one more edge.

    Raises:
        ValueError: If the graph is complete (no valid edge to add).
    """
    if genome.num_nodes < 2:
        raise ValueError("mutate_add_edge: need at least 2 nodes")

    g = genome.clone()
    n = g.num_nodes

    existing = set()
    if g.num_edges > 0:
        for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
            existing.add((s, d))

    candidates = [
        (i, j) for i in range(n) for j in range(n)
        if i != j and (i, j) not in existing
    ]
    if not candidates:
        raise ValueError("mutate_add_edge: graph is complete, no edge to add")

    idx = int(torch.randint(len(candidates), (1,), generator=generator).item())
    src, tgt = candidates[idx]

    new_edge = torch.tensor([[src], [tgt]], dtype=torch.long, device=g.device)
    g.edge_index = torch.cat([g.edge_index, new_edge], dim=1)

    if g.edge_features is not None:
        feat_shape = list(g.edge_features.shape[1:])
        new_ef = torch.zeros(1, *feat_shape, dtype=g.edge_features.dtype, device=g.device)
        g.edge_features = torch.cat([g.edge_features, new_ef], dim=0)

    if g.edge_types is not None:
        new_et = torch.tensor([edge_type], dtype=torch.long, device=g.device)
        g.edge_types = torch.cat([g.edge_types, new_et], dim=0)

    return g


def mutate_remove_edge(
    genome: GraphGenome,
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Remove a random edge.

    Args:
        genome: Source genome.
        generator: Optional RNG.

    Returns:
        New genome with one fewer edge.

    Raises:
        ValueError: If genome has no edges.
    """
    if genome.num_edges == 0:
        raise ValueError("mutate_remove_edge: genome has no edges")

    g = genome.clone()
    e = g.num_edges
    eid = int(torch.randint(e, (1,), generator=generator).item())

    keep_mask = torch.ones(e, dtype=torch.bool, device=g.device)
    keep_mask[eid] = False

    g.edge_index = g.edge_index[:, keep_mask]
    if g.edge_features is not None:
        g.edge_features = g.edge_features[keep_mask]
    if g.edge_types is not None:
        g.edge_types = g.edge_types[keep_mask]

    return g


def mutate_rewire_edge(
    genome: GraphGenome,
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Change one endpoint of a random existing edge.

    Args:
        genome: Source genome.
        generator: Optional RNG.

    Returns:
        New genome with one rewired edge.

    Raises:
        ValueError: If no valid rewiring exists.
    """
    if genome.num_edges == 0:
        raise ValueError("mutate_rewire_edge: genome has no edges")
    if genome.num_nodes < 2:
        raise ValueError("mutate_rewire_edge: need at least 2 nodes")

    g = genome.clone()
    e = g.num_edges
    n = g.num_nodes

    eid = int(torch.randint(e, (1,), generator=generator).item())
    src = int(g.edge_index[0, eid].item())
    tgt = int(g.edge_index[1, eid].item())

    # Try to rewire the target endpoint
    existing = set()
    for s, d in zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()):
        existing.add((s, d))
    existing.discard((src, tgt))

    candidates = [j for j in range(n) if j != src and (src, j) not in existing]
    if not candidates:
        # Try rewiring source
        candidates = [i for i in range(n) if i != tgt and (i, tgt) not in existing]
        if not candidates:
            raise ValueError("mutate_rewire_edge: no valid rewiring found")
        new_src = candidates[int(torch.randint(len(candidates), (1,), generator=generator).item())]
        new_tgt = tgt
    else:
        new_src = src
        new_tgt = candidates[int(torch.randint(len(candidates), (1,), generator=generator).item())]

    g.edge_index = g.edge_index.clone()
    g.edge_index[0, eid] = new_src
    g.edge_index[1, eid] = new_tgt

    return g


def mutate_node_feature(
    genome: GraphGenome,
    node_id: int,
    noise_scale: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Add Gaussian noise to a node feature. Preserves shape and dtype.

    Args:
        genome: Source genome.
        node_id: Node whose feature to perturb.
        noise_scale: Standard deviation of Gaussian noise.
        generator: Optional RNG.

    Returns:
        New genome with perturbed node feature.

    Raises:
        ValueError: If node_features is None or node_id out of range.
    """
    if genome.node_features is None:
        raise ValueError("mutate_node_feature: genome has no node_features")
    if node_id < 0 or node_id >= genome.num_nodes:
        raise ValueError(f"mutate_node_feature: node_id={node_id} out of range")

    g = genome.clone()
    feat_shape = g.node_features.shape[1:]
    noise = torch.randn(*feat_shape, generator=generator).to(
        dtype=g.node_features.dtype, device=g.node_features.device
    ) * noise_scale
    g.node_features = g.node_features.clone()
    g.node_features[node_id] = g.node_features[node_id] + noise
    return g


def mutate_edge_feature(
    genome: GraphGenome,
    edge_id: int,
    noise_scale: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Add Gaussian noise to an edge feature. Preserves shape and dtype.

    Args:
        genome: Source genome.
        edge_id: Edge whose feature to perturb.
        noise_scale: Standard deviation of Gaussian noise.
        generator: Optional RNG.

    Returns:
        New genome with perturbed edge feature.
    """
    if genome.edge_features is None:
        raise ValueError("mutate_edge_feature: genome has no edge_features")
    if edge_id < 0 or edge_id >= genome.num_edges:
        raise ValueError(f"mutate_edge_feature: edge_id={edge_id} out of range")

    g = genome.clone()
    feat_shape = g.edge_features.shape[1:]
    noise = torch.randn(*feat_shape, generator=generator).to(
        dtype=g.edge_features.dtype, device=g.edge_features.device
    ) * noise_scale
    g.edge_features = g.edge_features.clone()
    g.edge_features[edge_id] = g.edge_features[edge_id] + noise
    return g


def mutate_node_type(
    genome: GraphGenome,
    node_id: int,
    allowed_types: List[int],
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Change a node's type.

    Args:
        genome: Source genome.
        node_id: Node to modify.
        allowed_types: List of allowed type integers.
        generator: Optional RNG.

    Returns:
        New genome with changed node type.
    """
    if genome.node_types is None:
        raise ValueError("mutate_node_type: genome has no node_types")
    if node_id < 0 or node_id >= genome.num_nodes:
        raise ValueError(f"mutate_node_type: node_id={node_id} out of range")
    if not allowed_types:
        raise ValueError("mutate_node_type: allowed_types is empty")

    g = genome.clone()
    idx = int(torch.randint(len(allowed_types), (1,), generator=generator).item())
    new_type = allowed_types[idx]
    g.node_types = g.node_types.clone()
    g.node_types[node_id] = new_type
    return g


def mutate_edge_type(
    genome: GraphGenome,
    edge_id: int,
    allowed_types: List[int],
    generator: Optional[torch.Generator] = None,
) -> GraphGenome:
    """Change an edge's type.

    Args:
        genome: Source genome.
        edge_id: Edge to modify.
        allowed_types: List of allowed type integers.
        generator: Optional RNG.

    Returns:
        New genome with changed edge type.
    """
    if genome.edge_types is None:
        raise ValueError("mutate_edge_type: genome has no edge_types")
    if edge_id < 0 or edge_id >= genome.num_edges:
        raise ValueError(f"mutate_edge_type: edge_id={edge_id} out of range")
    if not allowed_types:
        raise ValueError("mutate_edge_type: allowed_types is empty")

    g = genome.clone()
    idx = int(torch.randint(len(allowed_types), (1,), generator=generator).item())
    g.edge_types = g.edge_types.clone()
    g.edge_types[edge_id] = allowed_types[idx]
    return g


_MUTATION_DISPATCH = {
    "add_node": mutate_add_node,
    "remove_node": mutate_remove_node,
    "add_edge": mutate_add_edge,
    "remove_edge": mutate_remove_edge,
    "rewire_edge": mutate_rewire_edge,
}


def apply_mutation(
    genome: GraphGenome,
    mutation_type: str,
    rate: float,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> GraphGenome:
    """Dispatcher for mutations.

    Args:
        genome: Source genome.
        mutation_type: One of 'add_node', 'remove_node', 'add_edge',
            'remove_edge', 'rewire_edge', 'node_feature', 'edge_feature'.
        rate: Probability of applying the mutation.
        generator: Optional RNG.
        **kwargs: Extra keyword arguments passed to the mutation function.

    Returns:
        (Possibly mutated) new genome.
    """
    if torch.rand(1, generator=generator).item() > rate:
        return genome.clone()

    if mutation_type == "node_feature":
        nid = kwargs.get("node_id", 0)
        return mutate_node_feature(genome, nid, generator=generator,
                                   noise_scale=kwargs.get("noise_scale", 0.1))
    elif mutation_type == "edge_feature":
        eid = kwargs.get("edge_id", 0)
        return mutate_edge_feature(genome, eid, generator=generator,
                                   noise_scale=kwargs.get("noise_scale", 0.1))
    elif mutation_type in _MUTATION_DISPATCH:
        fn = _MUTATION_DISPATCH[mutation_type]
        try:
            return fn(genome, generator=generator, **kwargs)
        except ValueError:
            return genome.clone()
    else:
        raise ValueError(
            f"Unknown mutation_type={mutation_type!r}. "
            f"Choose from {list(_MUTATION_DISPATCH)} or 'node_feature', 'edge_feature'."
        )


# ── Crossover operators ──────────────────────────────────────────────────────


def edge_set_crossover(
    parent_a: GraphGenome,
    parent_b: GraphGenome,
    generator: Optional[torch.Generator] = None,
) -> Tuple[GraphGenome, GraphGenome]:
    """Union edge sets and split randomly.

    Takes the union of edges from both parents, then randomly partitions
    back into two children.

    Args:
        parent_a: First parent.
        parent_b: Second parent.
        generator: Optional RNG.

    Returns:
        (child_a, child_b) — two new genomes.
    """
    # Use the larger num_nodes
    num_nodes = max(parent_a.num_nodes, parent_b.num_nodes)

    # Collect all edges from both parents, normalized to num_nodes range
    edge_set = set()
    if parent_a.num_edges > 0:
        for s, d in zip(parent_a.edge_index[0].tolist(), parent_a.edge_index[1].tolist()):
            if s < num_nodes and d < num_nodes:
                edge_set.add((s, d))
    if parent_b.num_edges > 0:
        for s, d in zip(parent_b.edge_index[0].tolist(), parent_b.edge_index[1].tolist()):
            if s < num_nodes and d < num_nodes:
                edge_set.add((s, d))

    edges_list = list(edge_set)
    if not edges_list:
        return parent_a.clone(), parent_b.clone()

    # Random split
    perm = torch.randperm(len(edges_list), generator=generator)
    split = len(edges_list) // 2

    def _build(idxs) -> GraphGenome:
        sel = [edges_list[i] for i in idxs.tolist()]
        if sel:
            ei = torch.tensor([[s for s, d in sel], [d for s, d in sel]], dtype=torch.long)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)
        return GraphGenome(
            edge_index=ei,
            num_nodes=num_nodes,
            node_features=parent_a.node_features.clone() if parent_a.node_features is not None else None,
        )

    child_a = _build(perm[:split])
    child_b = _build(perm[split:])
    return child_a, child_b


def node_induced_crossover(
    parent_a: GraphGenome,
    parent_b: GraphGenome,
    fraction: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> Tuple[GraphGenome, GraphGenome]:
    """Take node-induced subgraphs from each parent.

    Selects a fraction of nodes from each parent, then takes the induced subgraph.

    Args:
        parent_a: First parent.
        parent_b: Second parent.
        fraction: Fraction of nodes to select from each parent.
        generator: Optional RNG.

    Returns:
        (child_a, child_b) — two subgraph-based children.
    """
    def _induced(g: GraphGenome, frac: float) -> GraphGenome:
        n = g.num_nodes
        k = max(1, int(n * frac))
        sel_nodes = torch.randperm(n, generator=generator)[:k].sort().values
        sel_set = set(sel_nodes.tolist())

        keep_mask = torch.zeros(g.num_edges, dtype=torch.bool)
        if g.num_edges > 0:
            for i, (s, d) in enumerate(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist())):
                if s in sel_set and d in sel_set:
                    keep_mask[i] = True

        # Remap node IDs
        remap = {old: new for new, old in enumerate(sel_nodes.tolist())}

        if keep_mask.any():
            kept_ei = g.edge_index[:, keep_mask]
            new_src = torch.tensor([remap.get(s.item(), 0) for s in kept_ei[0]], dtype=torch.long)
            new_dst = torch.tensor([remap.get(d.item(), 0) for d in kept_ei[1]], dtype=torch.long)
            new_ei = torch.stack([new_src, new_dst], dim=0)
        else:
            new_ei = torch.zeros((2, 0), dtype=torch.long)

        new_nf = g.node_features[sel_nodes] if g.node_features is not None else None
        new_nt = g.node_types[sel_nodes] if g.node_types is not None else None

        new_ef = g.edge_features[keep_mask] if (g.edge_features is not None and keep_mask.any()) else None

        return GraphGenome(
            edge_index=new_ei,
            num_nodes=k,
            node_features=new_nf,
            node_types=new_nt,
            edge_features=new_ef,
        )

    return _induced(parent_a, fraction), _induced(parent_b, fraction)


def feature_crossover(
    parent_a: GraphGenome,
    parent_b: GraphGenome,
    generator: Optional[torch.Generator] = None,
) -> Tuple[GraphGenome, GraphGenome]:
    """Swap node/edge features where shapes align.

    Swaps features node-by-node for aligned nodes (up to min(n_a, n_b)).

    Args:
        parent_a: First parent.
        parent_b: Second parent.
        generator: Optional RNG.

    Returns:
        (child_a, child_b) with swapped features.
    """
    child_a = parent_a.clone()
    child_b = parent_b.clone()

    # Swap node features where possible
    if (child_a.node_features is not None and
            child_b.node_features is not None and
            child_a.node_features.shape[1:] == child_b.node_features.shape[1:]):
        n = min(child_a.num_nodes, child_b.num_nodes)
        swap_mask = torch.rand(n, generator=generator) < 0.5

        fa = child_a.node_features.clone()
        fb = child_b.node_features.clone()

        fa_swapped = fa[:n].clone()
        fb_swapped = fb[:n].clone()

        fa_swapped[swap_mask] = fb[:n][swap_mask]
        fb_swapped[swap_mask] = fa[:n][swap_mask]

        child_a.node_features = torch.cat([fa_swapped, fa[n:]], dim=0) if n < child_a.num_nodes else fa_swapped
        child_b.node_features = torch.cat([fb_swapped, fb[n:]], dim=0) if n < child_b.num_nodes else fb_swapped

    return child_a, child_b
