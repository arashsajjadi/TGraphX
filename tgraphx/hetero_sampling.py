"""Sampling utilities for :class:`~tgraphx.HeteroGraph` (v0.2.8).

These helpers extend the homogeneous samplers in :mod:`tgraphx.sampling`
to typed-node / typed-edge graphs.  The API parallels the homogeneous
versions:

* :func:`hetero_induced_subgraph` — keep a per-type subset of nodes and
  the relations between them.
* :func:`hetero_neighbor_sample` — multi-hop, per-relation fanout
  neighbour sampling rooted at typed seed nodes.

Both helpers return new :class:`HeteroGraph` instances.  Edge weights,
edge features, and node labels are carried through where present, and
``metadata["sampling"]`` records the sampling configuration.

Notes:
    * Sampling never mutates the input :class:`HeteroGraph`.
    * Determinism: ``hetero_neighbor_sample`` accepts a ``seed`` and
      uses a per-call ``torch.Generator`` (no global RNG side effects).
    * When ``relabel_nodes=True`` (default), per-type node ids are
      remapped to consecutive ``[0, K_t)`` ranges, edge_index entries
      are rewritten to point into the surviving sub-blocks, and
      ``metadata["sampling"]["original_node_ids"]`` records the
      ``type -> LongTensor`` mapping back to the global ids.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .core.hetero_graph import EdgeType, HeteroGraph, NodeType

__all__ = [
    "hetero_induced_subgraph",
    "hetero_neighbor_sample",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _to_long_1d(name: str, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor; got {type(t)}")
    if t.dim() != 1:
        raise ValueError(f"{name} must be 1-D; got shape {tuple(t.shape)}")
    return t.to(device=device, dtype=torch.long)


def _build_hetero_from_masks(
    g: HeteroGraph,
    keep_node_ids_dict: Dict[NodeType, torch.Tensor],
    relabel_nodes: bool,
    sampling_kind: str,
    extra_metadata: Optional[dict] = None,
) -> HeteroGraph:
    """Construct a HeteroGraph from per-type kept-node id tensors.

    All inputs ``keep_node_ids_dict[ntype]`` must already be on the
    correct device, sorted is not required (but sortedness is preserved
    for predictable ordering).
    """
    device = g.device

    # Per-type new node features and labels.
    new_node_stores: Dict[NodeType, torch.Tensor] = {}
    new_node_label_stores: Dict[NodeType, torch.Tensor] = {}
    relabel_maps: Dict[NodeType, Optional[torch.Tensor]] = {}

    for ntype in g.node_types:
        keep = keep_node_ids_dict.get(ntype)
        if keep is None or keep.numel() == 0:
            keep = torch.empty(0, dtype=torch.long, device=device)
        else:
            keep = keep.to(device=device, dtype=torch.long)
        new_node_stores[ntype] = g.node_features(ntype).index_select(0, keep)
        nl = g.node_labels(ntype)
        if nl is not None:
            new_node_label_stores[ntype] = nl.index_select(0, keep)
        if relabel_nodes:
            N = g.num_nodes(ntype)
            relabel = torch.full((N,), -1, dtype=torch.long, device=device)
            relabel[keep] = torch.arange(keep.numel(), device=device, dtype=torch.long)
            relabel_maps[ntype] = relabel
        else:
            relabel_maps[ntype] = None

    # Per-relation new edge_index, edge_weight, edge_features.
    new_edge_stores: Dict[EdgeType, torch.Tensor] = {}
    new_edge_weight_stores: Dict[EdgeType, torch.Tensor] = {}
    new_edge_feature_stores: Dict[EdgeType, torch.Tensor] = {}
    original_edge_ids: Dict[EdgeType, torch.Tensor] = {}

    for etype in g.edge_types:
        src_t, _, dst_t = etype
        ei = g.edge_index(etype)
        if ei.numel() == 0:
            new_edge_stores[etype] = ei.clone()
            original_edge_ids[etype] = torch.empty(0, dtype=torch.long, device=device)
            continue

        src_keep = keep_node_ids_dict.get(src_t)
        dst_keep = keep_node_ids_dict.get(dst_t)
        if src_keep is None or dst_keep is None:
            keep_mask = torch.zeros(ei.size(1), dtype=torch.bool, device=device)
        else:
            src_keep_set = torch.zeros(g.num_nodes(src_t), dtype=torch.bool, device=device)
            dst_keep_set = torch.zeros(g.num_nodes(dst_t), dtype=torch.bool, device=device)
            src_keep_set[src_keep.to(device=device)] = True
            dst_keep_set[dst_keep.to(device=device)] = True
            keep_mask = src_keep_set[ei[0]] & dst_keep_set[ei[1]]

        keep_eids = torch.where(keep_mask)[0]
        original_edge_ids[etype] = keep_eids.detach().cpu()

        kept_ei = ei.index_select(1, keep_eids)
        if relabel_nodes:
            kept_ei = torch.stack(
                [relabel_maps[src_t][kept_ei[0]], relabel_maps[dst_t][kept_ei[1]]],
                dim=0,
            )
        new_edge_stores[etype] = kept_ei

        ew = g.edge_weight(etype)
        if ew is not None:
            new_edge_weight_stores[etype] = ew.index_select(0, keep_eids)
        ef = g.edge_features(etype)
        if ef is not None:
            new_edge_feature_stores[etype] = ef.index_select(0, keep_eids)

    base_meta = dict(g.metadata) if isinstance(g.metadata, dict) else {}
    sampling_meta: Dict[str, object] = {
        "kind": sampling_kind,
        "original_node_ids": {
            ntype: keep.detach().cpu()
            for ntype, keep in keep_node_ids_dict.items()
        },
        "original_edge_ids": original_edge_ids,
    }
    if extra_metadata:
        sampling_meta.update(extra_metadata)
    base_meta = dict(base_meta)
    base_meta["sampling"] = sampling_meta

    return HeteroGraph(
        node_stores=new_node_stores,
        edge_stores=new_edge_stores,
        edge_weight_stores=new_edge_weight_stores or None,
        edge_feature_stores=new_edge_feature_stores or None,
        node_label_stores=new_node_label_stores or None,
        graph_label=g.graph_label,
        metadata=base_meta,
    )


# ── Public API ───────────────────────────────────────────────────────────────


def hetero_induced_subgraph(
    hetero_graph: HeteroGraph,
    node_ids_dict: Dict[NodeType, torch.Tensor],
    relabel_nodes: bool = True,
) -> HeteroGraph:
    """Return the typed induced subgraph over ``node_ids_dict``.

    For each node type ``t``, ``node_ids_dict[t]`` lists the ids to keep
    in that type's store.  Relations whose endpoints both survive are
    kept; orphan-edges (one endpoint dropped) are removed.  Node types
    omitted from ``node_ids_dict`` keep zero rows.

    Args:
        hetero_graph: Source :class:`HeteroGraph`.
        node_ids_dict: ``{node_type: 1-D LongTensor}`` of unique global
            ids per type.
        relabel_nodes: If ``True`` (default), edge_index entries are
            remapped to per-type local ids ``[0, K_t)``; otherwise the
            original ids are kept (and node-feature buffers stay sized
            to original ``N_t`` with zero rows for dropped entries).

    Returns:
        New :class:`HeteroGraph` with sampling metadata recorded.
    """
    if not isinstance(node_ids_dict, dict):
        raise TypeError("node_ids_dict must be a dict {node_type: LongTensor}")

    device = hetero_graph.device
    cleaned: Dict[NodeType, torch.Tensor] = {}
    for ntype, ids in node_ids_dict.items():
        if ntype not in hetero_graph.node_types:
            raise KeyError(
                f"Unknown node type {ntype!r}; available: {hetero_graph.node_types}"
            )
        ids_long = _to_long_1d(f"node_ids_dict[{ntype!r}]", ids, device)
        if ids_long.numel() == 0:
            cleaned[ntype] = ids_long
            continue
        if (ids_long < 0).any() or (ids_long >= hetero_graph.num_nodes(ntype)).any():
            raise ValueError(
                f"node_ids_dict[{ntype!r}] out of range "
                f"[0, {hetero_graph.num_nodes(ntype)})"
            )
        if ids_long.unique().numel() != ids_long.numel():
            raise ValueError(f"node_ids_dict[{ntype!r}] contains duplicates")
        cleaned[ntype] = ids_long

    # Backfill empty for missing types so downstream code keeps types stable.
    for ntype in hetero_graph.node_types:
        cleaned.setdefault(ntype, torch.empty(0, dtype=torch.long, device=device))

    if relabel_nodes:
        return _build_hetero_from_masks(
            hetero_graph, cleaned, relabel_nodes=True,
            sampling_kind="hetero_induced_subgraph",
        )

    # Non-relabel mode: return zero-row stores for dropped entries while
    # preserving original node count per type.  Re-pad inside helper.
    device = hetero_graph.device
    new_node_stores: Dict[NodeType, torch.Tensor] = {}
    new_node_label_stores: Dict[NodeType, torch.Tensor] = {}
    for ntype in hetero_graph.node_types:
        keep = cleaned[ntype]
        full = hetero_graph.node_features(ntype).new_zeros(
            (hetero_graph.num_nodes(ntype),
             *hetero_graph.node_features(ntype).shape[1:])
        )
        if keep.numel():
            full[keep] = hetero_graph.node_features(ntype).index_select(0, keep)
        new_node_stores[ntype] = full
        nl = hetero_graph.node_labels(ntype)
        if nl is not None:
            full_lbl = nl.new_zeros((hetero_graph.num_nodes(ntype), *nl.shape[1:]))
            if keep.numel():
                full_lbl[keep] = nl.index_select(0, keep)
            new_node_label_stores[ntype] = full_lbl

    new_edge_stores: Dict[EdgeType, torch.Tensor] = {}
    new_edge_weight_stores: Dict[EdgeType, torch.Tensor] = {}
    new_edge_feature_stores: Dict[EdgeType, torch.Tensor] = {}
    original_edge_ids: Dict[EdgeType, torch.Tensor] = {}
    for etype in hetero_graph.edge_types:
        src_t, _, dst_t = etype
        ei = hetero_graph.edge_index(etype)
        if ei.numel() == 0:
            new_edge_stores[etype] = ei.clone()
            original_edge_ids[etype] = torch.empty(0, dtype=torch.long, device=device)
            continue
        src_set = torch.zeros(hetero_graph.num_nodes(src_t), dtype=torch.bool, device=device)
        dst_set = torch.zeros(hetero_graph.num_nodes(dst_t), dtype=torch.bool, device=device)
        src_set[cleaned[src_t]] = True
        dst_set[cleaned[dst_t]] = True
        keep_mask = src_set[ei[0]] & dst_set[ei[1]]
        keep_eids = torch.where(keep_mask)[0]
        original_edge_ids[etype] = keep_eids.detach().cpu()
        new_edge_stores[etype] = ei.index_select(1, keep_eids)
        ew = hetero_graph.edge_weight(etype)
        if ew is not None:
            new_edge_weight_stores[etype] = ew.index_select(0, keep_eids)
        ef = hetero_graph.edge_features(etype)
        if ef is not None:
            new_edge_feature_stores[etype] = ef.index_select(0, keep_eids)

    base_meta = dict(hetero_graph.metadata) if isinstance(hetero_graph.metadata, dict) else {}
    base_meta["sampling"] = {
        "kind": "hetero_induced_subgraph",
        "original_node_ids": {
            ntype: cleaned[ntype].detach().cpu() for ntype in hetero_graph.node_types
        },
        "original_edge_ids": original_edge_ids,
        "relabel_nodes": False,
    }
    return HeteroGraph(
        node_stores=new_node_stores,
        edge_stores=new_edge_stores,
        edge_weight_stores=new_edge_weight_stores or None,
        edge_feature_stores=new_edge_feature_stores or None,
        node_label_stores=new_node_label_stores or None,
        graph_label=hetero_graph.graph_label,
        metadata=base_meta,
    )


def hetero_neighbor_sample(
    hetero_graph: HeteroGraph,
    seed_nodes_dict: Dict[NodeType, torch.Tensor],
    fanouts: Sequence[Dict[EdgeType, int]],
    seed: Optional[int] = None,
    direction: str = "in",
    relabel_nodes: bool = True,
) -> HeteroGraph:
    """Multi-layer typed neighbour sampling.

    Each entry of ``fanouts`` is a per-relation ``{edge_type: int}`` map
    describing how many neighbours to sample at the corresponding hop.
    A fanout of ``-1`` means "keep every neighbour at this hop".

    Args:
        hetero_graph: Source :class:`HeteroGraph`.
        seed_nodes_dict: ``{node_type: 1-D LongTensor}`` of seed ids per
            type.  At least one type must have non-empty seeds.
        fanouts: Sequence (one entry per hop) of per-relation fanout
            dicts.  Relations not listed for a given hop contribute no
            new nodes at that hop.
        seed: Optional RNG seed.
        direction: ``"in"`` (default; expand predecessors) or ``"out"``
            (expand successors).
        relabel_nodes: See :func:`hetero_induced_subgraph`.

    Returns:
        New :class:`HeteroGraph` whose stores cover the seeds and all
        sampled neighbours per type.  Edges between any pair of
        surviving nodes are kept (not just sampled edges); use
        ``metadata["sampling"]["seed_nodes"]`` to identify the original
        seed set.
    """
    if direction not in ("in", "out"):
        raise ValueError(f"direction must be 'in' or 'out'; got {direction!r}")
    if not isinstance(fanouts, (list, tuple)) or len(fanouts) == 0:
        raise ValueError("fanouts must be a non-empty sequence of dicts")
    for h, f in enumerate(fanouts):
        if not isinstance(f, dict):
            raise TypeError(f"fanouts[{h}] must be a dict {{edge_type: int}}")
        for etype, k in f.items():
            if etype not in hetero_graph.edge_types:
                raise KeyError(
                    f"fanouts[{h}] references unknown edge type {etype!r}"
                )
            if not isinstance(k, int):
                raise TypeError(
                    f"fanouts[{h}][{etype!r}] must be int; got {type(k)}"
                )
            if k == 0 or k < -1:
                raise ValueError(
                    f"fanouts[{h}][{etype!r}] must be >= 1 or -1 (all); got {k}"
                )

    device = hetero_graph.device
    visited: Dict[NodeType, torch.Tensor] = {}
    frontier: Dict[NodeType, torch.Tensor] = {}
    cleaned_seeds: Dict[NodeType, torch.Tensor] = {}

    for ntype in hetero_graph.node_types:
        seeds = seed_nodes_dict.get(ntype)
        N = hetero_graph.num_nodes(ntype)
        v = torch.zeros(N, dtype=torch.bool, device=device)
        if seeds is None or seeds.numel() == 0:
            cleaned_seeds[ntype] = torch.empty(0, dtype=torch.long, device=device)
            frontier[ntype] = torch.empty(0, dtype=torch.long, device=device)
        else:
            seeds_long = _to_long_1d(f"seed_nodes_dict[{ntype!r}]", seeds, device)
            if (seeds_long < 0).any() or (seeds_long >= N).any():
                raise ValueError(
                    f"seed_nodes_dict[{ntype!r}] out of range [0, {N})"
                )
            cleaned_seeds[ntype] = seeds_long
            v[seeds_long] = True
            frontier[ntype] = seeds_long
        visited[ntype] = v

    if all(s.numel() == 0 for s in cleaned_seeds.values()):
        raise ValueError(
            "seed_nodes_dict must contain at least one non-empty seed list"
        )

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    for hop_idx, hop_fanouts in enumerate(fanouts):
        new_frontier: Dict[NodeType, list] = {nt: [] for nt in hetero_graph.node_types}

        for etype, fanout in hop_fanouts.items():
            src_t, _, dst_t = etype
            ei = hetero_graph.edge_index(etype)
            if ei.numel() == 0:
                continue

            if direction == "in":
                # We are at dst-type frontier; pick predecessors of src_t.
                anchor_type = dst_t
                anchor_row = 1   # filter on dst row
                pick_row = 0     # take src ids
                grow_type = src_t
            else:  # "out"
                anchor_type = src_t
                anchor_row = 0
                pick_row = 1
                grow_type = dst_t

            anchor_frontier = frontier.get(anchor_type)
            if anchor_frontier is None or anchor_frontier.numel() == 0:
                continue

            # Edges whose anchor is in current frontier.
            mask_anchor = torch.zeros(
                hetero_graph.num_nodes(anchor_type), dtype=torch.bool, device=device,
            )
            mask_anchor[anchor_frontier] = True
            edge_keep = mask_anchor[ei[anchor_row]]
            cand_eids = torch.where(edge_keep)[0]
            if cand_eids.numel() == 0:
                continue
            anchors_per_edge = ei[anchor_row, cand_eids]

            picked: list[int] = []
            # Iterate per anchor node in the frontier.
            for u in anchor_frontier.tolist():
                edges_u = cand_eids[anchors_per_edge == u]
                if edges_u.numel() == 0:
                    continue
                if fanout == -1 or edges_u.numel() <= fanout:
                    chosen = edges_u
                else:
                    idx = torch.randperm(edges_u.numel(), generator=gen)[:fanout]
                    chosen = edges_u[idx]
                picked.extend(ei[pick_row, chosen].tolist())

            if picked:
                new_frontier[grow_type].extend(picked)

        next_frontier: Dict[NodeType, torch.Tensor] = {}
        for ntype in hetero_graph.node_types:
            if not new_frontier[ntype]:
                next_frontier[ntype] = torch.empty(0, dtype=torch.long, device=device)
                continue
            ids = torch.tensor(
                sorted(set(new_frontier[ntype])),
                dtype=torch.long, device=device,
            )
            novel_mask = ~visited[ntype][ids]
            new_ids = ids[novel_mask]
            visited[ntype][new_ids] = True
            next_frontier[ntype] = new_ids
        frontier = next_frontier

        if all(t.numel() == 0 for t in frontier.values()):
            break

    keep_node_ids_dict: Dict[NodeType, torch.Tensor] = {
        ntype: torch.where(visited[ntype])[0] for ntype in hetero_graph.node_types
    }

    return _build_hetero_from_masks(
        hetero_graph, keep_node_ids_dict, relabel_nodes=relabel_nodes,
        sampling_kind="hetero_neighbor_sample",
        extra_metadata={
            "seed_nodes": {
                nt: cleaned_seeds[nt].detach().cpu()
                for nt in hetero_graph.node_types
            },
            "fanouts": [dict(f) for f in fanouts],
            "direction": direction,
        },
    )
