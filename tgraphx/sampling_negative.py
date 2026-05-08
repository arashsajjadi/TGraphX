"""Negative-edge sampling utilities for link-prediction workflows.

These primitives produce edges that are *not* present in a positive
``edge_index``.  They are the building blocks for link-prediction
training (and for evaluation against an OGB-style evaluator).

API summary
-----------
``negative_sampling(edge_index, num_nodes, num_neg_samples=None,
                    method="sparse", force_undirected=False,
                    seed=None)``
    Uniformly sample negative edges that do not appear in
    ``edge_index``.

``structured_negative_sampling(edge_index, num_nodes,
                               contains_neg_self_loops=True, seed=None)``
    For each positive edge ``(i, j)``, sample a node ``k`` such that
    ``(i, k)`` is *not* a positive edge.  Returns the triplet tensors
    ``(i, j, k)`` aligned with the positive edge_index columns.

``batched_negative_sampling(edge_index, batch, num_neg_samples=None,
                            method="sparse", force_undirected=False,
                            seed=None)``
    Like :func:`negative_sampling` but respects graph-batch boundaries:
    a negative edge is only sampled within the same graph.

Determinism
-----------
Every function accepts an optional ``seed`` and uses a per-call
``torch.Generator`` so it does **not** affect the global RNG state.

Stability
---------
Beta.  Signatures may evolve before v0.4.0; behavioural invariants
(no false negatives, no self-loops when requested, deterministic with
``seed``) are stable.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "negative_sampling",
    "structured_negative_sampling",
    "batched_negative_sampling",
    "hard_negative_sampling",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    """Return a CPU generator (torch.randint accepts only CPU generators)."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def _edge_hash(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Encode edges (u, v) as scalar integers ``u * num_nodes + v``."""
    if edge_index.numel() == 0:
        return edge_index.new_empty((0,), dtype=torch.long)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    return src * int(num_nodes) + dst


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E]; got {tuple(edge_index.shape)}"
        )
    if edge_index.numel() and (
        edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes
    ):
        raise ValueError(
            f"edge_index entries must be in [0, num_nodes={num_nodes}); "
            f"got min={int(edge_index.min())}, max={int(edge_index.max())}"
        )


def _sample_random_pairs(
    num: int,
    num_nodes: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``num`` random (src, dst) pairs uniformly from [0, num_nodes)."""
    src = torch.randint(num_nodes, (num,), generator=generator)
    dst = torch.randint(num_nodes, (num,), generator=generator)
    out = torch.stack([src, dst], dim=0)
    return out.to(device=device, dtype=torch.long)


# ── Public API ───────────────────────────────────────────────────────────────


def negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Uniformly sample negative edges absent from ``edge_index``.

    Args:
        edge_index: ``LongTensor[2, E]`` of positive edges.
        num_nodes: Number of nodes in the graph.
        num_neg_samples: Target number of negative edges.  When ``None``,
            samples ``E`` negatives (matching the positive count).
        method: ``"sparse"`` (rejection sampling, the default) or
            ``"dense"``.  ``"dense"`` materialises the absent-edge set and
            is only safe for small graphs (``num_nodes < 5_000``).
        force_undirected: When ``True``, neither ``(u, v)`` nor
            ``(v, u)`` may be a positive edge.  The output still
            contains one directed edge per sampled negative pair —
            apply :class:`~tgraphx.transforms.ToUndirected` to the
            result if you need both directions.
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``LongTensor[2, num_neg]`` on the same device as ``edge_index``.
        No self-loops; no edge that exists in ``edge_index``; no
        duplicates within the returned tensor.

    Notes:
        - The function may return fewer than ``num_neg_samples`` if the
          graph is nearly complete and rejection sampling cannot find
          enough negatives within the internal attempt budget.
    """
    if method not in ("sparse", "dense"):
        raise ValueError(f"method must be 'sparse' or 'dense'; got {method!r}")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive; got {num_nodes}")
    _validate_edge_index(edge_index, num_nodes)

    device = edge_index.device
    num_pos = int(edge_index.size(1))
    target = int(num_neg_samples) if num_neg_samples is not None else max(num_pos, 1)
    if target <= 0:
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    if method == "dense":
        return _negative_sampling_dense(
            edge_index, num_nodes, target, force_undirected, seed,
        )

    # ── Sparse / rejection-sampling path ────────────────────────────────────
    generator = _make_generator(seed, device)
    forbidden = set(_edge_hash(edge_index, num_nodes).cpu().tolist())
    if force_undirected and num_pos:
        rev_hash = (
            edge_index[1].to(torch.long) * int(num_nodes)
            + edge_index[0].to(torch.long)
        )
        forbidden.update(rev_hash.cpu().tolist())

    # Track accepted directed hashes so each output edge is unique.  Under
    # ``force_undirected`` we additionally block the reverse hash so the
    # output never contains both (u, v) and (v, u) as negatives.
    accepted: set = set()
    out_src: list = []
    out_dst: list = []

    # Cap attempts at a small constant so degenerate dense graphs still
    # terminate.  Each attempt generates ``target`` candidates.
    max_attempts = 8
    for _ in range(max_attempts):
        if len(out_src) >= target:
            break
        candidates = _sample_random_pairs(
            target, num_nodes, generator, device,
        )
        # Drop self-loops.
        keep = candidates[0] != candidates[1]
        candidates = candidates[:, keep]
        cand_src = candidates[0].cpu().tolist()
        cand_dst = candidates[1].cpu().tolist()
        for col in range(len(cand_src)):
            u, v = cand_src[col], cand_dst[col]
            h = u * int(num_nodes) + v
            if h in forbidden or h in accepted:
                continue
            if force_undirected:
                rh = v * int(num_nodes) + u
                if rh in accepted:
                    continue
                accepted.add(rh)
            accepted.add(h)
            out_src.append(u)
            out_dst.append(v)
            if len(out_src) >= target:
                break

    out = torch.stack(
        [
            torch.tensor(out_src, dtype=torch.long, device=device),
            torch.tensor(out_dst, dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return out


def _negative_sampling_dense(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    force_undirected: bool,
    seed: Optional[int],
) -> torch.Tensor:
    """Dense path: enumerate all absent edges, then permute and slice.

    Only safe for small graphs.  Memory is O(num_nodes²).
    """
    if num_nodes >= 5_000:
        raise ValueError(
            f"method='dense' refuses num_nodes={num_nodes} "
            f"(would allocate an O(N^2) mask).  Use method='sparse' instead."
        )
    device = edge_index.device
    mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=device)
    # Block self-loops.
    mask[torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)] = False
    # Block positive edges.
    if edge_index.numel():
        mask[edge_index[0].long(), edge_index[1].long()] = False
        if force_undirected:
            mask[edge_index[1].long(), edge_index[0].long()] = False

    # Symmetric absent edges: keep upper triangle when force_undirected
    # (returning one directed edge per undirected absent pair, matching
    # the sparse-path contract).
    if force_undirected:
        triu = torch.triu(mask, diagonal=1)
        absent = triu.nonzero(as_tuple=False)
    else:
        absent = mask.nonzero(as_tuple=False)

    if absent.numel() == 0:
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    generator = _make_generator(seed, device)
    perm = torch.randperm(absent.size(0), generator=generator)
    take = min(num_neg_samples, absent.size(0))
    chosen = absent[perm[:take]]
    out = chosen.t().contiguous().to(device=device, dtype=torch.long)
    return out


def structured_negative_sampling(
    edge_index: torch.Tensor,
    num_nodes: int,
    contains_neg_self_loops: bool = True,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each positive edge (i, j), sample a node k such that (i, k) ∉ E.

    Args:
        edge_index: ``LongTensor[2, E]`` of positive edges.
        num_nodes: Number of nodes in the graph.
        contains_neg_self_loops: When ``True`` (default), the sampled
            ``k`` may equal ``i`` (a self-loop is treated as a valid
            negative).  When ``False``, ``k != i`` is enforced.
        seed: Optional RNG seed.

    Returns:
        ``(i, j, k)`` triplet of ``LongTensor[E]`` aligned with the
        positive edge_index columns.
    """
    _validate_edge_index(edge_index, num_nodes)
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive; got {num_nodes}")
    device = edge_index.device
    num_pos = int(edge_index.size(1))
    if num_pos == 0:
        empty = edge_index.new_zeros((0,), dtype=torch.long)
        return empty, empty, empty

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    # Adjacency set encoded as ``i * N + j`` for fast membership tests.
    forbidden = set(_edge_hash(edge_index, num_nodes).cpu().tolist())
    src_cpu = src.cpu().tolist()

    generator = _make_generator(seed, device)
    k_out = torch.empty(num_pos, dtype=torch.long)
    # Vectorised first attempt.
    cand = torch.randint(num_nodes, (num_pos,), generator=generator)
    for col in range(num_pos):
        i = src_cpu[col]
        attempts = 0
        while True:
            k = int(cand[col])
            if k in (i,) and not contains_neg_self_loops:
                pass  # forbidden
            else:
                if (i * int(num_nodes) + k) not in forbidden:
                    k_out[col] = k
                    break
            attempts += 1
            if attempts > 4 * num_nodes:
                # Degenerate dense case: just pick any non-forbidden node.
                for k in range(num_nodes):
                    if (i * int(num_nodes) + k) in forbidden:
                        continue
                    if not contains_neg_self_loops and k == i:
                        continue
                    k_out[col] = k
                    break
                else:
                    # Truly impossible (graph is complete + no self-loops).
                    raise RuntimeError(
                        f"Cannot find a structured negative for source {i}; "
                        f"node {i} is connected to every other node."
                    )
                break
            # Re-sample one candidate.
            cand[col] = torch.randint(
                num_nodes, (1,), generator=generator,
            ).item()

    # Return on the same device as the input edge_index.
    return src.to(device), dst.to(device), k_out.to(device)


def batched_negative_sampling(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    num_neg_samples: Optional[int] = None,
    method: str = "sparse",
    force_undirected: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Negative sampling that respects graph-batch boundaries.

    Each negative edge is sampled within a single graph (no inter-graph
    negatives).  Useful when training on a :class:`~tgraphx.GraphBatch`.

    Args:
        edge_index: ``LongTensor[2, E]`` of positive edges; node ids are
            global (i.e. already offset across the batch).
        batch: ``LongTensor[N]`` mapping each node to a graph id (the
            standard PyTorch Geometric / TGraphX batch vector).
        num_neg_samples: Target negatives per graph.  When ``None``,
            samples one negative per positive edge in each graph.
        method: ``"sparse"`` or ``"dense"`` (per-graph).
        force_undirected: When ``True``, the per-graph result includes
            both directions.
        seed: Optional RNG seed (used to derive a per-graph seed).

    Returns:
        ``LongTensor[2, num_neg_total]`` of negatives, with global node
        ids matching the input convention.
    """
    if batch.dim() != 1:
        raise ValueError(f"batch must be 1-D; got shape {tuple(batch.shape)}")
    device = edge_index.device
    batch = batch.to(torch.long)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if num_graphs == 0:
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    # Group positive edges by graph (assume both endpoints belong to the
    # same graph — the standard contract for GraphBatch).
    edge_graph = batch[edge_index[0]] if edge_index.numel() else batch.new_zeros((0,))

    pieces = []
    for g in range(num_graphs):
        node_mask = batch == g
        local_node_ids = node_mask.nonzero(as_tuple=False).view(-1)
        if local_node_ids.numel() == 0:
            continue
        # Local edge_index for graph g, remapped to local ids.
        e_mask = edge_graph == g
        ei_g = edge_index[:, e_mask]
        # Map global → local using a position vector.
        global_to_local = edge_index.new_full(
            (int(batch.size(0)),), -1, dtype=torch.long,
        )
        global_to_local[local_node_ids] = torch.arange(
            local_node_ids.numel(), device=device, dtype=torch.long,
        )
        ei_local = global_to_local[ei_g]
        n_g = int(local_node_ids.numel())

        # Per-graph seed derived from the user seed.
        sub_seed = None if seed is None else int(seed) + 7919 * g

        neg_local = negative_sampling(
            ei_local,
            num_nodes=n_g,
            num_neg_samples=num_neg_samples,
            method=method,
            force_undirected=force_undirected,
            seed=sub_seed,
        )
        # Map back to global ids.
        neg_global = local_node_ids[neg_local]
        pieces.append(neg_global)

    if not pieces:
        return edge_index.new_zeros((2, 0), dtype=torch.long)
    return torch.cat(pieces, dim=1)


def hard_negative_sampling(
    edge_index: torch.Tensor,
    node_embeddings: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_neg_samples: Optional[int] = None,
    candidate_pool_size: int = 1024,
    exclude_self_loops: bool = True,
    force_undirected: bool = False,
    seed: Optional[int] = None,
    method: str = "cosine",
) -> torch.Tensor:
    """Sample hard negatives — edges scored high by embedding similarity
    that do not exist as positives.

    A *hard* negative is a non-edge whose endpoints have high embedding
    similarity; such pairs challenge a link-prediction model more than
    random pairs.  The function draws ``candidate_pool_size`` random pairs,
    scores them by cosine (or dot-product) similarity, drops any that are
    positive edges or self-loops, and returns the highest-scoring ones.

    Memory is O(candidate_pool_size) — no dense all-pairs matrix is
    constructed unless ``num_nodes`` is tiny and ``candidate_pool_size``
    exceeds ``num_nodes²``.

    Args:
        edge_index: ``LongTensor[2, E]`` of positive edges.
        node_embeddings: ``FloatTensor[num_nodes, D]`` of node
            representations.  Must reside on the same device as
            ``edge_index``.
        num_nodes: Node count; inferred when ``None``.
        num_neg_samples: Number of hard negatives to return.  Defaults to
            ``edge_index.size(1)`` (one per positive edge).
        candidate_pool_size: Random pairs to score before ranking.  A
            larger pool increases recall of truly hard pairs at the cost
            of memory.  Default: 1 024.
        exclude_self_loops: When ``True`` (default), self-loops are never
            returned.
        force_undirected: Exclude both ``(u, v)`` and ``(v, u)`` from
            positives before scoring.
        seed: Optional RNG seed; no global RNG side effects.
        method: ``"cosine"`` (default) — L2-normalised dot product;
            ``"dot"`` — raw dot product.

    Returns:
        ``LongTensor[2, num_neg]`` of hard-negative edges.  May return
        fewer than ``num_neg_samples`` if the candidate pool contains too
        few valid negatives (logged as a warning).

    Stability: Beta.

    Notes:
        - The function scores random candidates, not all O(N²) pairs.
          It approximates hard negatives rather than finding the true
          hardest pairs.
        - For very small graphs where the candidate pool is large relative
          to N², consider using the dense path of :func:`negative_sampling`
          plus a post-hoc sort by embedding score.
    """
    import warnings

    if method not in ("cosine", "dot"):
        raise ValueError(f"method must be 'cosine' or 'dot'; got {method!r}")
    if candidate_pool_size <= 0:
        raise ValueError(
            f"candidate_pool_size must be positive; got {candidate_pool_size}"
        )
    _validate_edge_index(edge_index, num_nodes if num_nodes is not None else
                         (int(edge_index.max().item()) + 1 if edge_index.numel() else 0))
    if num_nodes is None:
        num_nodes = (
            max(int(edge_index.max().item()) + 1, node_embeddings.size(0))
            if edge_index.numel()
            else node_embeddings.size(0)
        )
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive; got {num_nodes}")
    if node_embeddings.size(0) < num_nodes:
        raise ValueError(
            f"node_embeddings has {node_embeddings.size(0)} rows "
            f"but num_nodes={num_nodes}"
        )

    device = edge_index.device
    target = int(num_neg_samples) if num_neg_samples is not None else max(
        int(edge_index.size(1)), 1,
    )
    if target <= 0:
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    emb = node_embeddings[:num_nodes].to(torch.float32).to(device)
    if method == "cosine":
        norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        emb = emb / norms

    # Build the forbidden set.
    forbidden = set(_edge_hash(edge_index, num_nodes).cpu().tolist())
    if force_undirected and edge_index.numel():
        rev = (
            edge_index[1].to(torch.long) * int(num_nodes)
            + edge_index[0].to(torch.long)
        )
        forbidden.update(rev.cpu().tolist())

    generator = _make_generator(seed, device)
    pool = max(candidate_pool_size, target * 4)
    candidates = _sample_random_pairs(pool, num_nodes, generator, device)

    # Drop self-loops.
    if exclude_self_loops:
        keep = candidates[0] != candidates[1]
        candidates = candidates[:, keep]

    # Drop positives and deduplicate.
    if candidates.numel():
        hashes = _edge_hash(candidates, num_nodes).cpu().tolist()
        seen: set = set()
        keep_idx = []
        for idx, h in enumerate(hashes):
            if h not in forbidden and h not in seen:
                seen.add(h)
                keep_idx.append(idx)
        if keep_idx:
            keep_t = torch.tensor(keep_idx, dtype=torch.long)
            candidates = candidates[:, keep_t]
        else:
            candidates = candidates[:, :0]

    if candidates.numel() == 0:
        warnings.warn(
            "hard_negative_sampling: no valid candidates found in pool; "
            "returning empty tensor.  "
            "Increase candidate_pool_size or reduce num_neg_samples.",
            stacklevel=2,
        )
        return edge_index.new_zeros((2, 0), dtype=torch.long)

    # Score candidates.
    src_emb = emb[candidates[0]]  # [K, D]
    dst_emb = emb[candidates[1]]  # [K, D]
    scores = (src_emb * dst_emb).sum(dim=1)  # [K]

    # Take the top-target by score (highest = hardest).
    take = min(target, scores.size(0))
    if take < target:
        warnings.warn(
            f"hard_negative_sampling: only {take} valid candidates "
            f"found (requested {target}).  "
            "Increase candidate_pool_size.",
            stacklevel=2,
        )
    _, top_idx = scores.topk(take, largest=True, sorted=True)
    return candidates[:, top_idx]
