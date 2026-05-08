"""Graph builder utilities and patch helpers for TGraphX.

All graph builders return a ``torch.LongTensor`` of shape ``[2, E]``
(``edge_index``) ready for use with any TGraphX GNN layer or the
``Graph`` / ``GraphBatch`` constructors.  Pure PyTorch — no scikit-learn,
PyG, DGL, or networkx dependency.

Complexity notes
----------------
* ``build_knn_graph`` and ``build_radius_graph`` call ``torch.cdist``,
  which is **O(N²)** in both time and memory.  Use approximate-NN
  libraries for large N (e.g. > 10 000 nodes).
* ``build_fully_connected_graph`` emits **O(N²)** edges.  Memory grows
  quadratically; keep N small.
* All other builders run in O(E) where E is the number of output edges.

Directedness convention
-----------------------
``directed=True``  — edges go in a single canonical direction; each
``(u → v)`` pair is emitted once.

``directed=False`` — both ``(u → v)`` and ``(v → u)`` are included;
this is the standard undirected representation used by GNNs.

Self-loop convention
--------------------
``self_loops=True``  — exactly one ``i → i`` edge is added per node.

``self_loops=False`` — no self-loops; the builders never produce
duplicate self-loops.

Determinism
-----------
All builders except ``build_random_graph`` produce a deterministic edge
order.  ``build_random_graph`` is deterministic when ``seed`` is set.
These builders create *fixed, rule-based* adjacency structures; they
do **not** implement learned adjacency.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import torch

# Thresholds above which O(N²) builders emit a runtime warning.
_KNN_RADIUS_WARN_THRESHOLD = 10_000
_FC_IOU_WARN_THRESHOLD = 5_000


# --------------------------------------------------------------------------- #
# Internal helpers                                                              #
# --------------------------------------------------------------------------- #

def _normalize_2d(val: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(val, int):
        return val, val
    a, b = val
    return int(a), int(b)


def _normalize_3d(val: Union[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if isinstance(val, int):
        return val, val, val
    a, b, c = val
    return int(a), int(b), int(c)


def _dedup(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate edges and sort by (src, dst) using a flat key."""
    if src.numel() == 0:
        return src, dst
    keys = src * num_nodes + dst
    unique_keys = torch.unique(keys, sorted=True)
    return unique_keys // num_nodes, unique_keys % num_nodes


# --------------------------------------------------------------------------- #
# Grid graphs                                                                   #
# --------------------------------------------------------------------------- #

def build_grid_graph(
    rows: int,
    cols: int,
    directed: bool = False,
    self_loops: bool = True,
    device=None,
) -> torch.LongTensor:
    """Build a 2-D 4-connected grid graph.

    Node ``(r, c)`` maps to index ``r * cols + c``.  Edges connect
    horizontal and vertical neighbours only (no diagonals).

    Args:
        rows: Number of rows (>= 1).
        cols: Number of columns (>= 1).
        directed: If ``False`` (default), include both ``(u, v)`` and
            ``(v, u)`` for every neighbour pair.
        self_loops: If ``True`` (default), add one ``i → i`` per node.
        device: Target device for the output tensor.

    Returns:
        ``edge_index`` shape ``[2, E]``, dtype ``torch.long``.

    Edge count (no self-loops):
        directed=True:  ``rows*(cols-1) + (rows-1)*cols``
        directed=False: ``2 * (rows*(cols-1) + (rows-1)*cols)``
    """
    if rows < 1 or cols < 1:
        raise ValueError(
            f"rows and cols must be >= 1; got rows={rows}, cols={cols}"
        )
    N = rows * cols
    dev = device

    # Horizontal edges: (r, c) → (r, c+1)
    if cols > 1:
        r = torch.arange(rows, device=dev).repeat_interleave(cols - 1)
        c = torch.arange(cols - 1, device=dev).repeat(rows)
        h_src = r * cols + c
        h_dst = h_src + 1
    else:
        h_src = torch.zeros(0, dtype=torch.long, device=dev)
        h_dst = torch.zeros(0, dtype=torch.long, device=dev)

    # Vertical edges: (r, c) → (r+1, c)
    if rows > 1:
        r = torch.arange(rows - 1, device=dev).repeat_interleave(cols)
        c = torch.arange(cols, device=dev).repeat(rows - 1)
        v_src = r * cols + c
        v_dst = v_src + cols
    else:
        v_src = torch.zeros(0, dtype=torch.long, device=dev)
        v_dst = torch.zeros(0, dtype=torch.long, device=dev)

    if directed:
        all_src = torch.cat([h_src, v_src])
        all_dst = torch.cat([h_dst, v_dst])
    else:
        all_src = torch.cat([h_src, h_dst, v_src, v_dst])
        all_dst = torch.cat([h_dst, h_src, v_dst, v_src])

    if self_loops:
        loop = torch.arange(N, dtype=torch.long, device=dev)
        all_src = torch.cat([all_src, loop])
        all_dst = torch.cat([all_dst, loop])

    return torch.stack([all_src, all_dst], dim=0).long()


def build_grid_graph_3d(
    depth: int,
    rows: int,
    cols: int,
    directed: bool = False,
    self_loops: bool = True,
    device=None,
) -> torch.LongTensor:
    """Build a 3-D 6-connected grid graph (face neighbours only).

    Node ``(d, r, c)`` maps to index ``d * rows * cols + r * cols + c``.

    Args:
        depth: Grid depth (>= 1).
        rows: Grid rows (>= 1).
        cols: Grid columns (>= 1).
        directed: If ``False`` (default), include both directions per pair.
        self_loops: If ``True`` (default), add one self-loop per node.
        device: Target device.

    Returns:
        ``edge_index`` shape ``[2, E]``, dtype ``torch.long``.

    Edge count (no self-loops, directed=True):
        ``(depth-1)*rows*cols + depth*(rows-1)*cols + depth*rows*(cols-1)``
    """
    if depth < 1 or rows < 1 or cols < 1:
        raise ValueError(
            f"depth, rows, cols must be >= 1; "
            f"got depth={depth}, rows={rows}, cols={cols}"
        )
    N = depth * rows * cols
    RC = rows * cols
    dev = device

    # Depth edges: (d, r, c) → (d+1, r, c)
    if depth > 1:
        di = torch.arange(depth - 1, device=dev).repeat_interleave(RC)
        ri = torch.arange(RC, device=dev).repeat(depth - 1)
        dep_src = di * RC + ri
        dep_dst = dep_src + RC
    else:
        dep_src = dep_dst = torch.zeros(0, dtype=torch.long, device=dev)

    # Row edges: (d, r, c) → (d, r+1, c)
    if rows > 1:
        di = torch.arange(depth, device=dev).repeat_interleave((rows - 1) * cols)
        ri = torch.arange(rows - 1, device=dev).repeat_interleave(cols).repeat(depth)
        ci = torch.arange(cols, device=dev).repeat((rows - 1) * depth)
        row_src = di * RC + ri * cols + ci
        row_dst = row_src + cols
    else:
        row_src = row_dst = torch.zeros(0, dtype=torch.long, device=dev)

    # Col edges: (d, r, c) → (d, r, c+1)
    if cols > 1:
        di = torch.arange(depth, device=dev).repeat_interleave(rows * (cols - 1))
        ri = torch.arange(rows, device=dev).repeat_interleave(cols - 1).repeat(depth)
        ci = torch.arange(cols - 1, device=dev).repeat(rows * depth)
        col_src = di * RC + ri * cols + ci
        col_dst = col_src + 1
    else:
        col_src = col_dst = torch.zeros(0, dtype=torch.long, device=dev)

    if directed:
        all_src = torch.cat([dep_src, row_src, col_src])
        all_dst = torch.cat([dep_dst, row_dst, col_dst])
    else:
        all_src = torch.cat([dep_src, dep_dst, row_src, row_dst, col_src, col_dst])
        all_dst = torch.cat([dep_dst, dep_src, row_dst, row_src, col_dst, col_src])

    if self_loops:
        loop = torch.arange(N, dtype=torch.long, device=dev)
        all_src = torch.cat([all_src, loop])
        all_dst = torch.cat([all_dst, loop])

    return torch.stack([all_src, all_dst], dim=0).long()


# --------------------------------------------------------------------------- #
# Complete graph                                                                #
# --------------------------------------------------------------------------- #

def build_fully_connected_graph(
    num_nodes: int,
    directed: bool = True,
    self_loops: bool = False,
    device=None,
) -> torch.LongTensor:
    """Build a fully-connected (complete) graph.

    For a complete graph every pair ``(i, j)`` with ``i != j`` appears in
    both directions; the ``directed`` flag does not change the edge count
    but is provided for API consistency.

    .. warning::
        Edge count is **O(N²)**.  Memory use grows quadratically.

    Args:
        num_nodes: Number of nodes (>= 1).
        directed: Accepted for API consistency (complete graph is always
            symmetric).
        self_loops: If ``True``, include ``(i, i)`` for every node.
        device: Target device.

    Returns:
        ``edge_index`` ``[2, N*(N-1) + N*self_loops]``, dtype ``torch.long``.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    N = num_nodes
    if N > _FC_IOU_WARN_THRESHOLD:
        warnings.warn(
            f"build_fully_connected_graph: num_nodes={N} > {_FC_IOU_WARN_THRESHOLD}. "
            f"Edge count is O(N²) ({N * (N - 1)} edges). "
            f"Memory use grows quadratically — consider a sparser graph builder.",
            stacklevel=2,
        )
    idx = torch.arange(N, device=device, dtype=torch.long)
    src = idx.repeat_interleave(N)
    dst = idx.repeat(N)
    if not self_loops:
        mask = src != dst
        src, dst = src[mask], dst[mask]
    return torch.stack([src, dst], dim=0)


# --------------------------------------------------------------------------- #
# kNN graph                                                                     #
# --------------------------------------------------------------------------- #

def build_knn_graph(
    coords: torch.Tensor,
    k: int,
    directed: bool = False,
    self_loops: bool = True,
    chunk_size: Optional[int] = None,
) -> torch.LongTensor:
    """Build a k-nearest-neighbour graph from node coordinates.

    .. warning::
        Computation is **O(N²)** in time regardless of ``chunk_size``.
        Without ``chunk_size`` the full ``N×N`` distance matrix is
        materialised (**O(N²) memory**).  Setting ``chunk_size=K``
        processes ``K`` rows at a time (**O(K×N) peak memory**), which
        can make large-N runs feasible.

    Self-loops are never counted as neighbours; they are appended
    separately when ``self_loops=True``.

    Args:
        coords: ``[N, D]`` coordinate tensor.
        k: Number of nearest neighbours per node (self excluded).
        directed: If ``False`` (default), include both ``(u→v)`` and
            ``(v→u)`` for every kNN pair (with deduplication).
        self_loops: If ``True`` (default), add one ``i→i`` per node.
        chunk_size: Row-chunk size for distance computation.  ``None``
            (default) materialises the full ``[N, N]`` distance matrix.
            Values > 0 cap peak memory to ``O(chunk_size × N)`` while
            total computation remains O(N²).

    Returns:
        ``edge_index`` ``[2, E]``, dtype ``torch.long``.
    """
    if coords.dim() != 2:
        raise ValueError(
            f"coords must be 2-D [N, D]; got shape {tuple(coords.shape)}"
        )
    N = coords.size(0)
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}")
    if k >= N:
        raise ValueError(
            f"k={k} must be less than num_nodes={N} "
            f"(self is excluded from kNN)"
        )
    if N > _KNN_RADIUS_WARN_THRESHOLD and chunk_size is None:
        warnings.warn(
            f"build_knn_graph: num_nodes={N} > {_KNN_RADIUS_WARN_THRESHOLD}. "
            f"torch.cdist allocates an O(N²) distance matrix ({N}×{N} floats). "
            f"Pass chunk_size=K to reduce peak memory to O(K×N), or use an "
            f"approximate-NN library for very large N.",
            stacklevel=2,
        )

    coords_f = coords.float()

    if chunk_size is None:
        dist = torch.cdist(coords_f, coords_f)  # [N, N], O(N²)
        inf_mask = torch.diag(torch.full((N,), float("inf"), device=coords.device))
        dist = dist + inf_mask
        _, nn_idx = torch.topk(dist, k, largest=False, dim=1)  # [N, k]
        src = torch.arange(N, device=coords.device, dtype=torch.long).repeat_interleave(k)
        dst = nn_idx.reshape(-1).long()
    else:
        # Chunked path: O(chunk_size × N) peak memory per iteration.
        all_src: list[torch.Tensor] = []
        all_dst: list[torch.Tensor] = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_c = end - start
            chunk_f = coords_f[start:end]
            dist_chunk = torch.cdist(chunk_f, coords_f)  # [n_c, N]
            # Vectorised: set self-distances to inf (row i → col start+i).
            local_rows = torch.arange(n_c, device=coords.device)
            global_cols = torch.arange(start, end, device=coords.device)
            dist_chunk[local_rows, global_cols] = float("inf")
            _, nn_c = torch.topk(dist_chunk, k, largest=False, dim=1)  # [n_c, k]
            src_c = torch.arange(start, end, device=coords.device, dtype=torch.long).repeat_interleave(k)
            dst_c = nn_c.reshape(-1).long()
            all_src.append(src_c)
            all_dst.append(dst_c)
        src = torch.cat(all_src)
        dst = torch.cat(all_dst)

    if not directed:
        src, dst = _dedup(
            torch.cat([src, dst]),
            torch.cat([dst, src]),
            N,
        )

    if self_loops:
        loop = torch.arange(N, device=coords.device, dtype=torch.long)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

    return torch.stack([src, dst], dim=0)


# --------------------------------------------------------------------------- #
# Radius graph                                                                  #
# --------------------------------------------------------------------------- #

def build_radius_graph(
    coords: torch.Tensor,
    radius: float,
    directed: bool = False,
    self_loops: bool = True,
    chunk_size: Optional[int] = None,
) -> torch.LongTensor:
    """Build a radius graph: connect all pairs within Euclidean ``radius``.

    .. warning::
        Computation is **O(N²)** in time regardless of ``chunk_size``.
        Without ``chunk_size`` the full ``N×N`` distance matrix is
        materialised (**O(N²) memory**).  Setting ``chunk_size=K``
        processes ``K`` rows at a time (**O(K×N) peak memory**).

    Args:
        coords: ``[N, D]`` coordinate tensor.
        radius: Distance threshold (inclusive).
        directed: If ``False`` (default), include both directions per pair.
        self_loops: If ``True`` (default), add one self-loop per node.
        chunk_size: Row-chunk size.  ``None`` materialises the full
            ``[N, N]`` distance matrix.  Values > 0 cap peak memory to
            ``O(chunk_size × N)``.

    Returns:
        ``edge_index`` ``[2, E]``, dtype ``torch.long``.
    """
    if coords.dim() != 2:
        raise ValueError(
            f"coords must be 2-D [N, D]; got shape {tuple(coords.shape)}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be > 0; got {radius}")

    N = coords.size(0)
    if N > _KNN_RADIUS_WARN_THRESHOLD and chunk_size is None:
        warnings.warn(
            f"build_radius_graph: num_nodes={N} > {_KNN_RADIUS_WARN_THRESHOLD}. "
            f"torch.cdist allocates an O(N²) distance matrix ({N}×{N} floats). "
            f"Pass chunk_size=K to reduce peak memory to O(K×N), or use an "
            f"approximate-NN library for very large N.",
            stacklevel=2,
        )

    coords_f = coords.float()

    if chunk_size is None:
        dist = torch.cdist(coords_f, coords_f)  # [N, N]
        eye = torch.eye(N, dtype=torch.bool, device=coords.device)
        mask = (dist <= radius) & ~eye

        if directed:
            upper = torch.triu(torch.ones(N, N, dtype=torch.bool, device=coords.device), diagonal=1)
            mask = mask & upper

        src, dst = torch.where(mask)
        src, dst = src.long(), dst.long()
    else:
        # Chunked path: fully vectorised, O(chunk_size × N) peak memory.
        all_src: list[torch.Tensor] = []
        all_dst: list[torch.Tensor] = []
        col_idx = torch.arange(N, device=coords.device, dtype=torch.long)  # [N]
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_c = end - start
            chunk_f = coords_f[start:end]
            dist_chunk = torch.cdist(chunk_f, coords_f)  # [n_c, N]

            # Self-exclusion mask: row i must not connect to col start+i.
            self_mask = torch.zeros(n_c, N, dtype=torch.bool, device=coords.device)
            local_rows = torch.arange(n_c, device=coords.device)
            self_mask[local_rows, torch.arange(start, end, device=coords.device)] = True

            conn = (dist_chunk <= radius) & ~self_mask  # [n_c, N]

            if directed:
                # Keep only (i, j) where j > global_i (upper triangle).
                global_rows = torch.arange(start, end, device=coords.device, dtype=torch.long).unsqueeze(1)
                conn = conn & (col_idx.unsqueeze(0) > global_rows)

            chunk_row, chunk_col = torch.where(conn)
            if chunk_row.numel() > 0:
                all_src.append((chunk_row + start).long())
                all_dst.append(chunk_col.long())

        if all_src:
            src = torch.cat(all_src)
            dst = torch.cat(all_dst)
        else:
            src = torch.empty(0, dtype=torch.long, device=coords.device)
            dst = torch.empty(0, dtype=torch.long, device=coords.device)

    if self_loops:
        loop = torch.arange(N, device=coords.device, dtype=torch.long)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])

    return torch.stack([src, dst], dim=0)


# --------------------------------------------------------------------------- #
# IoU graph                                                                     #
# --------------------------------------------------------------------------- #

def build_iou_graph(
    boxes: torch.Tensor,
    threshold: float,
    directed: bool = False,
    self_loops: bool = True,
    chunk_size: Optional[int] = None,
) -> torch.LongTensor:
    """Build a graph where nodes (bounding boxes) are connected if IoU ≥ threshold.

    ``IoU(i, i) = 1.0`` so every self-loop is present whenever
    ``threshold <= 1.0`` and ``self_loops=True``.

    .. warning::
        Computation is **O(N²)** in time.  Without ``chunk_size`` the full
        ``N×N`` IoU matrix is materialised (**O(N²) memory**).  Setting
        ``chunk_size=K`` processes ``K`` boxes at a time (**O(K×N) peak
        memory**).

    Args:
        boxes: ``[N, 4]`` in ``(x1, y1, x2, y2)`` format.
        threshold: Minimum IoU to create an edge (inclusive, in [0, 1]).
        directed: If ``False`` (default), include both ``(i,j)`` and
            ``(j,i)`` for every connected pair.
        self_loops: If ``True`` (default), include ``(i, i)`` when
            ``IoU(i,i) >= threshold``.
        chunk_size: Row-chunk size.  ``None`` materialises the full
            ``[N, N]`` IoU matrix.  Values > 0 cap peak memory to
            ``O(chunk_size × N)``.

    Returns:
        ``edge_index`` ``[2, E]``, dtype ``torch.long``.
    """
    if boxes.dim() != 2 or boxes.size(1) != 4:
        raise ValueError(
            f"boxes must have shape [N, 4]; got {tuple(boxes.shape)}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}")

    N = boxes.size(0)
    if N > _FC_IOU_WARN_THRESHOLD and chunk_size is None:
        warnings.warn(
            f"build_iou_graph: num_nodes={N} > {_FC_IOU_WARN_THRESHOLD}. "
            f"IoU computation allocates an O(N²) matrix ({N}×{N} elements). "
            f"Pass chunk_size=K to reduce peak memory to O(K×N).",
            stacklevel=2,
        )
    b = boxes.float()
    areas = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    def _iou_row_chunk(b_chunk: torch.Tensor, areas_chunk: torch.Tensor) -> torch.Tensor:
        """Return IoU matrix of shape [len(b_chunk), N] against all N boxes."""
        x1 = torch.max(b_chunk[:, 0].unsqueeze(1), b[:, 0].unsqueeze(0))
        y1 = torch.max(b_chunk[:, 1].unsqueeze(1), b[:, 1].unsqueeze(0))
        x2 = torch.min(b_chunk[:, 2].unsqueeze(1), b[:, 2].unsqueeze(0))
        y2 = torch.min(b_chunk[:, 3].unsqueeze(1), b[:, 3].unsqueeze(0))
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = areas_chunk.unsqueeze(1) + areas.unsqueeze(0) - inter
        return inter / union.clamp(min=1e-8)

    if chunk_size is None:
        iou = _iou_row_chunk(b, areas)  # [N, N]
        conn = iou >= threshold

        if not self_loops:
            conn.fill_diagonal_(False)

        if directed:
            diag_offset = 0 if self_loops else 1
            upper = torch.triu(
                torch.ones(N, N, dtype=torch.bool, device=boxes.device),
                diagonal=diag_offset,
            )
            conn = conn & upper

        src, dst = torch.where(conn)
        return torch.stack([src.long(), dst.long()], dim=0)
    else:
        # Chunked path: O(chunk_size × N) peak memory.
        all_src: list[torch.Tensor] = []
        all_dst: list[torch.Tensor] = []
        col_idx = torch.arange(N, device=boxes.device, dtype=torch.long)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            n_c = end - start
            b_c = b[start:end]
            areas_c = areas[start:end]
            iou_c = _iou_row_chunk(b_c, areas_c)  # [n_c, N]
            conn_c = iou_c >= threshold  # [n_c, N]

            if not self_loops:
                local_rows = torch.arange(n_c, device=boxes.device)
                conn_c[local_rows, torch.arange(start, end, device=boxes.device)] = False

            if directed:
                global_rows = torch.arange(start, end, device=boxes.device, dtype=torch.long).unsqueeze(1)
                diag_offset = 0 if self_loops else 1
                conn_c = conn_c & (col_idx.unsqueeze(0) >= global_rows + diag_offset)

            chunk_row, chunk_col = torch.where(conn_c)
            if chunk_row.numel() > 0:
                all_src.append((chunk_row + start).long())
                all_dst.append(chunk_col.long())

        if all_src:
            return torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
        return torch.zeros(2, 0, dtype=torch.long, device=boxes.device)


# --------------------------------------------------------------------------- #
# Random graph                                                                  #
# --------------------------------------------------------------------------- #

def _build_random_graph_sample(
    N: int,
    num_edges: int,
    directed: bool,
    self_loops: bool,
    seed: Optional[int],
    device,
) -> torch.LongTensor:
    """O(num_edges) memory random graph sampling for large N.

    Only supports ``directed=True, self_loops=False`` (the common case).
    Uses a flat-index → (src, dst) mapping:
        flat_idx in [0, N*(N-1))
        src = flat_idx // (N-1)
        raw = flat_idx % (N-1)
        dst = raw if raw < src else raw + 1   # skip self
    Samples without replacement via rejection sampling (O(num_edges) expected).
    """
    if not directed or self_loops:
        raise ValueError(
            "algorithm='sample' currently supports only "
            "directed=True and self_loops=False. "
            "For other configurations use algorithm='exact'."
        )
    n_cand = N * (N - 1)
    if num_edges < 0:
        raise ValueError(f"num_edges must be >= 0; got {num_edges}")
    if num_edges > n_cand:
        raise ValueError(
            f"Cannot sample {num_edges} directed edges from {n_cand} candidates "
            f"(num_nodes={N}, directed=True, self_loops=False)."
        )
    if num_edges == 0:
        ei = torch.zeros(2, 0, dtype=torch.long)
        if device is not None:
            ei = ei.to(device)
        return ei

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # Rejection sampling: over-sample and deduplicate until we have enough.
    collected: set[int] = set()
    oversample = max(int(num_edges * 1.5) + 16, num_edges + 16)
    while len(collected) < num_edges:
        batch = torch.randint(0, n_cand, (oversample,), generator=gen).tolist()
        for idx in batch:
            collected.add(idx)
            if len(collected) == num_edges:
                break
        oversample = min(oversample * 2, n_cand)

    flat = sorted(collected)  # deterministic order
    flat_t = torch.tensor(flat, dtype=torch.long)
    src = flat_t // (N - 1)
    raw = flat_t % (N - 1)
    dst = torch.where(raw < src, raw, raw + 1)

    if device is not None:
        src = src.to(device)
        dst = dst.to(device)
    return torch.stack([src, dst], dim=0)


def build_random_graph(
    num_nodes: int,
    num_edges: int,
    directed: bool = True,
    self_loops: bool = False,
    seed: Optional[int] = None,
    device=None,
    algorithm: str = "exact",
) -> torch.LongTensor:
    """Build a random graph by sampling edges without replacement.

    For ``directed=True``, exactly ``num_edges`` directed edges are
    returned (``edge_index.size(1) == num_edges``).

    For ``directed=False``, ``num_edges`` unique undirected pairs are
    sampled from the candidate pool and their reverse edges are appended
    for non-self pairs, so ``edge_index.size(1)`` is between
    ``num_edges`` (all self-loops) and ``2 * num_edges`` (no
    self-loops).

    Args:
        num_nodes: Number of nodes (>= 1).
        num_edges: Edges to sample (see directedness note above).
        directed: If ``True`` (default), sample directed edges.
        self_loops: If ``False`` (default), exclude self-loops from
            the candidate pool.
        seed: Optional RNG seed for reproducibility.
        device: Target device.
        algorithm: Sampling algorithm.

            * ``"exact"`` (default) — builds the full O(N²) candidate pool
              and samples via ``torch.randperm``.  Deterministic and exact;
              use for small/medium N.
            * ``"sample"`` — uses O(``num_edges``) memory by sampling flat
              edge indices directly without materialising the full candidate
              pool.  Suitable for large N where ``num_edges ≪ N²``.
              Requires ``directed=True, self_loops=False`` (the most common
              case); raises ``ValueError`` for unsupported combinations.

    Returns:
        ``edge_index`` ``[2, E]``, dtype ``torch.long``.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
    if num_edges < 0:
        raise ValueError(f"num_edges must be >= 0; got {num_edges}")
    if algorithm not in ("exact", "sample"):
        raise ValueError(f"algorithm must be 'exact' or 'sample'; got {algorithm!r}")

    N = num_nodes

    if algorithm == "sample":
        return _build_random_graph_sample(N, num_edges, directed, self_loops, seed, device)

    # ── "exact" path: O(N²) candidate pool ──────────────────────────────────
    # Build candidate pool on CPU (randperm is CPU-only for large N)
    idx = torch.arange(N)
    src_cand = idx.repeat_interleave(N)
    dst_cand = idx.repeat(N)

    if not self_loops:
        keep = src_cand != dst_cand
        src_cand, dst_cand = src_cand[keep], dst_cand[keep]

    if not directed:
        # Only upper triangle to avoid symmetry duplicates before adding reverses
        if self_loops:
            keep = src_cand <= dst_cand
        else:
            keep = src_cand < dst_cand
        src_cand, dst_cand = src_cand[keep], dst_cand[keep]

    n_cand = src_cand.size(0)
    if num_edges > n_cand:
        raise ValueError(
            f"Cannot sample {num_edges} edges from {n_cand} candidates "
            f"(num_nodes={N}, directed={directed}, self_loops={self_loops})."
        )

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    perm = torch.randperm(n_cand, generator=gen)[:num_edges]
    perm, _ = perm.sort()  # deterministic order within sample

    src = src_cand[perm]
    dst = dst_cand[perm]

    if not directed:
        non_self = src != dst
        fwd_src, fwd_dst = src, dst
        src = torch.cat([fwd_src, fwd_dst[non_self]])
        dst = torch.cat([fwd_dst, fwd_src[non_self]])

    if device is not None:
        src = src.to(device)
        dst = dst.to(device)

    return torch.stack([src, dst], dim=0).long()


# --------------------------------------------------------------------------- #
# 2-D patch helpers                                                             #
# --------------------------------------------------------------------------- #

def patch_grid_shape(
    height: int,
    width: int,
    patch_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int], None] = None,
) -> Tuple[int, int]:
    """Return the ``(n_rows, n_cols)`` patch grid for an image.

    Raises ``ValueError`` if ``height`` or ``width`` is not exactly
    covered by ``patch_size`` and ``stride``.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        patch_size: Patch height and width as int or ``(ph, pw)``.
        stride: Stride as int or ``(sh, sw)``.  Defaults to
            ``patch_size`` (non-overlapping).

    Returns:
        ``(n_h, n_w)`` number of patch rows and columns.
    """
    ph, pw = _normalize_2d(patch_size)
    sh, sw = _normalize_2d(stride) if stride is not None else (ph, pw)

    for size, patch, step, name in [
        (height, ph, sh, "height"),
        (width, pw, sw, "width"),
    ]:
        if patch <= 0 or step <= 0:
            raise ValueError(
                f"patch_size and stride must be positive; "
                f"got patch={patch}, stride={step}"
            )
        if patch > size:
            raise ValueError(
                f"{name}={size} is smaller than patch_size={patch}"
            )
        if (size - patch) % step != 0:
            raise ValueError(
                f"{name}={size} is not exactly covered by "
                f"patch_size={patch} and stride={step}: "
                f"({size} - {patch}) % {step} = {(size - patch) % step} != 0. "
                f"Choose dimensions so (size - patch) is divisible by stride."
            )

    return (height - ph) // sh + 1, (width - pw) // sw + 1


def image_to_patches(
    images: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int], None] = None,
    padding: str = "none",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Extract patches from a batch of 2-D images.

    Patch order is row-major (top-left to bottom-right), matching the
    node order of ``build_grid_graph(n_h, n_w)``.

    Args:
        images: ``[B, C, H, W]`` image batch.
        patch_size: Patch size as int or ``(ph, pw)``.
        stride: Stride as int or ``(sh, sw)``.  Defaults to
            ``patch_size`` (non-overlapping patches).
        padding: Padding mode.

            * ``"none"`` (default) — raises ``ValueError`` if the image
              dimensions are not exactly divisible by ``patch_size`` and
              ``stride``.  Identical to the v0.2.3 behaviour.
            * ``"auto"`` — pads the image on the right and bottom so that
              the dimensions are exactly covered.  The padded region is
              filled with ``pad_value`` (default ``0.0``).  The number of
              patches therefore equals ``ceil((H - ph) / sh) + 1`` ×
              ``ceil((W - pw) / sw) + 1``.

        pad_value: Constant fill value used when ``padding="auto"``.

    Returns:
        ``[B, P, C, ph, pw]`` where ``P = n_h * n_w``.

    Raises:
        ValueError: If ``images`` is not 4-D; if ``padding="none"`` and
            dimensions are not exactly covered; if ``padding`` is unknown.
    """
    if images.dim() != 4:
        raise ValueError(
            f"images must be a 4-D tensor [B, C, H, W]; "
            f"got shape {tuple(images.shape)}"
        )
    if padding not in ("none", "auto"):
        raise ValueError(
            f"padding must be 'none' or 'auto'; got {padding!r}"
        )
    B, C, H, W = images.shape
    ph, pw = _normalize_2d(patch_size)
    sh, sw = _normalize_2d(stride) if stride is not None else (ph, pw)

    if padding == "auto":
        import math
        n_h = math.ceil(max(H - ph, 0) / sh) + 1
        n_w = math.ceil(max(W - pw, 0) / sw) + 1
        H_needed = (n_h - 1) * sh + ph
        W_needed = (n_w - 1) * sw + pw
        pad_h = max(H_needed - H, 0)
        pad_w = max(W_needed - W, 0)
        if pad_h > 0 or pad_w > 0:
            images = torch.nn.functional.pad(
                images, (0, pad_w, 0, pad_h), mode="constant", value=pad_value
            )
    else:
        n_h, n_w = patch_grid_shape(H, W, (ph, pw), (sh, sw))

    x = images.unfold(2, ph, sh).unfold(3, pw, sw)
    # [B, C, n_h, n_w, ph, pw]
    x = x.contiguous().view(B, C, n_h * n_w, ph, pw)
    # [B, C, P, ph, pw]
    return x.permute(0, 2, 1, 3, 4)
    # [B, P, C, ph, pw]


# --------------------------------------------------------------------------- #
# 3-D patch helpers                                                             #
# --------------------------------------------------------------------------- #

def volume_patch_grid_shape(
    depth: int,
    height: int,
    width: int,
    patch_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int], None] = None,
) -> Tuple[int, int, int]:
    """Return the ``(n_d, n_h, n_w)`` patch grid for a volume.

    Raises ``ValueError`` if any dimension is not exactly covered.

    Args:
        depth: Volume depth.
        height: Volume height.
        width: Volume width.
        patch_size: Patch size as int or ``(pd, ph, pw)``.
        stride: Stride as int or ``(sd, sh, sw)``.  Defaults to
            ``patch_size`` (non-overlapping).

    Returns:
        ``(n_d, n_h, n_w)`` patch counts along each axis.
    """
    pd, ph, pw = _normalize_3d(patch_size)
    sd, sh, sw = _normalize_3d(stride) if stride is not None else (pd, ph, pw)

    for size, patch, step, name in [
        (depth, pd, sd, "depth"),
        (height, ph, sh, "height"),
        (width, pw, sw, "width"),
    ]:
        if patch <= 0 or step <= 0:
            raise ValueError(
                f"patch_size and stride must be positive; "
                f"got patch={patch}, stride={step}"
            )
        if patch > size:
            raise ValueError(
                f"{name}={size} is smaller than patch_size={patch}"
            )
        if (size - patch) % step != 0:
            raise ValueError(
                f"{name}={size} is not exactly covered by "
                f"patch_size={patch} and stride={step}: "
                f"({size} - {patch}) % {step} = {(size - patch) % step} != 0. "
                f"Choose dimensions so (size - patch) is divisible by stride."
            )

    return (depth - pd) // sd + 1, (height - ph) // sh + 1, (width - pw) // sw + 1


def volume_to_patches(
    volumes: torch.Tensor,
    patch_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int], None] = None,
    padding: str = "none",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Extract patches from a batch of 3-D volumes.

    Patch order is depth-row-col (C-order), matching the node order of
    ``build_grid_graph_3d(n_d, n_h, n_w)``.

    Args:
        volumes: ``[B, C, D, H, W]`` volume batch.
        patch_size: Patch size as int or ``(pd, ph, pw)``.
        stride: Stride as int or ``(sd, sh, sw)``.  Defaults to
            ``patch_size`` (non-overlapping).
        padding: ``"none"`` (default, exact tiling required) or ``"auto"``
            (pad to make dimensions exactly divisible).
        pad_value: Constant fill value used when ``padding="auto"``.

    Returns:
        ``[B, P, C, pd, ph, pw]`` where ``P = n_d * n_h * n_w``.

    Raises:
        ValueError: If ``volumes`` is not 5-D, or if ``padding="none"`` and
            dimensions are not exactly covered.
    """
    if volumes.dim() != 5:
        raise ValueError(
            f"volumes must be a 5-D tensor [B, C, D, H, W]; "
            f"got shape {tuple(volumes.shape)}"
        )
    if padding not in ("none", "auto"):
        raise ValueError(
            f"padding must be 'none' or 'auto'; got {padding!r}"
        )
    B, C, D, H, W = volumes.shape
    pd, ph, pw = _normalize_3d(patch_size)
    sd, sh, sw = _normalize_3d(stride) if stride is not None else (pd, ph, pw)

    if padding == "auto":
        import math
        n_d = math.ceil(max(D - pd, 0) / sd) + 1
        n_h = math.ceil(max(H - ph, 0) / sh) + 1
        n_w = math.ceil(max(W - pw, 0) / sw) + 1
        D_needed = (n_d - 1) * sd + pd
        H_needed = (n_h - 1) * sh + ph
        W_needed = (n_w - 1) * sw + pw
        pad_d = max(D_needed - D, 0)
        pad_h = max(H_needed - H, 0)
        pad_w = max(W_needed - W, 0)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volumes = torch.nn.functional.pad(
                volumes, (0, pad_w, 0, pad_h, 0, pad_d),
                mode="constant", value=pad_value,
            )
    else:
        n_d, n_h, n_w = volume_patch_grid_shape(D, H, W, (pd, ph, pw), (sd, sh, sw))

    x = volumes.unfold(2, pd, sd).unfold(3, ph, sh).unfold(4, pw, sw)
    # [B, C, n_d, n_h, n_w, pd, ph, pw]
    x = x.contiguous().view(B, C, n_d * n_h * n_w, pd, ph, pw)
    # [B, C, P, pd, ph, pw]
    return x.permute(0, 2, 1, 3, 4, 5)
    # [B, P, C, pd, ph, pw]


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

__all__ = [
    "build_grid_graph",
    "build_grid_graph_3d",
    "build_fully_connected_graph",
    "build_knn_graph",
    "build_radius_graph",
    "build_iou_graph",
    "build_random_graph",
    "patch_grid_shape",
    "image_to_patches",
    "volume_patch_grid_shape",
    "volume_to_patches",
]
