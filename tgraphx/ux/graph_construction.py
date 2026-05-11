"""Graph construction helpers (kNN, class prototypes, image-to-patch).

These are the reusable, leakage-aware, tensor-native versions of the patterns
that the advanced notebooks 31, 32, and others previously had to inline.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def _flatten_for_similarity(x: torch.Tensor) -> torch.Tensor:
    """Flatten any tensor-valued node features to 2-D ONLY for similarity computation.

    The original tensor is NOT modified. This satisfies the tensor-native flag:
    no silent flattening of the graph's node features.
    """
    if x.dim() == 2:
        return x
    return x.reshape(x.size(0), -1)


def knn_graph(
    x: torch.Tensor,
    k: int = 10,
    *,
    metric: str = "cosine",
    chunk_size: int = 1024,
    exclude_self: bool = True,
    make_symmetric: bool = True,
) -> torch.Tensor:
    """Build a kNN edge_index from node features.

    Supports tensor-valued features `[N, C, H, W]` etc. — they are flattened
    only for similarity, never overwritten.

    Args:
        x: Node-feature tensor, shape `[N, ...]`.
        k: Number of nearest neighbors per node.
        metric: ``"cosine"`` (default) or ``"euclidean"``.
        chunk_size: Process at most this many query nodes per inner loop to
            avoid O(N^2) memory.
        exclude_self: If True, the source node itself is excluded.
        make_symmetric: If True, output is undirected (both directions present).

    Returns:
        ``LongTensor[2, E]`` edge_index.
    """
    if x.dim() < 2:
        raise ValueError(f"x must have at least 2 dims; got shape {tuple(x.shape)}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if metric not in ("cosine", "euclidean"):
        raise ValueError(
            f"metric must be 'cosine' or 'euclidean'; got {metric!r}"
        )

    N = x.size(0)
    if k > N - (1 if exclude_self else 0):
        raise ValueError(
            f"k={k} too large for N={N} nodes "
            f"(max with exclude_self={exclude_self}: {N - (1 if exclude_self else 0)})"
        )

    flat = _flatten_for_similarity(x).float()
    if metric == "cosine":
        flat = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    src_list, dst_list = [], []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk = flat[i:end]
        if metric == "cosine":
            sims = chunk @ flat.T          # [C, N] higher = closer
            if exclude_self:
                # mark self-self with -inf
                idx = torch.arange(i, end, device=sims.device)
                sims[torch.arange(end - i, device=sims.device), idx] = float("-inf")
            _, topk = sims.topk(k, dim=1)
        else:  # euclidean
            diff = chunk.unsqueeze(1) - flat.unsqueeze(0)
            d = diff.pow(2).sum(-1)
            if exclude_self:
                idx = torch.arange(i, end, device=d.device)
                d[torch.arange(end - i, device=d.device), idx] = float("inf")
            _, topk = d.topk(k, dim=1, largest=False)
        base = torch.arange(i, end, device=topk.device).unsqueeze(1).expand_as(topk)
        src_list.append(base.reshape(-1))
        dst_list.append(topk.reshape(-1))

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    ei = torch.stack([src, dst], dim=0)
    if make_symmetric:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ei = torch.unique(ei, dim=1)
    return ei.long()


def build_class_prototypes(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build per-class prototype features from TRAIN nodes only.

    Tensor-native: returns prototypes with the same node-feature shape as `x`.
    For an [N, C, H, W] input, prototypes are [num_classes, C, H, W].

    Args:
        x: Node features `[N, ...]`.
        y: Class labels `[N]`.
        train_mask: Boolean mask of training nodes `[N]`.
        num_classes: Number of classes.

    Returns:
        Prototype tensor `[num_classes, ...]`. Missing classes get zeros.
    """
    if train_mask is None:
        raise ValueError(
            "build_class_prototypes requires train_mask to enforce the "
            "no-label-leakage policy. Validation/test labels must not be used."
        )
    if x.size(0) != y.size(0):
        raise ValueError(f"x has {x.size(0)} nodes but y has {y.size(0)} labels")
    if x.size(0) != train_mask.size(0):
        raise ValueError(
            f"x has {x.size(0)} nodes but train_mask has {train_mask.size(0)} entries"
        )
    feat_shape = x.shape[1:]
    proto = torch.zeros(num_classes, *feat_shape, dtype=x.dtype, device=x.device)
    train_bool = train_mask.bool()
    for c in range(num_classes):
        m = train_bool & (y == c)
        if int(m.sum().item()) > 0:
            proto[c] = x[m].mean(0)
    return proto


def build_prototype_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    *,
    k_proto: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build prototype-membership edges connecting nodes to their nearest class prototypes.

    No-leakage policy:
      Prototypes are computed from train labels only. Validation/test nodes
      connect by visual similarity to prototype features, NEVER by their labels.

    Args:
        x: Node features `[N, ...]`.
        y: Class labels `[N]` (only train entries are used).
        train_mask: `[N]` bool.
        num_classes: number of classes; prototype node IDs are N..N+num_classes-1.
        k_proto: number of prototype neighbors per node.

    Returns:
        Tuple of:
          - proto_features `[num_classes, ...]`
          - proto_edges `[2, N*k_proto]` (src image-node → dst prototype-node)
          - all_features `[N + num_classes, ...]` (original x stacked with prototypes)
    """
    proto = build_class_prototypes(x, y, train_mask, num_classes)
    flat_x = _flatten_for_similarity(x).float()
    flat_p = _flatten_for_similarity(proto).float()
    flat_x = flat_x / flat_x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    flat_p = flat_p / flat_p.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = flat_x @ flat_p.T  # [N, num_classes]
    _, best = sims.topk(k_proto, dim=1)
    N = x.size(0)
    src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k_proto).reshape(-1)
    dst = best.reshape(-1) + N
    proto_edges = torch.stack([src, dst], dim=0).long()
    all_features = torch.cat([x, proto.to(x.dtype)], dim=0)
    return proto, proto_edges, all_features


def image_to_patch_graph(
    image: torch.Tensor,
    patch_size: int = 8,
    *,
    self_loops: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a single image `[C, H, W]` into a patch graph.

    Patches are `[C, patch_size, patch_size]` tensor nodes connected by 4-grid
    spatial adjacency.

    Args:
        image: `[C, H, W]` tensor. H and W must be multiples of `patch_size`,
            otherwise raise ValueError with the fix.
        patch_size: side length of each square patch.
        self_loops: if True, include self-loops on each patch node.

    Returns:
        Tuple of (node_features `[num_patches, C, patch_size, patch_size]`,
                  edge_index `[2, E]`).
    """
    if image.dim() != 3:
        raise ValueError(
            f"image must be [C, H, W]; got {tuple(image.shape)}. "
            "Pass a single image, not a batch."
        )
    C, H, W = image.shape
    if H % patch_size or W % patch_size:
        raise ValueError(
            f"Image size {H}x{W} not divisible by patch_size={patch_size}. "
            f"Either pad to a multiple, or use tgraphx.image_to_patches "
            "(which supports padding) directly."
        )
    from ..graph_builders import image_to_patches, build_grid_graph
    patches = image_to_patches(image.unsqueeze(0), patch_size=patch_size)[0]
    n_h, n_w = H // patch_size, W // patch_size
    edge_index = build_grid_graph(n_h, n_w, directed=False, self_loops=self_loops)
    return patches, edge_index
