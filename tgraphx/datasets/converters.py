"""Conversion utilities used by external dataset adapters.

Re-exports the homogeneous + hetero converters from
:mod:`tgraphx.interop` and adds a few helpers that are dataset-specific
(e.g. ``ogb_item_to_graph``, ``torchvision_image_to_patch_graph``).

Heavy upstream packages (PyG / DGL / OGB / torchvision) are imported
**lazily** inside the helpers — never at module import time.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ..core.graph import Graph
from ..graph_builders import (
    build_grid_graph,
    build_knn_graph,
    build_radius_graph,
    image_to_patches,
)
# Re-export the converter functions already implemented in tgraphx/interop.py.
from ..interop import (  # noqa: F401
    from_dgl_graph,
    from_dgl_heterograph,
    from_pyg_data,
    from_pyg_heterodata,
    to_dgl_graph,
    to_dgl_heterograph,
    to_pyg_data,
    to_pyg_heterodata,
)

__all__ = [
    "from_pyg_data",
    "to_pyg_data",
    "from_pyg_heterodata",
    "to_pyg_heterodata",
    "from_dgl_graph",
    "to_dgl_graph",
    "from_dgl_heterograph",
    "to_dgl_heterograph",
    "ogb_item_to_graph",
    "torchvision_image_to_patch_graph",
]


# ── OGB ──────────────────────────────────────────────────────────────────────


def ogb_item_to_graph(item: Any, task_type: Optional[str] = None) -> Graph:
    """Convert a single OGB sample to a TGraphX :class:`Graph`.

    OGB datasets typically expose ``ds[i]`` as a PyG ``Data`` object
    (PygNodePropPredDataset / PygGraphPropPredDataset) so this helper
    delegates to :func:`from_pyg_data`.  When the item is a tuple
    ``(graph, label)``, the label is assigned to ``graph_label`` if it
    is graph-level, ``node_labels`` otherwise.

    Args:
        item: An OGB dataset item (PyG ``Data`` or ``(data, label)``).
        task_type: Optional hint (``"node"``, ``"graph"``, ``"link"``);
            used to disambiguate label placement when both
            interpretations are possible.

    Returns:
        :class:`Graph` with labels in the appropriate field.
    """
    if isinstance(item, tuple):
        data, label = item
        graph = from_pyg_data(data)
        if task_type == "graph" or graph.num_nodes != int(getattr(label, "numel", lambda: 0)()):
            graph.graph_label = label if isinstance(label, torch.Tensor) else torch.tensor(label)
        else:
            graph.node_labels = label if isinstance(label, torch.Tensor) else torch.tensor(label)
        return graph
    # Plain PyG Data — delegate.
    return from_pyg_data(item)


# ── Torchvision ──────────────────────────────────────────────────────────────


def torchvision_image_to_patch_graph(
    image: torch.Tensor,
    target: Optional[Any] = None,
    patch_size: int = 7,
    graph_builder: str = "grid",
    knn_k: int = 4,
    radius: float = 1.5,
    padding: str = "auto",
) -> Graph:
    """Turn a torchvision image (``[C, H, W]``) into a patch :class:`Graph`.

    Args:
        image: ``[C, H, W]`` float tensor (already normalised if you
            want it normalised).
        target: Optional class label.  Stored as ``graph_label``.
        patch_size, graph_builder, knn_k, radius, padding: Forwarded to
            the appropriate patch-graph builder.

    Returns:
        :class:`Graph` of shape ``[P, C, ph, pw]``.
    """
    if image.dim() != 3:
        raise ValueError(
            f"image must be [C, H, W]; got {tuple(image.shape)}"
        )
    C, H, W = image.shape
    if padding == "auto":
        pad_h = (-H) % patch_size
        pad_w = (-W) % patch_size
        if pad_h or pad_w:
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), value=0.0)
    elif padding == "none":
        if H % patch_size or W % patch_size:
            raise ValueError(
                f"image size {H}x{W} not divisible by patch_size={patch_size}; "
                f"pass padding='auto'"
            )
    else:
        raise ValueError(f"padding must be 'none' or 'auto'; got {padding!r}")

    _, H2, W2 = image.shape
    patches = image_to_patches(image.unsqueeze(0), patch_size=patch_size)[0]
    n_h, n_w = H2 // patch_size, W2 // patch_size
    if graph_builder == "grid":
        ei = build_grid_graph(n_h, n_w, directed=False, self_loops=True)
    elif graph_builder in ("knn", "radius"):
        coords = torch.stack(torch.meshgrid(
            torch.arange(n_h, dtype=torch.float),
            torch.arange(n_w, dtype=torch.float),
            indexing="ij",
        ), dim=-1).view(-1, 2)
        if graph_builder == "knn":
            ei = build_knn_graph(coords, k=knn_k, directed=False, self_loops=True)
        else:
            ei = build_radius_graph(coords, radius=radius, directed=False, self_loops=True)
    else:
        raise ValueError(f"Unknown graph_builder {graph_builder!r}")

    label_tensor = None
    if target is not None:
        if isinstance(target, torch.Tensor):
            label_tensor = target
        else:
            label_tensor = torch.tensor(target, dtype=torch.long)

    return Graph(
        node_features=patches,
        edge_index=ei,
        graph_label=label_tensor,
        metadata={
            "patch_size": patch_size,
            "grid_shape": (n_h, n_w),
            "graph_builder": graph_builder,
            "padding": padding,
        },
    )
