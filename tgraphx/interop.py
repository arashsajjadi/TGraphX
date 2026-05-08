"""Optional converters between TGraphX :class:`~tgraphx.Graph` objects and
PyTorch Geometric / DGL graph representations.

All heavy imports (``torch_geometric``, ``dgl``) are **lazy** — they are only
attempted when you call a converter function.  A clear :class:`ImportError`
is raised with installation instructions if the package is absent.

This module does NOT claim PyG/DGL API compatibility.  It provides
**data-format converters only**: node features, edge index, edge weights, and
edge features are preserved where possible; semantics and API differ.

Functions
---------
to_pyg_data(graph)        → ``torch_geometric.data.Data``
from_pyg_data(data)       → :class:`~tgraphx.Graph`
to_dgl_graph(graph)       → ``dgl.DGLGraph``
from_dgl_graph(g, ...)    → :class:`~tgraphx.Graph`
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .core.graph import Graph

__all__ = [
    "to_pyg_data",
    "from_pyg_data",
    "to_dgl_graph",
    "from_dgl_graph",
    "to_pyg_heterodata",
    "from_pyg_heterodata",
    "to_dgl_heterograph",
    "from_dgl_heterograph",
]


# ── PyTorch Geometric ─────────────────────────────────────────────────────────

def to_pyg_data(graph: "Graph") -> Any:
    """Convert a TGraphX :class:`~tgraphx.Graph` to a
    ``torch_geometric.data.Data`` object.

    Fields mapped:
    - ``node_features`` → ``data.x``
    - ``edge_index`` → ``data.edge_index``
    - ``edge_weight`` → ``data.edge_weight``
    - ``edge_features`` → ``data.edge_attr``
    - ``node_labels`` → ``data.y`` (if present)
    - ``graph_label`` → ``data.y`` (only if ``node_labels`` is absent)

    Args:
        graph: A TGraphX :class:`~tgraphx.Graph` instance.

    Returns:
        ``torch_geometric.data.Data``

    Raises:
        ImportError: If ``torch_geometric`` is not installed.
    """
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "to_pyg_data requires torch_geometric.\n"
            "Install it with: pip install torch-geometric\n"
            "(see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)"
        ) from exc

    kwargs: dict[str, Any] = {"x": graph.node_features}
    if graph.edge_index is not None:
        kwargs["edge_index"] = graph.edge_index
    if graph.has_edge_weight:
        kwargs["edge_weight"] = graph.edge_weight
    if graph.has_edge_features:
        kwargs["edge_attr"] = graph.edge_features
    if graph.node_labels is not None:
        kwargs["y"] = graph.node_labels
    elif graph.graph_label is not None:
        kwargs["y"] = graph.graph_label
    if graph.metadata is not None:
        kwargs["metadata"] = graph.metadata
    return Data(**kwargs)


def from_pyg_data(data: Any) -> "Graph":
    """Convert a ``torch_geometric.data.Data`` object to a TGraphX
    :class:`~tgraphx.Graph`.

    Fields mapped:
    - ``data.x`` → ``node_features`` (required)
    - ``data.edge_index`` → ``edge_index``
    - ``data.edge_weight`` → ``edge_weight``
    - ``data.edge_attr`` → ``edge_features``
    - ``data.y`` → ``node_labels`` (when ``data.y.size(0) == data.num_nodes``)
             or ``graph_label`` otherwise

    Args:
        data: ``torch_geometric.data.Data`` instance.

    Returns:
        :class:`~tgraphx.Graph`

    Raises:
        ImportError: If ``torch_geometric`` is not installed.
        ValueError: If ``data.x`` is absent.
    """
    try:
        import torch_geometric  # noqa: F401 — confirm import
    except ImportError as exc:
        raise ImportError(
            "from_pyg_data requires torch_geometric. "
            "Install it with: pip install torch-geometric"
        ) from exc
    from .core.graph import Graph

    if data.x is None:
        raise ValueError(
            "from_pyg_data requires data.x (node features) to be present."
        )
    node_features = data.x
    edge_index = getattr(data, "edge_index", None)
    edge_weight = getattr(data, "edge_weight", None)
    edge_features = getattr(data, "edge_attr", None)

    node_labels = None
    graph_label = None
    y = getattr(data, "y", None)
    if y is not None:
        if y.dim() >= 1 and y.size(0) == node_features.size(0):
            node_labels = y
        else:
            graph_label = y

    return Graph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_features=edge_features,
        node_labels=node_labels,
        graph_label=graph_label,
    )


# ── Deep Graph Library ────────────────────────────────────────────────────────

def to_dgl_graph(graph: "Graph") -> Any:
    """Convert a TGraphX :class:`~tgraphx.Graph` to a ``dgl.DGLGraph``.

    Fields mapped:
    - ``node_features`` → ``g.ndata['x']``
    - ``edge_weight`` → ``g.edata['w']``
    - ``edge_features`` → ``g.edata['e']``
    - ``node_labels`` → ``g.ndata['y']``

    Args:
        graph: A TGraphX :class:`~tgraphx.Graph` instance.

    Returns:
        ``dgl.DGLGraph``

    Raises:
        ImportError: If ``dgl`` is not installed.
        ValueError: If ``graph.edge_index`` is absent.
    """
    try:
        import dgl
    except ImportError as exc:
        raise ImportError(
            "to_dgl_graph requires dgl.\n"
            "Install it with: pip install dgl  "
            "(see https://www.dgl.ai/pages/start.html)"
        ) from exc

    if graph.edge_index is None:
        raise ValueError(
            "to_dgl_graph requires edge_index to be present on the Graph."
        )
    import torch
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=graph.num_nodes)
    g.ndata["x"] = graph.node_features
    if graph.has_edge_weight:
        g.edata["w"] = graph.edge_weight
    if graph.has_edge_features:
        g.edata["e"] = graph.edge_features
    if graph.node_labels is not None:
        g.ndata["y"] = graph.node_labels
    return g


def from_dgl_graph(
    g: Any,
    node_feature_key: str = "x",
    edge_weight_key: Optional[str] = "w",
    edge_feature_key: Optional[str] = "e",
    node_label_key: Optional[str] = "y",
) -> "Graph":
    """Convert a ``dgl.DGLGraph`` to a TGraphX :class:`~tgraphx.Graph`.

    Args:
        g: ``dgl.DGLGraph`` instance.
        node_feature_key: Key in ``g.ndata`` for node features.
        edge_weight_key: Key in ``g.edata`` for edge weights (1-D tensor).
        edge_feature_key: Key in ``g.edata`` for edge features.
        node_label_key: Key in ``g.ndata`` for node labels.

    Returns:
        :class:`~tgraphx.Graph`

    Raises:
        ImportError: If ``dgl`` is not installed.
        ValueError: If the specified ``node_feature_key`` is absent in ndata.
    """
    try:
        import dgl  # noqa: F401 — confirm import
    except ImportError as exc:
        raise ImportError(
            "from_dgl_graph requires dgl. Install it with: pip install dgl"
        ) from exc
    from .core.graph import Graph
    import torch

    if node_feature_key not in g.ndata:
        raise ValueError(
            f"from_dgl_graph: node_feature_key {node_feature_key!r} not found "
            f"in g.ndata.  Available keys: {list(g.ndata.keys())}"
        )
    node_features = g.ndata[node_feature_key]

    src, dst = g.edges()
    edge_index = torch.stack([src.long(), dst.long()], dim=0)

    edge_weight = None
    if edge_weight_key and edge_weight_key in g.edata:
        ew = g.edata[edge_weight_key]
        if ew.dim() == 1:
            edge_weight = ew

    edge_features = None
    if edge_feature_key and edge_feature_key in g.edata:
        edge_features = g.edata[edge_feature_key]

    node_labels = None
    if node_label_key and node_label_key in g.ndata:
        node_labels = g.ndata[node_label_key]

    return Graph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_features=edge_features,
        node_labels=node_labels,
    )


# ── Heterogeneous PyG converters ─────────────────────────────────────────────

def to_pyg_heterodata(hetero_graph: "HeteroGraph") -> Any:  # noqa: F821
    """Convert a TGraphX :class:`HeteroGraph` to ``torch_geometric.data.HeteroData``.

    Preserves: per-type node features, per-relation edge_index, edge_weight,
    edge_attr, per-type node labels, graph_label, metadata.

    Raises:
        ImportError: torch_geometric not installed.
    """
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "to_pyg_heterodata requires torch_geometric.\n"
            "Install: pip install torch-geometric"
        ) from exc

    data = HeteroData()
    for ntype in hetero_graph.node_types:
        data[ntype].x = hetero_graph.node_features(ntype)
        if hetero_graph.has_node_labels(ntype):
            data[ntype].y = hetero_graph.node_labels(ntype)
    for etype in hetero_graph.edge_types:
        data[etype].edge_index = hetero_graph.edge_index(etype)
        if hetero_graph.has_edge_weight(etype):
            data[etype].edge_weight = hetero_graph.edge_weight(etype)
        if hetero_graph.has_edge_features(etype):
            data[etype].edge_attr = hetero_graph.edge_features(etype)
    if hetero_graph.graph_label is not None:
        data.graph_label = hetero_graph.graph_label
    return data


def from_pyg_heterodata(data: Any) -> "HeteroGraph":  # noqa: F821
    """Convert a ``torch_geometric.data.HeteroData`` to TGraphX :class:`HeteroGraph`.

    Raises:
        ImportError: torch_geometric not installed.
        ValueError: a node type lacks ``x`` features.
    """
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_pyg_heterodata requires torch_geometric. "
            "Install: pip install torch-geometric"
        ) from exc
    from .core.hetero_graph import HeteroGraph

    node_stores: Dict[str, Any] = {}
    edge_stores: Dict[Any, Any] = {}
    edge_weight_stores: Dict[Any, Any] = {}
    edge_feature_stores: Dict[Any, Any] = {}
    node_label_stores: Dict[str, Any] = {}

    for ntype in data.node_types:
        store = data[ntype]
        if "x" not in store or store.x is None:
            raise ValueError(
                f"from_pyg_heterodata: node type {ntype!r} has no .x features."
            )
        node_stores[ntype] = store.x
        y = getattr(store, "y", None)
        if y is not None:
            node_label_stores[ntype] = y

    for etype in data.edge_types:
        store = data[etype]
        ei = getattr(store, "edge_index", None)
        if ei is None:
            continue
        edge_stores[tuple(etype)] = ei
        ew = getattr(store, "edge_weight", None)
        if ew is not None:
            edge_weight_stores[tuple(etype)] = ew
        ef = getattr(store, "edge_attr", None)
        if ef is not None:
            edge_feature_stores[tuple(etype)] = ef

    return HeteroGraph(
        node_stores=node_stores,
        edge_stores=edge_stores,
        edge_weight_stores=edge_weight_stores or None,
        edge_feature_stores=edge_feature_stores or None,
        node_label_stores=node_label_stores or None,
        graph_label=getattr(data, "graph_label", None),
    )


# ── Heterogeneous DGL converters ─────────────────────────────────────────────

def to_dgl_heterograph(hetero_graph: "HeteroGraph") -> Any:  # noqa: F821
    """Convert a TGraphX :class:`HeteroGraph` to ``dgl.DGLHeteroGraph``.

    Raises:
        ImportError: dgl not installed.
    """
    try:
        import dgl
    except ImportError as exc:
        raise ImportError(
            "to_dgl_heterograph requires dgl.\n"
            "Install: pip install dgl"
        ) from exc

    data_dict: Dict[Any, tuple] = {}
    num_nodes_dict: Dict[str, int] = {}
    for ntype in hetero_graph.node_types:
        num_nodes_dict[ntype] = hetero_graph.num_nodes(ntype)
    for etype in hetero_graph.edge_types:
        ei = hetero_graph.edge_index(etype)
        data_dict[etype] = (ei[0], ei[1])
    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    for ntype in hetero_graph.node_types:
        g.nodes[ntype].data["x"] = hetero_graph.node_features(ntype)
        if hetero_graph.has_node_labels(ntype):
            g.nodes[ntype].data["y"] = hetero_graph.node_labels(ntype)
    for etype in hetero_graph.edge_types:
        if hetero_graph.has_edge_weight(etype):
            g.edges[etype].data["w"] = hetero_graph.edge_weight(etype)
        if hetero_graph.has_edge_features(etype):
            g.edges[etype].data["e"] = hetero_graph.edge_features(etype)
    return g


def from_dgl_heterograph(
    g: Any,
    node_feature_key: str = "x",
    node_label_key: Optional[str] = "y",
    edge_weight_key: Optional[str] = "w",
    edge_feature_key: Optional[str] = "e",
) -> "HeteroGraph":  # noqa: F821
    """Convert a ``dgl.DGLHeteroGraph`` to TGraphX :class:`HeteroGraph`.

    Raises:
        ImportError: dgl not installed.
    """
    try:
        import dgl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_dgl_heterograph requires dgl. Install: pip install dgl"
        ) from exc
    import torch
    from .core.hetero_graph import HeteroGraph

    node_stores: Dict[str, Any] = {}
    edge_stores: Dict[Any, Any] = {}
    edge_weight_stores: Dict[Any, Any] = {}
    edge_feature_stores: Dict[Any, Any] = {}
    node_label_stores: Dict[str, Any] = {}

    for ntype in g.ntypes:
        nd = g.nodes[ntype].data
        if node_feature_key not in nd:
            raise ValueError(
                f"from_dgl_heterograph: node type {ntype!r} has no "
                f"{node_feature_key!r} in its ndata."
            )
        node_stores[ntype] = nd[node_feature_key]
        if node_label_key and node_label_key in nd:
            node_label_stores[ntype] = nd[node_label_key]

    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        edge_stores[tuple(etype)] = torch.stack([src.long(), dst.long()], dim=0)
        ed = g.edges[etype].data
        if edge_weight_key and edge_weight_key in ed:
            ew = ed[edge_weight_key]
            if ew.dim() == 1:
                edge_weight_stores[tuple(etype)] = ew
        if edge_feature_key and edge_feature_key in ed:
            edge_feature_stores[tuple(etype)] = ed[edge_feature_key]

    return HeteroGraph(
        node_stores=node_stores,
        edge_stores=edge_stores,
        edge_weight_stores=edge_weight_stores or None,
        edge_feature_stores=edge_feature_stores or None,
        node_label_stores=node_label_stores or None,
    )
