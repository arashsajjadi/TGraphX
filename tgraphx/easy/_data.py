"""Synthetic data creation for TGraphX easy mode."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def synthetic_tensor_node_classification(
    num_nodes: int = 1000,
    node_shape: Tuple[int, ...] = (16, 8, 8),
    num_classes: int = 10,
    num_edges: Optional[int] = None,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Any:
    """Create a synthetic graph for tensor node classification.

    Node features have shape ``[num_nodes, *node_shape]`` (e.g. image-like).
    Labels are random integers in ``[0, num_classes)``.

    Returns:
        :class:`~tgraphx.Graph` with ``node_features``, ``edge_index``, and ``y``.
    """
    from tgraphx import Graph

    if num_edges is None:
        num_edges = num_nodes * 5

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    dev = torch.device(device)
    x = torch.randn(num_nodes, *node_shape, generator=gen)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    y = torch.randint(0, num_classes, (num_nodes,), generator=gen)

    return Graph(
        node_features=x.to(dev),
        edge_index=edge_index.to(dev),
        y=y.to(dev),
        metadata={
            "synthetic": True,
            "node_shape": list(node_shape),
            "num_classes": num_classes,
        },
    )


def synthetic_vector_node_classification(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_classes: int = 10,
    num_edges: Optional[int] = None,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Any:
    """Create a synthetic graph for vector-feature node classification."""
    return synthetic_tensor_node_classification(
        num_nodes=num_nodes,
        node_shape=(num_features,),
        num_classes=num_classes,
        num_edges=num_edges,
        seed=seed,
        device=device,
    )


def synthetic_link_prediction(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_edges: int = 5000,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Create a synthetic graph for link prediction.

    Returns a dict with keys ``graph``, ``train_edges``, ``val_edges``,
    ``test_edges`` (as LongTensor[2, E]).
    """
    from tgraphx import Graph

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    dev = torch.device(device)
    x = torch.randn(num_nodes, num_features, generator=gen)
    all_edges = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    split = [int(num_edges * 0.8), int(num_edges * 0.1)]
    split.append(num_edges - split[0] - split[1])
    parts = torch.split(all_edges, split, dim=1)
    graph = Graph(node_features=x.to(dev), edge_index=parts[0].to(dev))
    return {
        "graph": graph,
        "train_edges": parts[0].to(dev),
        "val_edges": parts[1].to(dev),
        "test_edges": parts[2].to(dev),
    }


def synthetic_graph_classification(
    num_graphs: int = 100,
    num_nodes_per_graph: int = 20,
    num_features: int = 16,
    num_classes: int = 4,
    num_edges_per_graph: int = 40,
    seed: Optional[int] = 42,
) -> List[Any]:
    """Create a list of synthetic graphs for graph classification."""
    from tgraphx import Graph

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    graphs = []
    for i in range(num_graphs):
        N = num_nodes_per_graph
        x = torch.randn(N, num_features, generator=gen)
        ei = torch.randint(0, N, (2, num_edges_per_graph), generator=gen)
        label = torch.tensor(i % num_classes, dtype=torch.long)
        graphs.append(Graph(node_features=x, edge_index=ei, graph_label=label))
    return graphs
