"""dgl_dataset_adapter_demo.py — convert a tiny DGL graph to a TGraphX Graph.

Skips cleanly when DGL is not installed (DGL wheels are
platform-sensitive).  No network — we build a small in-memory DGL
graph rather than downloading citation datasets.
"""
from __future__ import annotations

import sys

try:
    import dgl  # type: ignore[import]
except ImportError as exc:
    print(f"Skipping demo: {exc.__class__.__name__}: {exc}")
    sys.exit(0)

import torch

from tgraphx.datasets import from_dgl_graph


def main() -> None:
    g = dgl.graph(
        (torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])),
        num_nodes=4,
    )
    g.ndata["feat"] = torch.randn(4, 3)
    graph = from_dgl_graph(g)
    print(f"Converted DGL graph → TGraphX Graph")
    print(f"  num_nodes={graph.num_nodes}  num_edges={graph.num_edges}")
    print(f"  node_features shape={tuple(graph.node_features.shape)}")


if __name__ == "__main__":
    main()
