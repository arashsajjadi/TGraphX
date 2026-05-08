"""pyg_dataset_adapter_demo.py — convert a tiny PyG ``Data`` to a TGraphX Graph.

Skips cleanly if torch_geometric is not installed.  No network — we
build a small ``Data`` object in-memory rather than downloading
Planetoid/Cora.
"""
from __future__ import annotations

import sys

try:
    from torch_geometric.data import Data  # type: ignore[import]
except ImportError as exc:
    print(f"Skipping demo: {exc.__class__.__name__}: {exc}")
    sys.exit(0)

import torch

from tgraphx.datasets import from_pyg_data


def main() -> None:
    data = Data(
        x=torch.randn(8, 4),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long,
        ),
        edge_attr=torch.randn(7, 2),
        y=torch.randint(0, 3, (8,)),
    )
    graph = from_pyg_data(data)
    print(f"Converted PyG Data → TGraphX Graph")
    print(f"  num_nodes={graph.num_nodes}  num_edges={graph.num_edges}")
    print(f"  node_labels shape={tuple(graph.node_labels.shape)}")
    print(f"  edge_features shape={tuple(graph.edge_features.shape)}")


if __name__ == "__main__":
    main()
