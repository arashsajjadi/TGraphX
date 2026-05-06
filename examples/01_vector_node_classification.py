"""01_vector_node_classification.py

Vector-feature node classification with a factory-built model.

Graph:  kNN graph on synthetic 2-D point cloud (N=20, k=4)
Model:  build_model("node_classification", "linear", ...)
Output: [N, num_classes] class logits
"""
import torch
from tgraphx import build_knn_graph, build_model


def main():
    N, D, num_classes = 20, 32, 4
    coords = torch.randn(N, 2)
    x = torch.randn(N, D)
    edge_index = build_knn_graph(coords, k=4, directed=False, self_loops=True)
    print(f"Nodes         : {N}  |  in_dim={D}")
    print(f"Edge index    : {tuple(edge_index.shape)}")

    model = build_model(
        task="node_classification",
        layer="linear",
        in_shape=(D,),
        hidden_shape=(64,),
        num_layers=3,
        num_classes=num_classes,
        aggr="mean",
    )
    print(f"Model         : {model.__class__.__name__}")

    out = model(x, edge_index)
    print(f"Output logits : {tuple(out.shape)}  (expected [{N}, {num_classes}])")

    out.sum().backward()
    print("Backward      : OK")
    print("\nDone.")


if __name__ == "__main__":
    main()
