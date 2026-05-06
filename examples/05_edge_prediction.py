"""05_edge_prediction.py

Edge prediction using:
  (a) EdgePredictor standalone — on pre-computed node embeddings
  (b) build_model("edge_prediction", ...) — GNN stack + MLP edge scorer

Graph: kNN graph on a 2-D random point cloud.
"""
import torch
from tgraphx import EdgePredictor, build_knn_graph, build_model


def main():
    print("=" * 56)
    print("  Edge prediction demo")
    print("=" * 56)

    N, D = 16, 32
    coords = torch.randn(N, 2)
    edge_index = build_knn_graph(coords, k=4, directed=False, self_loops=False)
    E = edge_index.shape[1]

    print(f"\nNodes={N}, in_dim={D}, edges={E}")

    # ------------------------------------------------------------------ #
    # (a) EdgePredictor on pre-computed embeddings                         #
    # ------------------------------------------------------------------ #
    print("\n(a) EdgePredictor  — standalone MLP edge scorer")
    node_emb = torch.randn(N, D)
    predictor = EdgePredictor(in_dim=D, hidden_dim=64, out_dim=1)
    scores = predictor(node_emb, edge_index)
    print(f"    Edge scores   : {tuple(scores.shape)}  (expected [{E}, 1])")
    probs = torch.sigmoid(scores.squeeze(-1))
    print(f"    Sigmoid probs : min={probs.min():.3f}  max={probs.max():.3f}")

    # ------------------------------------------------------------------ #
    # (b) build_model — GNN stack + EdgePredictor head                    #
    # ------------------------------------------------------------------ #
    print("\n(b) build_model('edge_prediction', 'linear', ...)")
    model = build_model(
        task="edge_prediction",
        layer="linear",
        in_shape=(D,),
        hidden_shape=(64,),
        num_layers=2,
        out_dim=1,
        aggr="mean",
        edge_predictor_hidden=64,
    )
    x = torch.randn(N, D)
    out = model(x, edge_index)
    print(f"    Model output  : {tuple(out.shape)}  (expected [{E}, 1])")

    out.sum().backward()
    print("    Backward      : OK")

    # Spatial features → same pattern (spatial pool happens inside EdgePredictor)
    print("\n(c) EdgePredictor  — spatial [N, C, H, W] node features")
    x_spatial = torch.randn(N, 4, 4, 4)
    predictor_sp = EdgePredictor(in_dim=4, hidden_dim=32, out_dim=1)
    out_sp = predictor_sp(x_spatial, edge_index)
    print(f"    Output        : {tuple(out_sp.shape)}  (expected [{E}, 1])")

    print("\nDone.")


if __name__ == "__main__":
    main()
