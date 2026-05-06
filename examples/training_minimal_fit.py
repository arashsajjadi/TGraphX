"""training_minimal_fit.py — fit() with no logging, no files written.

Demonstrates the simplest usage of fit() on a synthetic graph-classification
task.  Nothing is written to disk unless you add a logger explicitly.
"""
import torch
import torch.nn.functional as F

from tgraphx import Graph, build_grid_graph, build_model
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset
from tgraphx.training import fit, set_seed


def make_dataset(n_graphs: int = 20, nodes: int = 9, in_dim: int = 8, num_classes: int = 3):
    torch.manual_seed(0)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    graphs = []
    for _ in range(n_graphs):
        nf = torch.randn(nodes, in_dim)
        gl = torch.randint(0, num_classes, (1,))
        graphs.append(Graph(nf, ei, graph_label=gl))
    return GraphDataset(graphs)


def main() -> None:
    set_seed(42)

    dataset = make_dataset(n_graphs=24, in_dim=8, num_classes=3)
    train_ds = dataset.graphs[:18]
    val_ds   = dataset.graphs[18:]

    train_loader = GraphDataLoader(GraphDataset(train_ds), batch_size=6, shuffle=True)
    val_loader   = GraphDataLoader(GraphDataset(val_ds),   batch_size=6, shuffle=False)

    model = build_model(
        task="graph_classification",
        layer="linear",
        in_shape=(8,),
        hidden_shape=(32,),
        num_layers=2,
        num_classes=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training with fit()  (logger=None, no files written)")
    history = fit(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=5,
        optimizer=optimizer,
        loss_fn=F.cross_entropy,
        device="cpu",
        log_level=1,    # print per-epoch summary
    )

    print(f"\nFinal epoch — train_loss: {history[-1]['train_loss']:.4f}  "
          f"val_loss: {history[-1]['val_loss']:.4f}")
    print(f"History entries: {len(history)}")
    print("\nDone.  (no files were written)")


if __name__ == "__main__":
    main()
