"""training_with_csvlogger.py — fit() + CSVLogger integration.

Writes metrics.csv to a temporary directory.  The file path is printed at
the end so you can inspect it or point the dashboard at the directory.
"""
import os
import tempfile

import torch
import torch.nn.functional as F

from tgraphx import Graph, build_grid_graph, build_model
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset
from tgraphx.tracking import CSVLogger
from tgraphx.training import accuracy, fit, set_seed


def make_dataset(n: int = 20, in_dim: int = 8, num_classes: int = 3):
    torch.manual_seed(1)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    graphs = []
    for _ in range(n):
        nf = torch.randn(9, in_dim)
        gl = torch.randint(0, num_classes, (1,))
        graphs.append(Graph(nf, ei, graph_label=gl))
    return graphs


def main() -> None:
    set_seed(7)
    graphs = make_dataset(n=24, in_dim=8, num_classes=3)
    train_loader = GraphDataLoader(GraphDataset(graphs[:18]), batch_size=6)
    val_loader   = GraphDataLoader(GraphDataset(graphs[18:]), batch_size=6)

    model = build_model(
        task="graph_classification",
        layer="linear",
        in_shape=(8,),
        hidden_shape=(32,),
        num_layers=2,
        num_classes=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    logdir = tempfile.mkdtemp(prefix="tgraphx_csvlogger_")
    print(f"Writing metrics to: {logdir}/metrics.csv")

    with CSVLogger(logdir) as logger:
        history = fit(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=5,
            optimizer=optimizer,
            loss_fn=F.cross_entropy,
            metrics={"accuracy": accuracy},
            device="cpu",
            logger=logger,
            log_level=1,
        )

    # Show what was written
    csv_path = os.path.join(logdir, "metrics.csv")
    print(f"\nCSV written: {csv_path}")
    with open(csv_path) as f:
        lines = f.readlines()
    print(f"Rows: {len(lines) - 1} (+ 1 header)")
    print("Header:", lines[0].strip())
    print("Last row:", lines[-1].strip())

    print(f"\nTo open the dashboard:")
    print(f"  tgraphx-dashboard --logdir {logdir}")


if __name__ == "__main__":
    main()
