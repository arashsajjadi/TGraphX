"""training_with_tensorboard.py — fit() + TensorBoardLogger.

Skips gracefully if TensorBoard is not installed.

Install TensorBoard:
    pip install tensorboard
    # or
    pip install "tgraphx[tracking]"

Then run:
    python examples/training_with_tensorboard.py

Inspect results:
    tensorboard --logdir runs/tb_example
"""
import tempfile

import torch
import torch.nn.functional as F

from tgraphx import Graph, build_grid_graph, build_model
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset
from tgraphx.training import fit, set_seed


def make_dataset(n: int = 20, in_dim: int = 8, num_classes: int = 3):
    torch.manual_seed(2)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    graphs = []
    for _ in range(n):
        nf = torch.randn(9, in_dim)
        gl = torch.randint(0, num_classes, (1,))
        graphs.append(Graph(nf, ei, graph_label=gl))
    return graphs


def main() -> None:
    # Lazy import: TensorBoardLogger raises ImportError if tensorboard absent
    try:
        from tgraphx.tracking import TensorBoardLogger
    except ImportError:
        pass  # module always importable; error happens on instantiation

    set_seed(3)
    graphs = make_dataset(n=24)
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

    logdir = tempfile.mkdtemp(prefix="tgraphx_tb_")
    print(f"TensorBoard event dir: {logdir}")

    try:
        from tgraphx.tracking import TensorBoardLogger
        logger = TensorBoardLogger(logdir)
    except ImportError as exc:
        print(f"\n[SKIP] {exc}")
        print("Training without TensorBoard logger (no files written).")
        history = fit(
            model, train_loader, val_loader=val_loader, epochs=5,
            optimizer=optimizer, loss_fn=F.cross_entropy,
            device="cpu", log_level=1,
        )
        print("Done (TensorBoard not available).")
        return

    with logger:
        history = fit(
            model,
            train_loader,
            val_loader=val_loader,
            epochs=5,
            optimizer=optimizer,
            loss_fn=F.cross_entropy,
            device="cpu",
            logger=logger,
            log_level=1,
        )

    print(f"\nFinal — train_loss: {history[-1]['train_loss']:.4f}  "
          f"val_loss: {history[-1]['val_loss']:.4f}")
    print(f"\nView results:  tensorboard --logdir {logdir}")


if __name__ == "__main__":
    main()
