"""checkpoint_save_load.py — model checkpoint round-trip demonstration.

Uses a small factory-built model with synthetic data.
No datasets, no internet, CPU-only, fast.
"""
import os
import tempfile

import torch

from tgraphx import build_grid_graph, build_model
from tgraphx.training import (
    accuracy,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def main() -> None:
    set_seed(42)

    # ── Build a small graph-classification model ─────────────────────────────
    model = build_model(
        task="graph_classification",
        layer="linear",
        in_shape=(16,),
        hidden_shape=(32,),
        num_layers=2,
        num_classes=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Model parameters : {count_parameters(model):,}")

    # ── Synthetic graph (3×3 grid, 9 nodes) ──────────────────────────────────
    N = 9
    x = torch.randn(N, 16)
    edge_index = build_grid_graph(3, 3, directed=False, self_loops=True)
    batch = torch.zeros(N, dtype=torch.long)
    labels = torch.randint(0, 3, (1,))   # one graph, one label

    # ── One training step ────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index, batch=batch)      # [1, 3]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Loss after 1 step: {loss.item():.4f}")
    acc = accuracy(logits.detach(), labels)
    print(f"Accuracy          : {acc:.2f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "checkpoints", "epoch_01.pt")
        save_checkpoint(
            model, optimizer, epoch=1, path=ckpt,
            loss=loss.item(),
            model_config={"task": "graph_classification", "layer": "linear"},
        )
        size_kb = os.path.getsize(ckpt) / 1024
        print(f"Checkpoint saved  : {ckpt}  ({size_kb:.1f} KB)")

        # ── Load into a fresh model ───────────────────────────────────────────
        model2 = build_model(
            task="graph_classification",
            layer="linear",
            in_shape=(16,),
            hidden_shape=(32,),
            num_layers=2,
            num_classes=3,
        )
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        restored_epoch = load_checkpoint(model2, optimizer2, path=ckpt, map_location="cpu")
        print(f"Loaded from epoch : {restored_epoch}")

        # ── Verify outputs are identical ──────────────────────────────────────
        model.eval();  model2.eval()
        with torch.no_grad():
            out1 = model(x, edge_index, batch=batch)
            out2 = model2(x, edge_index, batch=batch)
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model outputs differ!"
        print("Output match      : OK")

    print("\nDone.")


if __name__ == "__main__":
    main()
