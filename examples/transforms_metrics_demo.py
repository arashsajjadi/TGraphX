"""transforms_metrics_demo.py — composing transforms + computing metrics."""
from __future__ import annotations

import torch

from tgraphx.datasets import SyntheticNodeClassificationDataset
from tgraphx.metrics import accuracy, classification_report, mae, regression_report
from tgraphx.transforms import (
    AddDegreeFeatures,
    AddSelfLoops,
    Compose,
    NormalizeFeatures,
    RandomNodeSplit,
)


def main() -> None:
    transform = Compose([
        NormalizeFeatures(),
        AddSelfLoops(),
        AddDegreeFeatures(direction="both", normalize=True),
        RandomNodeSplit(0.6, 0.2, seed=0),
    ])

    ds = SyntheticNodeClassificationDataset(
        num_nodes=80, num_classes=4, feature_dim=8, seed=0,
        transform=transform,
    )
    g = ds[0]
    print(f"After transform: feature dim = {g.node_features.size(1)} "
          f"(was 8, +2 degree features = 10)")
    print(f"Self-loops added: "
          f"{int((g.edge_index[0] == g.edge_index[1]).sum())}")

    masks = g.metadata["masks"]
    # Pretend we have a (silly) classifier — pick the most common class.
    common_class = int(g.node_labels[masks["train_mask"]].mode().values)
    preds = torch.full((g.num_nodes,), common_class, dtype=torch.long)
    print(f"\nDumb-classifier accuracy on test: "
          f"{accuracy(preds[masks['test_mask']], g.node_labels[masks['test_mask']]):.3f}")

    report = classification_report(
        preds[masks["test_mask"]], g.node_labels[masks["test_mask"]],
        num_classes=ds.metadata.num_classes,
    )
    print(f"  macro F1: {report['f1']:.3f}")
    print(f"  confusion:\n  {report['confusion_matrix']}")

    # Quick regression demo.
    pred_reg = torch.tensor([1.0, 2.0, 3.0])
    target_reg = torch.tensor([1.5, 2.5, 2.5])
    print(f"\nRegression report (toy):")
    print(f"  {regression_report(pred_reg, target_reg)}")


if __name__ == "__main__":
    main()
