"""Neural graph pattern classifier demo.

Trains a GraphPatternClassifier to distinguish path / star / cycle /
complete graphs using both structural and feature signals.
"""
import torch
from tgraphx.mining import (
    GraphPatternClassifier,
    create_synthetic_pattern_dataset,
    train_graph_pattern_classifier_step,
)

print("=" * 60)
print("Neural Graph Pattern Classifier Demo")
print("=" * 60)

torch.manual_seed(0)

# Create dataset.
ds = create_synthetic_pattern_dataset(
    num_graphs_per_class=40, num_nodes=8, in_dim=4, seed=0, noise_std=0.05,
)
# Stratified 75/25 split: 30 train + 10 test per class.
per_class = {c: [g for g in ds if g["label"] == c] for c in range(4)}
train_ds = [g for c in range(4) for g in per_class[c][:30]]
test_ds  = [g for c in range(4) for g in per_class[c][30:]]
print(f"Dataset: {len(ds)} graphs ({len(train_ds)} train / {len(test_ds)} test)")
print(f"Pattern classes: {sorted(set(g['pattern'] for g in ds))}")

# Build and train classifier.
clf = GraphPatternClassifier(in_dim=4, hidden_dim=32, enc_dim=16, num_classes=4)
opt = torch.optim.Adam(clf.parameters(), lr=5e-3)

losses = []
for epoch in range(50):
    import random; random.shuffle(train_ds)
    epoch_loss = 0.0
    for g in train_ds:
        loss = train_graph_pattern_classifier_step(
            clf, opt, [g], torch.tensor([g["label"]])
        )
        epoch_loss += loss
    losses.append(epoch_loss / len(train_ds))
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}: loss={losses[-1]:.4f}")

# Evaluate.
clf.eval()
correct = 0
with torch.no_grad():
    for g in test_ds:
        pred = int(clf(g["node_features"], g["edge_index"], g["num_nodes"]).argmax().item())
        if pred == g["label"]:
            correct += 1
acc = correct / len(test_ds)
print(f"\nTest accuracy: {acc:.3f} ({correct}/{len(test_ds)})")
assert acc >= 0.70, f"Expected accuracy >= 0.70, got {acc:.3f}"
print("Sanity check: accuracy >= 70% OK")

# Per-class accuracy.
patterns = ["path", "star", "cycle", "complete"]
print("\nPer-class accuracy:")
for cls, name in enumerate(patterns):
    cls_items = [g for g in test_ds if g["label"] == cls]
    with torch.no_grad():
        cls_correct = sum(
            int(clf(g["node_features"], g["edge_index"], g["num_nodes"]).argmax().item()) == cls
            for g in cls_items
        )
    print(f"  {name:8s}: {cls_correct}/{len(cls_items)} = {cls_correct/len(cls_items):.3f}")

print("\nDemo complete.")
