"""Image-patch tensor graph: each node is an image patch [C, H, W].

This tutorial showcases TGraphX's tensor-native message passing on
image-patch graphs.  Each node carries a 4-D tensor (channels x patch_H x
patch_W), preserved through ConvMessagePassing layers.

A flatten-baseline is included to illustrate **why** preserving spatial
structure matters: flattening throws away the per-channel spatial layout,
which a 1x1 message-passing convolution cannot recover.

Synthetic image only — no network, no torchvision dependency.

Usage::

    python tutorials/image_patch_tensor_graph_demo.py
    python tutorials/image_patch_tensor_graph_demo.py --epochs 5
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _make_synthetic_image(seed: int):
    """Return a [C, H, W] synthetic image with mild spatial structure."""
    torch.manual_seed(seed)
    C, H, W = 3, 12, 12
    base = torch.randn(C, H, W) * 0.3
    # Add a horizontal gradient and a center bump so spatial layout matters.
    yy = torch.linspace(-1.0, 1.0, H).view(H, 1).expand(H, W)
    xx = torch.linspace(-1.0, 1.0, W).view(1, W).expand(H, W)
    bump = torch.exp(-(xx ** 2 + yy ** 2)) * 1.0
    base[0] += yy
    base[1] += xx
    base[2] += bump
    return base


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    from tgraphx import (
        Graph, ConvMessagePassing,
        image_to_patches, build_grid_graph, patch_grid_shape,
    )

    image = _make_synthetic_image(args.seed)
    C, H, W = image.shape
    patch_size, stride = 4, 4

    # ---- Patchify and build a 4-connected grid graph ------------------
    # image_to_patches expects [B, C, H, W]; squeeze the batch dim afterwards.
    patches_b = image_to_patches(image.unsqueeze(0), patch_size=patch_size, stride=stride)
    patches = patches_b.squeeze(0)
    N, Cp, ph, pw = patches.shape
    grid_h, grid_w = patch_grid_shape(H, W, patch_size, stride)
    edge_index = build_grid_graph(grid_h, grid_w, directed=False)

    # Per-patch label: 1 if mean intensity in channel 0 is positive (encodes
    # the gradient we baked into the image).
    y = (patches[:, 0].mean(dim=(-1, -2)) > 0).long()
    num_classes = int(y.max().item()) + 1

    g_tensor = Graph(node_features=patches, edge_index=edge_index, y=y).to(device)
    print(f"Tensor graph: {N} patches, each shape [{Cp}, {ph}, {pw}], "
          f"{g_tensor.num_edges} edges, {num_classes} classes")

    # ---- Tensor-native model ------------------------------------------
    class TensorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = ConvMessagePassing(
                in_shape=(Cp, ph, pw), out_shape=(8, ph, pw),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(8, num_classes)

        def forward(self, x, edge_index):
            z = self.conv(x, edge_index).relu()
            return self.head(self.pool(z).flatten(1))

    # ---- Flatten baseline (loses spatial layout) -----------------------
    flat = patches.flatten(1)  # [N, C*ph*pw]

    class FlatModel(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 32)
            self.fc2 = nn.Linear(32, num_classes)

        def forward(self, x, edge_index):
            # Vector graph aggregation: simple mean over neighbours per dim.
            src, dst = edge_index
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, x[src])
            counts = torch.zeros(x.size(0), device=x.device)
            counts.index_add_(0, dst, torch.ones_like(src, dtype=x.dtype))
            agg = agg / counts.clamp(min=1).unsqueeze(-1)
            return self.fc2(F.relu(self.fc1(agg)))

    tensor_model = TensorModel().to(device)
    flat_model = FlatModel(flat.size(1)).to(device)

    opt_t = torch.optim.Adam(tensor_model.parameters(), lr=1e-2)
    opt_f = torch.optim.Adam(flat_model.parameters(), lr=1e-2)

    flat_g = Graph(node_features=flat, edge_index=edge_index, y=y).to(device)

    print("\nTraining tensor-native model:")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        # Tensor model
        z_t = tensor_model(g_tensor.node_features, g_tensor.edge_index)
        loss_t = F.cross_entropy(z_t, g_tensor.node_labels)
        opt_t.zero_grad(); loss_t.backward(); opt_t.step()
        acc_t = (z_t.argmax(-1) == g_tensor.node_labels).float().mean().item()
        # Flat baseline
        z_f = flat_model(flat_g.node_features, flat_g.edge_index)
        loss_f = F.cross_entropy(z_f, flat_g.node_labels)
        opt_f.zero_grad(); loss_f.backward(); opt_f.step()
        acc_f = (z_f.argmax(-1) == flat_g.node_labels).float().mean().item()
        print(f"  epoch {epoch:>2d}/{args.epochs}  "
              f"tensor loss={loss_t.item():.4f} acc={acc_t:.3f}  |  "
              f"flat loss={loss_f.item():.4f} acc={acc_f:.3f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Tensor model params: {sum(p.numel() for p in tensor_model.parameters())}")
    print(f"Flat   model params: {sum(p.numel() for p in flat_model.parameters())}")
    print("\nNote: this synthetic demo cannot make a robust SOTA claim.  It "
          "shows that ConvMessagePassing operates directly on [C, H, W] "
          "node features and produces gradients — see "
          "tests/test_user_friendly_llm_snippets.py for shape contract tests.")
    print("\nTutorial PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
