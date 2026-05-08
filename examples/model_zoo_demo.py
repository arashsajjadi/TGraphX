"""model_zoo_demo.py — vector model-zoo layers added in v0.3.0."""
from __future__ import annotations

import torch

from tgraphx import APPNP, GATv2Conv, GCNConv, global_mean_pool


def main() -> None:
    torch.manual_seed(0)
    N, D = 8, 4
    x = torch.randn(N, D)
    src = torch.arange(N)
    dst = (src + 1) % N
    edge_index = torch.stack([src, dst]).long()
    batch = torch.zeros(N, dtype=torch.long)

    print("GCNConv:")
    out = GCNConv(D, 16)(x, edge_index)
    print(f"  forward shape = {tuple(out.shape)}, finite = {torch.isfinite(out).all().item()}")

    print("\nGATv2Conv (4 heads):")
    out = GATv2Conv(D, 16, num_heads=4)(x, edge_index)
    print(f"  forward shape = {tuple(out.shape)}, finite = {torch.isfinite(out).all().item()}")

    print("\nAPPNP (K=4, alpha=0.1):")
    out = APPNP(K=4, alpha=0.1)(x, edge_index)
    print(f"  forward shape = {tuple(out.shape)}, finite = {torch.isfinite(out).all().item()}")

    print("\nglobal_mean_pool:")
    pooled = global_mean_pool(out, batch)
    print(f"  pooled shape = {tuple(pooled.shape)}")


if __name__ == "__main__":
    main()
