"""explainability_attention_demo.py — TensorGATLayer attention → per-edge scores."""
from __future__ import annotations

import torch

from tgraphx import TensorGATLayer
from tgraphx.explain import attention_to_edge_scores


def main() -> None:
    torch.manual_seed(0)
    N, C, H, W = 6, 4, 4, 4
    x = torch.randn(N, C, H, W)
    src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 4])
    dst = torch.tensor([1, 2, 3, 4, 5, 0, 3, 5, 1])
    edge_index = torch.stack([src, dst]).long()

    layer = TensorGATLayer(in_channels=C, out_channels=8, num_heads=2,
                           add_self_loops=False).eval()
    with torch.no_grad():
        out, attn = layer(x, edge_index, return_attention=True)

    scores = attention_to_edge_scores(attn, edge_index, head_reduce="mean")
    top = scores.argsort(descending=True)[:3]
    print(f"output         : {tuple(out.shape)}")
    print(f"attention      : {tuple(attn.shape)}")
    print(f"per-edge score : sum (per-dest, per-head) = "
          f"{torch.zeros(N, 2).index_add_(0, edge_index[1], attn).mean().item():.4f}")
    print(f"top-3 edges    : {edge_index[:, top].tolist()}  scores={scores[top].tolist()}")


if __name__ == "__main__":
    main()
