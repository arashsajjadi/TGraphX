"""Tests for HAN/HGT layers."""
from __future__ import annotations

import torch

from tgraphx.layers.han import HANConv
from tgraphx.layers.hgt import HGTConv


def test_han_forward_backward():
    torch.manual_seed(0)
    N, D = 10, 8
    x = torch.randn(N, D, requires_grad=True)
    ei1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ei2 = torch.tensor([[5, 6, 7, 8], [6, 7, 8, 9]], dtype=torch.long)
    layer = HANConv(in_dim=D, out_dim=4, num_heads=2)
    out = layer(x, {"mp1": ei1, "mp2": ei2})
    assert out.shape == (N, 4)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_han_single_metapath_no_edges():
    layer = HANConv(in_dim=4, out_dim=4)
    x = torch.randn(5, 4)
    ei = torch.zeros((2, 0), dtype=torch.long)
    out = layer(x, {"mp": ei})
    assert out.shape == (5, 4)


def test_hgt_forward_backward():
    torch.manual_seed(0)
    node_types = ["A", "B"]
    edge_types = [("A", "to", "B"), ("B", "to", "A")]
    x_dict = {
        "A": torch.randn(4, 6, requires_grad=True),
        "B": torch.randn(3, 6, requires_grad=True),
    }
    edge_index_dict = {
        ("A", "to", "B"): torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        ("B", "to", "A"): torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    }
    layer = HGTConv(in_dim=6, out_dim=8, node_types=node_types,
                    edge_types=edge_types, num_heads=2)
    out = layer(x_dict, edge_index_dict)
    assert set(out.keys()) == {"A", "B"}
    assert out["A"].shape == (4, 8)
    assert out["B"].shape == (3, 8)
    (out["A"].sum() + out["B"].sum()).backward()
    assert torch.isfinite(x_dict["A"].grad).all()
    assert torch.isfinite(x_dict["B"].grad).all()


def test_hgt_invalid_dim_heads():
    try:
        HGTConv(in_dim=4, out_dim=5, node_types=["A"],
                edge_types=[("A", "to", "A")], num_heads=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-divisible dims")
