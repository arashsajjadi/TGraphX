"""Tests for TGNMemory + TGATConv."""
from __future__ import annotations

import torch

from tgraphx.temporal import TGNMemory, TGATConv


def test_tgn_memory_init_zero():
    m = TGNMemory(num_nodes=10, memory_dim=4, message_dim=4)
    assert torch.equal(m.memory, torch.zeros(10, 4))
    assert torch.equal(m.last_update, torch.zeros(10))


def test_tgn_memory_update_and_get():
    m = TGNMemory(num_nodes=5, memory_dim=4, message_dim=4)
    nodes = torch.tensor([0, 2])
    msg = torch.randn(2, 4)
    t = torch.tensor([1.0, 1.5])
    m.update(nodes, msg, t)
    # Updated nodes have non-zero memory.
    assert m.memory[0].abs().sum().item() > 0
    assert m.memory[2].abs().sum().item() > 0
    # Other nodes untouched.
    assert m.memory[1].abs().sum().item() == 0
    # last_update reflects timestamps.
    assert m.last_update[0].item() == 1.0
    assert m.last_update[2].item() == 1.5
    # get() returns a clone.
    out = m.get(torch.tensor([0]))
    assert out.shape == (1, 4)
    out[0, 0] = 999.0
    assert m.memory[0, 0].item() != 999.0


def test_tgn_memory_monotonic_check():
    m = TGNMemory(num_nodes=3, memory_dim=2, message_dim=2)
    m.update(torch.tensor([0]), torch.randn(1, 2), torch.tensor([5.0]))
    try:
        m.update(torch.tensor([0]), torch.randn(1, 2), torch.tensor([2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on past timestamp")
    # Without the check, accept it.
    m.update(torch.tensor([0]), torch.randn(1, 2), torch.tensor([2.0]),
             check_monotonic=False)


def test_tgn_memory_reset():
    m = TGNMemory(num_nodes=5, memory_dim=4, message_dim=4)
    m.update(torch.tensor([0]), torch.randn(1, 4), torch.tensor([1.0]))
    assert m.memory[0].abs().sum().item() > 0
    m.reset_state()
    assert torch.equal(m.memory, torch.zeros(5, 4))
    assert torch.equal(m.last_update, torch.zeros(5))


def test_tgat_forward_backward():
    torch.manual_seed(0)
    N, D = 6, 8
    x = torch.randn(N, D, requires_grad=True)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    et = torch.tensor([1.0, 2.0, 3.0, 4.0])
    qt = torch.tensor([5.0] * N)
    layer = TGATConv(in_dim=D, out_dim=8, time_dim=4, num_heads=2)
    out = layer(x, ei, et, qt)
    assert out.shape == (N, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_tgat_no_edges():
    layer = TGATConv(in_dim=4, out_dim=4, time_dim=4)
    x = torch.randn(3, 4)
    ei = torch.zeros((2, 0), dtype=torch.long)
    et = torch.zeros(0)
    qt = torch.zeros(3)
    out = layer(x, ei, et, qt)
    assert out.shape == (3, 4)
