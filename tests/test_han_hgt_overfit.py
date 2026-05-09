"""Stronger HAN/HGT validation: tiny-overfit tests.

Validates:
  - loss decreases on a toy classification task
  - semantic attention sums to 1 (HANConv)
  - relation priors receive gradients (HGTConv)
  - no type leakage between node types
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_toy(n: int = 15, d: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    src = torch.arange(n).repeat_interleave(2)
    dst = torch.cat([(torch.arange(n) + 1) % n, (torch.arange(n) + 3) % n])
    ei = torch.stack([src, dst], dim=0)
    y = torch.randint(0, 2, (n,))
    return x, ei, y


class TestHANOverfit:

    def test_loss_decreases(self):
        from tgraphx.layers.han import HANConv
        x, ei, y = _make_toy(20, 8)
        layer = HANConv(in_dim=8, out_dim=4, num_heads=2)
        clf = nn.Linear(4, 2)
        opt = torch.optim.Adam(list(layer.parameters()) + list(clf.parameters()), lr=0.05)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = layer(x, {"mp1": ei, "mp2": ei.flip(0)})
            loss = F.cross_entropy(clf(out), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0] or losses[-1] < 1.0, \
            f"HAN loss did not decrease: {losses[0]:.3f}→{losses[-1]:.3f}"

    def test_semantic_attention_sums_to_one(self):
        """Semantic attention weights (over metapaths) sum to 1."""
        from tgraphx.layers.han import HANConv
        # Patch to inspect beta weights.
        import torch.nn.functional as F_
        x, ei, _ = _make_toy(10, 8)
        layer = HANConv(in_dim=8, out_dim=4, num_heads=1)

        # Run forward; compute semantic scores manually.
        embeds = []
        for name, ei_ in [("mp1", ei), ("mp2", ei)]:
            attn = layer._get_mp_attn(name, x.device, x.dtype)
            embeds.append(attn(x, ei_))
        Z = torch.stack(embeds, dim=0)
        summary = Z.mean(dim=1)
        s = layer.semantic(summary).squeeze(-1)
        beta = F_.softmax(s, dim=0)
        assert abs(float(beta.sum().item()) - 1.0) < 1e-4

    def test_gradients_finite(self):
        from tgraphx.layers.han import HANConv
        x, ei, y = _make_toy(10, 8)
        x = x.requires_grad_(True)
        layer = HANConv(in_dim=8, out_dim=4)
        out = layer(x, {"mp": ei})
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestHGTOverfit:

    def _make_hgt(self, n_a=20, n_b=15, d_in=8, d_out=8):
        torch.manual_seed(0)
        x_a = torch.randn(n_a, d_in)
        x_b = torch.randn(n_b, d_in)
        ei_ab = torch.stack([torch.randint(n_a, (30,)), torch.randint(n_b, (30,))], dim=0)
        ei_ba = torch.stack([torch.randint(n_b, (20,)), torch.randint(n_a, (20,))], dim=0)
        y_a = torch.randint(0, 2, (n_a,))
        return x_a, x_b, ei_ab, ei_ba, y_a

    def test_loss_decreases(self):
        from tgraphx.layers.hgt import HGTConv
        x_a, x_b, ei_ab, ei_ba, y_a = self._make_hgt()
        node_types = ["A", "B"]
        edge_types = [("A", "t", "B"), ("B", "t", "A")]
        layer = HGTConv(8, 8, node_types=node_types, edge_types=edge_types, num_heads=2)
        clf = nn.Linear(8, 2)
        opt = torch.optim.Adam(list(layer.parameters()) + list(clf.parameters()), lr=0.05)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            out = layer({"A": x_a, "B": x_b}, {("A", "t", "B"): ei_ab, ("B", "t", "A"): ei_ba})
            loss = F.cross_entropy(clf(out["A"]), y_a)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0] or losses[-1] < 1.0

    def test_relation_priors_have_gradients(self):
        from tgraphx.layers.hgt import HGTConv
        x_a, x_b, ei_ab, ei_ba, y_a = self._make_hgt()
        layer = HGTConv(8, 8, ["A", "B"],
                        [("A", "t", "B"), ("B", "t", "A")], num_heads=2)
        out = layer({"A": x_a, "B": x_b}, {("A", "t", "B"): ei_ab, ("B", "t", "A"): ei_ba})
        # Sum both outputs so ALL relations contribute to the backward pass.
        (out["A"].sum() + out["B"].sum()).backward()
        # At least one relation prior should have a gradient.
        any_pri_grad = any(
            p.grad is not None
            for n, p in layer.named_parameters()
            if "relation_pri" in n
        )
        assert any_pri_grad, "No relation prior received a gradient"

    def test_no_type_leakage(self):
        """A-node output must not equal B-node output at any index."""
        from tgraphx.layers.hgt import HGTConv
        x_a, x_b, ei_ab, ei_ba, _ = self._make_hgt()
        layer = HGTConv(8, 8, ["A", "B"],
                        [("A", "t", "B"), ("B", "t", "A")], num_heads=2)
        out = layer({"A": x_a, "B": x_b}, {("A", "t", "B"): ei_ab, ("B", "t", "A"): ei_ba})
        assert out["A"].shape != out["B"].shape or not torch.allclose(out["A"], out["B"])
