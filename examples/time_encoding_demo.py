"""Temporal time encoding demo — v0.3.2 beta/experimental.

Demonstrates:
- sinusoidal_time_encoding (deterministic, parameter-free).
- LearnableTimeEncoding (Time2Vec style, trainable).
"""
from __future__ import annotations

import torch
from tgraphx.temporal import LearnableTimeEncoding, sinusoidal_time_encoding

print("=" * 60)
print("Temporal Time Encoding Demo (TGraphX v0.3.2)")
print("=" * 60)

# ── Sinusoidal encoding ────────────────────────────────────────────────────────
t = torch.tensor([0.0, 1.0, 10.0, 100.0])
enc = sinusoidal_time_encoding(t, dim=8)
print(f"\nsinusoidal_time_encoding (t={t.tolist()}, dim=8):")
print(f"  output shape: {tuple(enc.shape)}")
print(f"  output dtype: {enc.dtype}")
print(f"  t=0 even cols (should be 0): {enc[0, 0::2].tolist()}")
print(f"  t=0 odd  cols (should be 1): {enc[0, 1::2].tolist()}")

# Norm property: ||enc||^2 == dim/2 for any t.
norms = enc.pow(2).sum(dim=-1)
print(f"  ||enc||^2 per t (should be {8//2} = 4.0): {norms.tolist()}")

# Batched shape.
t_batch = torch.zeros(3, 5)
enc_batch = sinusoidal_time_encoding(t_batch, dim=4)
print(f"\nBatched [3,5] input → output shape: {tuple(enc_batch.shape)}")

# ── Learnable encoding ────────────────────────────────────────────────────────
print("\nLearnableTimeEncoding (dim=8):")
enc_module = LearnableTimeEncoding(dim=8)
t2 = torch.tensor([0.0, 0.5, 1.0, 2.0])
out = enc_module(t2)
print(f"  output shape: {tuple(out.shape)}")
print(f"  output dtype: {out.dtype}")
print(f"  all finite: {torch.isfinite(out).all().item()}")

# Gradient.
loss = out.sum()
loss.backward()
print(f"  linear_w.grad: {enc_module.linear_w.grad:.6f}")
print(f"  periodic_w.grad[:4]: {enc_module.periodic_w.grad[:4].tolist()}")

# Tiny optimiser step.
opt = torch.optim.Adam(enc_module.parameters(), lr=1e-3)
for step in range(3):
    opt.zero_grad()
    out_s = enc_module(t2)
    loss_s = out_s.pow(2).mean()
    loss_s.backward()
    opt.step()
    print(f"  step {step}: loss={loss_s.detach().item():.6f}")

print("\nDemo complete.")
