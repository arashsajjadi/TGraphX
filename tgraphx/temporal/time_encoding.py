"""Time-encoding helpers for temporal graph workflows.

Two encoders are provided:

- :func:`sinusoidal_time_encoding`: deterministic, parameter-free
  encoding inspired by Vaswani et al. (2017) Transformer positional
  encoding.  Useful as a feature for snapshot-loop and TGAT-style
  workflows.

- :class:`LearnableTimeEncoding`: Time2Vec-style trainable encoder
  (Kazemi et al., 2019), where one component is linear and the rest
  are sinusoidal with learned frequencies/phases.  Marked experimental
  until v0.3.4 evaluates it on real temporal benchmarks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = [
    "sinusoidal_time_encoding",
    "LearnableTimeEncoding",
]


def sinusoidal_time_encoding(
    timestamps: torch.Tensor,
    dim: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Deterministic Transformer-style sinusoidal time encoding.

    For an even ``dim`` returns a tensor in which each pair of columns
    encodes a different frequency:

        even columns 2k:   sin(t / base^(2k / dim))
        odd columns 2k+1:  cos(t / base^(2k / dim))

    Args:
        timestamps: ``Tensor[*]`` of timestamps (any shape, any
            floating-point or integer dtype).  Cast to ``float32``
            internally if integer.
        dim: Encoding dimension; must be even and positive.
        base: Frequency base; default ``10_000`` matches the
            Transformer paper.

    Returns:
        ``Tensor[*, dim]`` (the input shape with a final ``dim``
        dimension appended), dtype ``float32``.

    Stability: Beta.  Output values are deterministic for any given
    input and never depend on global RNG.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"dim must be a positive even integer; got {dim}")
    if base <= 0:
        raise ValueError(f"base must be positive; got {base}")
    t = timestamps.to(torch.float32)
    half = dim // 2
    # Frequency vector: shape [half]
    indices = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = torch.exp(
        -math.log(base) * indices * (2.0 / dim)
    )
    # Outer product: shape [..., half]
    angles = t.unsqueeze(-1) * freqs
    out = torch.empty(*angles.shape[:-1], dim, dtype=torch.float32, device=t.device)
    out[..., 0::2] = torch.sin(angles)
    out[..., 1::2] = torch.cos(angles)
    return out


class LearnableTimeEncoding(nn.Module):
    """Time2Vec-style trainable time encoder (Kazemi et al., 2019).

    Output channel 0 is a linear projection of the timestamp; the
    remaining ``dim - 1`` channels are sinusoidal with learned
    frequencies and phases.  Useful as a building block for TGAT-style
    layers planned for v0.3.4.

    Args:
        dim: Output dimension; must be ≥ 2.
        init_scale: Initial scale of the random frequency parameters
            (smaller values keep the encoding closer to identity at
            initialisation).

    Forward:
        ``timestamps`` of shape ``[*]`` → ``[*, dim]`` (float32).

    Stability: Experimental.  API may change in v0.3.4 once integration
    with a real TGAT-style layer is finalised.  Forward semantics
    (Time2Vec, with channel 0 as the linear component) are stable.
    """

    def __init__(self, dim: int, init_scale: float = 0.01) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError(f"dim must be at least 2; got {dim}")
        self.dim = int(dim)
        # Linear part: w0 * t + b0  (one channel).
        self.linear_w = nn.Parameter(torch.tensor(init_scale))
        self.linear_b = nn.Parameter(torch.zeros(()))
        # Periodic part: sin(w_k * t + b_k) for k = 1 .. dim - 1.
        self.periodic_w = nn.Parameter(torch.randn(dim - 1) * init_scale)
        self.periodic_b = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        t = timestamps.to(torch.float32)
        # Linear channel: [*]
        lin = self.linear_w * t + self.linear_b
        # Periodic channels: [*, dim-1]
        per = torch.sin(t.unsqueeze(-1) * self.periodic_w + self.periodic_b)
        return torch.cat([lin.unsqueeze(-1), per], dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
