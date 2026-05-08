"""Positional and structural encodings for GraphTransformerLayer.

These are pure-PyTorch utilities — no extra dependencies.

Functions
---------
``degree_encoding(edge_index, num_nodes, dim, direction="both")``
    Per-node positional encoding from in/out degree.

``laplacian_eigvec_encoding(edge_index, num_nodes, dim)``
    Smallest non-trivial eigenvectors of the symmetric normalised
    Laplacian; the sign of each eigenvector is ambiguous, so a random
    sign-flip is applied at training time when used as a regulariser.

``build_adjacency_bias(edge_index, num_nodes, value=0.0, neg_inf=False)``
    Dense ``[N, N]`` bias tensor.  ``value=0`` and the rest set to a
    finite negative number (default ``-inf`` when ``neg_inf=True``)
    masks attention to actual neighbours; otherwise the bias is additive.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "degree_encoding",
    "laplacian_eigvec_encoding",
    "build_adjacency_bias",
]


def degree_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int = 8,
    direction: str = "both",
) -> torch.Tensor:
    """Per-node positional encoding from node degrees.

    Computes in-degree, out-degree, or both, then projects to ``dim``
    via a deterministic sinusoidal-style mapping.  The encoding does
    not depend on learned parameters.

    Args:
        edge_index: ``[2, E]`` LongTensor.
        num_nodes: Total node count.
        dim: Encoding dimension (>= 1).
        direction: ``"in"`` / ``"out"`` / ``"both"`` (concatenated when
            both; in that case the returned dim is ``dim*2``).

    Returns:
        ``[N, dim]`` (or ``[N, 2*dim]`` for ``"both"``) FloatTensor.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if direction not in ("in", "out", "both"):
        raise ValueError(f"direction must be 'in', 'out', or 'both'; got {direction!r}")

    device = edge_index.device
    in_deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    out_deg = torch.zeros(num_nodes, dtype=torch.float, device=device)
    if edge_index.numel() > 0:
        in_deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
        out_deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))

    def _sinusoidal(deg: torch.Tensor, d: int) -> torch.Tensor:
        # Use a sinusoidal embedding of log(1 + degree) so degrees that
        # vary by orders of magnitude give distinguishable encodings.
        log_deg = torch.log1p(deg).unsqueeze(-1)  # [N, 1]
        # Frequencies geometrically spaced.
        freqs = torch.pow(
            10000.0,
            -torch.arange(d, device=deg.device, dtype=torch.float) / max(d, 1),
        )  # [d]
        angles = log_deg * freqs.unsqueeze(0)  # [N, d]
        # Interleave sin/cos in even/odd dims.
        out = torch.zeros_like(angles)
        out[:, 0::2] = torch.sin(angles[:, 0::2])
        out[:, 1::2] = torch.cos(angles[:, 1::2])
        return out

    if direction == "in":
        return _sinusoidal(in_deg, dim)
    if direction == "out":
        return _sinusoidal(out_deg, dim)
    return torch.cat([_sinusoidal(in_deg, dim), _sinusoidal(out_deg, dim)], dim=-1)


def laplacian_eigvec_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int = 8,
    sign_flip: bool = False,
) -> torch.Tensor:
    """Smallest non-trivial eigenvectors of the symmetric normalised Laplacian.

    .. warning::
        Computes the dense Laplacian — **O(N²)** memory.  Use only for
        moderate N.

    Args:
        edge_index: ``[2, E]`` LongTensor.
        num_nodes: ``N``.
        dim: Number of eigenvectors to keep (>= 1).
        sign_flip: If ``True``, randomly flip the sign of each
            eigenvector — useful as a training-time data augmentation
            because eigenvector signs are ambiguous.

    Returns:
        ``[N, dim]`` FloatTensor.

    Notes:
        The first eigenvector (constant) is dropped because it carries
        no positional information.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if dim >= num_nodes:
        raise ValueError(
            f"dim={dim} must be < num_nodes={num_nodes} "
            f"(first eigenvector is dropped)"
        )

    device = edge_index.device
    # Build dense adjacency (undirected via symmetrisation).
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    if edge_index.numel() > 0:
        src, dst = edge_index[0], edge_index[1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # symmetrise
    deg = A.sum(dim=1)
    # Symmetric normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
    d_inv_sqrt = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
    D = torch.diag(d_inv_sqrt)
    L = torch.eye(num_nodes, device=device) - D @ A @ D
    # Eigendecomposition (symmetric → eigh).
    eigvals, eigvecs = torch.linalg.eigh(L)
    # Drop the first (smallest) eigenvalue/vector and take the next `dim`.
    encoding = eigvecs[:, 1 : dim + 1]  # [N, dim]
    if sign_flip:
        signs = torch.randint(0, 2, (dim,), device=device).float() * 2 - 1
        encoding = encoding * signs.unsqueeze(0)
    return encoding


def build_adjacency_bias(
    edge_index: torch.Tensor,
    num_nodes: int,
    value: float = 0.0,
    neg_inf: bool = False,
) -> torch.Tensor:
    """Dense ``[N, N]`` bias tensor for GraphTransformer attention.

    Use as ``edge_bias_dense`` to inject structural information.  When
    ``neg_inf=True``, off-diagonal entries default to a large negative
    number, effectively masking attention to non-neighbours (useful for
    GAT-equivalent global-attention graphs).

    Args:
        edge_index: ``[2, E]`` LongTensor.
        num_nodes: ``N``.
        value: Bias value at edge positions (default ``0`` — additive bias
            on the *log-attention* logits, i.e., does nothing extra; the
            non-edge background is what carries the signal).
        neg_inf: If ``True``, set the non-edge background to ``-1e4``
            (a finite stand-in for ``-inf`` that plays nicely with float16).

    Returns:
        ``[N, N]`` FloatTensor.
    """
    device = edge_index.device
    bg = -1e4 if neg_inf else 0.0
    bias = torch.full((num_nodes, num_nodes), bg, dtype=torch.float, device=device)
    if edge_index.numel() > 0:
        bias[edge_index[0], edge_index[1]] = value
    # Allow self-attention regardless.
    diag = torch.arange(num_nodes, device=device)
    bias[diag, diag] = 0.0
    return bias
