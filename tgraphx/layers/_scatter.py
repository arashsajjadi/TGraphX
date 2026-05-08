"""Internal scatter and softmax helpers for tensor-aware GNN layers.

These functions reduce per-edge tensors by destination-node index using
pure PyTorch operations.  They are kept in a single private module so that
``TensorGATLayer``, ``TensorGraphSAGELayer``, and ``TensorGINLayer`` share
exactly one well-tested implementation.

No external dependencies (e.g. ``torch_scatter``) are introduced.
``Tensor.scatter_reduce_`` with ``reduce='amax'`` is available in
PyTorch >= 1.13 and is the only scatter primitive used here besides
``Tensor.index_add_`` and ``Tensor.scatter_add_``.

AMP / dtype notes
-----------------
* ``broadcast_edge_weight`` casts ``weight`` to match ``like``'s dtype so
  that caller-provided float32 edge weights are safe under float16/bfloat16
  autocast.
* ``edge_softmax`` upcasts to float32 when the input is float16 or bfloat16.
  The max-shift + exp + sum computation is numerically sensitive; computing
  it in float32 and casting back avoids overflow/underflow in attention
  weights.  This is the standard approach used by most production GNN
  libraries.
"""

from __future__ import annotations

import torch

# dtypes that benefit from fp32 upcast in softmax
_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _expand_index(target: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a 1-D index tensor so it can scatter along dim 0 of ``like``."""
    if like.dim() == 1:
        return target
    view = (target.size(0),) + (1,) * (like.dim() - 1)
    return target.view(view).expand_as(like)


def broadcast_edge_weight(
    weight: torch.Tensor,
    like: torch.Tensor,
    num_edges: int,
) -> torch.Tensor:
    """Validate ``weight`` and reshape it to broadcast over ``like``'s trailing dims.

    Used by every layer that supports ``edge_weight`` to apply a per-edge
    scalar to messages of shape ``[E, ...]``.

    Under ``torch.autocast`` the message tensor ``like`` may be float16 or
    bfloat16 while the caller-supplied ``weight`` remains float32.  This
    function casts ``weight`` to ``like.dtype`` before returning so that
    the caller's element-wise multiplication does not raise a dtype mismatch.

    Raises:
        TypeError if ``weight`` is not a Tensor.
        ValueError on shape, length, or device mismatch with ``like``.
    """
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"edge_weight must be a torch.Tensor or None, "
            f"got {type(weight).__name__}"
        )
    if weight.dim() != 1:
        raise ValueError(
            f"edge_weight must be a 1-D tensor of shape [E], "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.size(0) != num_edges:
        raise ValueError(
            f"edge_weight has {weight.size(0)} entries but "
            f"{num_edges} edges were given"
        )
    if weight.device != like.device:
        raise ValueError(
            f"edge_weight device ({weight.device}) must match "
            f"the device of the messages it scales ({like.device})"
        )
    # Cast to message dtype so float32 edge_weight is safe under autocast.
    if weight.dtype != like.dtype:
        weight = weight.to(dtype=like.dtype)
    view = (num_edges,) + (1,) * (like.dim() - 1)
    return weight.view(view)


def edge_softmax(
    scores: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Softmax normalisation over incoming edges per destination node.

    For every destination ``j`` and every trailing index, the returned weights
    satisfy ``sum_{i : target[i] == j} out[i, ...] == 1`` modulo floating-point
    error.  Destinations with no incoming edges are simply not indexed.

    When ``scores`` has dtype ``float16`` or ``bfloat16`` the computation is
    performed in ``float32`` for numerical stability (the max-shift trick and
    the exp/sum are sensitive to precision) and the result is cast back to the
    original dtype before returning.  This matches the approach used by most
    production GNN libraries and prevents inf/NaN under low-precision autocast.

    Args:
        scores: ``[E, *]`` raw attention logits per edge.  ``*`` is any
            (possibly empty) trailing shape; common values are ``[E]`` and
            ``[E, num_heads]``.
        target: ``[E]`` long tensor of destination node indices.
        num_nodes: total number of nodes.  Must satisfy
            ``num_nodes >= int(target.max()) + 1`` when E > 0.

    Returns:
        Tensor of the same shape **and dtype** as ``scores``.
    """
    if scores.size(0) != target.size(0):
        raise ValueError(
            f"scores and target must agree on dim 0; "
            f"got {scores.size(0)} vs {target.size(0)}"
        )
    if target.dtype != torch.long:
        raise TypeError(f"target must have dtype torch.long, got {target.dtype}")
    if scores.size(0) == 0:
        return scores  # nothing to normalise; dtype preserved

    # Upcast to float32 for the softmax computation when input is low-precision.
    orig_dtype = scores.dtype
    if orig_dtype in _LOW_PRECISION_DTYPES:
        scores = scores.to(torch.float32)

    trailing = scores.shape[1:]
    target_b = _expand_index(target, scores)

    # Per-destination max for numerical stability.  scatter_reduce_ with
    # include_self=True means ``max(initial, src)`` — initial is -inf, so
    # for any destination that does receive at least one edge the result is
    # simply max(src for that edge group).  Destinations with no incoming
    # edges keep their -inf placeholder, but we never index into them.
    max_per_dest = scores.new_full((num_nodes, *trailing), float("-inf"))
    max_per_dest.scatter_reduce_(0, target_b, scores, reduce="amax", include_self=True)
    scores_shifted = scores - max_per_dest.index_select(0, target)

    # Exponentiate and sum per destination.
    exp_scores = scores_shifted.exp()
    sum_per_dest = scores.new_zeros((num_nodes, *trailing))
    sum_per_dest.scatter_add_(0, target_b, exp_scores)

    # Divide.  Any destination with no incoming edges has sum 0, but we never
    # index into those rows.  We still clamp the denominator for safety.
    denom = sum_per_dest.index_select(0, target).clamp_min(1e-16)
    result = exp_scores / denom

    # Cast back to the caller's dtype (no-op when orig_dtype is float32).
    return result.to(orig_dtype)


def scatter_sum(
    src: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sum ``src`` rows by destination index along dim 0.

    Returns a tensor ``out`` with shape ``[num_nodes, *src.shape[1:]]`` such
    that ``out[j] = sum_{i : target[i] == j} src[i]``.  Destinations with no
    matching rows stay at zero.
    """
    out = src.new_zeros((num_nodes, *src.shape[1:]))
    out.index_add_(0, target, src)
    return out


def scatter_mean(
    src: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean of ``src`` rows by destination index along dim 0.

    Destinations with no matching rows are returned as zero (a count clamp of
    1 is applied so we never divide by zero).
    """
    summed = scatter_sum(src, target, num_nodes)
    counts = src.new_zeros(num_nodes)
    counts.index_add_(0, target, src.new_ones(src.size(0)))
    counts = counts.clamp_min(1.0).view((num_nodes,) + (1,) * (summed.dim() - 1))
    return summed / counts


def scatter_max(
    src: torch.Tensor,
    target: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Max of ``src`` rows by destination index along dim 0.

    Destinations with no matching rows would otherwise return ``-inf``
    (the identity for max); they are replaced with ``0`` to keep downstream
    computations finite.
    """
    trailing = src.shape[1:]
    out = src.new_full((num_nodes, *trailing), float("-inf"))
    target_b = _expand_index(target, src)
    out.scatter_reduce_(0, target_b, src, reduce="amax", include_self=True)
    return out.masked_fill(torch.isinf(out) & (out < 0), 0.0)
