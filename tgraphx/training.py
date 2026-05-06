"""Lightweight training utilities for TGraphX.

These are thin helpers — they do **not** implement a full training framework.
All logging, checkpointing, and dashboard behavior is **off by default**.

What IS provided
----------------
set_seed          — reproducible seeds across torch / numpy / random
count_parameters  — total trainable parameter count
save_checkpoint   — torch.save wrapper (model + optimizer state + epoch)
load_checkpoint   — matching loader; returns the saved epoch number
accuracy          — multi-class: fraction of correct argmax predictions
mean_absolute_error / mean_squared_error — regression metrics

train_epoch  — one supervised training epoch over a DataLoader
evaluate     — evaluation loop (no_grad, no file writes)
fit          — thin loop wrapper: train_epoch + evaluate + optional logger

What is NOT provided
--------------------
No full training framework.  No hidden checkpointing.  No hidden dashboard.
If you need more control, write your own loop using the primitives above.
train_epoch/evaluate/fit handle GraphBatch (from GraphDataLoader) and plain
(x, y) tuple batches.  For other batch formats, write your own loop.
"""
from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set seeds for :mod:`torch`, :mod:`numpy`, and :mod:`random`.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Model utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in ``model``.

    Args:
        model: Any ``torch.nn.Module``.
        trainable_only: If ``True`` (default), count only parameters where
            ``requires_grad=True``.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    **extra: Any,
) -> None:
    """Save model and optimizer state to ``path``.

    The file is a plain ``torch.save`` dict with keys:
    ``epoch``, ``model_state_dict``, ``optimizer_state_dict``, plus any
    additional keyword arguments you pass.

    Args:
        model:     Model to save.
        optimizer: Optimizer to save (pass ``None`` to skip).
        epoch:     Current epoch number (stored for reference).
        path:      Destination file path.
        **extra:   Any extra values (e.g. ``loss=0.42``) stored in the dict.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        **extra,
    }
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    map_location: Optional[Any] = None,
) -> int:
    """Load model and optimizer state from ``path``.

    Args:
        model:        Target model (modified in-place).
        optimizer:    Target optimizer (modified in-place; ignored if ``None``).
        path:         Checkpoint file written by :func:`save_checkpoint`.
        map_location: Passed to ``torch.load`` (e.g. ``"cpu"``).

    Returns:
        The ``epoch`` value stored in the checkpoint.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("epoch", 0))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class accuracy: fraction of ``argmax(logits)`` matching ``labels``.

    Args:
        logits: ``[N, C]`` raw logits (not softmax).
        labels: ``[N]`` integer class indices.

    Returns:
        Float in ``[0, 1]``.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2-D [N, C]; got shape {tuple(logits.shape)}"
        )
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error (MAE).

    Args:
        predictions: Predicted values, any shape.
        targets:     Ground truth, same shape as ``predictions``.

    Returns:
        Float MAE.
    """
    return float((predictions - targets).abs().mean().item())


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean squared error (MSE).

    Args:
        predictions: Predicted values, any shape.
        targets:     Ground truth, same shape as ``predictions``.

    Returns:
        Float MSE.
    """
    return float(((predictions - targets) ** 2).mean().item())


# ─────────────────────────────────────────────────────────────────────────────
# Internal batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if device == "auto":
        from tgraphx.performance import recommended_device
        return recommended_device()
    return torch.device(device)


def _unpack_batch(batch: Any, device: torch.device):
    """Return (args, kwargs, targets) for a single batch.

    Supported formats
    -----------------
    ``GraphBatch`` — calls ``model(node_features, edge_index, batch=...,
                                   edge_features=..., edge_weight=...)``
                     and uses ``graph_labels`` or ``node_labels`` as targets.

    ``(x, y)`` tuple — calls ``model(x)`` and uses ``y`` as targets.

    Raises ``ValueError`` for unsupported formats.
    """
    # Lazy import to avoid circular imports and heavy load at module level.
    from tgraphx.core.graph import GraphBatch

    if isinstance(batch, GraphBatch):
        nf = batch.node_features.to(device)
        ei = batch.edge_index.to(device)
        kw: Dict[str, Any] = {}
        if batch.edge_features is not None:
            kw["edge_features"] = batch.edge_features.to(device)
        if batch.edge_weight is not None:
            kw["edge_weight"] = batch.edge_weight.to(device)
        kw["batch"] = batch.batch.to(device)

        if batch.graph_labels is not None:
            targets = batch.graph_labels.to(device)
        elif batch.node_labels is not None:
            targets = batch.node_labels.to(device)
        else:
            raise ValueError(
                "GraphBatch has neither graph_labels nor node_labels. "
                "Set labels on your Graph objects before batching, "
                "or write a custom training loop."
            )
        # GraphBatch stacks [1]-shaped labels into [B, 1]; squeeze to [B]
        # so that cross_entropy and similar losses get the expected 1-D input.
        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        return (nf, ei), kw, targets

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        if isinstance(x, torch.Tensor):
            return (x.to(device),), {}, y.to(device)
        raise ValueError(
            f"Expected batch[0] to be a Tensor; got {type(x).__name__}. "
            f"For custom batch formats, write your own training loop."
        )

    raise ValueError(
        f"Unsupported batch type: {type(batch).__name__}. "
        f"Expected a GraphBatch or a (x, y) tuple of Tensors. "
        f"For custom formats, write your own training loop."
    )


def _call_model(model: nn.Module, args: tuple, kwargs: dict) -> torch.Tensor:
    """Call model, falling back to fewer kwargs on TypeError."""
    try:
        return model(*args, **kwargs)
    except TypeError:
        try:
            return model(*args)
        except Exception as exc:
            raise RuntimeError(
                "Could not call model with the provided batch inputs. "
                "If your model has a custom forward signature, write "
                "your own training loop."
            ) from exc


def _compute_metrics(
    metrics: Optional[Dict[str, Callable]],
    output: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if not metrics:
        return {}
    result = {}
    with torch.no_grad():
        for name, fn in metrics.items():
            try:
                result[name] = float(fn(output.detach(), targets.detach()))
            except Exception:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training loop utilities
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    *,
    device: Union[str, torch.device] = "auto",
    metrics: Optional[Dict[str, Callable]] = None,
    logger=None,
    log_level: int = 0,
    epoch: Optional[int] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Run one supervised training epoch.

    Args:
        model:     ``nn.Module`` to train.
        loader:    DataLoader yielding ``GraphBatch`` or ``(x, y)`` tuples.
        optimizer: PyTorch optimizer (must already be constructed).
        loss_fn:   ``loss_fn(output, targets) -> Tensor`` (scalar).
        device:    ``"auto"`` selects CUDA > MPS > CPU automatically.
        metrics:   Dict ``{name: fn}`` where ``fn(output, targets) -> float``.
        logger:    Object with a ``log(**kwargs)`` method (e.g.
                   ``CSVLogger``, ``TensorBoardLogger``).  ``None`` writes nothing.
        log_level: 0 = silent; 1 = print epoch summary; 2 = per-batch progress.
        epoch:     Epoch index (passed to logger; does not affect training).
        amp:       If ``True`` and device is CUDA, wrap forward in
                   ``torch.autocast("cuda")``.  No GradScaler is used;
                   for stable float16 training, manage a GradScaler yourself.
        grad_clip: If set, clip gradient norm to this value before step.

    Returns:
        Dict with ``"loss"`` and any requested metric names, averaged over
        all batches.
    """
    dev = _resolve_device(device)
    model.train()
    model.to(dev)

    use_amp = amp and dev.type == "cuda"
    amp_ctx = torch.autocast("cuda") if use_amp else _NullCtx()

    total_loss = 0.0
    metric_acc: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    n_batches = 0

    for batch in loader:
        args, kwargs, targets = _unpack_batch(batch, dev)

        optimizer.zero_grad()
        with amp_ctx:
            output = _call_model(model, args, kwargs)
            loss = loss_fn(output, targets)

        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if metrics:
            for k, v in _compute_metrics(metrics, output, targets).items():
                metric_acc[k] += v

        if log_level >= 2:
            print(f"  [batch {n_batches}] loss={loss.item():.4f}")

    if n_batches == 0:
        return {"loss": float("nan")}

    results: Dict[str, float] = {"loss": total_loss / n_batches}
    for k in metric_acc:
        results[k] = metric_acc[k] / n_batches

    if log_level >= 1:
        ep_str = f"epoch {epoch} " if epoch is not None else ""
        parts = [f"loss={results['loss']:.4f}"]
        parts += [f"{k}={v:.4f}" for k, v in results.items() if k != "loss"]
        print(f"  {ep_str}{' | '.join(parts)}")

    if logger is not None:
        log_kwargs: Dict[str, Any] = {}
        if epoch is not None:
            log_kwargs["epoch"] = epoch
        log_kwargs["train_loss"] = results["loss"]
        for k, v in results.items():
            if k != "loss":
                log_kwargs[f"train_{k}"] = v
        logger.log(**log_kwargs)

    return results


class _NullCtx:
    """No-op context manager for non-AMP paths (zero overhead)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


def evaluate(
    model: nn.Module,
    loader,
    loss_fn: Callable,
    *,
    metrics: Optional[Dict[str, Callable]] = None,
    device: Union[str, torch.device] = "auto",
) -> Dict[str, float]:
    """Evaluate model on a DataLoader with no gradient computation.

    Args:
        model:    ``nn.Module`` to evaluate (set to ``eval()`` internally).
        loader:   DataLoader yielding ``GraphBatch`` or ``(x, y)`` tuples.
        loss_fn:  ``loss_fn(output, targets) -> Tensor`` (scalar).
        metrics:  Dict ``{name: fn}`` of additional metric functions.
        device:   ``"auto"`` selects CUDA > MPS > CPU.

    Returns:
        Dict with ``"loss"`` and any requested metric names, averaged over
        all batches.  No file writes.
    """
    dev = _resolve_device(device)
    model.eval()
    model.to(dev)

    total_loss = 0.0
    metric_acc: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            args, kwargs, targets = _unpack_batch(batch, dev)
            output = _call_model(model, args, kwargs)
            loss = loss_fn(output, targets)
            total_loss += loss.item()
            n_batches += 1
            if metrics:
                for k, v in _compute_metrics(metrics, output, targets).items():
                    metric_acc[k] += v

    if n_batches == 0:
        return {"loss": float("nan")}

    results: Dict[str, float] = {"loss": total_loss / n_batches}
    for k in metric_acc:
        results[k] = metric_acc[k] / n_batches
    return results


def fit(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 10,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[Callable] = None,
    device: Union[str, torch.device] = "auto",
    metrics: Optional[Dict[str, Callable]] = None,
    logger=None,
    log_level: int = 0,
    amp: bool = False,
    grad_clip: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Train a model for ``epochs`` epochs, optionally evaluating after each.

    This is a thin convenience wrapper around :func:`train_epoch` and
    :func:`evaluate`.  It does **not** perform checkpointing, start a
    dashboard, or write any files unless you explicitly provide a
    ``logger``.

    Args:
        model:        ``nn.Module`` to train.
        train_loader: Training DataLoader.
        val_loader:   Optional validation DataLoader.
        epochs:       Number of training epochs.
        optimizer:    PyTorch optimizer.  **Required** — raises ``ValueError``
                      if not provided.
        loss_fn:      Loss function ``fn(output, targets) -> Tensor``.
                      **Required** — raises ``ValueError`` if not provided.
        device:       ``"auto"`` selects CUDA > MPS > CPU.
        metrics:      Dict ``{name: fn}`` of metric functions.
        logger:       Logger with ``log(**kwargs)`` (e.g. ``CSVLogger``,
                      ``TensorBoardLogger``).  ``None`` writes nothing.
        log_level:    0 = silent; 1 = print per-epoch summary.
        amp:          Wrap forward in CUDA autocast (CUDA only).
        grad_clip:    Gradient norm clip value (``None`` to disable).

    Returns:
        List of per-epoch result dicts, each containing
        ``"epoch"``, ``"train_loss"``, and optionally ``"val_loss"``
        and any metric keys.
    """
    if optimizer is None:
        raise ValueError(
            "fit() requires an explicit optimizer. "
            "Example: optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)"
        )
    if loss_fn is None:
        raise ValueError(
            "fit() requires an explicit loss_fn. "
            "Example: loss_fn=torch.nn.CrossEntropyLoss()"
        )

    history: List[Dict[str, float]] = []

    for ep in range(epochs):
        train_res = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device=device, metrics=metrics,
            logger=None,  # we handle logging here after val
            log_level=0,  # we handle printing here
            epoch=ep,
            amp=amp,
            grad_clip=grad_clip,
        )

        epoch_row: Dict[str, float] = {"epoch": float(ep)}
        epoch_row["train_loss"] = train_res["loss"]
        for k, v in train_res.items():
            if k != "loss":
                epoch_row[f"train_{k}"] = v

        if val_loader is not None:
            val_res = evaluate(model, val_loader, loss_fn,
                               metrics=metrics, device=device)
            epoch_row["val_loss"] = val_res["loss"]
            for k, v in val_res.items():
                if k != "loss":
                    epoch_row[f"val_{k}"] = v

        history.append(epoch_row)

        if log_level >= 1:
            parts = [f"epoch {ep}/{epochs - 1}"]
            parts += [f"{k}={v:.4f}" for k, v in epoch_row.items() if k != "epoch"]
            print("  " + "  |  ".join(parts))

        if logger is not None:
            logger.log(**{k: v for k, v in epoch_row.items()})

    return history


__all__ = [
    "set_seed",
    "count_parameters",
    "save_checkpoint",
    "load_checkpoint",
    "accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "train_epoch",
    "evaluate",
    "fit",
]
