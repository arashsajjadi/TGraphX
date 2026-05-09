"""High-level training workflows for TGraphX easy mode."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._exceptions import (
    TGraphXConfigError, TGraphXLabelError, TGraphXUnknownNameError,
)
from ._discovery import _SAMPLERS
from ._models import _resolve_device, _resolve_model_name, _build_model
from ._results import EasyConfig, EasyResult


def train_node_classifier(
    graph: Any,
    model: Optional[Union[str, nn.Module]] = "tensor_gcn",
    sampler: str = "neighbor",
    fanouts: Optional[List[int]] = None,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    hidden_channels: int = 16,
    device: str = "auto",
    seed: Optional[int] = None,
    mask: Optional[Any] = None,
    verbose: bool = True,
    config: Optional[EasyConfig] = None,
) -> EasyResult:
    """Train a node classifier on a graph.

    This is the canonical easy-mode entry point for node classification.
    No direct ``import torch`` is required for common use.

    Args:
        graph: A :class:`~tgraphx.Graph` with ``y`` (node labels) set.
        model: ``"tensor_gcn"``, ``"vector_gcn"``, ``"auto"``, or a custom
            ``nn.Module`` with ``forward(x, edge_index)``.
        sampler: ``"neighbor"`` (default) or ``"full"``.
        fanouts: Neighbor fanouts per hop (default: ``[15, 10]``).
        batch_size: Seed nodes per batch.
        epochs: Training epochs.
        lr: Learning rate for Adam optimizer.
        hidden_channels: Hidden channels in auto-built models.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        seed: Random seed for reproducibility.
        mask: Optional ``BoolTensor[N]`` selecting which nodes to train on.
        verbose: Print epoch progress.
        config: Optional :class:`EasyConfig` that overrides keyword args.

    Returns:
        :class:`EasyResult` with metrics, history, model, graph, config, loader.
    """
    if config is not None:
        model = model if model != "tensor_gcn" else config.model or model
        sampler = config.sampler
        fanouts = fanouts or config.fanouts
        batch_size = config.batch_size
        epochs = config.epochs
        lr = config.lr
        hidden_channels = config.hidden_channels
        device = config.device
        seed = config.seed if config.seed is not None else seed
        verbose = config.verbose

    from tgraphx import Graph, NeighborLoader

    if not isinstance(graph, Graph):
        raise TGraphXConfigError(
            f"'graph' must be a tgraphx.Graph, got {type(graph).__name__}.\n"
            f"Create one with:\n"
            f"    from tgraphx import Graph\n"
            f"    g = Graph(node_features=x, edge_index=edge_index, y=y)\n"
            f"or use tgx.easy.synthetic_tensor_node_classification(...)"
        )

    if not graph.has_labels():
        raise TGraphXLabelError(
            "Node labels are required for node classification.\n"
            "Likely cause: the Graph was created without y/labels.\n"
            "Fix:\n"
            "    g = Graph(node_features=x, edge_index=edge_index, y=y)\n"
            "or:\n"
            "    g.y = y\n"
            "See docs/graph_basics.md#labels"
        )

    if sampler not in _SAMPLERS:
        raise TGraphXUnknownNameError(
            f"Unknown sampler '{sampler}'. Available: {list(_SAMPLERS)}.\n"
            f"Use list_samplers() for descriptions."
        )

    if seed is not None:
        torch.manual_seed(seed)
        from tgraphx.reproducibility import set_seed
        set_seed(seed)

    dev = _resolve_device(device)
    if fanouts is None:
        fanouts = [15, 10]

    node_shape = graph.feature_shape
    num_classes = int(graph.node_labels.max().item()) + 1

    if isinstance(model, str):
        model_name = _resolve_model_name(model, node_shape)
        net = _build_model(model_name, node_shape, num_classes, hidden_channels)
    else:
        net = model
        model_name = type(net).__name__

    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Move graph to device only if needed — avoid unnecessary clone.
    if graph.device != dev:
        graph = graph.clone()
        graph.to(dev)

    resolved_config = {
        "task": "node_classification",
        "model": model_name,
        "sampler": sampler,
        "optimizer": "adam",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(dev),
        "seed": seed,
        "fanouts": fanouts,
        "hidden_channels": hidden_channels,
        "num_classes": num_classes,
        "node_shape": list(node_shape),
    }

    history: List[Dict[str, float]] = []
    t0 = time.time()
    loader = None

    if sampler == "full":
        for epoch in range(1, epochs + 1):
            net.train()
            nf = graph.node_features
            ei = graph.edge_index
            logits = net(nf, ei)
            loss = F.cross_entropy(logits, graph.node_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = logits.detach().argmax(dim=-1)
            acc = float((preds == graph.node_labels).float().mean().item())
            epoch_metrics = {"loss": loss.detach().item(), "accuracy": acc}
            history.append(epoch_metrics)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  loss={epoch_metrics['loss']:.4f}  acc={acc:.4f}")
    else:
        loader = NeighborLoader(
            graph, fanouts=fanouts, mask=mask,
            batch_size=batch_size, shuffle=True, seed=seed,
        )
        for epoch in range(1, epochs + 1):
            net.train()
            total_loss = 0.0
            total_correct = 0
            total_seeds = 0

            for batch in loader:
                batch.to(dev)
                logits = net(batch.node_features, batch.edge_index)
                s_logits = batch.seed_logits(logits)
                s_y = batch.seed_y
                loss = F.cross_entropy(s_logits, s_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.detach().item() * batch.batch_size
                preds = s_logits.detach().argmax(dim=-1)
                total_correct += int((preds == s_y).sum())
                total_seeds += batch.batch_size

            avg_loss = total_loss / max(total_seeds, 1)
            acc = total_correct / max(total_seeds, 1)
            epoch_metrics = {"loss": avg_loss, "accuracy": acc}
            history.append(epoch_metrics)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    elapsed = time.time() - t0
    final_metrics = history[-1] if history else {}

    return EasyResult(
        metrics=final_metrics,
        history=history,
        model=net,
        graph=graph,
        config=resolved_config,
        artifacts={},
        loader=loader,
        optimizer=opt,
        elapsed=elapsed,
    )


fit_node_classifier = train_node_classifier
