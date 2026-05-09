"""Train/val/test split helpers for graph data."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "node_train_val_test_split",
    "edge_train_val_test_split",
    "temporal_train_val_test_split",
    "graph_train_test_split",
]


def _check_ratios(train: float, val: float, test: float) -> None:
    if not (0.0 < train < 1.0 and 0.0 <= val < 1.0 and 0.0 <= test < 1.0):
        raise ValueError("ratios must satisfy 0 < train < 1, 0 <= val < 1, 0 <= test < 1")
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {train + val + test}")


def node_train_val_test_split(
    num_nodes: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random split of ``num_nodes`` IDs into train/val/test masks.

    Returns three boolean tensors ``[N]`` summing to 1 per node.
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(num_nodes, generator=gen)
    n_train = int(round(train_ratio * num_nodes))
    n_val = int(round(val_ratio * num_nodes))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def edge_train_val_test_split(
    edge_index: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random split of edges into three ``LongTensor[2, *]`` slices."""
    _check_ratios(train_ratio, val_ratio, test_ratio)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    E = edge_index.size(1)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(E, generator=gen)
    n_train = int(round(train_ratio * E))
    n_val = int(round(val_ratio * E))
    train_e = edge_index[:, perm[:n_train]]
    val_e = edge_index[:, perm[n_train:n_train + n_val]]
    test_e = edge_index[:, perm[n_train + n_val:]]
    return train_e, val_e, test_e


def temporal_train_val_test_split(
    edge_index: torch.Tensor,
    edge_time: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Chronological split — earliest edges train, latest test.

    Returns three ``(edge_index, edge_time)`` pairs with no future
    leakage: every val edge has ``time >= max(train_time)`` and every
    test edge has ``time >= max(val_time)``.
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_time.dim() != 1 or edge_time.numel() != edge_index.size(1):
        raise ValueError("edge_time must have shape [E] matching edge_index")
    E = edge_index.size(1)
    sorted_t, perm = edge_time.sort(stable=True)
    sorted_ei = edge_index[:, perm]
    n_train = int(round(train_ratio * E))
    n_val = int(round(val_ratio * E))
    return (
        (sorted_ei[:, :n_train], sorted_t[:n_train]),
        (sorted_ei[:, n_train:n_train + n_val], sorted_t[n_train:n_train + n_val]),
        (sorted_ei[:, n_train + n_val:], sorted_t[n_train + n_val:]),
    )


def graph_train_test_split(
    num_graphs: int,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random split of graph indices."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(num_graphs, generator=gen)
    n_train = int(round(train_ratio * num_graphs))
    return perm[:n_train], perm[n_train:]
