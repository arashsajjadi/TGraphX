"""Split transforms: build train/val/test masks for nodes / edges / graphs."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..core.graph import Graph
from .graph import _shallow_copy


def _three_way_indices(
    n: int,
    train: float,
    val: float,
    seed: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(train_idx, val_idx, test_idx)`` from a deterministic permutation."""
    if not 0.0 <= train <= 1.0 or not 0.0 <= val <= 1.0:
        raise ValueError("train and val must be in [0, 1]")
    if train + val > 1.0 + 1e-9:
        raise ValueError(f"train + val must be <= 1; got {train + val}")
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    perm = torch.randperm(n, generator=gen)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _make_masks(n: int, train_idx, val_idx, test_idx) -> dict:
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


class RandomNodeSplit:
    """Stamp ``train_mask`` / ``val_mask`` / ``test_mask`` into ``graph.metadata['masks']``.

    Splits are *non-overlapping*; nodes with no positive mask end up in
    the test set.  Determinism is controlled by ``seed``.
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: Optional[int] = None,
        key: str = "masks",
    ) -> None:
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = seed
        self.key = key

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        N = new.num_nodes
        train_idx, val_idx, test_idx = _three_way_indices(
            N, self.train_ratio, self.val_ratio, self.seed,
        )
        masks = _make_masks(N, train_idx, val_idx, test_idx)
        meta = dict(new.metadata or {})
        meta[self.key] = masks
        new.metadata = meta
        return new


class RandomLinkSplit:
    """Random split over the *edge* set; stamps masks into ``metadata['edge_masks']``."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = seed

    def __call__(self, graph: Graph) -> Graph:
        if graph.edge_index is None or graph.num_edges == 0:
            raise ValueError("RandomLinkSplit requires a graph with edges.")
        new = _shallow_copy(graph)
        E = new.num_edges
        train_idx, val_idx, test_idx = _three_way_indices(
            E, self.train_ratio, self.val_ratio, self.seed,
        )
        masks = _make_masks(E, train_idx, val_idx, test_idx)
        meta = dict(new.metadata or {})
        meta["edge_masks"] = masks
        new.metadata = meta
        return new


class RandomGraphSplit:
    """Split a list of graphs into train/val/test; returns three index lists.

    This is a *list-level* helper (not a per-graph transform).  Use it
    on a dataset by manually slicing:

    .. code-block:: python

        splitter = RandomGraphSplit(0.7, 0.15, seed=0)
        train, val, test = splitter(len(dataset))
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = seed

    def __call__(self, num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _three_way_indices(
            int(num_graphs), self.train_ratio, self.val_ratio, self.seed,
        )


class FixedSplit:
    """Stamp pre-computed masks into ``graph.metadata['masks']``.

    Useful for re-using splits supplied by an upstream loader (PyG /
    DGL / OGB).  Inputs may be 1-D LongTensor index lists or boolean
    masks; both are accepted.
    """

    def __init__(self, train, val, test, key: str = "masks") -> None:
        self.train = train
        self.val = val
        self.test = test
        self.key = key

    def _to_mask(self, ids_or_mask, n: int) -> torch.Tensor:
        if ids_or_mask.dtype == torch.bool:
            if ids_or_mask.numel() != n:
                raise ValueError(
                    f"Boolean mask length {ids_or_mask.numel()} != num_nodes {n}"
                )
            return ids_or_mask.clone()
        m = torch.zeros(n, dtype=torch.bool)
        m[ids_or_mask] = True
        return m

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        N = new.num_nodes
        masks = {
            "train_mask": self._to_mask(self.train, N),
            "val_mask": self._to_mask(self.val, N),
            "test_mask": self._to_mask(self.test, N),
        }
        meta = dict(new.metadata or {})
        meta[self.key] = masks
        new.metadata = meta
        return new
