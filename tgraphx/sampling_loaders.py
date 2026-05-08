"""Lightweight sampling-based loaders for large-graph training.

These loaders are **plain Python iterables** — no hidden multiprocessing,
no DataLoader workers, no global RNG side effects.  They yield
:class:`~tgraphx.Graph` objects (or :class:`~tgraphx.GraphBatch` when
``batch_size > 1``) that can flow directly into existing training
utilities.

API
---
:class:`SubgraphDataLoader`
    Yield random subgraphs of fixed node count.

:class:`NeighborSamplerLoader`
    Yield k-hop neighbour-sampled subgraphs around batches of seed nodes.

Determinism
-----------
All loaders accept a ``seed`` and use a per-instance ``torch.Generator``.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import torch

from .core.graph import Graph
from .core.dataloader import GraphDataLoader
from .sampling import neighbor_sample, sample_nodes

__all__ = ["SubgraphDataLoader", "NeighborSamplerLoader"]


class SubgraphDataLoader:
    """Iterate over random subgraphs of fixed size.

    At each step samples ``num_nodes`` nodes uniformly without
    replacement and returns the induced :class:`~tgraphx.Graph`.

    Args:
        graph: Source :class:`~tgraphx.Graph` (must contain edges).
        num_nodes: Number of nodes per sampled subgraph.
        num_steps: Number of subgraphs per epoch (one ``__iter__``).
        relabel_nodes: Forwarded to :func:`tgraphx.sampling.sample_nodes`.
        seed: Optional RNG seed for reproducibility.

    Example::

        loader = SubgraphDataLoader(graph, num_nodes=64, num_steps=20, seed=0)
        for sub in loader:
            out = model(sub.node_features, sub.edge_index)
    """

    def __init__(
        self,
        graph: Graph,
        num_nodes: int,
        num_steps: int,
        relabel_nodes: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")
        if num_nodes > graph.num_nodes:
            raise ValueError(
                f"num_nodes={num_nodes} > graph.num_nodes={graph.num_nodes}"
            )
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        self.graph = graph
        self.num_nodes = num_nodes
        self.num_steps = num_steps
        self.relabel_nodes = relabel_nodes
        self._seed = seed

    def __len__(self) -> int:
        return self.num_steps

    def __iter__(self) -> Iterator[Graph]:
        # Use a per-iteration sequence of seeds derived from the base seed
        # so that each call to __iter__ is deterministic but different
        # epochs are independent.
        if self._seed is None:
            seeds = [None] * self.num_steps
        else:
            base = int(self._seed)
            seeds = [base + i for i in range(self.num_steps)]
        for s in seeds:
            yield sample_nodes(
                self.graph,
                num_nodes=self.num_nodes,
                seed=s,
                relabel_nodes=self.relabel_nodes,
            )


class NeighborSamplerLoader:
    """Iterate batches of seed nodes, returning neighbour-sampled subgraphs.

    Each step picks ``batch_size`` seed nodes (from ``input_nodes`` or
    all nodes by default) and runs
    :func:`tgraphx.sampling.neighbor_sample` with the configured fanouts.

    Args:
        graph: Source graph.
        input_nodes: 1-D LongTensor of candidate seed ids.  ``None``
            means all nodes.
        batch_size: Number of seeds per step.
        fanouts: Per-layer fanout list (same semantics as
            :func:`neighbor_sample`).
        shuffle: If ``True``, shuffle the seed order each epoch.
        direction: ``"in"`` (GraphSAGE) or ``"out"``.
        relabel_nodes: Forwarded.
        seed: RNG seed for reproducibility.
        drop_last: If ``True``, drop the trailing partial batch.

    Example::

        loader = NeighborSamplerLoader(
            graph, batch_size=16, fanouts=[10, 5], shuffle=True, seed=0,
        )
        for sub in loader:
            seeds = sub.metadata["sampling"]["seed_nodes"]
            out = model(sub.node_features, sub.edge_index)
            # The first ``len(seeds)`` rows of ``out`` correspond to seeds
            # (when ``relabel_nodes=True`` they are placed first; see
            # _build_subgraph_from_node_mask docstring for ordering).
    """

    def __init__(
        self,
        graph: Graph,
        batch_size: int,
        fanouts: Sequence[int],
        input_nodes: Optional[torch.Tensor] = None,
        shuffle: bool = False,
        direction: str = "in",
        relabel_nodes: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        if not fanouts:
            raise ValueError("fanouts must be a non-empty sequence")
        if direction not in ("in", "out"):
            raise ValueError(f"direction must be 'in' or 'out'; got {direction!r}")
        if input_nodes is None:
            input_nodes = torch.arange(graph.num_nodes, dtype=torch.long,
                                        device=graph.node_features.device)
        else:
            if input_nodes.dim() != 1:
                raise ValueError("input_nodes must be 1-D")
            input_nodes = input_nodes.to(
                device=graph.node_features.device, dtype=torch.long,
            )
        self.graph = graph
        self.batch_size = int(batch_size)
        self.fanouts = list(fanouts)
        self.input_nodes = input_nodes
        self.shuffle = bool(shuffle)
        self.direction = direction
        self.relabel_nodes = relabel_nodes
        self._seed = seed
        self.drop_last = bool(drop_last)

    def __len__(self) -> int:
        n = self.input_nodes.numel()
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Graph]:
        gen = torch.Generator()
        if self._seed is not None:
            gen.manual_seed(int(self._seed))
        nodes = self.input_nodes
        if self.shuffle:
            perm = torch.randperm(nodes.numel(), generator=gen)
            nodes = nodes[perm]

        n_total = nodes.numel()
        n_batches = len(self)
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, n_total)
            seed_batch = nodes[start:end]
            if seed_batch.numel() == 0:
                break
            sub_seed = (
                None if self._seed is None
                else int(self._seed) + i + 1
            )
            yield neighbor_sample(
                self.graph,
                seed_nodes=seed_batch,
                fanouts=self.fanouts,
                seed=sub_seed,
                direction=self.direction,
                relabel_nodes=self.relabel_nodes,
            )
