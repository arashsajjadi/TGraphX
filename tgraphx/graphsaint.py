"""GraphSAINT-style subgraph samplers.

Implements the three GraphSAINT samplers from Zeng et al. (ICLR 2020) —
"GraphSAINT: Graph Sampling Based Inductive Learning Method":

* :class:`GraphSAINTNodeSampler` — uniform node sampling (then induce).
* :class:`GraphSAINTEdgeSampler` — degree-aware edge sampling.
* :class:`GraphSAINTRandomWalkSampler` — root + random-walk sampling.
* :class:`GraphSAINTLoader` — DataLoader-compatible iteration.
* :func:`estimate_norm_coefficients` — Monte-Carlo estimate of the
  per-node and per-edge normalisation coefficients used by GraphSAINT
  to produce unbiased aggregations on sampled subgraphs.

All samplers:
  * preserve original node/edge IDs in ``metadata['sampling']``;
  * preserve node features, edge_weight, edge_features, labels;
  * accept an optional ``seed`` for deterministic sampling;
  * never allocate a dense ``[N, N]`` adjacency.

The Monte-Carlo normalisation estimator is approximate: it runs
``num_norm_samples`` independent draws and counts how often each node
(or edge) is included.  The per-node aggregation coefficient is
``α_v = count_v / num_norm_samples``; the per-edge loss coefficient is
``λ_e = count_e / num_norm_samples``.  In the limit of infinitely many
draws, GraphSAINT proves these coefficients give unbiased aggregations.
For finite ``num_norm_samples`` we clamp denominators away from zero
and document the approximation explicitly.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .core.graph import Graph
from .sampling import induced_subgraph, edge_subgraph, random_walk_sample

__all__ = [
    "GraphSAINTNodeSampler",
    "GraphSAINTEdgeSampler",
    "GraphSAINTRandomWalkSampler",
    "GraphSAINTLoader",
    "estimate_norm_coefficients",
]


# ── Internal helpers ─────────────────────────────────────────────────────────


def _gen(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def _attach_norm(graph: Graph, node_norm: torch.Tensor, edge_norm: torch.Tensor) -> Graph:
    meta = dict(graph.metadata) if isinstance(graph.metadata, dict) else {}
    saint = dict(meta.get("graphsaint", {}))
    saint["node_norm"] = node_norm.detach().cpu()
    saint["edge_norm"] = edge_norm.detach().cpu()
    meta["graphsaint"] = saint
    graph.metadata = meta
    return graph


# ── Node sampler ─────────────────────────────────────────────────────────────


class GraphSAINTNodeSampler:
    """Uniform-node sampler.

    On each draw, samples ``budget`` distinct nodes uniformly at random
    and returns the induced subgraph.  This is the simplest GraphSAINT
    variant and a strong baseline for transductive node classification.

    Args:
        graph: Source :class:`~tgraphx.Graph`.
        budget: Number of nodes to draw per subgraph.
        num_steps: Number of subgraphs the loader will yield per epoch.
            (One "epoch" of GraphSAINT visits ``num_steps`` subgraphs;
            this loosely corresponds to ``num_steps * budget`` node
            updates.)
        seed: Optional RNG seed.  Each draw uses an independent stream
            derived from ``seed`` so iteration order is reproducible.
        replace: If ``True``, sample with replacement (some nodes may
            repeat).  Default is ``False`` which matches the GraphSAINT
            paper.

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        budget: int,
        num_steps: int = 50,
        seed: Optional[int] = None,
        replace: bool = False,
    ) -> None:
        if budget < 1:
            raise ValueError(f"budget must be >= 1; got {budget}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        if not replace and budget > graph.num_nodes:
            raise ValueError(
                f"budget={budget} > num_nodes={graph.num_nodes}; "
                f"use replace=True or shrink budget"
            )
        self.graph = graph
        self.budget = int(budget)
        self.num_steps = int(num_steps)
        self.seed = seed
        self.replace = bool(replace)

    def sample(self, step: int = 0) -> Graph:
        """Draw a single subgraph for ``step``."""
        seed = None if self.seed is None else int(self.seed) + step
        gen = _gen(seed)
        N = self.graph.num_nodes
        if self.replace:
            ids = torch.randint(N, (self.budget,), generator=gen)
            ids = ids.unique()
        else:
            ids = torch.randperm(N, generator=gen)[: self.budget]
        ids = ids.to(self.graph.node_features.device).long()
        sub = induced_subgraph(self.graph, ids, relabel_nodes=True)
        # Attach configuration to metadata.
        meta = dict(sub.metadata) if isinstance(sub.metadata, dict) else {}
        saint = dict(meta.get("graphsaint", {}))
        saint["sampler"] = "node"
        saint["budget"] = self.budget
        saint["step"] = int(step)
        meta["graphsaint"] = saint
        sub.metadata = meta
        return sub

    def __iter__(self) -> Iterator[Graph]:
        for step in range(self.num_steps):
            yield self.sample(step)

    def __len__(self) -> int:
        return self.num_steps


# ── Edge sampler ─────────────────────────────────────────────────────────────


class GraphSAINTEdgeSampler:
    """Edge sampler with the GraphSAINT degree-aware probability.

    Each edge ``(u, v)`` is independently sampled with probability
    ``p_e = budget * (1/deg(u) + 1/deg(v))`` clamped to ``[0, 1]``.
    Sampled edges then induce a node subgraph (orphan nodes dropped).

    Args:
        graph: Source graph.
        budget: Edge-budget hyperparameter ``B``.  Larger ``B`` →
            larger expected subgraph.  In the GraphSAINT paper this is
            the per-step edge budget.
        num_steps: Subgraphs per epoch.
        seed: RNG seed.

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        budget: int,
        num_steps: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        if graph.edge_index is None or graph.num_edges == 0:
            raise ValueError("GraphSAINTEdgeSampler requires graph with edges")
        if budget < 1:
            raise ValueError(f"budget must be >= 1; got {budget}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        self.graph = graph
        self.budget = int(budget)
        self.num_steps = int(num_steps)
        self.seed = seed
        # Pre-compute per-edge sampling probabilities (degree-aware).
        self._edge_prob = self._compute_edge_prob(graph)

    @staticmethod
    def _compute_edge_prob(graph: Graph) -> torch.Tensor:
        ei = graph.edge_index
        device = ei.device
        N = graph.num_nodes
        deg = torch.zeros(N, dtype=torch.float, device=device)
        ones = torch.ones(ei.size(1), dtype=torch.float, device=device)
        deg.scatter_add_(0, ei[0], ones)
        deg.scatter_add_(0, ei[1], ones)
        deg = deg.clamp(min=1.0)
        # 1/deg(u) + 1/deg(v) per edge.
        per_edge = 1.0 / deg[ei[0]] + 1.0 / deg[ei[1]]
        return per_edge

    def sample(self, step: int = 0) -> Graph:
        """Draw one edge-sampled subgraph for ``step``."""
        seed = None if self.seed is None else int(self.seed) + step
        gen = _gen(seed)
        E = self.graph.num_edges
        # GraphSAINT scaling: scale prob to expectation = budget.
        scaled = self._edge_prob * float(self.budget) / max(self._edge_prob.sum().item(), 1.0)
        scaled = scaled.clamp(0.0, 1.0)
        # Bernoulli per edge.
        u = torch.rand(E, generator=gen, device=scaled.device)
        keep = u < scaled
        keep_eids = torch.where(keep)[0]
        if keep_eids.numel() == 0:
            # Fallback: sample at least one edge.
            keep_eids = torch.randint(E, (1,), generator=gen)
        sub = edge_subgraph(self.graph, keep_eids, relabel_nodes=True)
        meta = dict(sub.metadata) if isinstance(sub.metadata, dict) else {}
        saint = dict(meta.get("graphsaint", {}))
        saint["sampler"] = "edge"
        saint["budget"] = self.budget
        saint["step"] = int(step)
        meta["graphsaint"] = saint
        sub.metadata = meta
        return sub

    def __iter__(self) -> Iterator[Graph]:
        for step in range(self.num_steps):
            yield self.sample(step)

    def __len__(self) -> int:
        return self.num_steps


# ── Random-walk sampler ─────────────────────────────────────────────────────


class GraphSAINTRandomWalkSampler:
    """Root + random-walk sampler.

    From each of ``num_roots`` uniformly-sampled root nodes, performs a
    random walk of length ``walk_length``.  The induced subgraph over
    the union of visited nodes is returned.

    Args:
        graph: Source graph.
        num_roots: Number of independent walk roots per draw.
        walk_length: Steps per walk.
        num_steps: Subgraphs per epoch.
        seed: RNG seed.
        direction: ``"out"`` (default) or ``"in"`` walk direction.

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        num_roots: int,
        walk_length: int,
        num_steps: int = 50,
        seed: Optional[int] = None,
        direction: str = "out",
    ) -> None:
        if graph.edge_index is None:
            raise ValueError("GraphSAINTRandomWalkSampler requires graph edges")
        if num_roots < 1:
            raise ValueError(f"num_roots must be >= 1; got {num_roots}")
        if walk_length < 0:
            raise ValueError(f"walk_length must be >= 0; got {walk_length}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps}")
        if direction not in ("out", "in"):
            raise ValueError(f"direction must be 'out' or 'in'; got {direction!r}")
        self.graph = graph
        self.num_roots = int(num_roots)
        self.walk_length = int(walk_length)
        self.num_steps = int(num_steps)
        self.seed = seed
        self.direction = direction

    def sample(self, step: int = 0) -> Graph:
        seed = None if self.seed is None else int(self.seed) + step
        gen = _gen(seed)
        N = self.graph.num_nodes
        # Pick roots.
        roots = torch.randint(N, (self.num_roots,), generator=gen)
        roots = roots.to(self.graph.node_features.device).long()
        sub = random_walk_sample(
            self.graph, roots,
            walk_length=self.walk_length,
            num_walks_per_seed=1,
            direction=self.direction,
            seed=seed,
            relabel_nodes=True,
        )
        meta = dict(sub.metadata) if isinstance(sub.metadata, dict) else {}
        saint = dict(meta.get("graphsaint", {}))
        saint["sampler"] = "random_walk"
        saint["num_roots"] = self.num_roots
        saint["walk_length"] = self.walk_length
        saint["step"] = int(step)
        meta["graphsaint"] = saint
        sub.metadata = meta
        return sub

    def __iter__(self) -> Iterator[Graph]:
        for step in range(self.num_steps):
            yield self.sample(step)

    def __len__(self) -> int:
        return self.num_steps


# ── Loader ────────────────────────────────────────────────────────────────────


class GraphSAINTLoader:
    """DataLoader-compatible wrapper around any GraphSAINT sampler.

    On each iteration step, draws a fresh subgraph and (optionally)
    attaches GraphSAINT normalisation coefficients estimated by
    :func:`estimate_norm_coefficients`.

    Args:
        sampler: A :class:`GraphSAINTNodeSampler`,
            :class:`GraphSAINTEdgeSampler`, or
            :class:`GraphSAINTRandomWalkSampler`.
        attach_norm: If ``True``, run a Monte-Carlo estimator at
            construction time and attach the per-subgraph
            ``node_norm`` / ``edge_norm`` coefficients.
        num_norm_samples: Draws used by the estimator when
            ``attach_norm=True``.

    Yields:
        :class:`~tgraphx.Graph` subgraphs.

    Stability: Beta.
    """

    def __init__(
        self,
        sampler,
        attach_norm: bool = True,
        num_norm_samples: int = 50,
    ) -> None:
        self.sampler = sampler
        self.attach_norm = bool(attach_norm)
        if self.attach_norm:
            self._node_norm, self._edge_norm = estimate_norm_coefficients(
                sampler, num_samples=num_norm_samples,
            )
        else:
            self._node_norm = None
            self._edge_norm = None

    def __iter__(self) -> Iterator[Graph]:
        for step, sub in enumerate(self.sampler):
            if self.attach_norm and self._node_norm is not None:
                # Slice the global norms back onto the sampled subgraph.
                node_ids = sub.metadata.get("sampling", {}).get("original_node_ids")
                edge_ids = sub.metadata.get("sampling", {}).get("original_edge_ids")
                if node_ids is not None and self._node_norm is not None:
                    node_norm_local = self._node_norm[node_ids]
                else:
                    node_norm_local = torch.ones(sub.num_nodes)
                if edge_ids is not None and self._edge_norm is not None and edge_ids.numel() > 0:
                    edge_norm_local = self._edge_norm[edge_ids]
                else:
                    edge_norm_local = torch.ones(sub.num_edges)
                sub = _attach_norm(sub, node_norm_local, edge_norm_local)
            yield sub

    def __len__(self) -> int:
        return len(self.sampler)


# ── Normalisation estimator ──────────────────────────────────────────────────


def estimate_norm_coefficients(
    sampler,
    num_samples: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo estimate of GraphSAINT node/edge normalisation.

    Runs ``num_samples`` independent draws, counts how often each node
    and edge appears, and returns the inclusion probabilities clamped
    to ``[1e-6, 1.0]``.

    For an aggregation ``Σ_{j ∈ N(i)} W * h_j`` over a sampled subgraph,
    GraphSAINT uses the per-edge coefficient ``1 / λ_e`` (where
    ``λ_e`` is the sampling probability of edge ``e``); the per-node
    coefficient ``1 / α_v`` is used for loss reweighting.

    Args:
        sampler: A GraphSAINT sampler (node/edge/random-walk).
        num_samples: Number of Monte-Carlo draws.  Larger → more
            accurate but slower.

    Returns:
        ``(node_prob, edge_prob)`` — two CPU ``FloatTensor`` of size
        ``[num_nodes]`` and ``[num_edges]`` (or empty if no edges).
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1; got {num_samples}")
    g = sampler.graph
    N = g.num_nodes
    E = g.num_edges
    node_count = torch.zeros(N, dtype=torch.float)
    edge_count = torch.zeros(E, dtype=torch.float)
    for step in range(num_samples):
        sub = sampler.sample(step)
        node_ids = sub.metadata.get("sampling", {}).get("original_node_ids")
        edge_ids = sub.metadata.get("sampling", {}).get("original_edge_ids")
        if node_ids is not None and node_ids.numel() > 0:
            node_count[node_ids.cpu()] += 1.0
        if edge_ids is not None and edge_ids.numel() > 0:
            edge_count[edge_ids.cpu()] += 1.0
    node_prob = (node_count / float(num_samples)).clamp(min=1e-6, max=1.0)
    edge_prob = (edge_count / float(num_samples)).clamp(min=1e-6, max=1.0)
    return node_prob, edge_prob
