"""Cluster-GCN-style partitioning and loading.

Implements partitioners and a :class:`ClusterLoader` that mini-batches
graph training by training on one (or a small group) of partitions per
step, following Chiang et al. (KDD 2019) — "Cluster-GCN: An Efficient
Algorithm for Training Deep and Large Graph Convolutional Networks".

Partitioners:

* :class:`RandomBalancedPartitioner` — random balanced split.
* :class:`BFSPartitioner` — BFS-grown clusters.
* :class:`ConnectedComponentPartitioner` — partition by connected
  components, splitting large components further when needed.
* :class:`SpectralPartitioner` — recursive spectral bisection (small
  graphs only; emits a clear error above ``max_nodes``).

Loader:

* :class:`ClusterLoader` — iterate over single partitions or batches
  of partitions (``num_clusters_per_batch`` > 1 to merge clusters and
  recover inter-cluster edges, as in the Cluster-GCN paper).

All partitioners produce a :class:`PartitionResult` with:

* ``partition_id`` — ``LongTensor[num_nodes]`` mapping node → cluster.
* ``num_partitions`` — number of clusters.
* ``cut_edges`` — count of edges crossing partition boundaries.
* ``balance_ratio`` — ``min_size / max_size``.
* ``intra_edge_count`` — per-partition intra-cluster edge counts.

Stability: Beta (v0.5.0+).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from .core.graph import Graph
from .sampling import induced_subgraph

__all__ = [
    "PartitionResult",
    "RandomBalancedPartitioner",
    "BFSPartitioner",
    "ConnectedComponentPartitioner",
    "SpectralPartitioner",
    "ClusterLoader",
]


# ── Partition result ─────────────────────────────────────────────────────────


@dataclass
class PartitionResult:
    """Result of a partitioning algorithm.

    Attributes:
        partition_id: ``LongTensor[N]`` cluster id per node, in ``[0, K)``.
        num_partitions: Number of clusters ``K``.
        cut_edges: Count of edges with endpoints in different clusters.
        intra_edge_count: ``LongTensor[K]`` intra-cluster edge counts.
        partition_sizes: ``LongTensor[K]`` cluster sizes (#nodes).
        balance_ratio: ``min_size / max_size`` (1.0 = perfectly balanced).
        algorithm: Name of the algorithm (e.g. ``"random_balanced"``).
        seed: RNG seed used (when applicable).
    """

    partition_id: torch.Tensor
    num_partitions: int
    cut_edges: int
    intra_edge_count: torch.Tensor
    partition_sizes: torch.Tensor
    balance_ratio: float
    algorithm: str
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (for dashboard reports)."""
        return {
            "algorithm": self.algorithm,
            "num_partitions": int(self.num_partitions),
            "cut_edges": int(self.cut_edges),
            "balance_ratio": float(self.balance_ratio),
            "partition_sizes": [int(x) for x in self.partition_sizes.tolist()],
            "intra_edge_count": [int(x) for x in self.intra_edge_count.tolist()],
            "seed": self.seed,
            **self.extra,
        }


def _summarise(
    graph: Graph,
    partition_id: torch.Tensor,
    num_partitions: int,
    algorithm: str,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> PartitionResult:
    pid = partition_id.long()
    if pid.numel() != graph.num_nodes:
        raise ValueError("partition_id must have shape [num_nodes]")
    if (pid < 0).any() or (pid >= num_partitions).any():
        raise ValueError("partition_id out of range")

    sizes = torch.zeros(num_partitions, dtype=torch.long)
    sizes.scatter_add_(0, pid.cpu(), torch.ones(pid.numel(), dtype=torch.long))
    nonempty = sizes[sizes > 0]
    if nonempty.numel() == 0:
        balance = 0.0
    else:
        balance = float(nonempty.min().item()) / float(nonempty.max().item())

    cut = 0
    intra_count = torch.zeros(num_partitions, dtype=torch.long)
    if graph.edge_index is not None and graph.num_edges:
        ei = graph.edge_index.cpu()
        src_p = pid[ei[0]]
        dst_p = pid[ei[1]]
        cross = src_p != dst_p
        cut = int(cross.sum().item())
        intra_mask = ~cross
        if intra_mask.any():
            intra_count.scatter_add_(0, src_p[intra_mask],
                                     torch.ones(int(intra_mask.sum().item()), dtype=torch.long))

    return PartitionResult(
        partition_id=pid,
        num_partitions=int(num_partitions),
        cut_edges=cut,
        intra_edge_count=intra_count,
        partition_sizes=sizes,
        balance_ratio=balance,
        algorithm=algorithm,
        seed=seed,
        extra=dict(extra or {}),
    )


# ── Random balanced ──────────────────────────────────────────────────────────


class RandomBalancedPartitioner:
    """Random balanced partitioner.

    Shuffles node IDs deterministically (given ``seed``) and splits
    them into ``num_partitions`` near-equal groups.  Cheap and a strong
    baseline.

    Args:
        num_partitions: Number of clusters ``K`` (>= 1).
        seed: Optional RNG seed.

    Stability: Beta.
    """

    def __init__(self, num_partitions: int, seed: Optional[int] = None) -> None:
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1; got {num_partitions}")
        self.num_partitions = int(num_partitions)
        self.seed = seed

    def fit(self, graph: Graph) -> PartitionResult:
        """Partition ``graph``."""
        N = graph.num_nodes
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(int(self.seed))
        perm = torch.randperm(N, generator=gen)
        K = self.num_partitions
        pid = torch.empty(N, dtype=torch.long)
        # Round-robin assignment after shuffle keeps balance within 1.
        ranks = torch.arange(N) % K
        pid[perm] = ranks
        return _summarise(graph, pid, K, "random_balanced", self.seed)


# ── BFS partitioner ─────────────────────────────────────────────────────────


class BFSPartitioner:
    """BFS-grown partitioner.

    Picks ``num_partitions`` random seed nodes (deterministic given
    ``seed``) and grows clusters in parallel via breadth-first
    expansion.  When BFS frontiers conflict the earliest cluster wins
    (deterministic).  Disconnected nodes left over are assigned to the
    smallest cluster (round-robin) so every node ends up in exactly one
    cluster.

    Stability: Beta.
    """

    def __init__(self, num_partitions: int, seed: Optional[int] = None) -> None:
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1; got {num_partitions}")
        self.num_partitions = int(num_partitions)
        self.seed = seed

    def fit(self, graph: Graph) -> PartitionResult:
        N = graph.num_nodes
        K = self.num_partitions
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(int(self.seed))
        # Pick distinct seed nodes.
        if K > N:
            raise ValueError(f"num_partitions={K} > num_nodes={N}")
        seeds = torch.randperm(N, generator=gen)[:K].tolist()

        # Build undirected adjacency.
        adj: List[List[int]] = [[] for _ in range(N)]
        if graph.edge_index is not None and graph.num_edges:
            ei = graph.edge_index.cpu().tolist()
            for u, v in zip(ei[0], ei[1]):
                if u != v:
                    adj[u].append(v)
                    adj[v].append(u)

        pid = [-1] * N
        queues: List[deque] = []
        for k, s in enumerate(seeds):
            pid[s] = k
            q = deque([s])
            queues.append(q)

        # Round-robin BFS.
        active = True
        while active:
            active = False
            for k in range(K):
                if queues[k]:
                    u = queues[k].popleft()
                    for v in adj[u]:
                        if pid[v] == -1:
                            pid[v] = k
                            queues[k].append(v)
                            active = True

        # Assign leftover (disconnected) nodes round-robin to smallest cluster.
        sizes = [0] * K
        for p in pid:
            if p >= 0:
                sizes[p] += 1
        for i in range(N):
            if pid[i] == -1:
                k = int(min(range(K), key=lambda kk: sizes[kk]))
                pid[i] = k
                sizes[k] += 1

        pid_t = torch.tensor(pid, dtype=torch.long)
        return _summarise(graph, pid_t, K, "bfs", self.seed)


# ── Connected-component partitioner ─────────────────────────────────────────


class ConnectedComponentPartitioner:
    """Connected-component partitioner.

    Computes the connected components and assigns each component its
    own cluster id.  Optionally splits any component that exceeds
    ``max_size`` into balanced sub-clusters (random within the
    component).

    Args:
        max_size: When set, components larger than ``max_size`` are
            split into ``ceil(size / max_size)`` random-balanced
            sub-clusters.
        seed: RNG seed used only when splitting large components.

    Stability: Beta.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if max_size is not None and max_size < 1:
            raise ValueError(f"max_size must be >= 1 or None; got {max_size}")
        self.max_size = max_size
        self.seed = seed

    @staticmethod
    def _components(graph: Graph) -> Tuple[torch.Tensor, int]:
        N = graph.num_nodes
        parent = list(range(N))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        if graph.edge_index is not None and graph.num_edges:
            ei = graph.edge_index.cpu().tolist()
            for u, v in zip(ei[0], ei[1]):
                union(u, v)

        roots: Dict[int, int] = {}
        comp = torch.empty(N, dtype=torch.long)
        for i in range(N):
            r = find(i)
            if r not in roots:
                roots[r] = len(roots)
            comp[i] = roots[r]
        return comp, len(roots)

    def fit(self, graph: Graph) -> PartitionResult:
        comp, num_comp = self._components(graph)

        if self.max_size is None:
            return _summarise(
                graph, comp, num_comp, "connected_components", self.seed,
                extra={"max_size": None, "split_large": False},
            )

        # Split components larger than max_size.
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(int(self.seed))
        new_pid = comp.clone()
        next_id = int(num_comp)
        for c in range(num_comp):
            members = torch.where(comp == c)[0]
            if members.numel() <= self.max_size:
                continue
            n = members.numel()
            n_groups = (n + self.max_size - 1) // self.max_size
            perm = torch.randperm(n, generator=gen)
            for g in range(n_groups):
                start = g * self.max_size
                end = min(start + self.max_size, n)
                if g == 0:
                    target = c
                else:
                    target = next_id
                    next_id += 1
                new_pid[members[perm[start:end]]] = target
        K = int(new_pid.max().item()) + 1
        return _summarise(
            graph, new_pid, K, "connected_components", self.seed,
            extra={"max_size": self.max_size, "split_large": True},
        )


# ── Spectral partitioner ─────────────────────────────────────────────────────


class SpectralPartitioner:
    """Recursive spectral bisection (small graphs only).

    Computes the Fiedler vector of the symmetric normalised Laplacian
    and bisects on its sign.  Recursively partitions until
    ``num_partitions`` clusters are produced or splits stop yielding
    improvement.

    The spectral computation densifies the adjacency, so this is
    restricted to graphs with ``num_nodes <= max_nodes``.  Above that
    bound, the partitioner raises ``ValueError`` rather than silently
    densifying a huge graph.

    Args:
        num_partitions: Number of clusters.  Will be a power of 2 in
            practice (rounded up).
        max_nodes: Hard upper bound on input graph size (default 4096).
        seed: RNG seed (used only to break exact-zero Fiedler ties).

    Stability: Beta (small-graph only).
    """

    def __init__(
        self,
        num_partitions: int,
        max_nodes: int = 4096,
        seed: Optional[int] = None,
    ) -> None:
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1; got {num_partitions}")
        self.num_partitions = int(num_partitions)
        self.max_nodes = int(max_nodes)
        self.seed = seed

    def fit(self, graph: Graph) -> PartitionResult:
        if graph.num_nodes > self.max_nodes:
            raise ValueError(
                f"SpectralPartitioner: num_nodes={graph.num_nodes} > "
                f"max_nodes={self.max_nodes}.  Spectral partitioning densifies "
                f"the adjacency and is intended for small graphs only.  "
                f"Use BFSPartitioner or ConnectedComponentPartitioner for "
                f"larger graphs."
            )
        N = graph.num_nodes
        if N == 0:
            return _summarise(graph, torch.empty(0, dtype=torch.long), 0,
                              "spectral", self.seed)
        # Build symmetric adjacency.
        A = torch.zeros((N, N), dtype=torch.float64)
        if graph.edge_index is not None and graph.num_edges:
            ei = graph.edge_index.cpu()
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                if u != v:
                    A[u, v] = 1.0
                    A[v, u] = 1.0
        deg = A.sum(dim=1)
        # Symmetric normalised Laplacian.
        d_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
        L_sym = torch.eye(N, dtype=torch.float64) - (d_inv_sqrt.view(-1, 1) * A * d_inv_sqrt.view(1, -1))

        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        labels = torch.zeros(N, dtype=torch.long)
        # Recursive bisection up to num_partitions clusters.
        clusters_to_split = [(torch.arange(N).long(), 0)]  # (node_ids, current_label)
        current_max = 0
        while len(clusters_to_split) + (current_max + 1) - len(clusters_to_split) < self.num_partitions and clusters_to_split:
            # Pop the largest cluster to split (heuristic: balance).
            clusters_to_split.sort(key=lambda x: -x[0].numel())
            members, label = clusters_to_split.pop(0)
            if members.numel() < 2:
                continue
            sub_L = L_sym.index_select(0, members).index_select(1, members)
            try:
                eigvals, eigvecs = torch.linalg.eigh(sub_L)
            except Exception:
                continue
            if eigvecs.shape[1] < 2:
                continue
            fiedler = eigvecs[:, 1]
            # Tie-break sign-zero entries with deterministic noise.
            tiny = torch.rand(members.numel(), dtype=torch.float64, generator=gen) * 1e-12
            fiedler = fiedler + tiny
            mask_a = fiedler >= 0
            mask_b = ~mask_a
            if int(mask_a.sum().item()) == 0 or int(mask_b.sum().item()) == 0:
                # Bisection failed (single eigenvalue cluster).
                continue
            current_max += 1
            new_label = current_max
            labels[members[mask_b]] = new_label
            clusters_to_split.append((members[mask_a], label))
            clusters_to_split.append((members[mask_b], new_label))
            if (current_max + 1) >= self.num_partitions:
                break
        K = int(labels.max().item()) + 1
        return _summarise(graph, labels, K, "spectral", self.seed,
                          extra={"max_nodes": self.max_nodes})


# ── ClusterLoader ────────────────────────────────────────────────────────────


class ClusterLoader:
    """Cluster-GCN-style mini-batch loader.

    Iterates over partitions and yields the induced subgraph for each
    (or for a small group of partitions when
    ``num_clusters_per_batch > 1``, which matches the Cluster-GCN
    "stochastic clustering" trick that recovers some inter-cluster
    edges).

    Args:
        graph: Source graph.
        partition: A :class:`PartitionResult`.
        num_clusters_per_batch: Number of clusters merged per batch.
        shuffle: Shuffle cluster order each epoch.
        seed: RNG seed for shuffle.
        drop_last: Drop the last incomplete batch.

    Yields:
        Tuples ``(subgraph, cluster_ids)`` where ``cluster_ids`` is a
        list of partition IDs covered by this batch.

    Stability: Beta.
    """

    def __init__(
        self,
        graph: Graph,
        partition: PartitionResult,
        num_clusters_per_batch: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        if num_clusters_per_batch < 1:
            raise ValueError(
                f"num_clusters_per_batch must be >= 1; got {num_clusters_per_batch}"
            )
        self.graph = graph
        self.partition = partition
        self.batch = int(num_clusters_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)

    def _order(self) -> List[int]:
        K = self.partition.num_partitions
        ids = list(range(K))
        if self.shuffle:
            gen = torch.Generator()
            if self.seed is not None:
                gen.manual_seed(int(self.seed))
            perm = torch.randperm(K, generator=gen).tolist()
            ids = perm
        return ids

    def __iter__(self) -> Iterator[Tuple[Graph, List[int]]]:
        K = self.partition.num_partitions
        order = self._order()
        pid = self.partition.partition_id
        device = self.graph.node_features.device
        i = 0
        while i < K:
            j = min(i + self.batch, K)
            cluster_ids = order[i:j]
            if len(cluster_ids) < self.batch and self.drop_last:
                break
            mask = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            for c in cluster_ids:
                mask |= (pid == c)
            node_ids = torch.where(mask)[0].to(device)
            sub = induced_subgraph(self.graph, node_ids, relabel_nodes=True)
            meta = dict(sub.metadata) if isinstance(sub.metadata, dict) else {}
            cluster_meta = {
                "algorithm": self.partition.algorithm,
                "cluster_ids": list(cluster_ids),
                "num_partitions": int(self.partition.num_partitions),
            }
            meta["cluster_gcn"] = cluster_meta
            sub.metadata = meta
            yield sub, list(cluster_ids)
            i = j

    def __len__(self) -> int:
        K = self.partition.num_partitions
        if self.drop_last:
            return K // self.batch
        return (K + self.batch - 1) // self.batch
