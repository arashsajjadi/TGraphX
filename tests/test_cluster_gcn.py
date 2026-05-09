"""Tests for Cluster-GCN partitioners and ClusterLoader."""
from __future__ import annotations

import torch

from tgraphx import Graph
from tgraphx.cluster_gcn import (
    RandomBalancedPartitioner,
    BFSPartitioner,
    ConnectedComponentPartitioner,
    SpectralPartitioner,
    ClusterLoader,
)


def _disk_union_graph(n_per_disk=10, n_disks=3, seed=0):
    """Three disconnected cliques."""
    torch.manual_seed(seed)
    N = n_per_disk * n_disks
    src, dst = [], []
    for d in range(n_disks):
        base = d * n_per_disk
        for i in range(n_per_disk):
            for j in range(n_per_disk):
                if i != j:
                    src.append(base + i)
                    dst.append(base + j)
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(N, 5)
    return Graph(node_features=x, edge_index=ei)


def test_random_balanced_partitions_complete_and_disjoint():
    g = _disk_union_graph()
    part = RandomBalancedPartitioner(num_partitions=3, seed=0).fit(g)
    assert part.num_partitions == 3
    # Every node assigned exactly once; sizes within ±1.
    assert part.partition_id.numel() == g.num_nodes
    sizes = part.partition_sizes.tolist()
    assert max(sizes) - min(sizes) <= 1


def test_bfs_partitioner_full_coverage():
    g = _disk_union_graph()
    part = BFSPartitioner(num_partitions=3, seed=0).fit(g)
    assert part.partition_id.numel() == g.num_nodes
    # Every cluster id in [0, 3)
    assert int(part.partition_id.min().item()) >= 0
    assert int(part.partition_id.max().item()) < part.num_partitions


def test_connected_components_isolates_disks():
    g = _disk_union_graph(n_per_disk=4, n_disks=3)
    part = ConnectedComponentPartitioner().fit(g)
    # Three disconnected disks ⇒ at least 3 components.
    assert part.num_partitions == 3
    # No cut edges across components.
    assert part.cut_edges == 0


def test_spectral_partitioner_small_graph():
    # Small graph to ensure spectral runs.
    g = _disk_union_graph(n_per_disk=3, n_disks=2)
    part = SpectralPartitioner(num_partitions=2, seed=0).fit(g)
    assert part.num_partitions >= 1
    # Every node assigned.
    assert part.partition_id.numel() == g.num_nodes


def test_spectral_partitioner_large_graph_raises():
    g = _disk_union_graph(n_per_disk=200, n_disks=3)  # 600 nodes
    try:
        SpectralPartitioner(num_partitions=4, max_nodes=100).fit(g)
    except ValueError as e:
        assert "max_nodes" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_cluster_loader_iteration():
    g = _disk_union_graph()
    part = ConnectedComponentPartitioner().fit(g)
    loader = ClusterLoader(g, part, num_clusters_per_batch=1, shuffle=False)
    batches = list(iter(loader))
    assert len(batches) == part.num_partitions
    for sub, ids in batches:
        assert sub.num_nodes > 0
        assert "cluster_gcn" in sub.metadata


def test_cluster_loader_multi_cluster_batch():
    g = _disk_union_graph()
    part = ConnectedComponentPartitioner().fit(g)
    loader = ClusterLoader(g, part, num_clusters_per_batch=2, shuffle=False)
    sub, ids = next(iter(loader))
    assert len(ids) == 2
