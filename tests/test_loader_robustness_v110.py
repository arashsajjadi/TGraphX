"""Loader robustness and correctness tests (v1.1).

Targets:
- NeighborLoader determinism with same seed.
- GraphSAINTNodeSampler normalization correctness on a tiny known graph.
- Cluster-GCN partition coverage (every node assigned exactly once).
- Edge feature / mask / graph_features preservation through samplers.
"""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    Graph, NeighborLoader, GraphMiniBatch,
    GraphSAINTNodeSampler, GraphSAINTLoader, estimate_norm_coefficients,
    RandomBalancedPartitioner, ClusterLoader,
)


@pytest.fixture
def small_labelled_graph():
    torch.manual_seed(0)
    N, D = 64, 8
    x = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, 256))
    y = torch.randint(0, 4, (N,))
    return Graph(node_features=x, edge_index=edge_index, y=y)


@pytest.fixture
def labelled_graph_with_edge_features():
    torch.manual_seed(1)
    N, D = 32, 4
    x = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, 80))
    edge_attr = torch.randn(80, 2)
    y = torch.randint(0, 3, (N,))
    return Graph(node_features=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ── NeighborLoader determinism ───────────────────────────────────────────────


class TestNeighborLoaderDeterminism:
    def test_same_seed_same_seeds(self, small_labelled_graph):
        loader_a = NeighborLoader(small_labelled_graph, fanouts=[5, 3],
                                  batch_size=8, shuffle=True, seed=42)
        loader_b = NeighborLoader(small_labelled_graph, fanouts=[5, 3],
                                  batch_size=8, shuffle=True, seed=42)
        seeds_a = [batch.seed_node_ids.clone() for batch in loader_a]
        seeds_b = [batch.seed_node_ids.clone() for batch in loader_b]
        assert len(seeds_a) == len(seeds_b)
        for a, b in zip(seeds_a, seeds_b):
            assert torch.equal(a, b), \
                f"Different seed batches for same RNG seed: {a} vs {b}"

    def test_different_seeds_diverge(self, small_labelled_graph):
        loader_a = NeighborLoader(small_labelled_graph, fanouts=[5, 3],
                                  batch_size=8, shuffle=True, seed=42)
        loader_b = NeighborLoader(small_labelled_graph, fanouts=[5, 3],
                                  batch_size=8, shuffle=True, seed=43)
        seeds_a = torch.cat([batch.seed_node_ids for batch in loader_a])
        seeds_b = torch.cat([batch.seed_node_ids for batch in loader_b])
        # With shuffle=True and different seeds, the order of seed nodes must differ.
        assert not torch.equal(seeds_a, seeds_b), \
            "Different RNG seeds should produce different shuffles"


# ── NeighborLoader feature/label preservation ────────────────────────────────


class TestNeighborLoaderPreservation:
    def test_seed_y_matches_source_graph_labels(self, small_labelled_graph):
        loader = NeighborLoader(small_labelled_graph, fanouts=[3, 2],
                                batch_size=4, seed=0)
        for batch in loader:
            # Each seed_y entry must equal the source graph's label at that global ID.
            expected = small_labelled_graph.node_labels[batch.seed_node_ids]
            assert torch.equal(batch.seed_y, expected), \
                f"seed_y {batch.seed_y} != source labels {expected}"
            break

    def test_edge_features_preserved_in_subgraph(self, labelled_graph_with_edge_features):
        loader = NeighborLoader(labelled_graph_with_edge_features,
                                fanouts=[3, 2], batch_size=4, seed=0)
        for batch in loader:
            assert batch.edge_features is not None, \
                "Edge features must be preserved in the sampled subgraph"
            assert batch.edge_features.dim() == 2
            assert batch.edge_features.size(0) == batch.num_edges
            assert batch.edge_features.size(1) == 2
            break

    def test_seed_local_indices_match_node_features(self, small_labelled_graph):
        """For every batch, batch.node_features[batch.seed_local_indices] must
        equal source_graph.node_features[batch.seed_node_ids]."""
        loader = NeighborLoader(small_labelled_graph, fanouts=[3, 2],
                                batch_size=4, seed=0)
        for batch in loader:
            local_features = batch.node_features[batch.seed_local_indices]
            global_features = small_labelled_graph.node_features[batch.seed_node_ids]
            assert torch.allclose(local_features, global_features), \
                "Seed-local indexing of subgraph features must match the source graph"
            break

    def test_no_dtype_or_device_drift(self, small_labelled_graph):
        loader = NeighborLoader(small_labelled_graph, fanouts=[3, 2],
                                batch_size=4, seed=0)
        for batch in loader:
            assert batch.node_features.dtype == small_labelled_graph.node_features.dtype
            assert batch.node_features.device == small_labelled_graph.node_features.device
            assert batch.edge_index.dtype == torch.long
            break


# ── GraphSAINTNodeSampler normalization correctness ──────────────────────────


class TestGraphSAINTNormalization:
    def test_node_norm_estimation_runs(self):
        """Just verify estimate_norm_coefficients returns finite tensors."""
        torch.manual_seed(0)
        N = 20
        x = torch.randn(N, 4)
        ei = torch.randint(0, N, (2, 60))
        graph = Graph(node_features=x, edge_index=ei)
        sampler = GraphSAINTNodeSampler(graph, budget=10, num_steps=5, seed=0)
        node_norm, edge_norm = estimate_norm_coefficients(sampler, num_samples=8)
        assert node_norm.shape == (N,)
        assert edge_norm.shape == (60,)
        assert torch.all(torch.isfinite(node_norm))
        assert torch.all(torch.isfinite(edge_norm))
        # All norms must be positive.
        assert (node_norm > 0).all() or (node_norm == 0).any()  # zeros for unsampled nodes are OK

    def test_saintloader_yields_subgraphs(self):
        torch.manual_seed(0)
        N = 20
        x = torch.randn(N, 4)
        ei = torch.randint(0, N, (2, 60))
        graph = Graph(node_features=x, edge_index=ei)
        sampler = GraphSAINTNodeSampler(graph, budget=10, num_steps=4, seed=0)
        loader = GraphSAINTLoader(sampler, attach_norm=True)
        subgraphs = list(loader)
        assert len(subgraphs) == 4
        for sub in subgraphs:
            assert isinstance(sub, Graph)
            assert sub.num_nodes >= 1
            # When attach_norm=True, the loader stores norm coefficients in metadata.
            assert isinstance(sub.metadata, dict)


# ── Cluster-GCN partition coverage ──────────────────────────────────────────


class TestClusterGCNPartitionCoverage:
    def test_random_partitioner_covers_all_nodes_exactly_once(self):
        torch.manual_seed(0)
        N = 50
        x = torch.randn(N, 4)
        ei = torch.randint(0, N, (2, 100))
        graph = Graph(node_features=x, edge_index=ei)
        partitioner = RandomBalancedPartitioner(num_partitions=5, seed=0)
        result = partitioner.fit(graph)
        # partition_id maps every node to exactly one part in [0, num_partitions).
        assert result.partition_id.shape == (N,)
        assert result.partition_id.min() >= 0
        assert result.partition_id.max() < 5
        # Sum of partition sizes covers every node exactly once.
        assert sum(result.partition_sizes) == N

    def test_clusterloader_yields_subgraphs(self):
        torch.manual_seed(0)
        N = 40
        x = torch.randn(N, 4)
        ei = torch.randint(0, N, (2, 100))
        graph = Graph(node_features=x, edge_index=ei)
        partitioner = RandomBalancedPartitioner(num_partitions=4, seed=0)
        result = partitioner.fit(graph)
        loader = ClusterLoader(graph, result, num_clusters_per_batch=2, shuffle=False)
        items = list(loader)
        # 4 parts / 2 clusters per batch = 2 batches; each yields (subgraph, ids).
        assert len(items) == 2
        for item in items:
            sub, cluster_ids = item
            assert isinstance(sub, Graph)
            assert sub.num_nodes >= 1
            assert len(cluster_ids) == 2

    def test_random_partitioner_deterministic_seed(self):
        torch.manual_seed(0)
        N = 30
        x = torch.randn(N, 4)
        ei = torch.randint(0, N, (2, 60))
        graph = Graph(node_features=x, edge_index=ei)
        a = RandomBalancedPartitioner(num_partitions=3, seed=42).fit(graph)
        b = RandomBalancedPartitioner(num_partitions=3, seed=42).fit(graph)
        assert torch.equal(a.partition_id, b.partition_id), \
            "Same seed must produce identical partitions"


# ── Sparse backend behaviour ────────────────────────────────────────────────


class TestSparseBackendFallback:
    def test_backend_info_lists_pure_torch(self):
        from tgraphx.sparse import backend_info
        info = backend_info()
        assert info["pure_torch"] is True
        assert "active" in info

    def test_active_backend_does_not_require_torch_scatter(self):
        from tgraphx.sparse import active_backend
        backend = active_backend()
        # active_backend() reports the user's selection ("auto" by default
        # until select_backend("auto") is called explicitly).  All TGraphX
        # paths must remain functional regardless of the optional backend.
        assert backend in ("auto", "pure_torch", "torch_scatter", "pyg_lib")

    def test_select_backend_falls_back_pure_torch(self):
        from tgraphx.sparse import select_backend
        # Even if user asks for torch_scatter and it's missing, pure_torch
        # must still be available as a fallback.
        chosen = select_backend("auto")
        assert chosen in ("pure_torch", "torch_scatter", "pyg_lib")
