"""Tests for graph generators, spectral analysis, label propagation, and embeddings."""
import pytest
import torch
from tgraphx.mining import (
    erdos_renyi_graph, barabasi_albert_graph, stochastic_block_model_graph,
    watts_strogatz_graph, complete_graph, cycle_graph, path_graph, star_graph,
    karate_club_graph, synthetic_anomaly_graph, motif_injected_graph,
    graph_laplacian, normalized_laplacian, laplacian_eigenvalues,
    fiedler_vector, algebraic_connectivity, laplacian_eigvec_positional_encoding,
    spectral_clustering, spectral_distance, dirichlet_energy,
    label_propagation, LabelPropagationClassifier,
    extract_node_embeddings, extract_graph_embeddings,
    embedding_similarity_matrix, embedding_pairwise_distances,
    embedding_nearest_neighbors,
    analyze_graph,
)


class TestGenerators:
    def test_erdos_renyi_deterministic(self):
        ei1, N1 = erdos_renyi_graph(20, 0.3, seed=0)
        ei2, N2 = erdos_renyi_graph(20, 0.3, seed=0)
        assert torch.equal(ei1, ei2) and N1 == N2 == 20

    def test_erdos_renyi_density(self):
        """Dense ER graph should have more edges than sparse one."""
        ei_dense, _ = erdos_renyi_graph(20, 0.8, seed=42)
        ei_sparse, _ = erdos_renyi_graph(20, 0.1, seed=42)
        assert ei_dense.size(1) > ei_sparse.size(1)

    def test_barabasi_albert_connects(self):
        ei, N = barabasi_albert_graph(20, m=2, seed=0)
        assert N == 20
        assert ei.size(1) > 0

    def test_sbm_communities_larger_p_in(self):
        """SBM with higher p_in should have more intra-community edges."""
        ei, N, labels = stochastic_block_model_graph([5, 5], p_in=0.8, p_out=0.05, seed=0)
        intra = sum(int(labels[ei[0,k]]) == int(labels[ei[1,k]]) for k in range(ei.size(1)))
        inter = ei.size(1) - intra
        assert intra > inter

    def test_watts_strogatz_shape(self):
        ei, N = watts_strogatz_graph(10, k=4, p=0.1, seed=0)
        assert N == 10

    def test_complete_graph_edges(self):
        ei, N = complete_graph(4)
        assert ei.size(1) == 4 * 3  # 12 directed edges in K4

    def test_cycle_graph_edges(self):
        ei, N = cycle_graph(5)
        assert ei.size(1) == 10  # 5 undirected * 2 directions

    def test_path_graph_edges(self):
        ei, N = path_graph(5)
        assert ei.size(1) == 8  # 4 undirected edges

    def test_star_graph_hub_degree(self):
        ei, N = star_graph(5)
        deg = torch.zeros(N, dtype=torch.long)
        ones = torch.ones(ei.size(1), dtype=torch.long)
        deg.scatter_add_(0, ei[0], ones)
        assert int(deg[0]) == 4  # hub has degree 4

    def test_karate_club(self):
        ei, N = karate_club_graph()
        assert N == 34

    def test_anomaly_graph_mask_matches(self):
        ei, N, mask = synthetic_anomaly_graph(20, num_anomalous=3, seed=0)
        assert int(mask.sum()) == 3

    def test_motif_injected(self):
        ei, N, tri_nodes = motif_injected_graph(20, num_triangles=2, seed=0)
        assert tri_nodes.numel() == 6


class TestSpectral:
    def test_laplacian_positive_semidefinite(self):
        """Laplacian eigenvalues should be >= 0."""
        ei, N = path_graph(5)
        L = graph_laplacian(ei, N)
        evals = torch.linalg.eigvalsh(L)
        assert (evals >= -1e-6).all()

    def test_laplacian_row_sums_zero(self):
        """L * 1 = 0 (row sums are zero)."""
        ei, N = complete_graph(4)
        L = graph_laplacian(ei, N)
        row_sums = L.sum(dim=1)
        assert (row_sums.abs() < 1e-5).all()

    def test_normalized_laplacian_spectrum_range(self):
        """Normalised Laplacian eigenvalues should be in [0, 2]."""
        ei, N = cycle_graph(6)
        L_norm = normalized_laplacian(ei, N)
        evals = torch.linalg.eigvalsh(L_norm)
        assert float(evals.min()) >= -1e-6
        assert float(evals.max()) <= 2.0 + 1e-6

    def test_fiedler_vector_orthogonal_to_ones(self):
        """Fiedler vector should be approximately orthogonal to the all-ones vector."""
        ei, N = path_graph(6)
        fv, lam2 = fiedler_vector(ei, N)
        dot = float(fv.sum().abs().item())
        assert dot < 0.1

    def test_algebraic_connectivity_zero_disconnected(self):
        ei = torch.tensor([[0,1],[1,0]], dtype=torch.long)  # 2 isolated components
        N = 4  # nodes 2,3 are isolated
        lam2 = algebraic_connectivity(ei, N)
        assert lam2 < 1e-6  # disconnected → λ₂ = 0

    def test_algebraic_connectivity_positive_connected(self):
        ei, N = complete_graph(4)
        lam2 = algebraic_connectivity(ei, N)
        assert lam2 > 0

    def test_laplacian_pe_shape(self):
        ei, N = path_graph(8)
        enc = laplacian_eigvec_positional_encoding(ei, N, k=4)
        assert enc.shape == (N, 4)
        assert enc.dtype == torch.float32

    def test_spectral_clustering_shape(self):
        ei, N, labels = stochastic_block_model_graph([4, 4], p_in=0.8, p_out=0.05, seed=0)
        cluster_labels = spectral_clustering(ei, N, num_clusters=2, seed=0)
        assert cluster_labels.shape == (N,)
        assert set(cluster_labels.tolist()) <= {0, 1}

    def test_spectral_distance_zero_same_graph(self):
        ei, N = path_graph(5)
        d = spectral_distance(ei, N, ei, N)
        assert d < 1e-5

    def test_spectral_distance_positive_different(self):
        ei1, N1 = path_graph(5)
        ei2, N2 = complete_graph(5)
        d = spectral_distance(ei1, N1, ei2, N2)
        assert d > 0

    def test_dirichlet_energy_smooth(self):
        """Constant node features should have zero Dirichlet energy."""
        ei, N = path_graph(5)
        x = torch.ones(N, 4)
        e = dirichlet_energy(x, ei, N)
        assert abs(e) < 1e-5

    def test_spectral_size_guard(self):
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes"):
            graph_laplacian(ei, 2001, max_nodes=2000)


class TestLabelPropagation:
    def _two_clique_data(self):
        """Two connected cliques + labels only on clique centers."""
        # Clique 1: 0,1,2  Clique 2: 3,4,5
        src = [0,0,1,3,3,4,0,1,2,3,4,5]
        dst = [1,2,2,4,5,5,0,1,2,3,4,5]
        ei = torch.tensor([src+dst, dst+src], dtype=torch.long)
        ei = torch.unique(ei, dim=1)
        N = 6
        y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        # Only reveal label for node 0 and node 3.
        mask = torch.tensor([True, False, False, True, False, False])
        return ei, N, y, mask

    def test_propagates_correct_labels(self):
        ei, N, y, mask = self._two_clique_data()
        Z = label_propagation(ei, N, y, mask, num_classes=2)
        preds = Z.argmax(dim=1)
        # Nodes 1,2 should get class 0; nodes 4,5 should get class 1.
        assert int(preds[1]) == 0 and int(preds[2]) == 0
        assert int(preds[4]) == 1 and int(preds[5]) == 1

    def test_output_shape(self):
        ei, N, y, mask = self._two_clique_data()
        Z = label_propagation(ei, N, y, mask, num_classes=2)
        assert Z.shape == (N, 2)

    def test_labeled_nodes_stay_close_to_seed(self):
        """Labeled nodes should keep high probability for their own class."""
        ei, N, y, mask = self._two_clique_data()
        Z = label_propagation(ei, N, y, mask, num_classes=2, alpha=0.5)
        # Node 0 labeled class 0 — still high probability class 0.
        assert float(Z[0, 0]) >= float(Z[0, 1])

    def test_classifier_api(self):
        ei, N, y, mask = self._two_clique_data()
        clf = LabelPropagationClassifier(alpha=0.9)
        Z = clf.fit_predict(ei, N, y, mask, num_classes=2)
        preds = clf.predict()
        assert preds.shape == (N,)
        assert Z.shape == (N, 2)

    def test_deterministic(self):
        ei, N, y, mask = self._two_clique_data()
        Z1 = label_propagation(ei, N, y, mask, num_classes=2)
        Z2 = label_propagation(ei, N, y, mask, num_classes=2)
        assert torch.equal(Z1, Z2)


class TestEmbeddings:
    def test_extract_node_embeddings_shape(self):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        # Model that ignores edge_index for simplicity.
        class _SimpleModel(nn.Module):
            def __init__(self): super().__init__(); self.lin = nn.Linear(4, 8)
            def forward(self, x, ei): return self.lin(x)
        m = _SimpleModel()
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        x = torch.randn(3, 4)
        emb = extract_node_embeddings(m, ei, x, no_grad=True)
        assert emb.shape == (3, 8)
        assert not emb.requires_grad

    def test_extract_graph_embeddings_shape(self):
        class _M(torch.nn.Module):
            def __init__(self): super().__init__(); self.lin = torch.nn.Linear(4, 6)
            def forward(self, x, ei): return self.lin(x)
        m = _M()
        graphs = [
            {"node_features": torch.randn(3, 4), "edge_index": torch.tensor([[0],[1]], dtype=torch.long)},
            {"node_features": torch.randn(4, 4), "edge_index": torch.tensor([[0,1],[1,2]], dtype=torch.long)},
        ]
        embs = extract_graph_embeddings(m, graphs, pooling="mean")
        assert embs.shape == (2, 6)

    def test_embedding_similarity_cosine_symmetric(self):
        emb = torch.randn(5, 8)
        S = embedding_similarity_matrix(emb, method="cosine")
        assert S.shape == (5, 5)
        assert torch.allclose(S, S.t(), atol=1e-5)
        # Diagonal should be ~1.
        assert (S.diagonal() - 1.0).abs().max() < 1e-4

    def test_embedding_pairwise_distances_non_negative(self):
        emb = torch.randn(4, 8)
        D = embedding_pairwise_distances(emb)
        assert (D >= 0).all()
        assert (D.diagonal() < 1e-5).all()

    def test_embedding_nearest_neighbors(self):
        emb = torch.eye(5)  # orthogonal — nearest is self
        idx, scores = embedding_nearest_neighbors(emb, emb, k=2, method="cosine")
        assert idx.shape == (5, 2)
        # First nearest neighbour of each is itself.
        assert (idx[:, 0] == torch.arange(5)).all()

    def test_analyze_graph_runs(self):
        ei, N = path_graph(8)
        result = analyze_graph(ei, N, include_spectral=True)
        assert "summary" in result
        assert "motifs" in result
        assert "centrality" in result
        assert isinstance(result["summary"], dict)
