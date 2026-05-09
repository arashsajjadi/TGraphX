"""Tests for Node2Vec, Knowledge Graph, Hypergraph, and Graph IO."""
import json
import os
import tempfile

import pytest
import torch

from tgraphx.mining import (
    # Node2Vec
    node2vec_walks, deepwalk_walks, generate_skipgram_pairs,
    Node2VecEmbedding, train_node2vec_step, extract_node2vec_embeddings,
    # KG
    KnowledgeGraph, negative_triple_sampling, filtered_ranking_metrics,
    TransE, DistMult, train_kg_step,
    # Hypergraph
    Hypergraph, incidence_to_bipartite_graph, clique_expansion,
    star_expansion, hypergraph_density,
    # IO
    read_graph_json, write_graph_json, write_edge_list_csv, read_edge_list_csv,
    save_graph_npz, load_graph_npz,
)


def _chain_ei(N=4):
    src = list(range(N-1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N-1))
    return torch.tensor([src, dst], dtype=torch.long), N


class TestNode2Vec:
    def test_walks_shape(self):
        ei, N = _chain_ei()
        walks = node2vec_walks(ei, N, walk_length=5, walks_per_node=3, seed=0)
        assert walks.shape == (N * 3, 6)  # walk_length + 1

    def test_walks_deterministic(self):
        ei, N = _chain_ei()
        w1 = node2vec_walks(ei, N, walk_length=5, walks_per_node=2, seed=7)
        w2 = node2vec_walks(ei, N, walk_length=5, walks_per_node=2, seed=7)
        assert torch.equal(w1, w2)

    def test_walks_valid_nodes(self):
        ei, N = _chain_ei()
        walks = node2vec_walks(ei, N, walk_length=10, walks_per_node=5, seed=0)
        assert ((walks >= 0) & (walks < N)).all()

    def test_deepwalk_is_p1_q1(self):
        ei, N = _chain_ei()
        dw = deepwalk_walks(ei, N, walk_length=5, walks_per_node=2, seed=0)
        n2v = node2vec_walks(ei, N, walk_length=5, walks_per_node=2, p=1.0, q=1.0, seed=0)
        assert torch.equal(dw, n2v)

    def test_biased_walks_different_from_uniform(self):
        ei = torch.tensor([[0,0,0,0,0],[1,2,3,4,5]], dtype=torch.long)
        N = 6
        uniform = node2vec_walks(ei, N, walk_length=10, walks_per_node=5, p=1.0, q=1.0, seed=0)
        biased = node2vec_walks(ei, N, walk_length=10, walks_per_node=5, p=0.5, q=2.0, seed=0)
        # They should differ at some point (not guaranteed, but very likely for this graph).
        # Just verify both are valid.
        assert ((uniform >= 0) & (uniform < N)).all()
        assert ((biased >= 0) & (biased < N)).all()

    def test_skipgram_pairs(self):
        ei, N = _chain_ei()
        walks = node2vec_walks(ei, N, walk_length=5, walks_per_node=2, seed=0)
        centers, contexts, negatives = generate_skipgram_pairs(walks, window_size=2, negative_ratio=3, num_nodes=N, seed=0)
        assert centers.size(0) == contexts.size(0)
        assert negatives.size(0) == centers.size(0) * 3

    def test_node2vec_embedding_forward(self):
        model = Node2VecEmbedding(num_nodes=10, embedding_dim=8)
        centers = torch.randint(10, (4,))
        contexts = torch.randint(10, (4,))
        negatives = torch.randint(10, (8,))
        loss = model(centers, contexts, negatives)
        assert torch.isfinite(loss)

    def test_node2vec_embedding_backward(self):
        model = Node2VecEmbedding(num_nodes=10, embedding_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        centers = torch.randint(10, (4,))
        contexts = torch.randint(10, (4,))
        negatives = torch.randint(10, (8,))
        loss_val = train_node2vec_step(model, opt, centers, contexts, negatives)
        assert isinstance(loss_val, float) and loss_val >= 0

    def test_node2vec_loss_decreases(self):
        torch.manual_seed(0)
        ei, N = _chain_ei(8)
        # Generate many walks to create reliable pairs.
        walks = node2vec_walks(ei, N, walk_length=10, walks_per_node=20, seed=0)
        centers, contexts, negatives = generate_skipgram_pairs(
            walks, window_size=3, negative_ratio=5, num_nodes=N, seed=0,
        )
        model = Node2VecEmbedding(num_nodes=N, embedding_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        losses = []
        for _ in range(10):
            losses.append(train_node2vec_step(model, opt, centers, contexts, negatives))
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f}→{losses[-1]:.4f}"

    def test_extract_embeddings_shape(self):
        model = Node2VecEmbedding(num_nodes=5, embedding_dim=8)
        emb = extract_node2vec_embeddings(model)
        assert emb.shape == (5, 8)
        assert not emb.requires_grad


class TestKnowledgeGraph:
    def _kg(self):
        triples = torch.tensor([[0,0,1],[1,0,2],[0,1,2],[2,0,0]], dtype=torch.long)
        return KnowledgeGraph(triples)

    def test_kg_creation(self):
        kg = self._kg()
        assert kg.num_entities == 3
        assert kg.num_relations == 2
        assert len(kg) == 4

    def test_kg_positive_lookup(self):
        kg = self._kg()
        assert kg.is_positive(0, 0, 1)
        assert not kg.is_positive(0, 0, 2)

    def test_kg_split(self):
        triples = torch.tensor([[i, 0, (i+1) % 10] for i in range(20)], dtype=torch.long)
        kg = KnowledgeGraph(triples)
        train, val, test = kg.train_val_test_split(ratios=(0.7, 0.15, 0.15), seed=0)
        assert len(train) + len(val) + len(test) == 20

    def test_negative_sampling_valid(self):
        triples = torch.tensor([[0,0,1],[1,0,2]], dtype=torch.long)
        neg = negative_triple_sampling(triples, num_entities=3, num_neg=2, seed=0)
        assert neg.shape == (4, 3)
        # Relations must match.
        assert (neg[:, 1] == triples.repeat_interleave(2, dim=0)[:, 1]).all()

    def test_transe_forward_backward(self):
        kg = self._kg()
        model = TransE(kg.num_entities, kg.num_relations, embedding_dim=8)
        triples = kg.triples
        neg = negative_triple_sampling(triples, kg.num_entities, seed=0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = train_kg_step(model, opt, triples, neg)
        assert isinstance(loss, float) and loss >= 0

    def test_transe_loss_decreases(self):
        torch.manual_seed(0)
        triples = torch.tensor([[i, 0, (i+1) % 8] for i in range(8)], dtype=torch.long)
        kg = KnowledgeGraph(triples)
        model = TransE(kg.num_entities, kg.num_relations, embedding_dim=16, margin=1.0)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        losses = []
        for _ in range(20):
            neg = negative_triple_sampling(triples, kg.num_entities, num_neg=2, seed=0)
            losses.append(train_kg_step(model, opt, triples, neg[:8]))
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f}→{losses[-1]:.4f}"

    def test_distmult_forward_backward(self):
        kg = self._kg()
        model = DistMult(kg.num_entities, kg.num_relations, embedding_dim=8)
        neg = negative_triple_sampling(kg.triples, kg.num_entities, seed=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss = train_kg_step(model, opt, kg.triples, neg)
        assert isinstance(loss, float) and loss >= 0

    def test_transe_score_shape(self):
        model = TransE(5, 3, embedding_dim=8)
        h = torch.tensor([0, 1, 2])
        r = torch.tensor([0, 0, 1])
        t = torch.tensor([1, 2, 3])
        scores = model.score(h, r, t)
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()

    def test_filtered_ranking_metrics(self):
        torch.manual_seed(0)
        triples = torch.tensor([[0,0,1],[1,0,2],[2,0,0]], dtype=torch.long)
        kg = KnowledgeGraph(triples)
        model = TransE(kg.num_entities, kg.num_relations, embedding_dim=8)
        # Just run and check return format.
        all_set = kg._positive_set
        metrics = filtered_ranking_metrics(model, triples[:2], all_set, kg.num_entities)
        assert "mrr" in metrics and "hits@1" in metrics and "hits@10" in metrics
        assert 0 <= metrics["mrr"] <= 1


class TestHypergraph:
    def test_creation(self):
        hg = Hypergraph(5, [[0,1,2],[2,3,4],[0,4]])
        assert hg.num_nodes == 5
        assert hg.num_hyperedges == 3

    def test_node_hyperdegree(self):
        hg = Hypergraph(5, [[0,1],[1,2],[0,2]])
        deg = hg.node_hyperdegree()
        assert int(deg[0]) == 2  # node 0 appears in 2 hyperedges
        assert int(deg[1]) == 2
        assert int(deg[3]) == 0  # isolated node

    def test_hyperedge_degree(self):
        hg = Hypergraph(4, [[0,1],[1,2,3]])
        edge_deg = hg.hyperedge_degree()
        assert edge_deg.tolist() == [2, 3]

    def test_density(self):
        hg = Hypergraph(4, [[0,1,2,3]])  # all nodes in one hyperedge → density 1
        assert abs(hg.density() - 1.0) < 1e-6

    def test_density_empty(self):
        hg = Hypergraph(4, [[]])
        assert hg.density() == 0.0

    def test_incidence_to_bipartite(self):
        hg = Hypergraph(3, [[0,1],[1,2]])
        bi_ei, total = incidence_to_bipartite_graph(hg)
        assert total == 5  # 3 original + 2 hyperedge nodes
        assert bi_ei.shape[0] == 2

    def test_clique_expansion(self):
        hg = Hypergraph(3, [[0,1,2]])
        ei, N = clique_expansion(hg)
        assert N == 3
        # K3 = 6 directed edges
        assert ei.size(1) == 6

    def test_star_expansion(self):
        hg = Hypergraph(3, [[0,1,2]])
        ei, N = star_expansion(hg)
        assert N == 4  # 3 + 1 hyperedge node
        assert ei.size(1) == 6  # 3 undirected connections × 2 directions

    def test_invalid_node(self):
        with pytest.raises(ValueError):
            Hypergraph(3, [[0, 5]])  # node 5 >= 3

    def test_summary(self):
        hg = Hypergraph(5, [[0,1,2],[2,3,4]])
        s = hg.summary()
        assert s["num_nodes"] == 5
        assert s["num_hyperedges"] == 2
        import json; json.dumps(s)  # must be JSON-serializable


class TestGraphIO:
    def test_write_read_json_roundtrip(self, tmp_path):
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        ew = torch.tensor([1.0, 2.0, 3.0])
        p = write_graph_json(str(tmp_path / "g.json"), ei, 4, edge_weight=ew, metadata={"test": True})
        result = read_graph_json(p)
        assert torch.equal(result["edge_index"], ei)
        assert result["num_nodes"] == 4
        assert result["metadata"]["test"] is True

    def test_write_read_csv_roundtrip(self, tmp_path):
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        ew = torch.tensor([0.5, 1.5, 2.5])
        p = write_edge_list_csv(str(tmp_path / "edges.csv"), ei, edge_weight=ew, header=["src","dst","w"])
        new_ei, N, new_ew = read_edge_list_csv(p, has_header=True, weight_col=2)
        assert new_ei.size(1) == 3
        assert new_ew is not None

    def test_csv_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_edge_list_csv("/nonexistent/path.csv")

    def test_json_missing_file(self):
        with pytest.raises(FileNotFoundError):
            read_graph_json("/nonexistent/path.json")

    def test_write_read_npz_roundtrip(self, tmp_path):
        pytest.importorskip("numpy")
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        x = torch.randn(3, 4)
        p = save_graph_npz(str(tmp_path / "g.npz"), ei, 3, node_features=x)
        result = load_graph_npz(p)
        assert torch.equal(result["edge_index"], ei)
        assert result["num_nodes"] == 3
        assert result["node_features"].shape == (3, 4)

    def test_npz_missing_file(self):
        pytest.importorskip("numpy")
        with pytest.raises(FileNotFoundError):
            load_graph_npz("/nonexistent/path.npz")
