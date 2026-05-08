"""Tests for graph learning, self-supervised, and augmentation utilities."""
import pytest
import torch
from tgraphx.mining import (
    contrastive_loss, supervised_contrastive_loss, triplet_loss,
    bpr_loss, reconstruction_loss,
    drop_edges, drop_nodes, mask_node_features, add_random_edges,
    subgraph_sampling, DGIObjective, GraphCLObjective,
    create_negative_pairs, create_positive_pairs_from_batch,
    degree_encoding, random_walk_structural_encoding,
    shortest_path_anchor_encoding, centrality_encoding,
    StructuralEncodingModule, attach_structural_encodings,
    GraphSequenceEncoder, GraphSequenceClassifier, GraphRNNEdgeGenerator,
    bfs_sequence_encode, random_walk_sequence_encode, pad_sequences,
)


def _chain_ei(N=4):
    src = list(range(N-1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N-1))
    return torch.tensor([src, dst], dtype=torch.long), N


class TestContrastiveLosses:
    def test_contrastive_loss_forward(self):
        z1 = torch.randn(4, 8, requires_grad=True)
        z2 = torch.randn(4, 8, requires_grad=True)
        loss = contrastive_loss(z1, z2)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_contrastive_loss_backward(self):
        z1 = torch.randn(4, 8, requires_grad=True)
        z2 = torch.randn(4, 8, requires_grad=True)
        loss = contrastive_loss(z1, z2)
        loss.backward()
        assert z1.grad is not None and torch.isfinite(z1.grad).all()

    def test_contrastive_loss_batch_too_small(self):
        z1 = torch.randn(1, 8)
        z2 = torch.randn(1, 8)
        with pytest.raises(ValueError, match="batch size"):
            contrastive_loss(z1, z2)

    def test_supervised_contrastive_loss(self):
        emb = torch.randn(6, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = supervised_contrastive_loss(emb, labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert emb.grad is not None

    def test_triplet_loss(self):
        a = torch.randn(4, 8, requires_grad=True)
        p = torch.randn(4, 8, requires_grad=True)
        n = torch.randn(4, 8, requires_grad=True)
        loss = triplet_loss(a, p, n)
        assert loss >= 0
        loss.backward()
        assert a.grad is not None

    def test_bpr_loss(self):
        pos = torch.randn(4, requires_grad=True)
        neg = torch.randn(4, requires_grad=True)
        loss = bpr_loss(pos, neg)
        assert torch.isfinite(loss)
        loss.backward()

    def test_reconstruction_loss(self):
        x = torch.randn(4, 8)
        recon = torch.randn(4, 8, requires_grad=True)
        loss = reconstruction_loss(x, recon)
        assert torch.isfinite(loss)
        loss.backward()


class TestAugmentations:
    def test_drop_edges_deterministic(self):
        ei, N = _chain_ei()
        new1, _ = drop_edges(ei, p=0.3, seed=42)
        new2, _ = drop_edges(ei, p=0.3, seed=42)
        assert torch.equal(new1, new2)

    def test_drop_edges_all(self):
        ei, N = _chain_ei()
        new_ei, _ = drop_edges(ei, p=1.0)
        assert new_ei.size(1) == 0

    def test_drop_edges_none(self):
        ei, N = _chain_ei()
        new_ei, _ = drop_edges(ei, p=0.0)
        assert torch.equal(new_ei, ei)

    def test_drop_nodes_shape(self):
        ei, N = _chain_ei()
        x = torch.randn(N, 4)
        new_ei, new_N, new_x, kept = drop_nodes(ei, N, p=0.3, seed=0, node_features=x)
        assert new_x.size(0) == new_N
        assert new_ei.size(0) == 2

    def test_drop_nodes_deterministic(self):
        ei, N = _chain_ei()
        _, _, _, k1 = drop_nodes(ei, N, p=0.3, seed=7)
        _, _, _, k2 = drop_nodes(ei, N, p=0.3, seed=7)
        assert torch.equal(k1, k2)

    def test_mask_node_features(self):
        x = torch.ones(5, 8)
        masked, mask = mask_node_features(x, p=0.5, seed=0)
        assert masked.shape == x.shape
        assert mask.dtype == torch.bool
        # Masked elements should be 0.
        assert (masked[mask] == 0.0).all()
        # Non-masked elements should be 1.
        assert (masked[~mask] == 1.0).all()

    def test_mask_features_deterministic(self):
        x = torch.ones(5, 8)
        _, m1 = mask_node_features(x, p=0.3, seed=1)
        _, m2 = mask_node_features(x, p=0.3, seed=1)
        assert torch.equal(m1, m2)

    def test_add_random_edges(self):
        ei, N = _chain_ei()
        new_ei = add_random_edges(ei, N, num_add=3, seed=0)
        assert new_ei.size(1) == ei.size(1) + 3

    def test_subgraph_sampling(self):
        ei, N = _chain_ei(8)
        x = torch.randn(8, 4)
        new_ei, new_N, new_x, sampled = subgraph_sampling(ei, 8, 4, seed=0, node_features=x)
        assert new_N == 4
        assert new_x.size(0) == 4
        assert sampled.size(0) == 4


class TestSelfSupervisedObjectives:
    def test_dgi_objective_forward(self):
        dgi = DGIObjective(embed_dim=8, summary_dim=8)
        pos_emb = torch.randn(5, 8, requires_grad=True)
        neg_emb = torch.randn(5, 8, requires_grad=True)
        loss = dgi(pos_emb, neg_emb)
        assert torch.isfinite(loss)
        loss.backward()
        assert pos_emb.grad is not None

    def test_graphcl_objective(self):
        gcl = GraphCLObjective(project_dim=8)
        z1 = torch.randn(4, 8, requires_grad=True)
        z2 = torch.randn(4, 8, requires_grad=True)
        loss = gcl(z1, z2)
        assert torch.isfinite(loss)
        loss.backward()


class TestStructuralEncodings:
    def test_degree_encoding_shape(self):
        ei, N = _chain_ei()
        enc = degree_encoding(ei, N)
        assert enc.shape == (N, 2)
        assert enc.dtype == torch.float32

    def test_degree_encoding_star_hub(self):
        from tgraphx.mining import star_graph
        ei, N = star_graph(5)
        enc = degree_encoding(ei, N, normalize=False)
        # Hub (node 0) should have highest degree.
        assert float(enc[0, 0]) == 4.0

    def test_centrality_encoding_shape(self):
        ei, N = _chain_ei(6)
        enc = centrality_encoding(ei, N, include=["degree", "pagerank"])
        assert enc.shape == (N, 2)

    def test_attach_concat(self):
        ei, N = _chain_ei()
        enc = degree_encoding(ei, N)
        x = torch.randn(N, 6)
        aug = attach_structural_encodings(x, enc)
        assert aug.shape == (N, 8)  # 6 + 2

    def test_attach_spatial_raises(self):
        ei, N = _chain_ei()
        enc = degree_encoding(ei, N)
        x = torch.randn(N, 3, 4, 4)  # spatial
        with pytest.raises(ValueError, match="mode='side'"):
            attach_structural_encodings(x, enc, mode="concat")

    def test_attach_side_mode(self):
        ei, N = _chain_ei()
        enc = degree_encoding(ei, N)
        x = torch.randn(N, 3, 4, 4)
        result = attach_structural_encodings(x, enc, mode="side")
        assert result is enc

    def test_structural_encoding_module(self):
        mod = StructuralEncodingModule(in_dim=2, out_dim=8)
        enc = torch.randn(5, 2)
        out = mod(enc)
        assert out.shape == (5, 8)
        out.sum().backward()
        assert all(p.grad is not None for p in mod.parameters())


class TestSequenceModels:
    def test_bfs_sequence_encode(self):
        ei, N = _chain_ei()
        x = torch.randn(N, 4)
        seq = bfs_sequence_encode(ei, N, node_features=x, start=0)
        assert seq.shape[1] == 4
        assert seq.shape[0] == N

    def test_random_walk_sequence_encode(self):
        ei, N = _chain_ei()
        x = torch.randn(N, 4)
        seq = random_walk_sequence_encode(ei, N, walk_length=5, node_features=x, seed=0)
        assert seq.shape == (6, 4)  # walk_length + 1

    def test_pad_sequences(self):
        seqs = [torch.randn(3, 4), torch.randn(5, 4), torch.randn(2, 4)]
        padded, lengths = pad_sequences(seqs)
        assert padded.shape == (3, 5, 4)
        assert lengths.tolist() == [3, 5, 2]

    def test_graph_sequence_encoder_forward(self):
        enc = GraphSequenceEncoder(input_dim=4, hidden_dim=16, num_layers=2)
        x = torch.randn(3, 7, 4)
        out = enc(x)
        assert out.shape == (3, 16)

    def test_graph_sequence_encoder_backward(self):
        enc = GraphSequenceEncoder(input_dim=4, hidden_dim=16, num_layers=2)
        x = torch.randn(3, 7, 4)
        out = enc(x)
        out.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None

    def test_graph_sequence_classifier_overfit(self):
        clf = GraphSequenceClassifier(input_dim=4, hidden_dim=32, num_classes=2)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        # Class 0: mostly-zero sequences; class 1: mostly-one sequences.
        x0 = torch.randn(3, 5, 4) * 0.01
        x1 = torch.randn(3, 5, 4) + 5.0
        X = torch.cat([x0, x1])
        y = torch.tensor([0,0,0,1,1,1])
        losses = []
        for _ in range(30):
            opt.zero_grad()
            logits = clf(X)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f}→{losses[-1]:.4f}"

    def test_graph_rnn_generator_shape(self):
        gen = GraphRNNEdgeGenerator(max_nodes=8)
        adj = gen.generate(5, seed=0)
        assert adj.shape == (5, 5)
        # Symmetric.
        assert torch.equal(adj, adj.t())

    def test_graph_rnn_generator_deterministic(self):
        gen = GraphRNNEdgeGenerator(max_nodes=8)
        adj1 = gen.generate(5, seed=42)
        adj2 = gen.generate(5, seed=42)
        assert torch.equal(adj1, adj2)

    def test_graph_rnn_generator_backward(self):
        gen = GraphRNNEdgeGenerator(max_nodes=6)
        # Teacher-forcing training step.
        opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
        # Target: random binary adjacency rows.
        N = 4
        B = 2
        target = torch.zeros(B, N, 6)
        for i in range(1, N):
            target[:, i, :i] = (torch.rand(B, i) > 0.5).float()
        logits, _ = gen(target)
        # BCEWithLogits loss on upper triangle.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        assert all(p.grad is not None for p in gen.parameters())


class TestNegativePairUtils:
    def test_create_negative_pairs_no_false_neg(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        neg = create_negative_pairs(ei, num_nodes=4, num_neg=4, seed=0)
        pos_set = {(int(ei[0,i]), int(ei[1,i])) for i in range(ei.size(1))}
        neg_set = {(int(neg[0,i]), int(neg[1,i])) for i in range(neg.size(1))}
        assert not (pos_set & neg_set)

    def test_create_positive_pairs(self):
        labels = torch.tensor([0, 0, 1, 1, 0])
        pa, pb = create_positive_pairs_from_batch(labels)
        assert pa.size(0) > 0
        for a, b in zip(pa.tolist(), pb.tolist()):
            assert labels[a] == labels[b]
            assert a != b
