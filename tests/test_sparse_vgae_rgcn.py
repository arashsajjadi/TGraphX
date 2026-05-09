"""Tests for tgraphx.sparse, VGAE, and RGCN."""
import pytest
import torch
from tgraphx.sparse import (
    coalesce_edge_index, sort_edge_index, remove_self_loops, add_self_loops,
    degree, in_degree, out_degree,
    segment_sum, segment_mean, segment_max, segment_min, segment_softmax,
    edge_index_to_csr, csr_to_edge_index,
    chunked_cosine_similarity, chunked_top_k,
    backend_info,
)
from tgraphx.mining import (
    GraphAutoencoder, VGAE, VGAEGCNEncoder, DotProductDecoder, MLPEdgeDecoder,
    train_gae_step, evaluate_link_prediction,
)
from tgraphx.layers.rgcn import RGCNConv


# ── Sparse tests ─────────────────────────────────────────────────────────────


class TestSparseOps:
    def test_coalesce_removes_duplicates(self):
        ei = torch.tensor([[0,0,1],[1,1,2]], dtype=torch.long)
        ei_c, _ = coalesce_edge_index(ei, num_nodes=3)
        # After deduplication: should have 2 unique edges (0→1) becomes 1, (1→2) stays.
        assert ei_c.size(1) == 2

    def test_coalesce_sorted(self):
        ei = torch.tensor([[2,0,1],[0,1,2]], dtype=torch.long)
        ei_c, _ = coalesce_edge_index(ei, num_nodes=3)
        # Should be sorted by src then dst.
        src = ei_c[0].tolist()
        assert src == sorted(src)

    def test_sort_edge_index(self):
        ei = torch.tensor([[2,0,1],[0,1,2]], dtype=torch.long)
        ei_s, _ = sort_edge_index(ei, num_nodes=3)
        assert ei_s[0].tolist() == sorted(ei_s[0].tolist())

    def test_remove_self_loops(self):
        ei = torch.tensor([[0,1,1,2],[0,1,2,2]], dtype=torch.long)
        ei_out, _ = remove_self_loops(ei)
        # Self-loops: (0,0) and (2,2) removed.
        for k in range(ei_out.size(1)):
            assert ei_out[0, k] != ei_out[1, k]

    def test_add_self_loops(self):
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        ei_out, _ = add_self_loops(ei, num_nodes=3)
        # Node 0 already has edge to 1 but no self-loop; 1 has edge to 2; 2 has no edge.
        self_loops = ei_out[:, ei_out[0] == ei_out[1]]
        assert 0 in self_loops[0].tolist()
        assert 1 in self_loops[0].tolist()
        assert 2 in self_loops[0].tolist()

    def test_degree(self):
        ei = torch.tensor([[0,0,1],[1,2,2]], dtype=torch.long)
        d = degree(ei, num_nodes=3)
        assert d.tolist() == [2, 1, 0]  # out-degrees

    def test_in_degree(self):
        ei = torch.tensor([[0,0,1],[1,2,2]], dtype=torch.long)
        d = in_degree(ei, num_nodes=3)
        assert d.tolist() == [0, 1, 2]

    def test_segment_sum_hand_computed(self):
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        idx = torch.tensor([0, 0, 1, 1])
        result = segment_sum(src, idx, 2)
        assert result.tolist() == [3.0, 7.0]

    def test_segment_mean_hand_computed(self):
        src = torch.tensor([1.0, 3.0, 2.0, 8.0])
        idx = torch.tensor([0, 0, 1, 1])
        result = segment_mean(src, idx, 2)
        assert abs(float(result[0]) - 2.0) < 1e-5
        assert abs(float(result[1]) - 5.0) < 1e-5

    def test_segment_max_hand_computed(self):
        src = torch.tensor([1.0, 5.0, 2.0, 3.0])
        idx = torch.tensor([0, 0, 1, 1])
        result = segment_max(src, idx, 2)
        assert float(result[0]) == 5.0
        assert float(result[1]) == 3.0

    def test_segment_softmax_sums_to_one(self):
        src = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0])
        idx = torch.tensor([0, 0, 0, 1, 1])
        result = segment_softmax(src, idx, 2)
        # Group 0 sum.
        g0 = result[idx == 0].sum()
        g1 = result[idx == 1].sum()
        assert abs(float(g0) - 1.0) < 1e-5
        assert abs(float(g1) - 1.0) < 1e-5

    def test_csr_roundtrip(self):
        ei = torch.tensor([[0,0,1,2],[1,2,2,0]], dtype=torch.long)
        row_ptr, col_idx = edge_index_to_csr(ei, num_nodes=3)
        ei2 = csr_to_edge_index(row_ptr, col_idx)
        # Same number of edges.
        assert ei2.size(1) == ei.size(1)

    def test_chunked_cosine_similarity(self):
        x = torch.randn(10, 8)
        y = torch.randn(5, 8)
        sim = chunked_cosine_similarity(x, y, chunk_size=3)
        assert sim.shape == (10, 5)
        # Diagonal check: cosine of x with itself = 1.
        sim_self = chunked_cosine_similarity(x, x)
        diag = sim_self.diagonal()
        assert (diag - 1.0).abs().max() < 1e-4

    def test_chunked_top_k(self):
        scores = torch.randn(8, 20)
        vals, idx = chunked_top_k(scores, k=5, chunk_size=3)
        assert vals.shape == (8, 5) and idx.shape == (8, 5)
        # Verify top-1 is correct.
        expected_top1 = scores.topk(1, dim=1).indices
        assert (idx[:, 0] == expected_top1.squeeze()).all()

    def test_backend_info(self):
        info = backend_info()
        assert info["pure_torch"] is True
        assert isinstance(info["torch_scatter"], bool)


# ── VGAE tests ────────────────────────────────────────────────────────────────


def _toy_graph(N=10, D=8):
    ei = torch.tensor([list(range(N)), [(i+1) % N for i in range(N)]], dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    x = torch.randn(N, D)
    return x, ei, N


class TestGAE:
    def test_forward_backward(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        gae = GraphAutoencoder(enc)
        pos_ei = ei[:, :5]
        neg_ei = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]], dtype=torch.long)
        loss = gae(x, ei, pos_ei, neg_ei)
        assert torch.isfinite(loss)
        loss.backward()
        for p in gae.parameters():
            assert p.grad is not None

    def test_gradients_finite(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        gae = GraphAutoencoder(enc)
        pos_ei = ei[:, :5]; neg_ei = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]])
        gae(x, ei, pos_ei, neg_ei).backward()
        for p in gae.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_loss_decreases(self):
        torch.manual_seed(0)
        x, ei, N = _toy_graph(8, 4)
        enc = VGAEGCNEncoder(4, 16, 8)
        gae = GraphAutoencoder(enc)
        opt = torch.optim.Adam(gae.parameters(), lr=1e-2)
        pos_ei = ei[:, :6]; neg_ei = ei[:, 6:12]
        losses = [train_gae_step(gae, opt, x, ei, pos_ei, neg_ei) for _ in range(20)]
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f}→{losses[-1]:.4f}"

    def test_evaluate_link_prediction(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        gae = GraphAutoencoder(enc)
        pos_ei = ei[:, :5]; neg_ei = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]])
        metrics = evaluate_link_prediction(gae, x, ei, pos_ei, neg_ei)
        assert "auroc" in metrics and "auprc" in metrics and "accuracy" in metrics
        assert 0 <= metrics["auroc"] <= 1

    def test_mlp_decoder(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        decoder = MLPEdgeDecoder(in_dim=8, hidden_dim=16)
        gae = GraphAutoencoder(enc, decoder=decoder)
        pos_ei = ei[:, :5]; neg_ei = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]])
        loss = gae(x, ei, pos_ei, neg_ei)
        assert torch.isfinite(loss)
        loss.backward()


class TestVGAE:
    def test_forward_backward(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        vgae = VGAE(enc)
        pos_ei = ei[:, :5]; neg_ei = torch.tensor([[0,1,2,3,4],[5,6,7,8,9]])
        loss = vgae(x, ei, pos_ei, neg_ei)
        assert torch.isfinite(loss)
        loss.backward()

    def test_kl_finite(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        vgae = VGAE(enc)
        vgae.encode(x, ei, N)
        kl = vgae.kl_loss()
        assert torch.isfinite(kl)

    def test_eval_mode_no_sampling(self):
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        vgae = VGAE(enc)
        vgae.eval()
        z1 = vgae.encode(x, ei, N)
        z2 = vgae.encode(x, ei, N)
        # In eval mode, z = mu (deterministic).
        assert torch.allclose(z1, z2)

    def test_train_mode_stochastic(self):
        torch.manual_seed(0)
        x, ei, N = _toy_graph()
        enc = VGAEGCNEncoder(8, 16, 8)
        vgae = VGAE(enc)
        vgae.train()
        z1 = vgae.encode(x, ei, N)
        z2 = vgae.encode(x, ei, N)
        # In train mode, sampling → different outputs.
        assert not torch.allclose(z1, z2)

    def test_kl_loss_requires_encode(self):
        enc = VGAEGCNEncoder(8, 16, 8)
        vgae = VGAE(enc)
        with pytest.raises(RuntimeError, match="encode"):
            vgae.kl_loss()


# ── RGCN tests ────────────────────────────────────────────────────────────────


class TestRGCN:
    def test_output_shape(self):
        rgcn = RGCNConv(8, 16, num_relations=3)
        x = torch.randn(10, 8)
        ei_by_rel = {0: torch.tensor([[0,1],[1,2]], dtype=torch.long)}
        out = rgcn(x, ei_by_rel, 10)
        assert out.shape == (10, 16)

    def test_backward(self):
        rgcn = RGCNConv(8, 16, num_relations=3, num_bases=2)
        x = torch.randn(10, 8)
        ei_by_rel = {0: torch.tensor([[0,1],[1,2]], dtype=torch.long),
                     1: torch.tensor([[3,4],[4,5]], dtype=torch.long)}
        out = rgcn(x, ei_by_rel, 10)
        out.sum().backward()
        for p in rgcn.parameters():
            assert p.grad is not None

    def test_finite_gradients(self):
        rgcn = RGCNConv(4, 8, num_relations=2, num_bases=2)
        x = torch.randn(8, 4)
        out = rgcn(x, {0: torch.tensor([[0,1],[1,2]], dtype=torch.long)}, 8)
        out.sum().backward()
        for p in rgcn.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()

    def test_basis_decomposition_fewer_params(self):
        rgcn_no_basis = RGCNConv(8, 16, num_relations=10, num_bases=0)
        rgcn_basis = RGCNConv(8, 16, num_relations=10, num_bases=2)
        count_no = sum(p.numel() for p in rgcn_no_basis.parameters())
        count_basis = sum(p.numel() for p in rgcn_basis.parameters())
        assert count_basis < count_no

    def test_empty_relation(self):
        rgcn = RGCNConv(4, 8, num_relations=2)
        x = torch.randn(6, 4)
        # No edges for any relation.
        out = rgcn(x, {0: torch.zeros((2, 0), dtype=torch.long)}, 6)
        assert out.shape == (6, 8)
        assert torch.isfinite(out).all()

    def test_tiny_overfit(self):
        torch.manual_seed(0)
        rgcn = RGCNConv(4, 8, num_relations=2, num_bases=2)
        head = torch.nn.Linear(8, 3)
        model = torch.nn.Sequential()
        opt = torch.optim.Adam(list(rgcn.parameters()) + list(head.parameters()), lr=5e-3)
        x = torch.randn(8, 4)
        ei_by_rel = {0: torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)}
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        losses = []
        for _ in range(30):
            opt.zero_grad()
            emb = rgcn(x, ei_by_rel, 8)
            logits = head(torch.relu(emb))
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f}→{losses[-1]:.4f}"
