"""Extended Graph / GraphBatch / graph_utils API tests.

These cover the additions to the user-facing graph layer:
- new constructor fields (edge_weight, edge_features 2D/3D, labels, metadata)
- new properties (num_nodes/edges, feature_shape, has_*)
- topology helpers (add/remove self-loops, make_undirected, is_undirected)
- clone(), validate(), to(device, dtype)
- GraphBatch batching of every new field
- clear, descriptive errors on invalid input

Existing test_graph.py covers the original positional-API contract; this
file adds the kwarg-driven, fully-loaded surface.
"""

from __future__ import annotations

import copy

import pytest
import torch

from tgraphx import (
    Graph,
    GraphBatch,
    add_self_loops,
    coalesce_edges,
    is_undirected,
    make_undirected,
    remove_self_loops,
)


# ------------------------------------------------------------------ helpers #

def _x_vec(N=4, D=8):
    return torch.randn(N, D)


def _x_img(N=4, C=3, H=4, W=4):
    return torch.randn(N, C, H, W)


def _x_vol(N=4, C=2, D=3, H=4, W=4):
    return torch.randn(N, C, D, H, W)


def _ei_directed(N=4):
    src = torch.arange(N)
    return torch.stack([src, (src + 1) % N])


# =========================================================================== #
# Construction: shapes and optional fields                                     #
# =========================================================================== #

class TestGraphConstructionExtended:
    def test_vector_node_features(self):
        g = Graph(_x_vec(N=5, D=8), _ei_directed(5))
        assert g.feature_shape == (8,)
        assert g.num_nodes == 5
        assert g.num_edges == 5

    def test_image_node_features(self):
        g = Graph(_x_img(N=4, C=3, H=4, W=4), _ei_directed(4))
        assert g.feature_shape == (3, 4, 4)
        assert g.num_nodes == 4

    def test_volumetric_node_features_storage(self):
        """Storage-level support for 5D node tensors [N, C, D, H, W]."""
        x = _x_vol(N=3, C=2, D=3, H=4, W=4)
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        g = Graph(x, ei)
        assert g.feature_shape == (2, 3, 4, 4)
        assert g.node_features.dim() == 5

    def test_node_features_must_be_at_least_2d(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            Graph(torch.randn(5), None)  # 1-D node tensor

    def test_edge_weight_basic(self):
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        ew = torch.tensor([0.5, 1.0, 1.5])
        g = Graph(_x_vec(3, 4), ei, edge_weight=ew)
        assert g.has_edge_weight
        assert torch.equal(g.edge_weight, ew)

    def test_vector_edge_features(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef = torch.randn(2, 7)  # [E, D_e]
        g = Graph(_x_vec(3, 4), ei, edge_features=ef)
        assert g.edge_feature_shape == (7,)
        assert g.has_edge_features

    def test_tensor_edge_features_2d_image(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef = torch.randn(2, 4, 4, 4)  # [E, C_e, H, W]
        g = Graph(_x_img(3, 3, 4, 4), ei, edge_features=ef)
        assert g.edge_feature_shape == (4, 4, 4)

    def test_tensor_edge_features_3d_storage(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef = torch.randn(2, 2, 3, 4, 4)  # [E, C_e, D, H, W]
        g = Graph(_x_vec(3, 4), ei, edge_features=ef)
        assert g.edge_feature_shape == (2, 3, 4, 4)
        assert g.edge_features.dim() == 5

    def test_node_edge_graph_labels(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        nl = torch.tensor([0, 1, 0])
        el = torch.tensor([1, 1])
        gl = torch.tensor(7)  # scalar label
        g = Graph(
            _x_vec(3, 4),
            ei,
            node_labels=nl,
            edge_labels=el,
            graph_label=gl,
        )
        assert torch.equal(g.node_labels, nl)
        assert torch.equal(g.edge_labels, el)
        assert torch.equal(g.graph_label, gl)

    def test_metadata_preservation(self):
        meta = {"name": "g0", "tags": ["foo", "bar"], "step": 17}
        g = Graph(_x_vec(3, 4), None, metadata=meta)
        assert g.metadata == meta
        assert g.metadata is meta  # stored by reference, not copied
        # clone() should deep-copy metadata
        g2 = g.clone()
        assert g2.metadata == meta
        assert g2.metadata is not meta

    def test_has_flags(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g_plain = Graph(_x_vec(2, 3), None)
        assert not g_plain.has_edges
        assert not g_plain.has_edge_weight
        assert not g_plain.has_edge_features

        g_full = Graph(
            _x_vec(2, 3),
            ei,
            edge_weight=torch.tensor([1.0, 2.0]),
            edge_features=torch.randn(2, 5),
        )
        assert g_full.has_edges
        assert g_full.has_edge_weight
        assert g_full.has_edge_features


# =========================================================================== #
# Validation: clear errors                                                     #
# =========================================================================== #

class TestGraphValidationErrors:
    def test_edge_weight_2d_rejected(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="1-D tensor"):
            Graph(_x_vec(3, 4), ei, edge_weight=torch.randn(2, 3))

    def test_edge_weight_length_mismatch(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="edge_weight"):
            Graph(_x_vec(3, 4), ei, edge_weight=torch.tensor([1.0]))

    def test_edge_weight_without_edge_index(self):
        with pytest.raises(ValueError, match="edge_weight.*edge_index is None"):
            Graph(_x_vec(3, 4), None, edge_weight=torch.tensor([1.0]))

    def test_edge_features_must_be_2d_plus(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            Graph(_x_vec(3, 4), ei, edge_features=torch.tensor([1.0, 2.0]))

    def test_node_labels_length_mismatch(self):
        with pytest.raises(ValueError, match="node_labels"):
            Graph(_x_vec(3, 4), None, node_labels=torch.tensor([0, 1]))

    def test_edge_labels_without_edges(self):
        with pytest.raises(ValueError, match="edge_labels.*edge_index is None"):
            Graph(_x_vec(3, 4), None, edge_labels=torch.tensor([0]))

    def test_edge_labels_length_mismatch(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="edge_labels"):
            Graph(_x_vec(3, 4), ei, edge_labels=torch.tensor([0, 1, 2]))

    def test_metadata_wrong_type(self):
        with pytest.raises(TypeError, match="metadata must be a dict"):
            Graph(_x_vec(3, 4), None, metadata=[("k", "v")])

    def test_graph_label_wrong_type(self):
        with pytest.raises(TypeError, match="graph_label"):
            Graph(_x_vec(3, 4), None, graph_label="positive")

    def test_node_labels_device_mismatch_clear_error(self):
        x = _x_vec(3, 4)
        # Forcing a device mismatch on CPU-only by faking via meta device: skip if no meta
        if hasattr(torch, "device") and "meta" in dir(torch):
            try:
                bad = torch.zeros(3, device="meta")
                with pytest.raises(ValueError, match="device"):
                    Graph(x, None, node_labels=bad)
            except (RuntimeError, NotImplementedError):
                pytest.skip("meta device not usable here")


# =========================================================================== #
# Topology helpers: directed / undirected / self-loops                          #
# =========================================================================== #

class TestTopology:
    def test_directed_cycle(self):
        g = Graph(_x_vec(4, 4), _ei_directed(4))
        assert g.is_directed()
        assert not g.is_undirected()

    def test_make_undirected_set_only(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        g = Graph(_x_vec(3, 4), ei)
        g.make_undirected()
        assert g.is_undirected()
        # 2 forward + 2 reverse, no duplicates → 4 edges
        assert g.num_edges == 4

    def test_make_undirected_with_edge_weight(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ew = torch.tensor([0.5, 1.0])
        g = Graph(_x_vec(3, 4), ei, edge_weight=ew)
        g.make_undirected()
        assert g.is_undirected()
        # 4 edges, weight per reverse equals weight of its forward
        assert g.num_edges == 4
        # find weight of (0,1) and (1,0); should both equal 0.5
        w_by_pair = {
            (int(g.edge_index[0, k]), int(g.edge_index[1, k])): float(g.edge_weight[k])
            for k in range(g.num_edges)
        }
        assert w_by_pair[(0, 1)] == pytest.approx(0.5)
        assert w_by_pair[(1, 0)] == pytest.approx(0.5)
        assert w_by_pair[(1, 2)] == pytest.approx(1.0)
        assert w_by_pair[(2, 1)] == pytest.approx(1.0)

    def test_make_undirected_coalesces_existing_pair(self):
        # Both directions already present with different weights — coalesce by mean.
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew = torch.tensor([1.0, 3.0])
        g = Graph(_x_vec(2, 4), ei, edge_weight=ew)
        g.make_undirected()
        assert g.num_edges == 2
        for k in range(2):
            assert float(g.edge_weight[k]) == pytest.approx(2.0)

    def test_make_undirected_with_edge_features_2d(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ef = torch.randn(2, 3, 4, 4)
        g = Graph(_x_img(3, 3, 4, 4), ei, edge_features=ef)
        g.make_undirected()
        assert g.is_undirected()
        assert g.num_edges == 4
        assert g.edge_features.shape[1:] == (3, 4, 4)

    def test_add_self_loops_no_existing(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Graph(_x_vec(3, 4), ei)
        g.add_self_loops()
        # 3 nodes → 3 self-loops added on top of 2 original edges
        assert g.num_edges == 5
        # every node has at least one (i,i) row
        for n in range(3):
            assert ((g.edge_index[0] == n) & (g.edge_index[1] == n)).any()

    def test_add_self_loops_skips_existing(self):
        # Node 0 already has self-loop, others don't.
        ei = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        g = Graph(_x_vec(3, 4), ei)
        g.add_self_loops()
        # Should add self-loops only for nodes 1 and 2 (not 0).
        # Originals: (0,0), (1,2)
        # New: (1,1), (2,2)
        assert g.num_edges == 4
        pairs = {(int(g.edge_index[0, k]), int(g.edge_index[1, k])) for k in range(4)}
        assert (0, 0) in pairs and (1, 1) in pairs and (2, 2) in pairs and (1, 2) in pairs

    def test_add_self_loops_propagates_weight_and_features(self):
        ei = torch.tensor([[0, 1]], dtype=torch.long).t().contiguous()
        # That's just (0→1). 2 nodes, 1 edge.
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        ew = torch.tensor([3.0])
        ef = torch.randn(1, 4)
        g = Graph(_x_vec(2, 4), ei, edge_weight=ew, edge_features=ef)
        g.add_self_loops(fill_value=2.0)
        assert g.num_edges == 3  # (0,1) + (0,0) + (1,1)
        assert g.edge_weight.numel() == 3
        # The two new self-loops have weight = fill_value
        assert float(g.edge_weight[1]) == pytest.approx(2.0)
        assert float(g.edge_weight[2]) == pytest.approx(2.0)
        assert g.edge_features.shape == (3, 4)
        assert torch.allclose(g.edge_features[1], torch.full((4,), 2.0))

    def test_remove_self_loops(self):
        ei = torch.tensor([[0, 1, 2, 1], [0, 1, 1, 2]], dtype=torch.long)
        ew = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ef = torch.randn(4, 5)
        el = torch.tensor([10, 20, 30, 40])
        g = Graph(_x_vec(3, 4), ei, edge_weight=ew, edge_features=ef, edge_labels=el)
        g.remove_self_loops()
        # Removed (0,0) and (1,1); kept (2,1) and (1,2).
        assert g.num_edges == 2
        kept_pairs = {(int(g.edge_index[0, k]), int(g.edge_index[1, k])) for k in range(2)}
        assert kept_pairs == {(2, 1), (1, 2)}
        assert g.edge_weight.tolist() == [3.0, 4.0]
        assert g.edge_features.shape == (2, 5)
        assert g.edge_labels.tolist() == [30, 40]

    def test_add_self_loops_blocked_with_edge_labels(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        el = torch.tensor([0, 1])
        g = Graph(_x_vec(2, 4), ei, edge_labels=el)
        with pytest.raises(ValueError, match="edge_labels"):
            g.add_self_loops()

    def test_make_undirected_blocked_with_edge_labels(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        el = torch.tensor([0, 1])
        g = Graph(_x_vec(3, 4), ei, edge_labels=el)
        with pytest.raises(ValueError, match="edge_labels"):
            g.make_undirected()

    def test_is_undirected_pure_function(self):
        ei = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
        # (0,1), (1,0), (1,0), (0,1) — symmetric multiset
        assert is_undirected(ei)

    def test_coalesce_dedupes_and_means_weight(self):
        ei = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long)
        ew = torch.tensor([1.0, 3.0, 5.0])
        new_ei, new_w, _ = coalesce_edges(ei, ew, num_nodes=2, reduce="mean")
        # (0,1) appears twice with weights 1, 3 → mean 2; (1,0) once with 5.
        assert new_ei.size(1) == 2
        pair_to_w = {
            (int(new_ei[0, k]), int(new_ei[1, k])): float(new_w[k]) for k in range(2)
        }
        assert pair_to_w[(0, 1)] == pytest.approx(2.0)
        assert pair_to_w[(1, 0)] == pytest.approx(5.0)


# =========================================================================== #
# clone(), validate(), to(device, dtype)                                       #
# =========================================================================== #

class TestGraphClonValidateTo:
    def test_clone_independence(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew = torch.tensor([0.5, 0.5])
        meta = {"k": [1, 2, 3]}
        g = Graph(_x_vec(2, 4), ei, edge_weight=ew, metadata=meta)
        g2 = g.clone()
        # Tensors are distinct
        assert g2.node_features.data_ptr() != g.node_features.data_ptr()
        assert g2.edge_weight.data_ptr() != g.edge_weight.data_ptr()
        # Mutating original doesn't affect the clone
        g.edge_weight[0] = 99.0
        assert float(g2.edge_weight[0]) == pytest.approx(0.5)
        # metadata deep-copied
        meta["k"].append(4)
        assert g2.metadata["k"] == [1, 2, 3]

    def test_validate_reruns_checks(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Graph(_x_vec(3, 4), ei)
        # Manually corrupt edge_index range — should fail validate()
        g.edge_index = torch.tensor([[0, 1], [1, 99]], dtype=torch.long)
        with pytest.raises(ValueError, match="out-of-range"):
            g.validate()

    def test_to_moves_all_tensor_fields_cpu(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ew = torch.tensor([1.0, 2.0])
        ef = torch.randn(2, 3)
        nl = torch.tensor([0, 1, 0])
        el = torch.tensor([0, 1])
        gl = torch.tensor(1)
        g = Graph(
            _x_vec(3, 4), ei,
            edge_weight=ew, edge_features=ef,
            node_labels=nl, edge_labels=el, graph_label=gl,
        )
        g.to(device="cpu")
        for t in (g.node_features, g.edge_index, g.edge_weight, g.edge_features,
                  g.node_labels, g.edge_labels, g.graph_label):
            assert t.device.type == "cpu"

    def test_to_dtype_only_floats(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        nl = torch.tensor([0, 1, 0])  # integer labels
        g = Graph(_x_vec(3, 4), ei, node_labels=nl)
        g.to(dtype=torch.float64)
        assert g.node_features.dtype == torch.float64
        # Index tensor must remain torch.long
        assert g.edge_index.dtype == torch.long
        # Integer labels must NOT be coerced to float
        assert g.node_labels.dtype == nl.dtype


# =========================================================================== #
# GraphBatch                                                                   #
# =========================================================================== #

class TestGraphBatchExtended:
    def test_edge_weight_batching(self):
        x1 = _x_vec(3, 4)
        x2 = _x_vec(2, 4)
        ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ei2 = torch.tensor([[0], [1]], dtype=torch.long)
        ew1 = torch.tensor([1.0, 2.0])
        ew2 = torch.tensor([5.0])
        g1 = Graph(x1, ei1, edge_weight=ew1)
        g2 = Graph(x2, ei2, edge_weight=ew2)
        b = GraphBatch([g1, g2])
        assert b.edge_weight.tolist() == [1.0, 2.0, 5.0]
        assert b.edge_index[:, -1].tolist() == [3, 4]
        assert b.has_edge_weight

    def test_edge_features_batching_vector(self):
        ei1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ei2 = torch.tensor([[0], [1]], dtype=torch.long)
        ef1 = torch.randn(2, 6)
        ef2 = torch.randn(1, 6)
        g1 = Graph(_x_vec(3, 4), ei1, edge_features=ef1)
        g2 = Graph(_x_vec(2, 4), ei2, edge_features=ef2)
        b = GraphBatch([g1, g2])
        assert b.edge_features.shape == (3, 6)
        assert torch.equal(b.edge_features[:2], ef1)
        assert torch.equal(b.edge_features[2:], ef2)

    def test_edge_features_batching_image(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ef1 = torch.randn(2, 4, 4, 4)
        ef2 = torch.randn(2, 4, 4, 4)
        g1 = Graph(_x_img(3, 3, 4, 4), ei, edge_features=ef1)
        g2 = Graph(_x_img(2, 3, 4, 4), ei[:, :2].clone(), edge_features=ef2)
        b = GraphBatch([g1, g2])
        assert b.edge_features.shape == (4, 4, 4, 4)

    def test_edge_features_shape_mismatch_descriptive_error(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g1 = Graph(_x_vec(3, 4), ei, edge_features=torch.randn(2, 6))
        g2 = Graph(_x_vec(3, 4), ei, edge_features=torch.randn(2, 7))
        with pytest.raises(ValueError, match="per-edge feature shape"):
            GraphBatch([g1, g2])

    def test_partial_edge_weight_rejected(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g1 = Graph(_x_vec(3, 4), ei, edge_weight=torch.tensor([1.0, 2.0]))
        g2 = Graph(_x_vec(3, 4), ei)  # no edge_weight
        with pytest.raises(ValueError, match="edge_weight"):
            GraphBatch([g1, g2])

    def test_node_and_graph_labels_batching(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g1 = Graph(_x_vec(3, 4), ei,
                   node_labels=torch.tensor([0, 1, 0]),
                   graph_label=torch.tensor(0))
        g2 = Graph(_x_vec(2, 4),
                   torch.tensor([[0], [1]], dtype=torch.long),
                   node_labels=torch.tensor([1, 0]),
                   graph_label=torch.tensor(1))
        b = GraphBatch([g1, g2])
        assert b.node_labels.tolist() == [0, 1, 0, 1, 0]
        # Stacked scalars become a 1-D tensor of length B.
        assert b.graph_labels.tolist() == [0, 1]

    def test_metadata_preserved_as_list(self):
        meta1 = {"src": "a"}
        meta2 = {"src": "b"}
        g1 = Graph(_x_vec(2, 4), None, metadata=meta1)
        g2 = Graph(_x_vec(3, 4), None, metadata=meta2)
        b = GraphBatch([g1, g2])
        assert isinstance(b.metadata, list)
        assert len(b.metadata) == 2
        assert b.metadata[0] is meta1 and b.metadata[1] is meta2

    def test_metadata_can_include_none(self):
        g1 = Graph(_x_vec(2, 4), None, metadata={"k": 1})
        g2 = Graph(_x_vec(3, 4), None)  # metadata defaults to None
        b = GraphBatch([g1, g2])
        assert b.metadata == [{"k": 1}, None]

    def test_batch_to_device_moves_all_fields(self):
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g = Graph(
            _x_vec(2, 4), ei,
            edge_weight=torch.tensor([1.0, 1.0]),
            edge_features=torch.randn(2, 5),
            node_labels=torch.tensor([0, 1]),
            edge_labels=torch.tensor([0, 1]),
            graph_label=torch.tensor(7),
        )
        b = GraphBatch([g, g.clone()])
        b.to(device="cpu")
        for t in (b.node_features, b.edge_index, b.edge_weight, b.edge_features,
                  b.node_labels, b.edge_labels, b.graph_labels, b.batch):
            assert t.device.type == "cpu"

    def test_backward_compat_alias_batch_graphs(self):
        """The old `batch_graphs` instance alias still resolves."""
        g = Graph(_x_vec(2, 4), None)
        b = GraphBatch([g])
        assert hasattr(b, "batch_graphs")  # alias preserved


# =========================================================================== #
# Free helper functions                                                        #
# =========================================================================== #

class TestGraphUtilsHelpers:
    def test_remove_self_loops_pure(self):
        ei = torch.tensor([[0, 1, 1], [0, 1, 2]], dtype=torch.long)
        ew = torch.tensor([1.0, 2.0, 3.0])
        new_ei, new_w, _, _ = remove_self_loops(ei, ew)
        assert new_ei.tolist() == [[1], [2]]
        assert new_w.tolist() == [3.0]

    def test_add_self_loops_pure_no_existing(self):
        new_ei, new_w, new_ef = add_self_loops(
            edge_index=None, edge_weight=None, edge_features=None,
            num_nodes=3,
        )
        assert new_ei.shape == (2, 3)
        assert new_w is None and new_ef is None

    def test_make_undirected_pure(self):
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        new_ei, _, _ = make_undirected(ei)
        assert is_undirected(new_ei)
        assert new_ei.size(1) == 4
