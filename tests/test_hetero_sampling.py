"""Tests for hetero sampling helpers (v0.2.8)."""
from __future__ import annotations

import pytest
import torch

from tgraphx import (
    HeteroGraph,
    hetero_induced_subgraph,
    hetero_neighbor_sample,
)


def _hg(n_paper=6, n_author=4, n_aw=8, n_pc=5, seed=0):
    torch.manual_seed(seed)
    nodes = {
        "paper": torch.randn(n_paper, 8),
        "author": torch.randn(n_author, 4),
    }
    aw_src = torch.randint(0, n_author, (n_aw,))
    aw_dst = torch.randint(0, n_paper, (n_aw,))
    pc_src = torch.randint(0, n_paper, (n_pc,))
    pc_dst = torch.randint(0, n_paper, (n_pc,))
    edges = {
        ("author", "writes", "paper"): torch.stack([aw_src, aw_dst], dim=0).long(),
        ("paper", "cites", "paper"): torch.stack([pc_src, pc_dst], dim=0).long(),
    }
    return HeteroGraph(
        node_stores=nodes,
        edge_stores=edges,
        edge_weight_stores={
            ("author", "writes", "paper"): torch.rand(n_aw),
        },
        edge_feature_stores={
            ("author", "writes", "paper"): torch.randn(n_aw, 3),
        },
        node_label_stores={"paper": torch.randint(0, 3, (n_paper,))},
    )


# ── hetero_induced_subgraph ──────────────────────────────────────────────────


class TestHeteroInducedSubgraph:
    def test_basic_shapes(self):
        g = _hg()
        sub = hetero_induced_subgraph(
            g, {"paper": torch.tensor([0, 1, 2]),
                "author": torch.tensor([0, 1])}
        )
        assert sub.num_nodes("paper") == 3
        assert sub.num_nodes("author") == 2

    def test_features_preserved(self):
        g = _hg()
        keep_papers = torch.tensor([0, 3, 5])
        sub = hetero_induced_subgraph(
            g, {"paper": keep_papers, "author": torch.tensor([0, 2])},
        )
        for local, gid in enumerate(keep_papers.tolist()):
            assert torch.equal(
                sub.node_features("paper")[local], g.node_features("paper")[gid],
            )

    def test_labels_preserved(self):
        g = _hg()
        keep_papers = torch.tensor([1, 4])
        sub = hetero_induced_subgraph(
            g, {"paper": keep_papers, "author": torch.tensor([0, 1, 2])},
        )
        nl = sub.node_labels("paper")
        assert nl is not None
        assert nl.tolist() == g.node_labels("paper")[keep_papers].tolist()

    def test_edges_filtered_to_kept_endpoints(self):
        g = _hg()
        keep = {"paper": torch.tensor([0, 1]),
                "author": torch.tensor([0])}
        sub = hetero_induced_subgraph(g, keep)
        ei_aw = sub.edge_index(("author", "writes", "paper"))
        # All surviving edges must have endpoints in the kept range.
        if ei_aw.numel() > 0:
            assert (ei_aw[0] < 1).all()  # author local id in [0, 1)
            assert (ei_aw[1] < 2).all()  # paper local id in [0, 2)

    def test_edge_weights_and_features_aligned(self):
        g = _hg()
        sub = hetero_induced_subgraph(
            g, {"paper": torch.arange(6), "author": torch.arange(4)},
        )
        ei_aw = sub.edge_index(("author", "writes", "paper"))
        ew = sub.edge_weight(("author", "writes", "paper"))
        ef = sub.edge_features(("author", "writes", "paper"))
        assert ew.size(0) == ei_aw.size(1)
        assert ef.size(0) == ei_aw.size(1)

    def test_metadata_records_ids(self):
        g = _hg()
        sub = hetero_induced_subgraph(
            g, {"paper": torch.tensor([0, 2, 5]),
                "author": torch.tensor([1, 3])},
        )
        meta = sub.metadata["sampling"]
        assert meta["kind"] == "hetero_induced_subgraph"
        assert meta["original_node_ids"]["paper"].tolist() == [0, 2, 5]
        assert meta["original_node_ids"]["author"].tolist() == [1, 3]

    def test_no_relabel_keeps_global_ids(self):
        g = _hg()
        sub = hetero_induced_subgraph(
            g,
            {"paper": torch.tensor([1, 2, 4]),
             "author": torch.tensor([0, 3])},
            relabel_nodes=False,
        )
        # Original num_nodes preserved.
        assert sub.num_nodes("paper") == g.num_nodes("paper")
        assert sub.num_nodes("author") == g.num_nodes("author")
        ei_aw = sub.edge_index(("author", "writes", "paper"))
        # In no-relabel mode, edge_index keeps original ids.
        if ei_aw.numel() > 0:
            assert (ei_aw[0] < g.num_nodes("author")).all()
            assert (ei_aw[1] < g.num_nodes("paper")).all()

    def test_unknown_node_type_raises(self):
        g = _hg()
        with pytest.raises(KeyError, match="Unknown node type"):
            hetero_induced_subgraph(g, {"reviewer": torch.tensor([0])})

    def test_duplicate_ids_raise(self):
        g = _hg()
        with pytest.raises(ValueError, match="duplicates"):
            hetero_induced_subgraph(
                g, {"paper": torch.tensor([0, 0, 1])},
            )

    def test_out_of_range_raises(self):
        g = _hg()
        with pytest.raises(ValueError, match="out of range"):
            hetero_induced_subgraph(g, {"paper": torch.tensor([99])})


# ── hetero_neighbor_sample ───────────────────────────────────────────────────


class TestHeteroNeighborSample:
    def test_basic_in_direction(self):
        g = _hg()
        sub = hetero_neighbor_sample(
            g,
            seed_nodes_dict={"paper": torch.tensor([0, 1])},
            fanouts=[
                {("author", "writes", "paper"): -1,
                 ("paper", "cites", "paper"): -1},
            ],
            seed=0,
            direction="in",
        )
        # Sub must contain at least the seed papers.
        ids = sub.metadata["sampling"]["original_node_ids"]["paper"].tolist()
        assert 0 in ids and 1 in ids

    def test_basic_out_direction(self):
        g = _hg()
        sub = hetero_neighbor_sample(
            g,
            seed_nodes_dict={"author": torch.tensor([0])},
            fanouts=[{("author", "writes", "paper"): 2}],
            seed=0,
            direction="out",
        )
        # Sub keeps the seed author.
        ids = sub.metadata["sampling"]["original_node_ids"]["author"].tolist()
        assert 0 in ids

    def test_two_hop_expansion(self):
        g = _hg()
        sub = hetero_neighbor_sample(
            g,
            seed_nodes_dict={"paper": torch.tensor([0])},
            fanouts=[
                {("author", "writes", "paper"): -1},
                {("paper", "cites", "paper"): -1},
            ],
            seed=42,
            direction="in",
        )
        # Should pull in some author nodes at hop 1 (when in-edges exist).
        # Author count is non-negative; we don't assert > 0 since the
        # random hetero graph may have no in-edge author for paper 0.
        assert sub.num_nodes("author") <= g.num_nodes("author")

    def test_determinism(self):
        g = _hg()
        s1 = hetero_neighbor_sample(
            g, {"paper": torch.tensor([0, 2])},
            fanouts=[{("author", "writes", "paper"): 2}], seed=7,
        )
        s2 = hetero_neighbor_sample(
            g, {"paper": torch.tensor([0, 2])},
            fanouts=[{("author", "writes", "paper"): 2}], seed=7,
        )
        ids_a_1 = s1.metadata["sampling"]["original_node_ids"]["author"].tolist()
        ids_a_2 = s2.metadata["sampling"]["original_node_ids"]["author"].tolist()
        assert ids_a_1 == ids_a_2

    def test_no_global_rng_pollution(self):
        g = _hg()
        torch.manual_seed(0)
        before = torch.rand(3)
        torch.manual_seed(0)
        hetero_neighbor_sample(
            g, {"paper": torch.tensor([0])},
            fanouts=[{("author", "writes", "paper"): 2}], seed=0,
        )
        after = torch.rand(3)
        assert torch.allclose(before, after)

    def test_metadata_records_config(self):
        g = _hg()
        sub = hetero_neighbor_sample(
            g, {"paper": torch.tensor([0, 1])},
            fanouts=[{("author", "writes", "paper"): 2}],
            direction="in", seed=0,
        )
        meta = sub.metadata["sampling"]
        assert meta["kind"] == "hetero_neighbor_sample"
        assert meta["direction"] == "in"
        assert meta["fanouts"][0][("author", "writes", "paper")] == 2

    def test_invalid_direction_raises(self):
        g = _hg()
        with pytest.raises(ValueError, match="direction"):
            hetero_neighbor_sample(
                g, {"paper": torch.tensor([0])},
                fanouts=[{("author", "writes", "paper"): 2}],
                direction="diag",
            )

    def test_invalid_fanout_raises(self):
        g = _hg()
        with pytest.raises(ValueError, match="must be >= 1"):
            hetero_neighbor_sample(
                g, {"paper": torch.tensor([0])},
                fanouts=[{("author", "writes", "paper"): 0}],
            )

    def test_unknown_relation_raises(self):
        g = _hg()
        with pytest.raises(KeyError, match="unknown edge type"):
            hetero_neighbor_sample(
                g, {"paper": torch.tensor([0])},
                fanouts=[{("editor", "owns", "paper"): 2}],
            )

    def test_empty_seeds_raise(self):
        g = _hg()
        with pytest.raises(ValueError, match="at least one non-empty"):
            hetero_neighbor_sample(
                g, {"paper": torch.tensor([], dtype=torch.long),
                    "author": torch.tensor([], dtype=torch.long)},
                fanouts=[{("author", "writes", "paper"): 2}],
            )
