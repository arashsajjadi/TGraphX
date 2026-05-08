"""Tests for tgraphx.plotting — graph and mining visualization.

All tests use the Agg (headless) Matplotlib backend.
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

# Force headless backend before importing matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _close_mpl_figures():
    """Close all matplotlib figures after each test to avoid memory warnings."""
    yield
    plt.close("all")


class TestLayouts:
    def test_circular(self):
        from tgraphx.plotting import circular_layout
        import numpy as np
        pos = circular_layout(5)
        assert pos.shape == (5, 2)
        # All points on unit circle.
        norms = (pos ** 2).sum(axis=1) ** 0.5
        assert all(abs(n - 1.0) < 1e-9 for n in norms)

    def test_circular_zero_nodes(self):
        from tgraphx.plotting import circular_layout
        pos = circular_layout(0)
        assert pos.shape == (0, 2)

    def test_grid(self):
        from tgraphx.plotting import grid_layout
        pos = grid_layout(9)
        assert pos.shape == (9, 2)

    def test_random(self):
        from tgraphx.plotting import random_layout
        pos1 = random_layout(10, seed=42)
        pos2 = random_layout(10, seed=42)
        assert (pos1 == pos2).all()
        # Values in [0,1].
        assert (pos1 >= 0).all() and (pos1 <= 1).all()

    def test_spring_empty(self):
        from tgraphx.plotting import spring_layout
        ei = torch.zeros((2, 0), dtype=torch.long)
        pos = spring_layout(ei, 4, iterations=5, seed=0)
        assert pos.shape == (4, 2)

    def test_spring_chain(self):
        from tgraphx.plotting import spring_layout
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        pos = spring_layout(ei, 4, iterations=10, seed=0)
        assert pos.shape == (4, 2)


class TestGraphPlots:
    def test_plot_graph_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from tgraphx.plotting import plot_graph
        ei = torch.tensor([[0,1,2],[1,2,0]], dtype=torch.long)
        fig, ax = plot_graph(ei, 3, seed=0)
        assert fig is not None and ax is not None

    def test_plot_graph_empty(self):
        from tgraphx.plotting import plot_graph
        ei = torch.zeros((2, 0), dtype=torch.long)
        fig, ax = plot_graph(ei, 0, seed=0)
        assert fig is not None

    def test_plot_graph_with_values(self):
        from tgraphx.plotting import plot_graph
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        vals = torch.tensor([0.1, 0.5, 0.9])
        fig, ax = plot_graph(ei, 3, node_values=vals, layout="circular")
        assert fig is not None

    def test_plot_graph_circular_layout(self):
        from tgraphx.plotting import plot_graph
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        fig, ax = plot_graph(ei, 3, layout="circular")
        assert fig is not None

    def test_plot_graph_size_guard(self):
        from tgraphx.plotting import plot_graph
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes"):
            plot_graph(ei, 501, max_nodes=500)

    def test_plot_degree_distribution(self):
        from tgraphx.plotting import plot_degree_distribution
        ei = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
        fig, ax = plot_degree_distribution(ei, 4)
        assert fig is not None

    def test_plot_adjacency_matrix(self):
        from tgraphx.plotting import plot_adjacency_matrix
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        fig, ax = plot_adjacency_matrix(ei, 3)
        assert fig is not None

    def test_plot_adjacency_matrix_size_guard(self):
        from tgraphx.plotting import plot_adjacency_matrix
        ei = torch.zeros((2, 0), dtype=torch.long)
        with pytest.raises(ValueError, match="max_nodes"):
            plot_adjacency_matrix(ei, 101, max_nodes=100)

    def test_plot_connected_components(self):
        from tgraphx.plotting import plot_connected_components
        ei = torch.tensor([[0,1],[1,2]], dtype=torch.long)
        fig, ax = plot_connected_components(ei, 3)
        assert fig is not None


class TestMiningPlots:
    def test_plot_motif_summary(self):
        from tgraphx.plotting import plot_motif_summary
        fig, ax = plot_motif_summary({"edges": 3, "triangles": 1, "wedges": 3})
        assert fig is not None

    def test_plot_graph_mining_summary(self):
        from tgraphx.plotting import plot_graph_mining_summary
        s = {"num_nodes": 10, "num_edges": 15, "density": 0.3,
             "mean_total_degree": 3.0, "num_connected_components": 2}
        fig, ax = plot_graph_mining_summary(s)
        assert fig is not None

    def test_plot_link_prediction_scores(self):
        from tgraphx.plotting import plot_link_prediction_score_distribution
        scores = {"jaccard": [0.1, 0.3, 0.5, 0.2], "adamic_adar": [0.4, 0.6, 0.8]}
        fig, ax = plot_link_prediction_score_distribution(scores)
        assert fig is not None

    def test_plot_similarity_heatmap(self):
        from tgraphx.plotting import plot_graph_similarity_heatmap
        M = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
        fig, ax = plot_graph_similarity_heatmap(M, labels=["G1", "G2"])
        assert fig is not None

    def test_plot_similarity_heatmap_size_guard(self):
        from tgraphx.plotting import plot_graph_similarity_heatmap
        M = torch.zeros(60, 60)
        with pytest.raises(ValueError, match="max_size"):
            plot_graph_similarity_heatmap(M, max_size=50)

    def test_plot_anomaly_scores(self):
        from tgraphx.plotting import plot_anomaly_scores
        scores = torch.tensor([0.1, 2.5, 0.2, 1.8, 0.3])
        fig, ax = plot_anomaly_scores(scores, top_k=3)
        assert fig is not None

    def test_plot_prototype_membership_scores(self):
        from tgraphx.plotting import plot_prototype_membership_scores
        scores = {0: 0.8, 1: 0.3, 2: 0.1}
        fig, ax = plot_prototype_membership_scores(scores, true_label=0)
        assert fig is not None

    def test_plot_confusion_matrix(self):
        from tgraphx.plotting import plot_confusion_matrix
        M = [[5, 1], [2, 4]]
        fig, ax = plot_confusion_matrix(M, class_names=["A", "B"])
        assert fig is not None

    def test_plot_confusion_matrix_normalize(self):
        from tgraphx.plotting import plot_confusion_matrix
        M = [[10, 2], [3, 8]]
        fig, ax = plot_confusion_matrix(M, normalize=True)
        assert fig is not None

    def test_plot_training_curves_list(self):
        from tgraphx.plotting import plot_training_curves
        history = [{"train_loss": 0.9}, {"train_loss": 0.7}, {"train_loss": 0.5}]
        fig, ax = plot_training_curves(history)
        assert fig is not None

    def test_plot_training_curves_dict(self):
        from tgraphx.plotting import plot_training_curves
        data = {"train_loss": [0.9, 0.7, 0.5], "val_loss": [0.95, 0.8, 0.6]}
        fig, ax = plot_training_curves(data)
        assert fig is not None

    def test_plot_training_curves_empty(self):
        from tgraphx.plotting import plot_training_curves
        fig, ax = plot_training_curves([])
        assert fig is not None

    def test_plot_community_assignments(self):
        from tgraphx.plotting import plot_community_assignments
        ei = torch.tensor([[0,1,3,4],[1,2,4,5]], dtype=torch.long)
        comms = torch.tensor([0, 0, 0, 1, 1, 1])
        fig, ax = plot_community_assignments(ei, 6, comms)
        assert fig is not None


class TestSaveFigure:
    def test_save_png(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tgraphx.plotting import save_figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_figure(fig, os.path.join(tmp, "test"), formats=("png",))
            assert len(paths) == 1
            assert os.path.exists(paths[0])
            assert paths[0].endswith(".png")

    def test_save_svg(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tgraphx.plotting import save_figure
        fig, _ = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_figure(fig, os.path.join(tmp, "test"), formats=("svg",))
            assert paths[0].endswith(".svg")
            assert os.path.exists(paths[0])

    def test_save_multiple_formats(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tgraphx.plotting import save_figure
        fig, _ = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_figure(fig, os.path.join(tmp, "test"), formats=("png", "svg"))
            assert len(paths) == 2
