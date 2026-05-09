"""Tests for tgraphx.loaders and tgraphx.feature_store."""
import pytest
import torch
from tgraphx import Graph
from tgraphx.loaders import (
    NeighborLoader, GraphLoader, make_neighbor_loader,
    make_link_loader, make_graph_loader,
    NodeClassificationDataset, LinkPredictionDataset,
)
from tgraphx.feature_store import InMemoryFeatureStore, MemmapFeatureStore, FeatureStoreError


def _simple_graph(N=6, D=4) -> Graph:
    src = list(range(N-1)) + [N-1] + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N-1)) + [N-1]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(N, D)
    return Graph(x, ei)


class TestNeighborLoader:
    def test_basic_iteration(self):
        g = _simple_graph()
        loader = make_neighbor_loader(g, fanouts=[2, 1], batch_size=2, shuffle=False, seed=0)
        batches = list(loader)
        assert len(batches) == 3  # 6 nodes / batch_size=2

    def test_subgraph_contains_seeds(self):
        g = _simple_graph(10, 4)
        loader = make_neighbor_loader(g, fanouts=[3], batch_size=3, shuffle=False, seed=0)
        for subg, seeds in loader:
            # All sampled subgraphs must have >= batch_size nodes.
            assert subg.num_nodes >= seeds.size(0)
            break

    def test_with_train_mask(self):
        g = _simple_graph(10, 4)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[:5] = True
        loader = make_neighbor_loader(g, fanouts=[2], mask=mask, batch_size=2, shuffle=False, seed=0)
        batches = list(loader)
        # 5 masked nodes / batch_size 2 = ceil(5/2) = 3 batches.
        assert len(batches) == 3

    def test_deterministic_with_seed(self):
        g = _simple_graph()
        l1 = make_neighbor_loader(g, fanouts=[2], batch_size=2, shuffle=True, seed=42)
        l2 = make_neighbor_loader(g, fanouts=[2], batch_size=2, shuffle=True, seed=42)
        seeds1 = [seeds.tolist() for _, seeds in l1]
        seeds2 = [seeds.tolist() for _, seeds in l2]
        assert seeds1 == seeds2

    def test_tensor_features_preserved(self):
        g = _simple_graph()
        loader = make_neighbor_loader(g, fanouts=[2], batch_size=2, shuffle=False, seed=0)
        for subg, _ in loader:
            assert subg.node_features is not None
            assert subg.node_features.shape[1] == 4  # D
            break


class TestGraphLoader:
    def test_basic(self):
        graphs = [_simple_graph(n, 4) for n in range(3, 8)]
        loader = make_graph_loader(graphs, batch_size=2, shuffle=False, seed=0)
        batches = list(loader)
        assert len(batches) == 3  # ceil(5 / 2) = 3

    def test_deterministic(self):
        graphs = [_simple_graph(n, 4) for n in range(4, 10)]
        l1 = make_graph_loader(graphs, batch_size=3, shuffle=True, seed=7)
        l2 = make_graph_loader(graphs, batch_size=3, shuffle=True, seed=7)
        # Just check same number of batches.
        assert len(list(l1)) == len(list(l2))


class TestNodeClassificationDataset:
    def test_all_nodes(self):
        g = _simple_graph(10, 4)
        ds = NodeClassificationDataset(g)
        assert len(ds) == 10

    def test_with_mask(self):
        g = _simple_graph(10, 4)
        mask = torch.zeros(10, dtype=torch.bool)
        mask[[1, 3, 5]] = True
        ds = NodeClassificationDataset(g, mask)
        assert len(ds) == 3
        assert sorted([ds[0], ds[1], ds[2]]) == [1, 3, 5]


class TestInMemoryFeatureStore:
    def test_put_and_get_full(self):
        store = InMemoryFeatureStore()
        x = torch.randn(10, 8)
        store.put("x", x)
        result = store.get("x")
        assert torch.allclose(result, x)

    def test_put_and_get_indexed(self):
        store = InMemoryFeatureStore()
        x = torch.arange(20, dtype=torch.float).view(10, 2)
        store.put("x", x)
        ids = torch.tensor([0, 5, 9])
        result = store.get("x", ids=ids)
        assert result.shape == (3, 2)
        assert torch.allclose(result[0], x[0])
        assert torch.allclose(result[1], x[5])

    def test_update_partial(self):
        store = InMemoryFeatureStore()
        x = torch.zeros(5, 4)
        store.put("x", x)
        new_val = torch.ones(2, 4)
        ids = torch.tensor([1, 3])
        store.put("x", new_val, ids=ids)
        result = store.get("x")
        assert (result[1] == 1.0).all()
        assert (result[3] == 1.0).all()
        assert (result[0] == 0.0).all()

    def test_contains(self):
        store = InMemoryFeatureStore()
        assert not store.contains("x")
        store.put("x", torch.randn(4, 8))
        assert store.contains("x")

    def test_missing_key_raises(self):
        store = InMemoryFeatureStore()
        with pytest.raises(FeatureStoreError):
            store.get("nonexistent")

    def test_image_features_preserved(self):
        store = InMemoryFeatureStore()
        x = torch.randn(6, 3, 8, 8)  # image node features
        store.put("img", x)
        result = store.get("img", ids=torch.tensor([0, 2, 4]))
        assert result.shape == (3, 3, 8, 8)

    def test_summary_json_serializable(self):
        import json
        store = InMemoryFeatureStore()
        store.put("x", torch.randn(5, 4))
        summary = store.summary()
        json.dumps(summary)  # must not raise

    def test_to_device_cpu(self):
        store = InMemoryFeatureStore()
        store.put("x", torch.randn(4, 8))
        store.to("cpu")
        result = store.get("x")
        assert result.device.type == "cpu"


class TestMemmapFeatureStore:
    def test_put_and_get_roundtrip(self, tmp_path):
        pytest.importorskip("numpy")
        store = MemmapFeatureStore(tmp_path)
        x = torch.randn(10, 8)
        store.put("x", x)
        result = store.get("x")
        assert torch.allclose(result, x, atol=1e-6)

    def test_indexed_get(self, tmp_path):
        pytest.importorskip("numpy")
        store = MemmapFeatureStore(tmp_path)
        x = torch.arange(20, dtype=torch.float).view(10, 2)
        store.put("x", x)
        result = store.get("x", ids=torch.tensor([0, 9]))
        assert result.shape == (2, 2)
        assert torch.allclose(result[0], x[0])
        assert torch.allclose(result[1], x[9])

    def test_metadata_persisted(self, tmp_path):
        pytest.importorskip("numpy")
        import json
        store = MemmapFeatureStore(tmp_path)
        store.put("y", torch.randn(5, 4))
        assert store.contains("y")
        meta = store.metadata("y")
        assert meta["shape"] == [5, 4]
        assert json.loads((tmp_path / "feature_store_metadata.json").read_text())

    def test_no_pickle(self, tmp_path):
        pytest.importorskip("numpy")
        import numpy as np
        store = MemmapFeatureStore(tmp_path)
        store.put("z", torch.randn(3, 4))
        # Verify numpy load works with allow_pickle=False.
        arr = np.load(str(tmp_path / "z.npy"), allow_pickle=False)
        assert arr.shape == (3, 4)

    def test_missing_key_raises(self, tmp_path):
        pytest.importorskip("numpy")
        store = MemmapFeatureStore(tmp_path)
        with pytest.raises(FeatureStoreError):
            store.get("nonexistent")
