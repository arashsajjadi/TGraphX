"""v1.5.0 SetTransformer (learned implicit relations) test suite.

Covers the section-7B/7C requirements: shapes, variable node counts,
padding-mask correctness, permutation invariance/equivariance, gradients,
CPU/CUDA parity, serialization, deterministic construction, the
edge_index contract, registry/factory integration, batching/evaluator
integration, and two tiny non-scientific sanity checks (memorization and
relation-dependence).  The sanity checks are optimization smoke tests on
synthetic data — they are NOT empirical benchmark results.
"""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from tgraphx import (
    Graph,
    GraphDataLoader,
    GraphDataset,
    SetTransformerModel,
    TOPOLOGY_SOURCES,
    TopologyIgnoredWarning,
    accuracy,
    build_model,
    build_model_from_config,
    evaluate,
    fit,
    global_mean_pool,
    load_checkpoint,
    save_checkpoint,
    topology_source_of,
)

EMB = 16


def _model(task="graph_classification", in_shape=(8,), **kw) -> SetTransformerModel:
    defaults = dict(embed_dim=EMB, num_layers=2, num_heads=2,
                    num_classes=3 if task.endswith("classification") else None,
                    out_dim=None if task.endswith("classification") else 2)
    defaults.update(kw)
    return SetTransformerModel(task, in_shape, **defaults)


def _batch(n_per_graph=(4, 6), dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(sum(n_per_graph), dim, generator=g)
    batch = torch.cat([
        torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_per_graph)
    ])
    return x, batch


# ──────────────────────────────────────────────────────────────────── #
# Shapes / input validation                                             #
# ──────────────────────────────────────────────────────────────────── #

def test_graph_classification_output_shape() -> None:
    m = _model()
    x, b = _batch()
    assert m(x, None, batch=b).shape == (2, 3)


def test_graph_regression_output_shape() -> None:
    m = _model("graph_regression")
    x, b = _batch()
    assert m(x, None, batch=b).shape == (2, 2)


def test_node_level_output_shapes() -> None:
    x, b = _batch()
    assert _model("node_classification")(x, None, batch=b).shape == (10, 3)
    assert _model("node_regression")(x, None, batch=b).shape == (10, 2)


def test_variable_node_counts_single_forward() -> None:
    m = _model()
    x, b = _batch(n_per_graph=(1, 7, 3, 12))
    assert m(x, None, batch=b).shape == (4, 3)


def test_batch_none_treats_all_nodes_as_one_set() -> None:
    m = _model()
    x, _ = _batch()
    assert m(x, None).shape == (1, 3)


def test_spatial_and_volumetric_inputs() -> None:
    m2d = _model(in_shape=(3, 8, 8), num_layers=1)
    x = torch.randn(5, 3, 8, 8)
    assert m2d(x, None, batch=torch.tensor([0, 0, 1, 1, 1])).shape == (2, 3)
    m3d = _model(in_shape=(2, 4, 4, 4), num_layers=1)
    xv = torch.randn(4, 2, 4, 4, 4)
    assert m3d(xv, None, batch=torch.tensor([0, 0, 0, 1])).shape == (2, 3)


def test_wrong_input_shape_rejected() -> None:
    m = _model()
    with pytest.raises(ValueError, match="expected node features"):
        m(torch.randn(4, 5), None)
    with pytest.raises(ValueError, match="batch has"):
        m(torch.randn(4, 8), None, batch=torch.zeros(3, dtype=torch.long))


def test_empty_graph_in_batch_rejected() -> None:
    m = _model()
    x = torch.randn(3, 8)
    with pytest.raises(ValueError, match="zero"):
        m(x, None, batch=torch.tensor([0, 0, 2]))  # graph 1 empty


def test_invalid_constructor_args_rejected() -> None:
    with pytest.raises(ValueError, match="task"):
        SetTransformerModel("edge_prediction", (8,), num_classes=2)
    with pytest.raises(ValueError, match="num_classes"):
        SetTransformerModel("graph_classification", (8,))
    with pytest.raises(ValueError, match="out_dim"):
        SetTransformerModel("graph_regression", (8,))
    with pytest.raises(ValueError, match="pooling"):
        _model(pooling="median")
    with pytest.raises(ValueError, match="divisible"):
        _model(embed_dim=15, num_heads=2)
    with pytest.raises(ValueError, match="on_edge_index"):
        _model(on_edge_index="explode")
    with pytest.raises(ValueError, match="not both"):
        _model(encoder=nn.Linear(8, EMB), encoder_config={"dropout_prob": 0.0})


# ──────────────────────────────────────────────────────────────────── #
# Permutation properties / padding masks                                #
# ──────────────────────────────────────────────────────────────────── #

def test_permutation_invariance_of_graph_output() -> None:
    torch.manual_seed(0)
    m = _model().eval()
    x, b = _batch()
    perm = torch.randperm(x.size(0))
    out = m(x, None, batch=b)
    out_p = m(x[perm], None, batch=b[perm])
    assert torch.allclose(out, out_p, atol=1e-5)


@pytest.mark.parametrize("pooling", ["attention", "mean", "sum", "max"])
def test_permutation_invariance_all_poolings(pooling) -> None:
    torch.manual_seed(0)
    m = _model(pooling=pooling).eval()
    x, b = _batch()
    perm = torch.randperm(x.size(0))
    assert torch.allclose(
        m(x, None, batch=b), m(x[perm], None, batch=b[perm]), atol=1e-5
    )


def test_permutation_equivariance_of_node_embeddings() -> None:
    torch.manual_seed(0)
    m = _model().eval()
    x, b = _batch()
    perm = torch.randperm(x.size(0))
    emb = m.encode_nodes(x, b)
    emb_p = m.encode_nodes(x[perm], b[perm])
    assert emb.shape == (10, EMB)
    assert torch.allclose(emb[perm], emb_p, atol=1e-5)


def test_padding_mask_isolates_graphs() -> None:
    """Growing one graph in the batch must not change another graph's output."""
    torch.manual_seed(0)
    m = _model().eval()
    x, b = _batch(n_per_graph=(4, 6))
    out = m(x, None, batch=b)
    x2 = torch.cat([x, torch.randn(5, 8)])
    b2 = torch.cat([b, torch.full((5,), 1, dtype=torch.long)])
    out2 = m(x2, None, batch=b2)
    assert torch.allclose(out[0], out2[0], atol=1e-5)


def test_batched_equals_individual_forward() -> None:
    torch.manual_seed(0)
    m = _model().eval()
    x, b = _batch(n_per_graph=(3, 5))
    out = m(x, None, batch=b)
    out_a = m(x[:3], None)
    out_b = m(x[3:], None)
    assert torch.allclose(out, torch.cat([out_a, out_b]), atol=1e-5)


# ──────────────────────────────────────────────────────────────────── #
# edge_index contract                                                   #
# ──────────────────────────────────────────────────────────────────── #

def test_edge_index_warns_once_by_default() -> None:
    m = _model()
    x, b = _batch()
    ei = torch.tensor([[0, 1], [1, 0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m(x, ei, batch=b)
        m(x, ei, batch=b)
    hits = [w for w in caught if issubclass(w.category, TopologyIgnoredWarning)]
    assert len(hits) == 1, "must warn exactly once per instance"


def test_edge_index_ignore_mode_is_silent_and_output_independent() -> None:
    torch.manual_seed(0)
    m = _model(on_edge_index="ignore").eval()
    x, b = _batch()
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ei_shuffled = ei[:, torch.tensor([2, 0, 1])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_none = m(x, None, batch=b)
        out_ei = m(x, ei, batch=b)
        out_shuf = m(x, ei_shuffled, batch=b)
    assert not any(
        issubclass(w.category, TopologyIgnoredWarning) for w in caught
    )
    # Output cannot depend on the supplied edges or their order.
    assert torch.equal(out_none, out_ei)
    assert torch.equal(out_ei, out_shuf)


def test_edge_index_error_mode_rejects() -> None:
    m = _model(on_edge_index="error")
    x, b = _batch()
    with pytest.raises(ValueError, match="learned_implicit"):
        m(x, torch.tensor([[0], [1]]), batch=b)


def test_topology_metadata() -> None:
    m = _model()
    assert m.topology_source == "learned_implicit"
    assert m.model_family == "set_transformer"
    assert "learned_implicit" in TOPOLOGY_SOURCES
    assert topology_source_of("set_transformer") == "learned_implicit"
    assert topology_source_of("conv") == "given"
    assert topology_source_of("gat") == "given"
    assert topology_source_of("graph_transformer") == "learned_implicit"
    assert topology_source_of("graph_transformer", edge_bias=True) == "hybrid"
    with pytest.raises(KeyError):
        topology_source_of("not_a_family")


# ──────────────────────────────────────────────────────────────────── #
# Gradients / numerical health                                          #
# ──────────────────────────────────────────────────────────────────── #

def test_forward_backward_finite_and_grads_reach_all_parts() -> None:
    torch.manual_seed(0)
    m = _model(in_shape=(3, 8, 8), num_layers=2)
    x = torch.randn(6, 3, 8, 8, requires_grad=True)
    b = torch.tensor([0, 0, 0, 1, 1, 1])
    out = m(x, None, batch=b)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    def has_grad(module: nn.Module) -> bool:
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        return bool(grads) and all(
            g is not None and torch.isfinite(g).all() for g in grads
        )

    assert has_grad(m.encoder), "no gradient reached the node encoder"
    for i, block in enumerate(m.blocks):
        assert has_grad(block), f"no gradient reached attention block {i}"
    assert has_grad(m.pool), "no gradient reached the attention pooling"
    assert has_grad(m.head)


def test_dropout_explicit_and_visible() -> None:
    m = _model(dropout=0.3, attention_dropout=0.1)
    assert m.dropout == 0.3 and m.attention_dropout == 0.1
    r = repr(m)
    assert "dropout=0.3" in r and "attention_dropout=0.1" in r
    cfg = m.config()
    assert cfg["dropout"] == 0.3 and cfg["attention_dropout"] == 0.1
    # dropout=0 must have no active dropout modules with p > 0
    m0 = _model(dropout=0.0)
    assert all(
        d.p == 0.0
        for d in m0.modules()
        if isinstance(d, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
    )


# ──────────────────────────────────────────────────────────────────── #
# Determinism / serialization                                           #
# ──────────────────────────────────────────────────────────────────── #

def test_fixed_seed_construction_is_deterministic() -> None:
    torch.manual_seed(123)
    m1 = _model()
    torch.manual_seed(123)
    m2 = _model()
    for (n1, p1), (n2, p2) in zip(
        m1.state_dict().items(), m2.state_dict().items()
    ):
        assert n1 == n2 and torch.equal(p1, p2), n1


def test_config_round_trip_exact_reconstruction() -> None:
    torch.manual_seed(0)
    m = _model(dropout=0.1, pooling="attention", num_seeds=2,
               encoder_config={"hidden_channels": 24, "dropout_prob": 0.0})
    cfg = m.config()
    m2 = SetTransformerModel.from_config(cfg)
    assert m2.config() == cfg
    m2.load_state_dict(m.state_dict())
    m.eval()
    m2.eval()
    x, b = _batch()
    assert torch.equal(m(x, None, batch=b), m2(x, None, batch=b))


def test_custom_encoder_config_refuses_reconstruction() -> None:
    enc = nn.Linear(8, EMB)
    m = _model(encoder=enc)
    cfg = m.config()
    assert cfg["encoder"] == "custom"
    with pytest.raises(ValueError, match="custom"):
        SetTransformerModel.from_config(cfg)


def test_checkpoint_save_load_exact(tmp_path) -> None:
    torch.manual_seed(0)
    m = _model(dropout=0.0)
    path = str(tmp_path / "set_transformer.pt")
    save_checkpoint(m, None, epoch=3, path=path, config=m.config())
    payload = torch.load(path, weights_only=True)
    rebuilt = SetTransformerModel.from_config(payload["config"])
    load_checkpoint(rebuilt, None, path)
    m.eval()
    rebuilt.eval()
    x, b = _batch()
    assert torch.equal(m(x, None, batch=b), rebuilt(x, None, batch=b))


@pytest.mark.cuda
def test_cpu_cuda_parity() -> None:
    torch.manual_seed(0)
    m = _model(in_shape=(3, 8, 8), num_layers=1).eval()
    x = torch.randn(7, 3, 8, 8)
    b = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    out_cpu = m(x, None, batch=b)
    mc = m.cuda()
    out_gpu = mc(x.cuda(), None, batch=b.cuda()).cpu()
    assert torch.allclose(out_cpu, out_gpu, atol=1e-4)


# ──────────────────────────────────────────────────────────────────── #
# Factory / registry integration                                        #
# ──────────────────────────────────────────────────────────────────── #

def test_build_model_constructs_set_transformer() -> None:
    m = build_model(
        task="graph_classification", layer="set_transformer",
        in_shape=(8,), hidden_shape=(EMB,), num_layers=2, num_classes=3,
        heads=2, dropout=0.0,
    )
    assert isinstance(m, SetTransformerModel)
    assert m.model_family == "set_transformer"
    assert m.topology_source == "learned_implicit"
    x, b = _batch()
    assert m(x, None, batch=b).shape == (2, 3)


def test_build_model_family_alias() -> None:
    m = build_model(
        task="graph_classification", family="set_transformer",
        in_shape=(8,), hidden_shape=(EMB,), num_layers=1, num_classes=3,
    )
    assert isinstance(m, SetTransformerModel)
    with pytest.raises(ValueError, match="aliases"):
        build_model(
            task="graph_classification", layer="conv", family="gat",
            in_shape=(4, 4, 4), hidden_shape=(4, 4, 4), num_layers=1,
            num_classes=2,
        )


def test_build_model_tags_given_topology_families() -> None:
    m = build_model(
        "graph_classification", "conv", (4, 4, 4), (4, 4, 4),
        num_layers=1, num_classes=2, dropout=0.0,
    )
    assert m.model_family == "conv"
    assert m.topology_source == "given"


def test_build_model_set_transformer_rejects_edge_prediction() -> None:
    with pytest.raises(ValueError, match="edge_prediction"):
        build_model(
            "edge_prediction", "set_transformer", (8,), (EMB,),
            num_layers=1, out_dim=1,
        )
    with pytest.raises(ValueError, match="hidden_shape"):
        build_model(
            "graph_classification", "set_transformer", (8,), (EMB, 4, 4),
            num_layers=1, num_classes=2,
        )


def test_make_layer_rejects_set_transformer_with_hint() -> None:
    from tgraphx import make_layer
    with pytest.raises(ValueError, match="model-level family"):
        make_layer("set_transformer", (8,), (EMB,))


def test_build_model_from_config_set_transformer() -> None:
    cfg = {
        "model": {
            "task": "graph_classification",
            "layer": "set_transformer",
            "in_shape": [8],
            "hidden_shape": [EMB],
            "num_layers": 2,
            "num_classes": 3,
            "heads": 2,
            "dropout": 0.0,
            "pooling": "attention",
            "on_edge_index": "ignore",
        }
    }
    m = build_model_from_config(cfg)
    assert isinstance(m, SetTransformerModel)
    assert m.on_edge_index == "ignore"
    x, b = _batch()
    assert m(x, None, batch=b).shape == (2, 3)


def test_pipeline_fit_and_evaluate_with_graphbatch() -> None:
    torch.manual_seed(0)
    graphs = []
    for i in range(8):
        n = 3 + i % 3
        graphs.append(Graph(
            torch.randn(n, 8),
            torch.tensor([[j, (j + 1) % n] for j in range(n)]).t(),
            graph_label=torch.tensor(i % 2),
        ))
    loader = GraphDataLoader(GraphDataset(graphs), batch_size=4)
    m = build_model(
        "graph_classification", "set_transformer", (8,), (EMB,),
        num_layers=1, num_classes=2, heads=2, on_edge_index="ignore",
    )
    fit(m, loader, epochs=2, loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(m.parameters(), lr=1e-3), device="cpu")
    result = evaluate(m, loader, nn.CrossEntropyLoss(),
                      metrics={"acc": accuracy}, device="cpu")
    assert "acc" in result and 0.0 <= result["acc"] <= 1.0


# ──────────────────────────────────────────────────────────────────── #
# Tiny NON-SCIENTIFIC sanity checks (synthetic; not benchmark results)  #
# ──────────────────────────────────────────────────────────────────── #

def test_sanity_memorizes_tiny_synthetic_dataset() -> None:
    torch.manual_seed(0)
    m = SetTransformerModel("graph_classification", (8,), embed_dim=16,
                            num_layers=1, num_heads=2, num_classes=2)
    x = torch.randn(24, 8)
    b = torch.repeat_interleave(torch.arange(6), 4)
    y = torch.tensor([0, 1, 0, 1, 1, 0])
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    lf = nn.CrossEntropyLoss()
    for _ in range(200):
        m.train()
        opt.zero_grad()
        lf(m(x, None, batch=b), y).backward()
        opt.step()
    m.eval()
    acc = (m(x, None, batch=b).argmax(1) == y).float().mean().item()
    assert acc == 1.0, f"failed to memorize 6 tiny sets (acc={acc})"


def _make_key_query_sets(n_sets: int, seed: int):
    """Label = sign of the value carried by the token whose key matches the
    query token's key.  Requires token-token matching (relations); a
    mean-pool over per-token features loses the pairing."""
    num_keys, dim = 6, 8  # key one-hot | value | query flag
    g = torch.Generator().manual_seed(seed)
    xs, batches, ys = [], [], []
    for i in range(n_sets):
        keys = torch.randperm(num_keys, generator=g)[:5]
        vals = torch.randint(0, 2, (5,), generator=g).float() * 2 - 1
        toks = torch.zeros(6, dim)
        for j in range(5):
            toks[j, keys[j]] = 1.0
            toks[j, num_keys] = vals[j]
        qpos = int(torch.randint(0, 5, (1,), generator=g).item())
        toks[5, keys[qpos]] = 1.0
        toks[5, num_keys + 1] = 1.0
        perm = torch.randperm(6, generator=g)
        xs.append(toks[perm])
        batches.append(torch.full((6,), i, dtype=torch.long))
        ys.append(int(vals[qpos] > 0))
    return torch.cat(xs), torch.cat(batches), torch.tensor(ys)


class _MeanPoolOnlyBaseline(nn.Module):
    """Per-token MLP + mean pooling: no token-token relations at all."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),
                                  nn.Linear(64, 2))

    def forward(self, x, edge_index=None, batch=None):
        return self.head(global_mean_pool(self.enc(x), batch))


def test_sanity_relation_dependent_task_beats_pooling_only_baseline() -> None:
    """SetTransformer must exploit token-token relations that a pooling-only
    model cannot represent.  Fixed seeds; a coarse sanity margin, not a
    scientific benchmark."""
    x, b, y = _make_key_query_sets(300, seed=0)
    xv, bv, yv = _make_key_query_sets(120, seed=1)

    def train(model):
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lf = nn.CrossEntropyLoss()
        for _ in range(300):
            model.train()
            opt.zero_grad()
            lf(model(x, None, batch=b), y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            return (model(xv, None, batch=bv).argmax(1) == yv).float().mean().item()

    torch.manual_seed(0)
    st = SetTransformerModel("graph_classification", (8,), embed_dim=32,
                             num_layers=2, num_heads=4, num_classes=2)
    torch.manual_seed(0)
    baseline = _MeanPoolOnlyBaseline()
    acc_st = train(st)
    acc_base = train(baseline)
    assert acc_st >= 0.93, f"set attention failed the relation task ({acc_st})"
    assert acc_st >= acc_base + 0.05, (
        f"no relation advantage: set={acc_st} pool={acc_base}"
    )
