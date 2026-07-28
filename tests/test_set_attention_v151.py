"""Tests for the 1.5.1 set-attention update.

Covers: the canonical ``TGraphXSetAttention`` name and the
``SetTransformerModel`` compatibility alias; the factory family aliases;
the new architecture axes (``norm_order``, ``activation``,
``pool_attention_dropout``, ``head_hidden_dim``, strided encoder); the
evaluated reference configuration (``reference_config`` /
``map_reference_state_dict`` / ``from_reference_state_dict``) including a
self-contained torch-primitives replica parity check; config/checkpoint
round trips including v1.5.0-style configs; permutation properties and
padding-mask correctness for the post-norm configuration; CPU/CUDA parity;
and (when the external evidence tree is present) strict checkpoint mapping
from the completed revised experiment.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import tgraphx
from tgraphx import (
    AttentionPooling,
    SetAttentionBlock,
    SetTransformerModel,
    StridedConvEncoder,
    TGraphXSetAttention,
    build_model,
)
from tgraphx.models.topology import TopologyIgnoredWarning, topology_source_of

REVISED_CKPT_DIR = Path(
    "/home/arash/PycharmProjects/_families/TGraphX/TGraphX_revised/checkpoints/frozen_base"
)

IN_SHAPE = (13, 32, 32)
NUM_CLASSES = 18


def _reference_model(in_shape=IN_SHAPE, num_classes=NUM_CLASSES):
    cfg = TGraphXSetAttention.reference_config(in_shape, num_classes)
    return TGraphXSetAttention(**cfg)


class _TorchPrimitivesReference(nn.Module):
    """Replica of the evaluation program's SetTransformer built from torch
    primitives only — used to prove the state-dict mapping and numerical
    parity without external files."""

    def __init__(self, in_channels: int, num_classes: int, embed_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, ffn: int = 128):
        super().__init__()
        blocks, c_in = [], in_channels
        for i in range(3):
            c_out = 32 * (2 ** i)
            blocks += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                          stride=2 if i > 0 else 1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.encoder = nn.Module()
        self.encoder.conv = nn.Sequential(*blocks)
        self.encoder.pool = nn.AdaptiveAvgPool2d(1)
        self.encoder.proj = nn.Linear(c_in, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn,
            batch_first=True)  # post-norm, ReLU, dropout 0.1 (torch defaults)
        self.self_attn = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pma = nn.Module()
        self.pma.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pma.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                              batch_first=True)
        self.head = nn.Module()
        self.head.net = nn.Linear(embed_dim, num_classes)

    def forward(self, x, batch):
        h = self.encoder.proj(
            self.encoder.pool(self.encoder.conv(x)).flatten(1))
        num_graphs = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=num_graphs)
        max_n = int(counts.max())
        padded = h.new_zeros(num_graphs, max_n, h.size(-1))
        mask = torch.zeros(num_graphs, max_n, dtype=torch.bool)
        order = torch.argsort(batch, stable=True)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(batch.numel()) - starts[batch[order]]
        padded[batch[order], pos] = h[order]
        mask[batch[order], pos] = True
        enc = self.self_attn(padded, src_key_padding_mask=~mask)
        q = self.pma.query.expand(num_graphs, -1, -1)
        pooled, _ = self.pma.attn(q, enc, enc, key_padding_mask=~mask)
        return self.head.net(pooled.squeeze(1))


def _batch(num_graphs=3, nodes=(2, 4, 3), in_shape=IN_SHAPE, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(nodes[:num_graphs])
    x = torch.randn(n, *in_shape, generator=g)
    batch = torch.cat([
        torch.full((c,), i, dtype=torch.long)
        for i, c in enumerate(nodes[:num_graphs])
    ])
    return x, batch


# --------------------------------------------------------------------------- #
# Names, imports, metadata                                                      #
# --------------------------------------------------------------------------- #

def test_canonical_name_and_alias_identity():
    assert SetTransformerModel is TGraphXSetAttention
    from tgraphx.models import SetTransformerModel as m_alias
    from tgraphx.models import TGraphXSetAttention as m_canonical
    from tgraphx.models.set_transformer import (
        SetTransformerModel as s_alias,
        TGraphXSetAttention as s_canonical,
    )
    assert m_alias is m_canonical is s_alias is s_canonical is TGraphXSetAttention


def test_all_exports():
    for name in ("TGraphXSetAttention", "SetTransformerModel",
                 "StridedConvEncoder", "SetAttentionBlock", "AttentionPooling"):
        assert name in tgraphx.__all__
        assert getattr(tgraphx, name) is not None


def test_isinstance_and_metadata_through_alias():
    m = SetTransformerModel("graph_classification", (8,), num_classes=3)
    assert isinstance(m, TGraphXSetAttention)
    assert m.model_family == "set_transformer"
    assert m.topology_source == "learned_implicit"


def test_api_stability_registry():
    from tgraphx.ux.public_api import api_status, list_aliases
    assert api_status("TGraphXSetAttention") == "stable"
    assert api_status("SetTransformerModel") == "stable"
    assert api_status("StridedConvEncoder") == "stable"
    assert "SetTransformerModel" in list_aliases("TGraphXSetAttention")
    assert "set_attention" in list_aliases("TGraphXSetAttention")


def test_topology_source_of_aliases():
    for fam in ("set_transformer", "set_attention", "tgraphx_set_attention"):
        assert topology_source_of(fam) == "learned_implicit"


# --------------------------------------------------------------------------- #
# Factory aliases                                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "family", ["set_transformer", "set_attention", "tgraphx_set_attention"])
def test_factory_aliases_build_same_family(family):
    m = build_model("graph_classification", layer=family, in_shape=(3, 8, 8),
                    hidden_shape=(16,), num_layers=1, num_classes=2)
    assert type(m) is TGraphXSetAttention
    assert m.model_family == "set_transformer"
    assert m.topology_source == "learned_implicit"


def test_factory_aliases_identical_architecture():
    kw = dict(task="graph_classification", in_shape=(3, 8, 8),
              hidden_shape=(16,), num_layers=2, num_classes=4)
    keysets = []
    for family in ("set_transformer", "set_attention", "tgraphx_set_attention"):
        m = build_model(layer=family, **kw)
        keysets.append({k: tuple(v.shape) for k, v in m.state_dict().items()})
    assert keysets[0] == keysets[1] == keysets[2]


def test_factory_forwards_new_kwargs():
    m = build_model(
        "graph_classification", layer="tgraphx_set_attention",
        in_shape=(3, 8, 8), hidden_shape=(16,), num_layers=1, num_classes=2,
        norm_order="post", activation="relu", dropout=0.1,
        attention_dropout=0.1, pool_attention_dropout=0.0, head_hidden_dim=8,
    )
    assert m.norm_order == "post"
    assert m.activation == "relu"
    assert m.pool_attention_dropout == 0.0
    assert m.head_hidden_dim == 8
    assert isinstance(m.head, nn.Sequential)


# --------------------------------------------------------------------------- #
# New architecture axes                                                         #
# --------------------------------------------------------------------------- #

def test_invalid_norm_order_and_activation():
    with pytest.raises(ValueError, match="norm_order"):
        TGraphXSetAttention("graph_classification", (8,), num_classes=2,
                            norm_order="sandwich")
    with pytest.raises(ValueError, match="activation"):
        TGraphXSetAttention("graph_classification", (8,), num_classes=2,
                            activation="tanh")
    with pytest.raises(ValueError, match="norm_order"):
        SetAttentionBlock(16, 4, norm_order="mid")
    with pytest.raises(ValueError, match="activation"):
        SetAttentionBlock(16, 4, activation="swish")


def test_postnorm_block_matches_torch_transformer_layer():
    torch.manual_seed(0)
    ref = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                     batch_first=True)  # post-norm, relu, 0.1
    blk = SetAttentionBlock(32, 4, ffn_dim=64, dropout=0.1,
                            attention_dropout=0.1, norm_order="post",
                            activation="relu")
    mapping = {
        "self_attn.in_proj_weight": "attn.in_proj_weight",
        "self_attn.in_proj_bias": "attn.in_proj_bias",
        "self_attn.out_proj.weight": "attn.out_proj.weight",
        "self_attn.out_proj.bias": "attn.out_proj.bias",
        "linear1.weight": "ffn.0.weight",
        "linear1.bias": "ffn.0.bias",
        "linear2.weight": "ffn.3.weight",
        "linear2.bias": "ffn.3.bias",
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
    }
    ref_sd = ref.state_dict()
    blk.load_state_dict({dst: ref_sd[src] for src, dst in mapping.items()},
                        strict=True)
    ref.eval(); blk.eval()
    tokens = torch.randn(2, 5, 32)
    pad = torch.zeros(2, 5, dtype=torch.bool)
    pad[1, 3:] = True
    with torch.no_grad():
        out_ref = ref(tokens, src_key_padding_mask=pad)
        out_blk = blk(tokens, key_padding_mask=pad)
    # compare only real (non-padding) positions; torch's fast path may
    # leave padding rows unnormalized
    real = ~pad
    assert torch.allclose(out_ref[real], out_blk[real], atol=1e-5)


def test_strided_encoder_shapes_and_names():
    enc = StridedConvEncoder(13, 64)
    assert enc.channel_schedule == [32, 64, 128]
    x = torch.randn(4, 13, 32, 32)
    assert enc(x).shape == (4, 64)
    keys = set(enc.state_dict().keys())
    assert "conv.0.weight" in keys and "conv.3.weight" in keys
    assert "conv.6.weight" in keys and "proj.weight" in keys


def test_strided_encoder_explicit_schedule_and_validation():
    enc = StridedConvEncoder(3, 8, num_layers=2, channel_schedule=[4, 6])
    assert enc.channel_schedule == [4, 6]
    assert enc(torch.randn(2, 3, 8, 8)).shape == (2, 8)
    with pytest.raises(ValueError, match="channel_schedule"):
        StridedConvEncoder(3, 8, num_layers=3, channel_schedule=[4, 6])
    with pytest.raises(ValueError, match="num_layers"):
        StridedConvEncoder(3, 8, num_layers=0)


def test_strided_architecture_via_encoder_config():
    m = TGraphXSetAttention(
        "graph_classification", (3, 16, 16), num_classes=2,
        encoder_config={"architecture": "strided", "num_layers": 2},
    )
    assert isinstance(m.encoder, StridedConvEncoder)
    assert m.encoder_config["architecture"] == "strided"
    assert m.encoder_config["channel_multiplier"] == 2
    with pytest.raises(ValueError, match="strided"):
        TGraphXSetAttention("graph_classification", (8,), num_classes=2,
                            encoder_config={"architecture": "strided"})
    with pytest.raises(ValueError, match="architecture"):
        TGraphXSetAttention("graph_classification", (3, 8, 8), num_classes=2,
                            encoder_config={"architecture": "resnet"})


def test_pool_attention_dropout_decoupled():
    m = TGraphXSetAttention("graph_classification", (8,), num_classes=2,
                            attention_dropout=0.2, pool_attention_dropout=0.0)
    assert m.blocks[0].attn.dropout == pytest.approx(0.2)
    assert m.pool.attn.dropout == pytest.approx(0.0)
    m2 = TGraphXSetAttention("graph_classification", (8,), num_classes=2,
                             attention_dropout=0.2)
    assert m2.pool.attn.dropout == pytest.approx(0.2)  # follows by default


def test_head_hidden_dim():
    m = TGraphXSetAttention("graph_classification", (8,), num_classes=3,
                            head_hidden_dim=16)
    assert isinstance(m.head, nn.Sequential)
    x, batch = torch.randn(5, 8), torch.tensor([0, 0, 1, 1, 1])
    assert m(x, batch=batch).shape == (2, 3)
    rebuilt = TGraphXSetAttention.from_config(m.config())
    rebuilt.load_state_dict(m.state_dict(), strict=True)


# --------------------------------------------------------------------------- #
# Reference configuration + mapping (self-contained parity)                     #
# --------------------------------------------------------------------------- #

def test_reference_config_builds_exact_architecture():
    m = _reference_model()
    assert sum(p.numel() for p in m.parameters()) == 189650
    assert isinstance(m.encoder, StridedConvEncoder)
    assert m.norm_order == "post"
    assert m.activation == "relu"
    assert m.dropout == pytest.approx(0.1)
    assert m.attention_dropout == pytest.approx(0.1)
    assert m.pool_attention_dropout == pytest.approx(0.0)
    assert len(m.state_dict()) == 54


def test_reference_config_round_trip():
    m = _reference_model()
    rebuilt = TGraphXSetAttention.from_config(m.config())
    rebuilt.load_state_dict(m.state_dict(), strict=True)
    assert rebuilt.config() == m.config()


def test_map_reference_state_dict_strict_load_and_parity():
    torch.manual_seed(7)
    ref = _TorchPrimitivesReference(IN_SHAPE[0], NUM_CLASSES)
    ref.eval()
    mapped = TGraphXSetAttention.map_reference_state_dict(ref.state_dict())
    m = _reference_model()
    res = m.load_state_dict(mapped, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    m.eval()
    x, batch = _batch()
    with torch.no_grad():
        out_ref = ref(x, batch)
        with pytest.warns(TopologyIgnoredWarning):
            out_new = m(x, edge_index=torch.zeros(2, 0, dtype=torch.long),
                        batch=batch)
    assert torch.allclose(out_ref, out_new, atol=1e-5)
    assert torch.equal(out_ref.argmax(-1), out_new.argmax(-1))


def test_from_reference_state_dict():
    torch.manual_seed(3)
    ref = _TorchPrimitivesReference(IN_SHAPE[0], NUM_CLASSES)
    m = TGraphXSetAttention.from_reference_state_dict(
        ref.state_dict(), in_shape=IN_SHAPE, num_classes=NUM_CLASSES)
    mapped = TGraphXSetAttention.map_reference_state_dict(ref.state_dict())
    sd = m.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in mapped.items())


def test_map_reference_state_dict_rejects_unknown_keys():
    with pytest.raises(KeyError, match="Unrecognized"):
        TGraphXSetAttention.map_reference_state_dict({"foo.bar": torch.zeros(1)})


# --------------------------------------------------------------------------- #
# Config / checkpoint compatibility                                             #
# --------------------------------------------------------------------------- #

def test_v150_style_config_still_loads():
    """A config dict serialized by v1.5.0 (no norm_order/activation/
    pool_attention_dropout/head_hidden_dim, no encoder architecture key)
    must reconstruct the identical architecture."""
    v150_cfg = {
        "task": "graph_classification",
        "in_shape": [3, 8, 8],
        "embed_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_dim": 32,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "num_classes": 4,
        "out_dim": None,
        "pooling": "attention",
        "num_seeds": 1,
        "layer_norm": True,
        "encoder_config": {"num_layers": 3, "hidden_channels": 32,
                           "dropout_prob": 0.0, "use_batchnorm": True,
                           "use_residual": False, "pool_layers": 1},
        "on_edge_index": "warn",
        "model_family": "set_transformer",
        "topology_source": "learned_implicit",
    }
    m_old_style = TGraphXSetAttention.from_config(copy.deepcopy(v150_cfg))
    m_new = TGraphXSetAttention("graph_classification", (3, 8, 8),
                                embed_dim=16, num_layers=2, num_heads=4,
                                ffn_dim=32, num_classes=4)
    old_sd = m_old_style.state_dict()
    new_sd = m_new.state_dict()
    assert {k: tuple(v.shape) for k, v in old_sd.items()} == \
           {k: tuple(v.shape) for k, v in new_sd.items()}
    assert m_old_style.norm_order == "pre"
    assert m_old_style.activation == "gelu"


def test_checkpoint_round_trip_with_new_fields(tmp_path):
    from tgraphx.training import save_checkpoint
    m = _reference_model(in_shape=(3, 8, 8), num_classes=4)
    path = str(tmp_path / "ck.pt")
    save_checkpoint(m, None, epoch=1, path=path, config=m.config())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = TGraphXSetAttention.from_config(payload["config"])
    rebuilt.load_state_dict(payload["model_state_dict"], strict=True)
    assert rebuilt.norm_order == "post"
    assert isinstance(rebuilt.encoder, StridedConvEncoder)


# --------------------------------------------------------------------------- #
# Behavior of the reference configuration                                       #
# --------------------------------------------------------------------------- #

def test_reference_edge_index_modes():
    x, batch = _batch(num_graphs=2, nodes=(2, 3))
    ei = torch.tensor([[0], [1]])
    cfg = TGraphXSetAttention.reference_config(IN_SHAPE, NUM_CLASSES)
    m = TGraphXSetAttention(**cfg).eval()
    with pytest.warns(TopologyIgnoredWarning):
        m(x, edge_index=ei, batch=batch)
    cfg["on_edge_index"] = "error"
    with pytest.raises(ValueError, match="learned_implicit"):
        TGraphXSetAttention(**cfg)(x, edge_index=ei, batch=batch)
    cfg["on_edge_index"] = "ignore"
    TGraphXSetAttention(**cfg).eval()(x, edge_index=ei, batch=batch)


def test_reference_padding_mask_isolation():
    """Adding a graph with a different node count must not change another
    graph's output (post-norm path)."""
    m = _reference_model().eval()
    x1, b1 = _batch(num_graphs=1, nodes=(3,), seed=5)
    with torch.no_grad():
        solo = m(x1, batch=b1)
    x2, _ = _batch(num_graphs=1, nodes=(7,), seed=9)
    x = torch.cat([x1, x2])
    batch = torch.cat([torch.zeros(3, dtype=torch.long),
                       torch.ones(7, dtype=torch.long)])
    with torch.no_grad():
        joint = m(x, batch=batch)
    assert torch.allclose(solo[0], joint[0], atol=1e-5)


def test_reference_permutation_invariance_and_equivariance():
    m = _reference_model().eval()
    x, batch = _batch(num_graphs=2, nodes=(4, 5), seed=11)
    perm = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        out = m(x, batch=batch)
        out_p = m(x[perm], batch=batch[perm])
        nodes = m.encode_nodes(x, batch=batch)
        nodes_p = m.encode_nodes(x[perm], batch=batch[perm])
    assert torch.allclose(out, out_p, atol=1e-4)          # invariance
    assert torch.allclose(nodes[perm], nodes_p, atol=1e-4)  # equivariance


def test_reference_variable_node_counts():
    m = _reference_model().eval()
    for nodes in ((1,), (1, 6), (2, 3, 8)):
        x, batch = _batch(num_graphs=len(nodes), nodes=nodes, seed=13)
        with torch.no_grad():
            out = m(x, batch=batch)
        assert out.shape == (len(nodes), NUM_CLASSES)
        assert torch.isfinite(out).all()


def test_reference_train_mode_dropout_active():
    m = _reference_model()
    m.train()
    x, batch = _batch(num_graphs=2, nodes=(3, 4), seed=17)
    torch.manual_seed(0)
    a = m(x, batch=batch)
    torch.manual_seed(1)
    b = m(x, batch=batch)
    assert not torch.allclose(a, b)  # dropout 0.1 is really active


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reference_cpu_cuda_parity():
    m = _reference_model().eval()
    x, batch = _batch(num_graphs=2, nodes=(3, 5), seed=19)
    with torch.no_grad():
        out_cpu = m(x, batch=batch)
        m_gpu = m.to("cuda")
        out_gpu = m_gpu(x.cuda(), batch=batch.cuda()).cpu()
    assert torch.allclose(out_cpu, out_gpu, atol=1e-4)
    assert torch.equal(out_cpu.argmax(-1), out_gpu.argmax(-1))


# --------------------------------------------------------------------------- #
# External evidence (skipped when the evidence tree is absent, e.g. CI)         #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not REVISED_CKPT_DIR.exists(),
                    reason="external evidence tree not present")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_completed_experiment_checkpoint_maps_strictly(seed):
    ck = torch.load(REVISED_CKPT_DIR / f"set_transformer_s{seed}.pt",
                    map_location="cpu", weights_only=False)
    state = ck["best"]["state"]
    m = TGraphXSetAttention.from_reference_state_dict(
        state, in_shape=IN_SHAPE, num_classes=NUM_CLASSES)
    assert sum(p.numel() for p in m.parameters()) == 189650
    mapped = TGraphXSetAttention.map_reference_state_dict(state)
    sd = m.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in mapped.items())
    m.eval()
    x, batch = _batch(num_graphs=2, nodes=(4, 6), seed=seed)
    with torch.no_grad():
        out = m(x, batch=batch)
    assert out.shape == (2, NUM_CLASSES)
    assert torch.isfinite(out).all()
