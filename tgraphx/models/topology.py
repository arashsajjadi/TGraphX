"""Topology-source vocabulary for TGraphX model families.

TGraphX is a tensor-relational platform: tensor-valued entities are
combined by a *relation/fusion operator* and reduced by a *readout*.
Different model families obtain their relations from different sources,
and that source is scientifically meaningful — it determines whether a
supplied ``edge_index`` is required, optional, or ignored.

Vocabulary
----------
``"none"``
    No relations at all: per-entity processing followed by a pooling
    readout (DeepSets-style).
``"fixed"``
    Fixed ordered fusion: entity count, identity, order, and alignment
    are baked into the architecture (e.g. channel-stacked CNN fusion).
    No ``edge_index`` exists.
``"given"``
    Relations are supplied with each sample as ``edge_index`` and the
    operator only aggregates along those edges (ConvMessagePassing,
    TensorGAT, TensorGraphSAGE, TensorGIN, LinearMessagePassing).
``"learned_implicit"``
    Relations are inferred from node content by global self-attention;
    a supplied ``edge_index`` is *not* consumed (SetTransformer).  Such
    models are relation-aware but explicit-input-topology-blind.
``"learned_explicit"``
    A trainable component scores or constructs an explicit edge set
    which is then used for message passing
    (:mod:`tgraphx.learned_graph`: ``EdgeScorer``,
    ``top_k_edges_from_scores``, ``build_knn_graph_from_embeddings``).
``"hybrid"``
    Supplied edges combined with learned global or residual relations
    (e.g. ``GraphTransformerLayer`` with ``edge_bias=True``, which adds
    an adjacency bias from ``edge_index`` to otherwise-global
    attention).

Model families known to :func:`tgraphx.build_model` map as follows:

=====================  ====================  =============================
family                 topology source       supplied ``edge_index``
=====================  ====================  =============================
``conv``               ``given``             required, defines messages
``gat``                ``given``             required, defines attention
``sage``               ``given``             required
``gin``                ``given``             required
``linear``             ``given``             required
``legacy_attention``   ``given``             required
``graph_transformer``  ``learned_implicit``  ignored unless
                                             ``edge_bias``/``positional_encoding``
                                             is enabled (then ``hybrid``)
``set_transformer``    ``learned_implicit``  ignored (warn/error per
                                             ``on_edge_index``)
=====================  ====================  =============================

``"tgraphx_set_attention"`` and ``"set_attention"`` are factory aliases
for ``"set_transformer"`` — all three resolve to the same
learned-implicit family (canonical class
:class:`tgraphx.TGraphXSetAttention`).
"""
from __future__ import annotations

__all__ = [
    "TOPOLOGY_SOURCES",
    "TopologyIgnoredWarning",
    "topology_source_of",
]

#: Ordered vocabulary of relation/topology sources supported by the platform.
TOPOLOGY_SOURCES = (
    "none",
    "fixed",
    "given",
    "learned_implicit",
    "learned_explicit",
    "hybrid",
)

_FAMILY_TOPOLOGY = {
    "conv": "given",
    "gat": "given",
    "sage": "given",
    "gin": "given",
    "linear": "given",
    "legacy_attention": "given",
    "graph_transformer": "learned_implicit",
    "set_transformer": "learned_implicit",
    # Factory aliases for the same learned-implicit set-attention family
    # (canonical class: TGraphXSetAttention).
    "set_attention": "learned_implicit",
    "tgraphx_set_attention": "learned_implicit",
}


class TopologyIgnoredWarning(UserWarning):
    """Emitted when a supplied ``edge_index`` is ignored by a model whose
    topology source is not ``"given"`` (e.g. SetTransformer).

    The model still learns content-dependent relations through
    self-attention — it is relation-aware — but the explicit input
    topology does not influence the computation.
    """


def topology_source_of(family: str, **kwargs) -> str:
    """Return the topology source for a ``build_model`` family name.

    Args:
        family: A model-family / layer name accepted by
            :func:`tgraphx.build_model` (e.g. ``"conv"``,
            ``"set_transformer"``).
        **kwargs: The model kwargs; used to refine families whose source
            depends on configuration (``graph_transformer`` becomes
            ``"hybrid"`` when ``edge_bias`` or ``positional_encoding``
            consumes ``edge_index``).

    Raises:
        KeyError: Unknown family name.
    """
    family = family.lower().strip()
    if family not in _FAMILY_TOPOLOGY:
        raise KeyError(
            f"Unknown model family {family!r}. "
            f"Known: {', '.join(sorted(_FAMILY_TOPOLOGY))}."
        )
    source = _FAMILY_TOPOLOGY[family]
    if family == "graph_transformer" and (
        kwargs.get("edge_bias") or kwargs.get("positional_encoding")
    ):
        return "hybrid"
    return source
