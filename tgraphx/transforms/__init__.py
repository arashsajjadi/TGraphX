"""Graph transforms (v0.2.9).

Lightweight, deterministic-when-seeded, non-mutating-by-default
transforms.  Compose them with :class:`Compose`:

.. code-block:: python

    from tgraphx.transforms import (
        Compose, NormalizeFeatures, AddSelfLoops, RandomNodeSplit,
    )

    transform = Compose([
        NormalizeFeatures(),
        AddSelfLoops(),
        RandomNodeSplit(0.6, 0.2, seed=0),
    ])
    g = transform(graph)

This module imports nothing optional.
"""
from __future__ import annotations

from .compose import Compose, LambdaTransform, RandomApply
from .features import (
    AddConstantFeatures,
    AddDegreeFeatures,
    FeatureNoise,
    NodeFeatureMask,
    NormalizeEdgeFeatures,
    NormalizeFeatures,
    StandardizeFeatures,
)
from .graph import (
    AddSelfLoops,
    CoalesceEdges,
    DropEdges,
    RemoveSelfLoops,
    ToUndirected,
)
from .patch import (
    BuildGridGraph,
    BuildKNNGraph,
    BuildRadiusGraph,
    PatchifyImage,
    PatchifyVolume,
)
from .positional import (
    AddAdjacencyBias,
    AddDegreeEncoding,
    AddLaplacianEigenvectors,
)
from .splits import FixedSplit, RandomGraphSplit, RandomLinkSplit, RandomNodeSplit


__all__ = [
    "Compose",
    "LambdaTransform",
    "RandomApply",
    # graph structure
    "AddSelfLoops",
    "RemoveSelfLoops",
    "ToUndirected",
    "CoalesceEdges",
    "DropEdges",
    # features
    "NormalizeFeatures",
    "StandardizeFeatures",
    "NormalizeEdgeFeatures",
    "AddDegreeFeatures",
    "AddConstantFeatures",
    "FeatureNoise",
    "NodeFeatureMask",
    # splits
    "RandomNodeSplit",
    "RandomLinkSplit",
    "RandomGraphSplit",
    "FixedSplit",
    # positional / structural
    "AddDegreeEncoding",
    "AddLaplacianEigenvectors",
    "AddAdjacencyBias",
    # patch / structural
    "PatchifyImage",
    "PatchifyVolume",
    "BuildGridGraph",
    "BuildKNNGraph",
    "BuildRadiusGraph",
]
