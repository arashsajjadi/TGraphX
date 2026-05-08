"""Positional / structural encodings.

These transforms reuse the pure-PyTorch helpers in
:mod:`tgraphx.layers.transformer_encodings`; we wrap them as
transforms so they can be composed into dataset pipelines.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..core.graph import Graph
from ..layers.transformer_encodings import (
    build_adjacency_bias,
    degree_encoding,
    laplacian_eigvec_encoding,
)
from .graph import _shallow_copy


class AddDegreeEncoding:
    """Append degree-based positional encoding to node features."""

    def __init__(self, dim: int = 8, direction: str = "both") -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.direction = direction

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.node_features.dim() != 2:
            raise ValueError(
                "AddDegreeEncoding requires vector node features [N, D]"
            )
        if new.edge_index is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            ei = new.edge_index
        enc = degree_encoding(ei, new.num_nodes, dim=self.dim, direction=self.direction)
        new.node_features = torch.cat([new.node_features, enc], dim=-1)
        return new


class AddLaplacianEigenvectors:
    """Append the smallest non-trivial Laplacian eigenvectors.

    .. warning::
        Computes a dense Laplacian — **O(N²)** memory.  Only use on
        small graphs (a per-graph hard cap is enforced).
    """

    def __init__(
        self,
        dim: int = 4,
        sign_flip: bool = False,
        max_nodes: int = 5000,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = int(dim)
        self.sign_flip = bool(sign_flip)
        self.max_nodes = int(max_nodes)

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.num_nodes > self.max_nodes:
            raise ValueError(
                f"AddLaplacianEigenvectors refuses graphs with N > "
                f"{self.max_nodes} (O(N²) Laplacian).  Got {new.num_nodes}."
            )
        if new.node_features.dim() != 2:
            raise ValueError(
                "AddLaplacianEigenvectors requires vector node features [N, D]"
            )
        ei = new.edge_index if new.edge_index is not None else torch.zeros((2, 0), dtype=torch.long)
        enc = laplacian_eigvec_encoding(
            ei, new.num_nodes, dim=self.dim, sign_flip=self.sign_flip,
        )
        new.node_features = torch.cat([new.node_features, enc], dim=-1)
        return new


class AddAdjacencyBias:
    """Stamp a dense adjacency-bias matrix into ``metadata['edge_bias_dense']``.

    For use with :class:`tgraphx.layers.GraphTransformerLayer` when
    constructed with ``edge_bias=True``.
    """

    def __init__(
        self,
        value: float = 0.0,
        neg_inf: bool = False,
        max_nodes: int = 5000,
    ) -> None:
        self.value = float(value)
        self.neg_inf = bool(neg_inf)
        self.max_nodes = int(max_nodes)

    def __call__(self, graph: Graph) -> Graph:
        new = _shallow_copy(graph)
        if new.num_nodes > self.max_nodes:
            raise ValueError(
                f"AddAdjacencyBias refuses graphs with N > {self.max_nodes} "
                f"(O(N²) bias).  Got {new.num_nodes}."
            )
        if new.edge_index is None:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            ei = new.edge_index
        bias = build_adjacency_bias(
            ei, new.num_nodes, value=self.value, neg_inf=self.neg_inf,
        )
        meta = dict(new.metadata or {})
        meta["edge_bias_dense"] = bias
        new.metadata = meta
        return new
