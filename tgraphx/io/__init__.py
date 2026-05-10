"""TGraphX graph IO formats.

Currently implemented:

- :func:`write_graphml` / :func:`read_graphml` — simple GraphML round-trip
  using only the Python standard library (``xml.etree.ElementTree``).

GraphML is a widely-supported XML format used by NetworkX, Gephi, Cytoscape,
and many graph databases.  TGraphX's implementation focuses on **structure +
scalar metadata** round-trip.

Tensor-feature serialization is intentionally restricted: GraphML cannot
express arbitrary tensor shapes safely, so by default ``write_graphml``
**rejects** graphs that carry node or edge feature tensors with rank > 1.
Pass ``include_tensor_features=True`` to flatten them with shape metadata at
your own risk (a warning is emitted; round-trip is best-effort).

Other formats (GEXF, Pajek) are roadmapped.

Stability: Beta (v1.2+) — round-trip is tested for structure, edge_weight,
directionality, and 1-D node/edge features.
"""
from __future__ import annotations

from .graphml import read_graphml, write_graphml

__all__ = ["read_graphml", "write_graphml"]
