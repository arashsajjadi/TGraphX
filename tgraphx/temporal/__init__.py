"""Temporal-graph utilities (v0.3.2 foundation, growing through v0.3.4).

This package consolidates temporal-graph helpers under a single
namespace.  v0.3.2 ships only :mod:`time_encoding` here; the existing
``tgraphx.temporal_sampling`` (window sampling) and the snapshot-loop
classifiers in ``tgraphx.models.temporal_models`` are unchanged and
will be re-exported from this package in a later milestone.

Public surface
--------------
- :func:`sinusoidal_time_encoding` (Beta) — deterministic
  Transformer-style positional encoding for timestamps.
- :class:`LearnableTimeEncoding` (Experimental) — Time2Vec-style
  trainable encoder.

Stability levels are recorded in ``docs/api_stability.md`` and the
docstrings.
"""
from __future__ import annotations

from .time_encoding import (
    LearnableTimeEncoding,
    sinusoidal_time_encoding,
)
from .tgn import TGNMemory
from .tgat import TGATConv

__all__ = [
    "sinusoidal_time_encoding",
    "LearnableTimeEncoding",
    "TGNMemory",
    "TGATConv",
]
