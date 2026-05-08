# Experimental API Policy

This document defines what "🧪 Experimental" means in TGraphX, what
guarantees we make, and how an experimental API graduates to stable.

## What "Experimental" means

An API marked **🧪 Experimental** is:

- Functional, tested, documented at the smoke level.
- **Not** guaranteed to keep its exact signature, defaults, or behaviour
  across minor releases (v0.x.y).
- Not removed without prior notice within the v0.x line.
- Subject to refinement based on user feedback before promotion.

Experimental APIs are clearly labelled in:

- The class/function docstring (``.. experimental::`` block).
- The README support tables (🧪 marker).
- The class ``__repr__`` (``[🧪 Experimental]``).

## What "Stable" means

A **✅ Stable** API:

- Has its signature and observable behaviour fixed for the v0.x line
  unless a deprecation cycle is followed.
- Is covered by a test suite that exercises edge cases.
- Has at least one example or non-trivial docstring section.
- Will only break in a major-version release (e.g., v1.0).

## Promotion criteria

An experimental API is promoted to stable when:

1. **Tests** — Comprehensive unit tests covering forward, backward (where
   applicable), edge cases, errors, and shape invariants.
2. **Docs** — API reference entry plus at least one usage example.
3. **No known shape/device bugs** for at least one minor release cycle.
4. **No mandatory new dependency** introduced after promotion.
5. **Compatibility** — clear migration path for any rename / re-signing.

Promotion happens in a release that **also documents the change** in
`CHANGELOG.md` under "Changed" or "Promoted".

## Demotion / removal

An experimental API can be:

- **Renamed** at any time within v0.x with a deprecation alias if reasonable.
- **Removed** with one minor-release notice.

A stable API can only be removed via the deprecation policy (see
`deprecation_policy.md`).

## Current experimental APIs (as of v0.2.7 release prep)

| API | Module | Promotion target |
|-----|--------|---|
| `HeteroGraph` (data container with hetero stores + labels) | `tgraphx.core.hetero_graph` | v0.2.8 |
| `HeteroGraphBatch` | `tgraphx.core.hetero_batch` | v0.2.8 |
| `HeteroConv` | `tgraphx.layers.hetero` | v0.3.0 |
| `HeteroGraphClassifier` / `HeteroNodeClassifier` | `tgraphx.models.hetero_models` | v0.3.0 |
| `TemporalGraphSequence` / `TemporalGraphBatch` | `tgraphx.core.temporal*` | v0.2.8 |
| `temporal_readout`, `TemporalGraphClassifier`/`Regressor` | `tgraphx.layers.temporal_readout` / `tgraphx.models.temporal_models` | v0.3.0 |
| `GraphTransformerLayer` | `tgraphx.layers.graph_transformer` | v0.2.8 |
| `TensorGATLayer(attention_mode="channel")` | `tgraphx.layers.gat` | v0.2.8 |
| `tgraphx.interop` (PyG/DGL converters) | `tgraphx.interop` | v0.3.0 |
| `tgraphx.learned_graph` | `tgraphx.learned_graph` | v0.2.8 |

## How to use an experimental API safely

- Pin a specific TGraphX version in your project.
- Run your tests against the package; experimental APIs are tested at
  the smoke level but consumer-side tests catch real-world drift.
- Watch the CHANGELOG for the API.
- When you see a 🧪 marker in code or README, expect possible refinement.
