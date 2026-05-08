# Deprecation Policy

TGraphX follows a clear deprecation policy so consumers can adopt
changes with predictable migration cost.

## Versioning

TGraphX is currently in the **v0.x** series.  In v0.x:

- **Stable** APIs (without 🧪 marker) keep their signature/behaviour
  across minor releases unless a deprecation cycle is followed.
- **Experimental** APIs (🧪 marker) may change at any time within the
  v0.x line; see `experimental_policy.md`.

Once TGraphX hits **v1.0**, semantic versioning becomes strict:

- Patch releases (v1.0.x): bug fixes only.
- Minor releases (v1.x.0): backward-compatible additions.
- Major releases (v2.x.0): breaking changes only.

## Deprecating a stable API

1. **Annotation**: the deprecated API emits a `DeprecationWarning` on
   import or first use, with the suggested replacement.
2. **CHANGELOG**: noted under a `### Deprecated` subsection of the
   release that introduces the deprecation.
3. **Notice period**: the API remains functional for at least **one
   minor release** before removal (v0.x line) or **one major release**
   (v1.x line).
4. **Migration**: a one-paragraph migration recipe in
   `docs/migration_*.md` (where applicable).
5. **Removal**: noted under `### Removed` in the release that drops it.

## Removing an experimental API

Experimental APIs can be removed with **one minor-release notice** in
the v0.x line:

1. CHANGELOG: `### Deprecated (experimental)` — names what is going.
2. Next minor release: removal under `### Removed`.

## Guarantees we never make

- That every PyG/DGL idiom has a TGraphX equivalent.
- That experimental APIs will not change.
- That GPU/MPS-specific kernels will work on every PyTorch version.
- That a deprecated experimental API will be renamed-with-alias.

## Current deprecated APIs

None as of v0.2.7 release prep.

## Pre-1.0 stability summary

The following APIs are intended to remain stable through v0.3.0 and
into v1.0 unless explicitly deprecated:

- `Graph`, `GraphBatch`
- `ConvMessagePassing`, `TensorGATLayer`, `TensorGraphSAGELayer`,
  `TensorGINLayer`, `LinearMessagePassing`
- `make_layer`, `build_model`, `build_model_from_config`
- `train_epoch`, `evaluate`, `fit`, `set_seed`
- `CSVLogger`, `TensorBoardLogger`
- All graph builders: `build_grid_graph`, `build_grid_graph_3d`,
  `build_fully_connected_graph`, `build_knn_graph`, `build_radius_graph`,
  `build_iou_graph`, `build_random_graph`, patch helpers
- `tgraphx.sampling.*` (v0.2.6 stable)
- `tgraphx.distributed.*` helpers (v0.2.6 stable)
- `Dashboard` HTTP API and CLI flags
