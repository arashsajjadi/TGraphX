# Transforms

`tgraphx.transforms` is a small, deterministic-by-default,
non-mutating-by-default set of graph transforms.  Importing the
module does not pull in any optional dependency.

## Composition

```python
from tgraphx.transforms import Compose, NormalizeFeatures, AddSelfLoops

transform = Compose([NormalizeFeatures(), AddSelfLoops()])
g = transform(graph)
```

`Compose`, `LambdaTransform`, and `RandomApply` form the composition
layer.  `RandomApply(transform, p=..., seed=...)` is deterministic
when seeded.

## Available transforms

| Group | Class |
|-------|-------|
| **Graph structure** | `AddSelfLoops`, `RemoveSelfLoops`, `ToUndirected`, `CoalesceEdges`, `DropEdges` |
| **Feature transforms** | `NormalizeFeatures`, `StandardizeFeatures`, `NormalizeEdgeFeatures`, `AddDegreeFeatures`, `AddConstantFeatures`, `FeatureNoise`, `NodeFeatureMask` |
| **Splits** | `RandomNodeSplit`, `RandomLinkSplit`, `RandomGraphSplit`, `FixedSplit` |
| **Positional / structural** | `AddDegreeEncoding`, `AddLaplacianEigenvectors`, `AddAdjacencyBias` |
| **Patch builders** | `PatchifyImage`, `PatchifyVolume`, `BuildGridGraph`, `BuildKNNGraph`, `BuildRadiusGraph` |

Every class returns a new `Graph`; metadata is preserved.

## Mutation policy

* All transforms shallow-copy the input graph and re-assign attributes.
* `RemoveSelfLoops` re-aligns `edge_weight` / `edge_features` /
  `edge_labels` to the surviving edges.
* `ToUndirected` and `CoalesceEdges` drop `edge_labels` (they cannot
  survive an unambiguous coalesce).

## Determinism

Every transform that contains randomness (`RandomApply`, `DropEdges`,
`FeatureNoise`, `NodeFeatureMask`, splits) accepts a `seed=...`
argument and uses a per-instance `torch.Generator`.  No global RNG
state is touched.

## O(N²) guards

`AddLaplacianEigenvectors` and `AddAdjacencyBias` build a dense
adjacency.  Both refuse to run on graphs larger than ``max_nodes``
(default 5 000) — pass a higher cap explicitly if you really mean to.

## Examples

* `examples/transforms_metrics_demo.py`
* `examples/datasets_quickstart.py`
