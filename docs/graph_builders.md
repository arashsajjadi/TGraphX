# Graph Builders

All builders return `edge_index` as `[2, E]` LongTensor, ready for any GNN
layer or `Graph` constructor. Pure PyTorch — no scikit-learn, PyG, DGL, or
networkx dependency.

## Grid graphs

```python
from tgraphx import build_grid_graph, build_grid_graph_3d

# 2-D 4-connected grid: node (r,c) → index r*cols + c
ei = build_grid_graph(rows=5, cols=5, directed=False, self_loops=True)

# 3-D 6-connected grid: node (d,r,c) → d*rows*cols + r*cols + c
ei = build_grid_graph_3d(depth=4, rows=4, cols=4, directed=False, self_loops=True)
```

## Fully connected graph ⚠️ O(N²)

```python
from tgraphx import build_fully_connected_graph

ei = build_fully_connected_graph(num_nodes=50, self_loops=False)
# N*(N-1) = 2450 edges — memory grows quadratically
```

## kNN graph ⚠️ O(N²)

```python
from tgraphx import build_knn_graph
import torch

coords = torch.randn(100, 3)           # [N, D] coordinates
ei = build_knn_graph(coords, k=6, directed=False, self_loops=True)
# Uses torch.cdist — O(N²) in time and memory
```

## Radius graph ⚠️ O(N²)

```python
from tgraphx import build_radius_graph

ei = build_radius_graph(coords, radius=1.5, directed=False, self_loops=True)
```

## IoU graph ⚠️ O(N²)

```python
from tgraphx import build_iou_graph

boxes = torch.rand(50, 4).sort(dim=-1).values   # [N, 4] (x1,y1,x2,y2)
ei    = build_iou_graph(boxes * torch.tensor([1,1,2,2]), threshold=0.3)
```

## Random graph

```python
from tgraphx import build_random_graph

ei = build_random_graph(num_nodes=100, num_edges=400, directed=True, seed=42)
```

## Directedness convention

| `directed=True` | One edge per pair (canonical direction) |
|---|---|
| `directed=False` | Both `(u→v)` and `(v→u)` — standard undirected GNN representation |

## Self-loop convention

`self_loops=True` → one `i→i` edge per node, no duplicates.

## Patch helpers

Convert image / volume batches to patches and build matching grid graphs:

```python
from tgraphx import (
    image_to_patches, patch_grid_shape,
    volume_to_patches, volume_patch_grid_shape,
)

# 2-D patches
patches = image_to_patches(images, patch_size=4)      # [B, P, C, ph, pw]
n_h, n_w = patch_grid_shape(H, W, patch_size=4)       # (2, 2) for 8×8 → 4×4
ei = build_grid_graph(n_h, n_w)

# 3-D patches
patches = volume_to_patches(volumes, patch_size=4)    # [B, P, C, pd, ph, pw]
n_d, n_h, n_w = volume_patch_grid_shape(D, H, W, 4)
ei = build_grid_graph_3d(n_d, n_h, n_w)
```

Patch order is row-major (top-left to bottom-right), matching grid-node order.
Both functions raise `ValueError` if dimensions are not exactly covered.

## Complexity summary

| Builder | Time/Memory | Notes |
|---|---|---|
| `build_grid_graph` | O(E) | Deterministic |
| `build_grid_graph_3d` | O(E) | Deterministic |
| `build_fully_connected_graph` | **O(N²)** | Use small N |
| `build_knn_graph` | **O(N²)** | `torch.cdist` |
| `build_radius_graph` | **O(N²)** | `torch.cdist` |
| `build_iou_graph` | **O(N²)** | All-pairs IoU |
| `build_random_graph` | O(E) | Deterministic with `seed` |

These builders create **fixed, rule-based** adjacency — they do not implement
learned adjacency.

## See also

- [Spatial tensor GNN](spatial_tensor_gnn.md)
- [Volumetric 3-D](volumetric_3d.md)
