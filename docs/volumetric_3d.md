# 3-D Volumetric Node Features

TGraphX supports node features with shape `[N, C, D, H, W]` (depth × height × width).
This is useful for medical imaging (MRI/CT), 3-D point cloud processing, and
any domain where nodes carry volumetric data.

## Supported layers

All spatial GNN families support `spatial_rank=3`:

```python
from tgraphx.layers.gat  import TensorGATLayer
from tgraphx.layers.sage import TensorGraphSAGELayer
from tgraphx.layers.gin  import TensorGINLayer
from tgraphx.layers.conv_message import ConvMessagePassing

# ConvMessagePassing: in_shape must have 4 elements (C, D, H, W)
layer = ConvMessagePassing(in_shape=(4, 4, 4, 4), out_shape=(8, 4, 4, 4))

# TensorGATLayer
layer = TensorGATLayer(in_channels=4, out_channels=8, spatial_rank=3)

# TensorGraphSAGELayer
layer = TensorGraphSAGELayer(in_channels=4, out_channels=8, spatial_rank=3)

# TensorGINLayer
layer = TensorGINLayer(in_channels=4, out_channels=8, spatial_rank=3)
```

## Volume patch workflow

```python
import torch
from tgraphx import build_grid_graph_3d, volume_to_patches
from tgraphx.layers.gat import TensorGATLayer

volumes = torch.randn(1, 2, 8, 8, 8)                        # [B, C, D, H, W]
patches = volume_to_patches(volumes, patch_size=4)            # [1, 8, 2, 4, 4, 4]
x       = patches[0]                                          # [8, 2, 4, 4, 4]

ei  = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
gat = TensorGATLayer(in_channels=2, out_channels=4, spatial_rank=3)
out = gat(x, ei)                                              # [8, 4, 4, 4, 4]
```

## Factory API

```python
from tgraphx import build_model

model = build_model(
    task="graph_classification",
    layer="sage",
    in_shape=(2, 4, 4, 4),       # (C, D, H, W)
    hidden_shape=(4, 4, 4, 4),
    num_layers=2,
    num_classes=3,
)
```

## Limitations

- **Arbitrary-rank tensors are not supported.** Only vector `[N, D]`,
  2-D `[N, C, H, W]`, and 3-D `[N, C, D, H, W]` node features work.
- `AttentionMessagePassing` (legacy) only supports vector or 2-D inputs;
  3-D raises `NotImplementedError`.
- Patch helpers raise `ValueError` if dimensions are not exactly tiled.

## See also

- [Patch helpers in Graph Builders](graph_builders.md#patch-helpers)
- [Volumetric example](../examples/03_volumetric_graph_classification.py)
