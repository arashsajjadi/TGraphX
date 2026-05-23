# Heterogeneous GNN layers

TGraphX provides three hetero convolution layers in `tgraphx.layers`:

- `RGCNConv` — relational GCN with optional basis decomposition
  (Schlichtkrull et al., 2018).  **Experimental**.
- `HANConv` — metapath-based attention (Wang et al., WWW 2019).
  **Experimental**.
- `HGTConv` — Heterogeneous Graph Transformer with relation-specific
  attention (Hu et al., WWW 2020).  **Experimental**.

These pair with `HeteroGraph`, `HeteroGraphBatch`, and the
`hetero_neighbor_sample` typed sampler.

## HANConv

```python
from tgraphx.layers.han import HANConv

layer = HANConv(in_dim=64, out_dim=32, num_heads=4)
# metapath_edge_index_dict: {metapath_name: LongTensor[2, E_m]}
out = layer(x, metapath_edge_index_dict)  # [N, out_dim]
```

The user is responsible for materialising metapath neighbour edges from
the `HeteroGraph` (e.g. via repeated `hetero_neighbor_sample`).

## HGTConv

```python
from tgraphx.layers.hgt import HGTConv

layer = HGTConv(in_dim=64, out_dim=64,
                node_types=["paper", "author"],
                edge_types=[("paper", "writes", "author"),
                            ("author", "writes", "paper")],
                num_heads=4)
out_dict = layer(x_dict, edge_index_dict)
```

Per-node-type Q/K/V, per-relation prior weights, and a per-type residual
projection are learned end-to-end.

## Stability

Marked **Experimental**. Behaviour is validated on small regression
fixtures and unit tests for shape/gradient flow; no claim of numerical
or training-throughput parity with reference HAN/HGT implementations
is made until the v0.5.x benchmark suite is in place.
