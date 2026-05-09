# KG + GNN Integration

`tgraphx.kg.gnn` connects the KG data model to TGraphX's relational
GNN layers (`RGCNConv`).

**Stability: Experimental**

## KG to edge_index conversion

```python
from tgraphx.kg import KnowledgeGraph, kg_to_edge_index

kg = KnowledgeGraph.from_triples([("alice", "knows", "bob")])
edge_index, edge_type, edge_attr, edge_weight = kg_to_edge_index(kg)
# edge_index: LongTensor[2, N_t]
# edge_type:  LongTensor[N_t]
# edge_attr:  FloatTensor[N_t, F] if triple_features["edge_attr"] exists, else None
# edge_weight: FloatTensor[N_t] if kg.edge_weight exists, else None
```

This format is directly compatible with `RGCNConv`:

```python
from tgraphx.layers.rgcn import RGCNConv

layer = RGCNConv(in_channels=16, out_channels=32, num_relations=N_r)
# Build per-relation edge dict:
edge_index_by_rel = {int(r): edge_index[:, edge_type == r] for r in range(N_r)}
h = layer(x, edge_index_by_rel, num_nodes=N_e)
```

## RGCN KG completion (KGRGCNModel)

```python
from tgraphx.kg import KGRGCNModel, generate_synthetic_kg

kg = generate_synthetic_kg(50, 4, 150, seed=0)
model = KGRGCNModel(
    kg.num_entities, kg.num_relations,
    embedding_dim=32, num_rgcn_layers=1,
)
# Score triples using RGCN-encoded entities + DistMult decoder.
scores = model.forward_kg(kg, kg.triples[:8])
```

For entity features:

```python
model = KGRGCNModel(kg.num_entities, kg.num_relations, in_dim=16, embedding_dim=32)
embs = model.encode(edge_index_by_rel, entity_features=my_feat_tensor)
scores = model.score_triples(test_triples, embs)
```

## Limitations

- `KGRGCNModel` uses `RGCNConv` with a DistMult decoder; CompGCN is not yet implemented.
- No typed message passing beyond relation-specific weight matrices.
- Non-vector entity features (images, volumes) must be pre-projected before being passed as `entity_features`.
- `forward_kg` reconstructs `edge_index_by_rel` on every call; cache it for performance in training loops.

See `benchmarks/kg/benchmark_kg_transe.py` for a training loop pattern adaptable to RGCN.
