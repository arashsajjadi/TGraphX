# Graph IO formats

TGraphX provides a small, **honest** GraphML round-trip in `tgraphx.io`.

GEXF and Pajek are roadmapped (see [roadmap.md](roadmap.md)) but not
implemented yet — we prefer one well-tested format over three half-tested
formats.

---

## GraphML

```python
from tgraphx import Graph
from tgraphx.io import write_graphml, read_graphml
import torch

x = torch.zeros(4, 1)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
y = torch.tensor([0, 1, 0, 1])
g = Graph(node_features=x, edge_index=edge_index, y=y)

write_graphml(g, "out.graphml")     # writes structure + labels
g2 = read_graphml("out.graphml")    # round-trip
```

### What round-trips

- Number of nodes
- Edge index
- Directedness flag (recovered into `metadata["graphml_directed"]`)
- `edge_weight`
- `node_labels` (`y`) and `edge_labels`
- 1-D `node_features` / `edge_features` **only when `include_tensor_features=True`**

### What does NOT round-trip

| Field | Behavior |
|-------|----------|
| Multi-dimensional `node_features` (e.g. `[N, C, H, W]`) | Rejected with a clear error if `include_tensor_features=True`. Default is to drop them silently from the GraphML output (the read-side gives a `[N, 1]` zero placeholder). |
| `graph_features`, `graph_label` | Not serialised. Plain GraphML has no clean place for graph-level tensors. |
| Arbitrary `metadata` dict | Not serialised. Save it separately (`json.dump(graph.metadata, ...)`). |

If you need lossless persistence including tensor features, use
`torch.save({"graph": graph}, ...)` instead of GraphML.

---

## Implementation notes

- Pure Python standard library (`xml.etree.ElementTree`).  No additional
  dependencies.
- Pretty-prints with `ET.indent` on Python 3.9+.
- Output XML is GraphML-namespaced and parses cleanly in NetworkX / Gephi
  / Cytoscape (basic structure; their advanced visual attributes are not
  emitted).
- Integer labels are restored as `torch.long`; non-integer labels become
  `torch.float32`.

---

## Tests

See `tests/test_io_graphml_v120.py` for round-trip, edge weight, label
preservation, error paths, multi-dim rejection, and pathlib/string path
coverage (14 tests).

---

## Roadmap

- GEXF read/write (richer attribute typing).
- Pajek read/write (legacy format; useful for some graph DBs).
- Bipartite-aware exports.
- Tensor-feature-friendly format (likely a TGraphX-specific JSON+`.pt` pair
  rather than overloading existing formats).
