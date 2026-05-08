# Plotting utilities

`tgraphx.plotting` (Beta, v0.4.0+) provides Matplotlib-based graph
and mining visualization utilities.

**Requires:** `pip install matplotlib` (not installed automatically by
`pip install tgraphx` — add it to your environment separately or use
`pip install "tgraphx[dev]"` which includes it).

All plots are:
- Matplotlib-only (no seaborn, no NetworkX required).
- Headless-safe — work with the `Agg` backend.
- Colorblind-friendly using the Okabe-Ito palette by default.
- Saveable to PNG / SVG / PDF via `save_figure`.

## Import

```python
from tgraphx.plotting import (
    # Layouts
    circular_layout, grid_layout, random_layout, spring_layout,
    # Graph plots
    plot_graph, plot_degree_distribution, plot_adjacency_matrix,
    plot_connected_components,
    # Mining plots
    plot_motif_summary, plot_graph_mining_summary,
    plot_link_prediction_score_distribution,
    plot_graph_similarity_heatmap,
    plot_anomaly_scores, plot_prototype_membership_scores,
    plot_confusion_matrix, plot_training_curves,
    plot_community_assignments,
    # Utilities
    save_figure,
)
```

## Headless usage

Force the `Agg` backend before importing pyplot in scripts or CI:

```python
import matplotlib
matplotlib.use("Agg")
```

Or set the environment variable: `MPLBACKEND=Agg`.

## Layout algorithms

All layouts return `ndarray[N, 2]` of (x, y) coordinates.

| Function | Algorithm | Complexity |
|----------|-----------|-----------|
| `circular_layout(N)` | Equidistant on unit circle | O(N) |
| `grid_layout(N, width=None)` | Grid placement | O(N) |
| `random_layout(N, seed=None)` | Uniform [0,1]² | O(N) |
| `spring_layout(edge_index, N, iterations=50, seed=None)` | Fruchterman-Reingold | O(N²·iters) |

`spring_layout` is a pure Python/NumPy implementation — no NetworkX
required.  Suitable for graphs with up to ~200 nodes.

## Graph plots

### `plot_graph`

```python
from tgraphx.plotting import plot_graph
fig, ax = plot_graph(
    edge_index, num_nodes,
    node_values=None,    # FloatTensor[N] for colouring
    layout="spring",     # "spring" | "circular" | "random" | "grid"
    node_size=80.0,
    with_labels=True,    # node id labels (only for N ≤ 30)
    ax=None,             # optional Matplotlib Axes
    seed=42,
    max_nodes=500,       # size guard
)
# Returns (fig, ax)
```

### `plot_degree_distribution`

```python
fig, ax = plot_degree_distribution(edge_index, num_nodes, bins=20)
```

### `plot_adjacency_matrix`

```python
fig, ax = plot_adjacency_matrix(edge_index, num_nodes, max_nodes=100)
```

### `plot_connected_components`

Colours nodes by connected component.

```python
fig, ax = plot_connected_components(edge_index, num_nodes, layout="spring")
```

## Mining plots

### `plot_motif_summary`

```python
from tgraphx.mining import motif_counts
from tgraphx.plotting import plot_motif_summary

mc = motif_counts(edge_index, num_nodes)
fig, ax = plot_motif_summary(mc, title="Motifs")
```

### `plot_confusion_matrix`

```python
from tgraphx.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(
    matrix,              # [[int]] or ndarray or LongTensor [C, C]
    class_names=None,    # optional list of class label strings
    normalize=True,      # normalise rows to [0,1]
)
```

### `plot_training_curves`

Accepts a list of per-epoch dicts (output of `tgraphx.training.fit`)
or a `dict[str, list]`:

```python
from tgraphx.plotting import plot_training_curves
fig, ax = plot_training_curves(history, metrics=["train_loss", "val_loss"])
```

### `plot_anomaly_scores`

```python
from tgraphx.plotting import plot_anomaly_scores
fig, ax = plot_anomaly_scores(scores, top_k=20)
```

### `plot_graph_similarity_heatmap`

```python
from tgraphx.mining import wl_kernel_matrix
from tgraphx.plotting import plot_graph_similarity_heatmap
K = wl_kernel_matrix(graphs, normalize=True)
fig, ax = plot_graph_similarity_heatmap(K, labels=["G1","G2","G3"], max_size=50)
```

## `save_figure`

```python
from tgraphx.plotting import save_figure
paths = save_figure(fig, "/tmp/my_plot", formats=("png", "svg", "pdf"), dpi=150)
# Returns list of written paths.
```

## Colab usage

All plots work in Google Colab. For inline display simply call:
`plt.show()` or `display(fig)`.  No special setup is required.

## Performance notes

- `spring_layout` is O(N²·iters) — avoid for N > 300.
- `plot_graph` has a `max_nodes=500` guard to prevent accidental slow renders.
- `plot_adjacency_matrix` has a `max_nodes=100` guard.
- `plot_graph_similarity_heatmap` has a `max_size=50` guard.

## Limitations

- No interactive plots (no `plotly`, no `bokeh`).
- `spring_layout` is a simple FR implementation; not as well-tuned as
  NetworkX's layout engine.
- Plotting is CPU/NumPy only — it does not interact with GPU training.

## Related

- Tests: `tests/test_plotting.py`
- Examples: `examples/plot_graph_mining_demo.py`
