# Temporal graph learning

TGraphX provides Beta-quality temporal foundations in
`tgraphx.temporal`:

- `sinusoidal_time_encoding` (Beta) — Transformer-style positional
  encoding for timestamps.
- `LearnableTimeEncoding` (Experimental) — Time2Vec-style trainable
  encoder.
- `TGNMemory` (Experimental) — per-node memory module with a GRU
  update and a no-future-leakage guard.
- `TGATConv` (Experimental) — temporal attention layer with edge-level
  time encoding.

## TGNMemory

```python
from tgraphx.temporal import TGNMemory

mem = TGNMemory(num_nodes=N, memory_dim=64, message_dim=64, time_dim=16)
mem.update(node_ids, messages, timestamps, time_encoding=time_enc)
mem.detach()       # call between batches to keep BPTT tractable
mem.reset_state()  # call between epochs
```

`update` raises `ValueError` if any incoming timestamp is older than
the node's `last_update` — a strong no-future-leakage guard.  Pass
`check_monotonic=False` only when you intentionally allow out-of-order
updates.

## TGATConv

```python
from tgraphx.temporal import TGATConv

layer = TGATConv(in_dim=64, out_dim=64, time_dim=16, num_heads=4,
                 time_encoding="sinusoidal")
out = layer(x, edge_index, edge_time, query_time)
```

The caller supplies `query_time` (one cutoff time per destination
node), and the layer encodes `Δt = query_time[dst] - edge_time` into
the attention computation.  Pair this with `temporal_window_sample` to
ensure `edge_time <= query_time` for every kept edge.

## Stability

**Experimental** in v0.5.0+. APIs and semantics may evolve once the
v0.5.x temporal benchmark suite lands.
