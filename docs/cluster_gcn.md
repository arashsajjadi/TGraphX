# Cluster-GCN

Cluster-GCN ([Chiang et al., KDD 2019](https://arxiv.org/abs/1905.07953))
trains GNNs on large graphs by partitioning nodes into clusters and
loading one (or a few) clusters per mini-batch.  TGraphX provides four
partitioners and a `ClusterLoader`.

## Partitioners

| Class | Strategy | Use when |
|---|---|---|
| `RandomBalancedPartitioner` | random shuffle, round-robin | strong baseline; cheap |
| `BFSPartitioner` | parallel BFS frontier expansion | preserves locality |
| `ConnectedComponentPartitioner` | per connected component (optional split) | disconnected graphs |
| `SpectralPartitioner` | recursive spectral bisection | small graphs only (<= `max_nodes`) |

Each `.fit(graph)` returns a `PartitionResult` with:
`partition_id`, `num_partitions`, `cut_edges`, `intra_edge_count`,
`partition_sizes`, `balance_ratio`, and a JSON-serialisable
`.to_dict()` for dashboard integration.

Spectral partitioning densifies the adjacency, so it raises
`ValueError` above `max_nodes` (default 4096) rather than silently
allocating an `[N, N]` matrix.

## ClusterLoader

```python
from tgraphx import ConnectedComponentPartitioner, ClusterLoader

part = ConnectedComponentPartitioner(max_size=10_000, seed=0).fit(graph)
loader = ClusterLoader(graph, part,
                       num_clusters_per_batch=2,  # stochastic clustering
                       shuffle=True, seed=0)
for sub, cluster_ids in loader:
    out = model(sub)
```

`num_clusters_per_batch > 1` reproduces the Cluster-GCN trick where
multiple clusters are merged per batch to recover some inter-cluster
edges.

## Stability

**Beta** in v0.5.0+. All partitioners produce deterministic results
under a fixed `seed`.
