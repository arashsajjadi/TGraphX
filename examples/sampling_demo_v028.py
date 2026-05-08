"""sampling_demo_v028.py — random walks, hetero, and temporal sampling.

CPU-safe demo of the v0.2.8 sampling additions:

* :func:`tgraphx.random_walk_sample` — DeepWalk / node2vec-style random
  walk subgraph extraction, deterministic with ``seed``.
* :func:`tgraphx.hetero_induced_subgraph` and
  :func:`tgraphx.hetero_neighbor_sample` — typed-graph counterparts
  to the homogeneous samplers.
* :func:`tgraphx.temporal_window_sample` and
  :func:`tgraphx.temporal_window_sample_batch` — slice contiguous
  windows out of temporal sequences.

Synthetic data only.  Tests the API, not benchmark performance.
"""
from __future__ import annotations

import torch

from tgraphx import (
    Graph,
    HeteroGraph,
    TemporalGraphBatch,
    TemporalGraphSequence,
    hetero_induced_subgraph,
    hetero_neighbor_sample,
    random_walk_sample,
    temporal_window_sample,
    temporal_window_sample_batch,
)


# ── 1. Random walk on a directed line graph ──────────────────────────────────


def random_walk_demo() -> None:
    print("\n[1] Random walk sampling")
    n = 12
    x = torch.randn(n, 4)
    src = torch.arange(n - 1, dtype=torch.long)
    dst = src + 1
    g = Graph(x, torch.stack([src, dst], dim=0))

    sub = random_walk_sample(
        g,
        seed_nodes=torch.tensor([0]),
        walk_length=8,
        num_walks_per_seed=3,
        direction="out",
        seed=42,
    )
    visited = sub.metadata["sampling"]["original_node_ids"].tolist()
    print(f"    seeds=[0]  visited={visited}  num_nodes={sub.num_nodes}")


# ── 2. Hetero sampling on a small bipartite graph ────────────────────────────


def hetero_sampling_demo() -> None:
    print("\n[2] Hetero sampling")
    n_paper, n_author = 6, 4
    g = HeteroGraph(
        node_stores={
            "paper": torch.randn(n_paper, 8),
            "author": torch.randn(n_author, 4),
        },
        edge_stores={
            ("author", "writes", "paper"): torch.tensor(
                [[0, 0, 1, 1, 2, 3, 3], [0, 1, 1, 2, 4, 3, 5]],
                dtype=torch.long,
            ),
        },
    )

    induced = hetero_induced_subgraph(
        g,
        node_ids_dict={
            "paper": torch.tensor([0, 1, 4]),
            "author": torch.tensor([0, 1, 2]),
        },
    )
    print(
        f"    induced_subgraph: paper={induced.num_nodes('paper')}, "
        f"author={induced.num_nodes('author')}, "
        f"writes_edges={induced.num_edges(('author', 'writes', 'paper'))}"
    )

    sampled = hetero_neighbor_sample(
        g,
        seed_nodes_dict={"paper": torch.tensor([0, 1])},
        fanouts=[{("author", "writes", "paper"): 2}],
        seed=0,
        direction="in",
    )
    a_ids = sampled.metadata["sampling"]["original_node_ids"]["author"].tolist()
    p_ids = sampled.metadata["sampling"]["original_node_ids"]["paper"].tolist()
    print(f"    neighbor_sample: author={a_ids}, paper={p_ids}")


# ── 3. Temporal window sampling on equal- and variable-length batches ────────


def _snapshot(t: int) -> Graph:
    src = torch.arange(4, dtype=torch.long)
    dst = (src + 1) % 4
    return Graph(torch.randn(4, 3), torch.stack([src, dst], dim=0))


def temporal_window_demo() -> None:
    print("\n[3] Temporal window sampling")
    seq = TemporalGraphSequence(
        graphs=[_snapshot(t) for t in range(6)],
        timestamps=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    sub_seq = temporal_window_sample(seq, t_start=2, t_end=5)
    print(f"    sequence: full T=6 -> window [2,5) -> T={sub_seq.num_snapshots}")

    batch = TemporalGraphBatch([
        TemporalGraphSequence(graphs=[_snapshot(t) for t in range(5)]),
        TemporalGraphSequence(graphs=[_snapshot(t) for t in range(3)]),
        TemporalGraphSequence(graphs=[_snapshot(t) for t in range(4)]),
    ])
    sub_batch = temporal_window_sample_batch(batch, t_start=1, t_end=4)
    print(
        f"    batch lengths {batch.lengths} -> windowed lengths "
        f"{sub_batch.lengths} (variable-length ok)"
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(0)
    random_walk_demo()
    hetero_sampling_demo()
    temporal_window_demo()
    print("\nAll sampling demos completed.")


if __name__ == "__main__":
    main()
