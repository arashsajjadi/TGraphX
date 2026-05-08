"""datasets_quickstart.py — list/inspect/instantiate TGraphX datasets.

Demonstrates the v0.2.9 dataset registry, native synthetic datasets,
and optional adapter discovery.  No network access; ``--include-pyg``,
``--include-dgl``, and ``--include-ogb`` opt in to upstream-backed
adapters only when you have those packages installed.
"""
from __future__ import annotations

import argparse

from tgraphx.datasets import (
    available_dataset_groups,
    dataset_info,
    get_dataset,
    list_datasets,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-pyg", action="store_true",
                        help="Try to construct a small PyG dataset (needs torch_geometric).")
    parser.add_argument("--include-dgl", action="store_true")
    parser.add_argument("--include-ogb", action="store_true")
    args = parser.parse_args(argv)

    print(f"\n[1] Registered groups:")
    for group, names in available_dataset_groups().items():
        print(f"    {group}: {len(names)} dataset(s)")

    print(f"\n[2] First few synthetic datasets:")
    for name in list_datasets(tags=["synthetic"]):
        info = dataset_info(name)
        print(f"    {name:<36}  {info['metadata'].get('description', '')}")

    print(f"\n[3] Construct a synthetic patch-graph dataset:")
    ds = get_dataset("synthetic:patch_graph", num_graphs=4, seed=0)
    g = ds[0]
    print(f"    len={len(ds)}  shape={tuple(g.node_features.shape)}  "
          f"label={int(g.graph_label)}")
    print(f"    metadata: {ds.metadata.short_summary()}")

    print(f"\n[4] Synthetic node-classification with masks:")
    ds_n = get_dataset("synthetic:node_classification", num_nodes=24, seed=0)
    masks = ds_n[0].metadata["masks"]
    print(f"    num_nodes={ds_n[0].num_nodes}  "
          f"train/val/test = "
          f"{int(masks['train_mask'].sum())}/"
          f"{int(masks['val_mask'].sum())}/"
          f"{int(masks['test_mask'].sum())}")

    if args.include_pyg:
        print(f"\n[5] PyG (Cora) — only if torch_geometric installed:")
        try:
            ds = get_dataset("pyg:planetoid/cora",
                             root="data", download=False)
            print(f"    Cora num_nodes={ds[0].num_nodes}")
        except Exception as exc:  # noqa: BLE001
            print(f"    Skipped: {exc.__class__.__name__}: {exc}")

    if args.include_dgl:
        print(f"\n[6] DGL (Cora) — only if dgl installed:")
        try:
            ds = get_dataset("dgl:cora", root="data", download=False)
            print(f"    DGL Cora num_nodes={ds[0].num_nodes}")
        except Exception as exc:  # noqa: BLE001
            print(f"    Skipped: {exc.__class__.__name__}: {exc}")

    if args.include_ogb:
        print(f"\n[7] OGB (ogbn-arxiv) — only if ogb installed:")
        try:
            ds = get_dataset("ogb:ogbn-arxiv", root="data", download=False)
            print(f"    OGB ogbn-arxiv num_nodes={ds[0].num_nodes}")
        except Exception as exc:  # noqa: BLE001
            print(f"    Skipped: {exc.__class__.__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
