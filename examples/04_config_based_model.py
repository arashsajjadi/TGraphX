"""04_config_based_model.py

Build TGraphX models from Python dicts, JSON, and YAML configs.
All file I/O uses tempfile — no permanent files are written.
"""
import json
import os
import tempfile

import torch
from tgraphx import build_grid_graph, build_model_from_config

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def _run(model, x, edge_index, batch=None, label=""):
    out = model(x, edge_index, batch=batch)
    print(f"  [{label}] output: {tuple(out.shape)}")


def main():
    print("=" * 56)
    print("  Config-based model construction")
    print("=" * 56)

    # ------------------------------------------------------------------ #
    # 1. Python dict — node classification                                 #
    # ------------------------------------------------------------------ #
    print("\n1. From Python dict  (node_classification, linear)")
    cfg_node = {
        "model": {
            "task": "node_classification",
            "layer": "linear",
            "in_shape": [16],
            "hidden_shape": [32],
            "num_layers": 2,
            "num_classes": 4,
        }
    }
    model = build_model_from_config(cfg_node)
    x = torch.randn(9, 16)
    ei = build_grid_graph(3, 3, directed=False, self_loops=True)
    _run(model, x, ei, label="node_cls logits")

    # ------------------------------------------------------------------ #
    # 2. JSON tempfile — graph classification (2-D GAT)                   #
    # ------------------------------------------------------------------ #
    print("\n2. From JSON tempfile (graph_classification, gat)")
    cfg_gat = {
        "model": {
            "task": "graph_classification",
            "layer": "gat",
            "in_shape": [4, 4, 4],
            "hidden_shape": [8, 4, 4],
            "num_layers": 2,
            "num_classes": 3,
            "heads": 2,
            "residual": True,
            "dropout": 0.0,
        }
    }
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(cfg_gat, f)
        json_path = f.name

    try:
        model = build_model_from_config(json_path)
        x = torch.randn(4, 4, 4, 4)
        ei = build_grid_graph(2, 2, directed=False, self_loops=True)
        batch = torch.zeros(4, dtype=torch.long)
        _run(model, x, ei, batch=batch, label="graph_cls logits")
    finally:
        os.unlink(json_path)

    # ------------------------------------------------------------------ #
    # 3. YAML tempfile — 3-D SAGE graph regression                        #
    # ------------------------------------------------------------------ #
    if HAS_YAML:
        print("\n3. From YAML tempfile (graph_regression, sage, 3-D)")
        cfg_3d = {
            "model": {
                "task": "graph_regression",
                "layer": "sage",
                "in_shape": [2, 2, 2, 2],
                "hidden_shape": [4, 2, 2, 2],
                "num_layers": 2,
                "out_dim": 1,
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(cfg_3d, f)
            yaml_path = f.name
        try:
            model = build_model_from_config(yaml_path)
            x = torch.randn(8, 2, 2, 2, 2)
            from tgraphx import build_grid_graph_3d
            ei = build_grid_graph_3d(2, 2, 2, directed=False, self_loops=True)
            batch = torch.zeros(8, dtype=torch.long)
            _run(model, x, ei, batch=batch, label="graph_reg output")
        finally:
            os.unlink(yaml_path)
    else:
        print("\n3. YAML test skipped (PyYAML not installed)")

    print("\nDone.")


if __name__ == "__main__":
    main()
