"""Demonstrate classical graph generators with tensor features.

Shows: FeatureAwareERGraph, FeatureAwareBAGraph, TemporalEvolvingGraph,
       TypedGeneratedGraph, AnomalyInjectedGraph, MotifInjectedGraph.

Writes a generation report artifact to /tmp/generation_report.json.
"""
import tempfile
import os
from tgraphx.generation.classical import (
    FeatureAwareERGraph,
    FeatureAwareBAGraph,
    TemporalEvolvingGraph,
    TypedGeneratedGraph,
    AnomalyInjectedGraph,
    MotifInjectedGraph,
)
from tgraphx.generation.data_model import graph_generation_summary
from tgraphx.generation.reports import write_graph_generation_report


def main():
    print("=== Classical Graph Generation Demo ===\n")

    # ER graph
    g_er = FeatureAwareERGraph(n=20, p=0.3, node_feature_dim=8, edge_feature_dim=4, seed=42)
    g_er.validate()
    s = graph_generation_summary(g_er)
    print(f"ER graph:  n={s['num_nodes']}, e={s['num_edges']}, density={s['density']:.3f}")
    print(f"  node_features shape: {s['node_features_shape']}")
    print(f"  edge_features shape: {s['edge_features_shape']}")

    # BA graph
    g_ba = FeatureAwareBAGraph(n=20, m=2, node_feature_dim=16, seed=42)
    g_ba.validate()
    s = graph_generation_summary(g_ba)
    print(f"BA graph:  n={s['num_nodes']}, e={s['num_edges']}, density={s['density']:.3f}")

    # Temporal graph
    g_temp = TemporalEvolvingGraph(n=10, steps=5, edge_add_prob=0.2, edge_remove_prob=0.1, seed=42)
    g_temp.validate()
    print(f"Temporal:  n={g_temp.num_nodes}, e={g_temp.num_edges}, "
          f"timestamps shape={list(g_temp.timestamps.shape) if g_temp.timestamps is not None else None}")

    # Typed graph
    g_typed = TypedGeneratedGraph(
        n=12,
        node_types_list=[0] * 6 + [1] * 6,
        type_edge_probs={(0, 0): 0.4, (1, 1): 0.4, (0, 1): 0.1},
        node_feature_dims_by_type={0: 4, 1: 8},
        seed=42,
    )
    g_typed.validate()
    print(f"Typed:     n={g_typed.num_nodes}, e={g_typed.num_edges}, "
          f"node_types={g_typed.node_types.unique().tolist()}")

    # Anomaly injected
    g_anom = AnomalyInjectedGraph(g_er, anomaly_fraction=0.1, anomaly_type="clique", seed=0)
    g_anom.validate()
    print(f"Anomaly:   n={g_anom.num_nodes}, e={g_anom.num_edges}, "
          f"anomaly_labels={g_anom.metadata['anomaly_labels'].sum().item()} anomalous nodes")

    # Motif injected
    g_motif = MotifInjectedGraph(g_er, motif_type="triangle", motif_count=3, seed=0)
    g_motif.validate()
    print(f"Motif:     n={g_motif.num_nodes}, e={g_motif.num_edges}, "
          f"motif_labels={int((g_motif.metadata['motif_labels'] >= 0).sum())} nodes in motifs")

    # Write report
    out = os.path.join(tempfile.gettempdir(), "generation_report.json")
    write_graph_generation_report(
        out,
        generator_name="FeatureAwareERGraph",
        seed=42,
        params={"n": 20, "p": 0.3, "node_feature_dim": 8},
        graph_stats={"num_nodes": g_er.num_nodes, "num_edges": g_er.num_edges},
        feature_shapes={"node_features": list(g_er.node_features.shape)},
    )
    print(f"\nReport written to: {out}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
