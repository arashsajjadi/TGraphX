"""run_all_fast_examples.py — run every fast TGraphX example and report results.

Skips examples that are missing, require a GPU, or take longer than a
per-example timeout.  Never crashes the runner itself due to a single failure.

Usage::

    python examples/run_all_fast_examples.py
    python examples/run_all_fast_examples.py --timeout 30   # seconds per example
    python examples/run_all_fast_examples.py --verbose
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))

# Each entry: (script_name, description, always_skip_reason_or_None)
FAST_EXAMPLES = [
    # Factory / model examples
    ("01_vector_node_classification.py",    "Vector node classification",           None),
    ("02_spatial_graph_classification.py",  "2-D spatial graph classification",     None),
    ("03_volumetric_graph_classification.py","3-D volumetric graph classification",  None),
    ("04_config_based_model.py",            "Config-based model construction",       None),
    ("05_edge_prediction.py",               "Edge prediction",                       None),
    # Core data / graph API
    ("directed_vs_undirected_graphs.py",    "Directed vs undirected graphs",         None),
    ("weighted_edges.py",                   "Weighted edges",                        None),
    ("tensor_edge_features.py",             "Tensor (spatial) edge features",        None),
    # Patch helpers
    ("image_patch_graph.py",                "2-D image patch graph",                 None),
    ("volume_patch_graph.py",               "3-D volume patch graph",                None),
    # Graph builders
    ("gnn_family_with_graph_builders.py",   "GNN families with graph builders",      None),
    # Minimal layer examples
    ("minimal_spatial_message_passing.py",  "Minimal spatial message passing",       None),
    ("minimal_graph_classifier.py",         "Minimal graph classifier",              None),
    ("tensor_gat_minimal.py",               "Tensor-aware GAT (minimal)",            None),
    ("tensor_graphsage_minimal.py",         "Tensor-aware GraphSAGE (minimal)",      None),
    ("custom_message_passing.py",           "Custom message-passing layer",          None),
    # Spatial / volumetric node features
    ("volumetric_3d_node_features.py",      "3-D volumetric node features",          None),
    # Training utilities
    ("checkpoint_save_load.py",             "Checkpoint save / load",                None),
    ("training_minimal_fit.py",             "fit() with no logging",                 None),
    ("training_with_csvlogger.py",          "fit() + CSVLogger (writes to /tmp)",    None),
    ("training_with_tensorboard.py",        "fit() + TensorBoardLogger (optional)",  None),
    ("training_with_dashboard.py",          "Training with dashboard (writes runs/)",None),
    # Performance / hardware
    ("memory_report.py",                    "Memory report",                         None),
    ("mixed_precision_inference.py",        "Mixed precision inference",             None),
    # torch.compile (may be slow on first call; timeout guards it)
    ("torch_compile_benchmark.py",          "torch.compile benchmark",               None),
    # Overfitting sanity (slightly heavier; still fast)
    ("tiny_overfit_tensor_gat.py",          "Tiny overfit (GAT)",                    None),
    ("tiny_overfit_edge_features.py",       "Tiny overfit (edge features)",          None),
    ("gradient_sanity_stack.py",            "Gradient sanity (8-layer stack)",       None),
    # Sampling (v0.2.8)
    ("sampling_demo_v028.py",               "Random walk + hetero + temporal sampling", None),
    # Datasets / transforms / metrics (v0.2.9)
    ("datasets_quickstart.py",              "Datasets quickstart (no download)",        None),
    ("synthetic_datasets_demo.py",          "All synthetic datasets",                   None),
    ("transforms_metrics_demo.py",          "Compose + metrics demo",                   None),
    ("image_folder_patch_dataset_demo.py",  "ImageFolder → patch graph (PIL)",          None),
    ("pyg_dataset_adapter_demo.py",         "PyG → TGraphX conversion (skips if missing)", None),
    ("dgl_dataset_adapter_demo.py",         "DGL → TGraphX conversion (skips if missing)", None),
    ("ogb_dataset_adapter_demo.py",         "OGB evaluator wrapper (mock)",             None),
    ("mnist_patch_graph_demo.py",           "MNIST patch demo (FakeData by default)",   None),
    # v0.3.0 — experiments + explainability + model zoo
    ("experiment_config_quickstart.py",     "Run a YAML experiment config",              None),
    ("explainability_saliency_demo.py",     "Saliency + IG + perturbation",              None),
    ("explainability_attention_demo.py",    "Attention → per-edge scores",               None),
    ("model_zoo_demo.py",                   "Vector model zoo (GCN / GATv2 / APPNP)",     None),
    # v0.3.2 — negative sampling, algorithms, time encoding
    ("negative_sampling_demo.py",           "Negative sampling primitives demo",          None),
    ("graph_algorithms_demo.py",            "Graph algorithms (components, BFS, degree)",  None),
    ("time_encoding_demo.py",               "Temporal time encoding demo",                None),
    # v0.3.2 — neural mining + plotting
    ("neural_prototype_membership_demo.py",  "Neural prototype membership scorer",         None),
    ("neural_graph_anomaly_demo.py",         "Neural graph autoencoder anomaly detection", None),
    ("neural_graph_pattern_classifier_demo.py", "Neural graph pattern classifier",         None),
    ("plot_graph_mining_demo.py",            "Graph mining visualization (save to files)", None),
    # v0.3.2 — classical graph mining
    ("graph_mining_structural_demo.py",     "Graph mining: density, degrees, motifs",     None),
    ("graph_mining_link_prediction_demo.py","Graph mining: classical link prediction scores", None),
    ("graph_mining_wl_kernel_demo.py",      "Graph mining: WL features and kernel",       None),
    ("graph_mining_anomaly_demo.py",        "Graph mining: anomaly detection",            None),
    ("prototype_graph_membership_demo.py",  "Prototype graph membership classification",  None),
    # v0.4.3 — paths, graph learning, sequence models
    ("graph_paths_algorithms_demo.py",      "BFS/DFS, Dijkstra, MST, cuts",              None),
    ("graph_learning_demo.py",              "Self-supervised learning, contrastive, DGI", None),
    # v0.4.4 — advanced algorithms, KG, Node2Vec
    ("graph_algorithms_advanced_demo.py",     "Max-flow, matching, coloring, hypergraph, IO", None),
    ("knowledge_graph_demo.py",               "TransE/DistMult knowledge graph embeddings",  None),
    ("node2vec_demo.py",                      "Node2Vec/DeepWalk unsupervised embeddings",   None),
    # v0.5.0 — VGAE, loaders, feature store
    ("vgae_link_prediction_demo.py",          "VGAE/GAE link prediction demo",              None),
    ("neighbor_loader_demo.py",               "NeighborLoader + FeatureStore scalability",  None),
    # v0.6.0 / v1.0.0 — graph generation, evolutionary optimization, graph RL
    ("classical_graph_generation_demo.py",    "Classical graph generation (ER/BA/temporal)", None),
    ("neural_graph_generation_demo.py",       "Neural graph generation (VGAE/AR/transformer)", None),
    ("evolutionary_graph_optimization_demo.py","Evolutionary optimization (GA/SA/NSGA-II)", None),
    ("graph_rl_environments_demo.py",         "Graph RL environments survey",               None),
    ("graph_reinforce_demo.py",               "REINFORCE on graph navigation",              None),
    ("graph_dqn_demo.py",                     "DQN on graph coloring",                      None),
    ("graph_ppo_demo.py",                     "PPO on graph navigation",                    None),
    ("graph_td3_sac_demo.py",                 "TD3/SAC continuous graph RL",                None),
    ("generation_rl_high_level_api_demo.py",  "High-level API: generation + evolution + RL",None),
    ("graph_lstm_sequence_demo.py",           "GraphRNN sequence model",                    None),
    # v1.0.1 — UX / easy mode examples
    ("tensor_node_classification_neighbor_loader_demo.py", "GraphMiniBatch + seed_logits demo", None),
    ("easy_tensor_node_classification_no_torch.py", "Easy mode no-torch import demo",      None),
    ("easy_graph_generation_no_torch.py",     "Easy graph generation demo",                None),
    ("easy_graph_rl_no_torch.py",             "Easy graph RL demo",                        None),
    ("easy_kg_training_no_torch.py",          "Easy KG training demo",                     None),
]


def run_example(script: str, timeout: int, verbose: bool) -> tuple[str, float, str | None]:
    """Run one example. Returns (status, elapsed_s, error_msg_or_None)."""
    path = os.path.join(EXAMPLES_DIR, script)
    if not os.path.isfile(path):
        return "MISSING", 0.0, None

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=not verbose,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            return "OK", elapsed, None
        err = (result.stderr or b"").decode(errors="replace")[-200:]
        return "FAIL", elapsed, err
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, f"exceeded {timeout}s"
    except Exception as exc:
        return "ERROR", time.perf_counter() - t0, str(exc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run all fast TGraphX examples")
    p.add_argument("--timeout", type=int, default=60, help="Seconds per example (default 60)")
    p.add_argument("--verbose", action="store_true", help="Show example stdout/stderr")
    args = p.parse_args(argv)

    print(f"\nTGraphX — Fast Examples Runner  (timeout={args.timeout}s per example)\n")
    print(f"{'Script':<46} {'Status':<9} {'Time':>6}")
    print("-" * 64)

    n_ok = n_fail = n_skip = n_missing = n_timeout = 0

    for script, desc, skip_reason in FAST_EXAMPLES:
        if skip_reason:
            print(f"  {script:<44} {'SKIP':<9}  {skip_reason}")
            n_skip += 1
            continue

        status, elapsed, err = run_example(script, args.timeout, args.verbose)
        elapsed_str = f"{elapsed:5.1f}s"

        if status == "MISSING":
            print(f"  {script:<44} {'missing':<9}")
            n_missing += 1
        elif status == "OK":
            print(f"  {script:<44} {'ok':<9} {elapsed_str}")
            n_ok += 1
        elif status == "TIMEOUT":
            print(f"  {script:<44} {'TIMEOUT':<9} {elapsed_str}")
            n_timeout += 1
        else:
            short_err = (err or "")[:80].replace("\n", " ")
            print(f"  {script:<44} {'FAIL':<9} {elapsed_str}  {short_err}")
            n_fail += 1

    print("-" * 64)
    print(f"\n  OK {n_ok}  |  FAIL {n_fail}  |  TIMEOUT {n_timeout}"
          f"  |  MISSING {n_missing}  |  SKIP {n_skip}")

    if n_fail:
        print(f"\n  {n_fail} example(s) failed — see output above.")
        return 1
    print("\n  All present examples passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
