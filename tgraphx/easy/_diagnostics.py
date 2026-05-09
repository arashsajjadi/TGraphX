"""System diagnostics for TGraphX easy mode."""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import torch


def check_install() -> Dict[str, Any]:
    """Check TGraphX installation and return a status dict."""
    import sys
    import importlib

    status: Dict[str, Any] = {}
    status["python_version"] = sys.version.split()[0]

    import tgraphx
    status["tgraphx_version"] = tgraphx.__version__

    status["torch_version"] = torch.__version__
    status["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        status["cuda_version"] = torch.version.cuda
        status["gpu_count"] = torch.cuda.device_count()
        status["gpu_names"] = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]

    optional_deps = ["torch_geometric", "dgl", "ogb", "matplotlib", "mlflow", "tensorboard"]
    status["optional_deps"] = {}
    for dep in optional_deps:
        try:
            m = importlib.import_module(dep)
            status["optional_deps"][dep] = getattr(m, "__version__", "installed")
        except ImportError:
            status["optional_deps"][dep] = "not installed"

    return status


def doctor() -> Dict[str, Any]:
    """Print a system-readiness report and return a status dict."""
    status = check_install()

    lines = [
        "=" * 55,
        "TGraphX Doctor — System Status",
        "=" * 55,
        f"  tgraphx    : {status['tgraphx_version']}",
        f"  python     : {status['python_version']}",
        f"  torch      : {status['torch_version']}",
        f"  CUDA       : {'available (' + status.get('cuda_version', '') + ')' if status['cuda_available'] else 'not available'}",
    ]

    if status["cuda_available"]:
        for i, name in enumerate(status.get("gpu_names", [])):
            lines.append(f"    GPU {i}    : {name}")

    lines.append("")
    lines.append("Optional dependencies:")
    for dep, ver in status["optional_deps"].items():
        marker = "✓" if ver != "not installed" else "–"
        lines.append(f"  {marker} {dep}: {ver}")

    pkg_dir = pathlib.Path(__file__).parent.parent.parent
    examples_dir = pkg_dir / "examples"
    tutorials_dir = pkg_dir / "tutorials"
    lines.append("")
    lines.append(f"Examples : {examples_dir}")
    lines.append(f"Tutorials: {tutorials_dir}")
    lines.append("")
    lines.append("Quick start:")
    lines.append("  python tutorials/tensor_node_classification_neighbor_loader.py")
    lines.append("  python -m tgraphx.doctor")
    lines.append("=" * 55)
    print("\n".join(lines))
    return status


def show_capabilities() -> None:
    """Print a table of all TGraphX capabilities."""
    lines = [
        "=" * 65,
        "TGraphX v1.0+ Capabilities",
        "=" * 65,
        "",
        "Graph data structures:",
        "  Graph, GraphBatch, HeteroGraph, TemporalGraphSequence",
        "",
        "GNN layers (tensor-native):",
        "  ConvMessagePassing (2-D/3-D), AttentionMessagePassing,",
        "  TensorGATLayer, TensorGraphSAGELayer, TensorGINLayer,",
        "  GCNConv, GATv2Conv, APPNP (vector features)",
        "",
        "Loaders / samplers:",
        "  NeighborLoader, LinkNeighborLoader, GraphLoader,",
        "  GraphSAINT{Node,Edge,RandomWalk}Loader, ClusterLoader",
        "",
        "Easy mode (this module):",
        "  train_node_classifier, synthetic_tensor_node_classification,",
        "  doctor, list_tasks, list_models, list_samplers",
        "",
        "Knowledge graphs (tgraphx.kg):",
        "  TransE, DistMult, ComplEx, RotatE, KGPipeline,",
        "  multimodal tensor-aware KG",
        "",
        "Graph generation (tgraphx.generation):",
        "  VGAEGraphGenerator, AutoregressiveEdgeGenerator,",
        "  run_graph_generation, list_graph_generation_methods",
        "",
        "Evolutionary optimization (tgraphx.evolutionary):",
        "  GeneticAlgorithmOptimizer, NSGAIIOptimizer,",
        "  run_evolutionary_optimization",
        "",
        "Graph RL (tgraphx.rl):",
        "  DQN, DoubleDQN, PPO, SAC, TD3, DDPG, A2C, REINFORCE,",
        "  run_graph_rl, list_graph_rl_algorithms",
        "",
        "Graph mining (tgraphx.mining):",
        "  centrality, communities, motifs, anomaly, run_graph_mining",
        "",
        "Dashboard: tgraphx-dashboard (or python -m tgraphx.dashboard)",
        "Explainability: tgraphx.explain (saliency, IG, edge attribution)",
        "Experiments: tgraphx.experiments (config, runner, early-stop, CLI)",
        "=" * 65,
    ]
    print("\n".join(lines))
