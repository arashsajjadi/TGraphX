"""TGraphX Easy Mode — high-level, beginner-friendly API.

This module provides zero-boilerplate workflows for the most common TGraphX
tasks.  Advanced users can always access the low-level PyTorch objects through
the result objects.

Quick start::

    import tgraphx as tgx

    data = tgx.easy.synthetic_tensor_node_classification(
        num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
    )
    result = tgx.easy.train_node_classifier(
        data, model="tensor_gcn", epochs=5, seed=42,
    )
    print(result.metrics)
    result.summary()

Design principles:
- Every inferred default is visible in ``result.config``.
- No hidden defaults.
- Every result exposes the underlying PyTorch objects (``result.model``,
  ``result.graph``, ``result.loader``, etc.) for advanced use.
- No additional imports are added to the top-level ``tgraphx`` namespace on
  import — this module is opt-in.

Stability: Alpha (v1.0.1+).  Public function names are stable; return-object
fields may be extended in patch releases.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Heavy torch imports are deferred to avoid slowing down `import tgraphx`.
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # Data creation.
    "synthetic_tensor_node_classification",
    "synthetic_vector_node_classification",
    "synthetic_link_prediction",
    "synthetic_graph_classification",
    # Model creation.
    "make_tensor_node_classifier",
    "make_vector_node_classifier",
    # Training.
    "train_node_classifier",
    "fit_node_classifier",
    # Discovery.
    "list_tasks",
    "list_models",
    "list_samplers",
    "list_workflows",
    "explain_workflow",
    # Diagnostics.
    "doctor",
    "check_install",
    "show_capabilities",
    # Config.
    "EasyConfig",
    # Result object.
    "EasyResult",
]


# ── Exceptions ────────────────────────────────────────────────────────────────


class TGraphXError(ValueError):
    """Base user-facing error from TGraphX easy mode."""


class TGraphXConfigError(TGraphXError):
    """Invalid configuration for an easy-mode workflow."""


class TGraphXLabelError(TGraphXError):
    """Graph labels are missing or have an unsupported type."""


class TGraphXShapeError(TGraphXError):
    """Tensor shape contract violated."""


class TGraphXUnknownNameError(TGraphXError):
    """Unknown algorithm / model / sampler name."""


# ── Result object ─────────────────────────────────────────────────────────────


@dataclass
class EasyResult:
    """Structured result returned by all easy-mode training workflows.

    Attributes:
        metrics: Final metrics dict (``loss``, ``accuracy``, etc.).
        history: List of per-epoch metric dicts.
        model: Trained ``nn.Module``.
        graph: Source ``Graph`` object.
        config: Resolved configuration dict with all defaults expanded.
        artifacts: Optional dict of generated artefact paths.
        loader: The last-used data loader (for inspection or resuming).
        optimizer: The last-used ``torch.optim.Optimizer``.
        elapsed: Wall-clock seconds for the run.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    model: Optional[nn.Module] = None
    graph: Optional[Any] = None  # tgraphx.Graph
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    loader: Optional[Any] = None
    optimizer: Optional[Any] = None
    elapsed: float = 0.0

    def summary(self) -> str:
        """Print a human-readable summary and return it as a string."""
        lines = [
            "=" * 55,
            "TGraphX Easy Mode — Training Result",
            "=" * 55,
        ]
        lines.append("Metrics:")
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append(f"Epochs: {len(self.history)}")
        lines.append(f"Elapsed: {self.elapsed:.1f}s")
        lines.append("Config (resolved defaults):")
        for k, v in self.config.items():
            lines.append(f"  {k}: {v}")
        if self.artifacts:
            lines.append("Artifacts:")
            for k, v in self.artifacts.items():
                lines.append(f"  {k}: {v}")
        text = "\n".join(lines)
        print(text)
        return text

    def print_summary(self) -> None:
        """Print a human-readable summary."""
        self.summary()

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes non-serialisable objects)."""
        return {
            "metrics": self.metrics,
            "history": self.history,
            "config": self.config,
            "artifacts": self.artifacts,
            "elapsed": self.elapsed,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        """Return a markdown table of metrics."""
        if not self.metrics:
            return "_No metrics recorded._"
        lines = ["| Metric | Value |", "| --- | --- |"]
        for k, v in self.metrics.items():
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {val} |")
        return "\n".join(lines)

    def save_report(self, path: str) -> None:
        """Write a JSON report to ``path``."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(self.to_json())
        print(f"Report saved to {path}")

    def plot_loss(self) -> None:
        """Plot training loss history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(
                "matplotlib is not installed.  Install it with:\n"
                "    pip install matplotlib\n"
                "to enable plot_loss()."
            )
            return
        losses = [e.get("loss", float("nan")) for e in self.history]
        plt.figure()
        plt.plot(losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.show()

    def plot_metrics(self) -> None:
        """Plot all numeric metrics from history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available.  Install it to use plot_metrics().")
            return
        if not self.history:
            print("No history to plot.")
            return
        keys = [k for k in self.history[0] if isinstance(self.history[0][k], (int, float))]
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            vals = [e.get(k, float("nan")) for e in self.history]
            ax.plot(vals, marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(k)
            ax.set_title(k)
        plt.tight_layout()
        plt.show()


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EasyConfig:
    """Configuration dataclass for easy-mode workflows.

    Every field has a sensible default.  Pass an ``EasyConfig`` instance to
    any ``train_*`` or ``fit_*`` function instead of individual keyword args.

    Attributes:
        task: Workflow type.  One of ``"node_classification"``,
            ``"graph_classification"``, ``"link_prediction"``.
        model: Model name string or ``None`` (auto-select from task).
        sampler: Sampler name.  ``"neighbor"``, ``"full"``, ``"graphsaint_node"``,
            ``"cluster"``.
        optimizer: Optimizer name: ``"adam"``, ``"sgd"``, ``"adamw"``.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size (seed nodes per batch for node tasks).
        device: ``"auto"`` (pick CUDA if available), ``"cpu"``, or ``"cuda"``.
        seed: Global random seed.  ``None`` means non-deterministic.
        fanouts: Neighbor-sampling fanouts (per hop).
        hidden_channels: Hidden channel count.
        verbose: Print progress.
        dashboard_dir: Optional directory to write dashboard artefacts.
    """

    task: str = "node_classification"
    model: Optional[str] = None
    sampler: str = "neighbor"
    optimizer: str = "adam"
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 64
    device: str = "auto"
    seed: Optional[int] = None
    fanouts: List[int] = field(default_factory=lambda: [15, 10])
    hidden_channels: int = 16
    verbose: bool = True
    dashboard_dir: Optional[str] = None


# ── Discovery ─────────────────────────────────────────────────────────────────

_TASKS: Dict[str, str] = {
    "node_classification": "Predict a label for each node in the graph.",
    "graph_classification": "Predict a label for each whole graph.",
    "link_prediction": "Predict whether edges exist between node pairs.",
    "node_regression": "Predict a continuous value per node.",
    "graph_regression": "Predict a continuous value per graph.",
    "knowledge_graph_completion": "Complete missing (h, r, t) triples in a KG.",
    "graph_generation": "Generate new graphs with desired properties.",
    "evolutionary_optimization": "Optimize graph structure via evolutionary search.",
    "graph_rl": "Train RL agents that act on graph environments.",
    "graph_mining": "Structural analysis: communities, centrality, anomalies.",
}

_MODELS: Dict[str, Dict[str, str]] = {
    "node_classification": {
        "tensor_gcn": "Two-layer ConvMessagePassing for [N, C, H, W] node features.",
        "vector_gcn": "Two-layer GCNConv for [N, D] vector node features.",
        "tensor_sage": "Two-layer TensorGraphSAGELayer for tensor features.",
        "auto": "Automatically choose based on node feature shape.",
    },
    "graph_classification": {
        "tensor_gcn_pool": "ConvMessagePassing + global pool for graph classification.",
        "vector_gcn_pool": "GCNConv + global pool for vector features.",
        "auto": "Automatically choose based on node feature shape.",
    },
    "link_prediction": {
        "dot_product": "Dot-product link predictor.",
        "bilinear": "Bilinear link predictor.",
    },
}

_SAMPLERS: Dict[str, str] = {
    "neighbor": "NeighborLoader — multi-hop neighbor sampling (best for large graphs).",
    "full": "Use all nodes as a single batch (best for small graphs).",
    "graphsaint_node": "GraphSAINT node sampler.",
    "graphsaint_edge": "GraphSAINT edge sampler.",
    "graphsaint_rw": "GraphSAINT random-walk sampler.",
    "cluster": "Cluster-GCN partitioned loader.",
}

_WORKFLOWS: Dict[str, str] = {
    "train_node_classifier": "Train a GNN for node classification.",
    "fit_node_classifier": "Alias for train_node_classifier.",
    "synthetic_tensor_node_classification": "Generate synthetic tensor node classification data.",
    "synthetic_vector_node_classification": "Generate synthetic vector node classification data.",
    "doctor": "Check TGraphX installation and dependencies.",
    "show_capabilities": "Show all TGraphX capabilities.",
}


def list_tasks() -> Dict[str, str]:
    """Return a dict of task names and their descriptions."""
    return dict(_TASKS)


def list_models(task: Optional[str] = None) -> Dict[str, str]:
    """Return available model names for a given task.

    Args:
        task: Task name (e.g. ``"node_classification"``).  When ``None``,
            return all models for all tasks.

    Returns:
        Dict mapping model name to description.
    """
    if task is None:
        result: Dict[str, str] = {}
        for models in _MODELS.values():
            result.update(models)
        return result
    if task not in _MODELS:
        available = list(_MODELS)
        raise TGraphXUnknownNameError(
            f"Unknown task '{task}'. Available tasks: {available}."
        )
    return dict(_MODELS[task])


def list_samplers() -> Dict[str, str]:
    """Return available sampler names and their descriptions."""
    return dict(_SAMPLERS)


def list_workflows() -> Dict[str, str]:
    """Return available high-level workflow function names."""
    return dict(_WORKFLOWS)


def explain_workflow(name: str) -> str:
    """Print and return a description of a workflow function."""
    if name not in _WORKFLOWS:
        raise TGraphXUnknownNameError(
            f"Unknown workflow '{name}'.  Available: {list(_WORKFLOWS)}."
        )
    desc = _WORKFLOWS[name]
    print(f"{name}: {desc}")
    return desc


# ── Diagnostics ───────────────────────────────────────────────────────────────


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
        status["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

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
    """Print a system-readiness report and return a status dict.

    Checks:
    - TGraphX version
    - Python version
    - PyTorch version
    - CUDA availability
    - Optional dependency availability
    - Where examples and tutorials are

    Returns:
        Dict with all status fields.
    """
    import sys
    import pathlib

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

    # Locate examples and tutorials relative to the package.
    pkg_dir = pathlib.Path(__file__).parent.parent
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


# ── Data creation ─────────────────────────────────────────────────────────────


def synthetic_tensor_node_classification(
    num_nodes: int = 1000,
    node_shape: Tuple[int, ...] = (16, 8, 8),
    num_classes: int = 10,
    num_edges: Optional[int] = None,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Any:
    """Create a synthetic graph for tensor node classification.

    Node features have shape ``[num_nodes, *node_shape]`` (e.g. image-like).
    Labels are random integers in ``[0, num_classes)``.

    Args:
        num_nodes: Number of nodes.
        node_shape: Per-node feature shape, e.g. ``(C, H, W)`` for 2-D.
        num_classes: Number of classes.
        num_edges: Number of edges (default: ``5 * num_nodes``).
        seed: Random seed.
        device: Target device (``"cpu"`` or ``"cuda"``).

    Returns:
        :class:`~tgraphx.Graph` with ``node_features``, ``edge_index``,
        and ``y`` (labels).

    Example::

        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
        )
    """
    from tgraphx import Graph

    if num_edges is None:
        num_edges = num_nodes * 5

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    dev = torch.device(device)
    x = torch.randn(num_nodes, *node_shape, generator=gen)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    y = torch.randint(0, num_classes, (num_nodes,), generator=gen)

    return Graph(
        node_features=x.to(dev),
        edge_index=edge_index.to(dev),
        y=y.to(dev),
        metadata={"synthetic": True, "node_shape": list(node_shape), "num_classes": num_classes},
    )


def synthetic_vector_node_classification(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_classes: int = 10,
    num_edges: Optional[int] = None,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Any:
    """Create a synthetic graph for vector-feature node classification.

    Args:
        num_nodes: Number of nodes.
        num_features: Dimensionality of per-node vectors.
        num_classes: Number of classes.
        num_edges: Number of edges (default: ``5 * num_nodes``).
        seed: Random seed.
        device: Target device.

    Returns:
        :class:`~tgraphx.Graph`.
    """
    return synthetic_tensor_node_classification(
        num_nodes=num_nodes,
        node_shape=(num_features,),
        num_classes=num_classes,
        num_edges=num_edges,
        seed=seed,
        device=device,
    )


def synthetic_link_prediction(
    num_nodes: int = 1000,
    num_features: int = 64,
    num_edges: int = 5000,
    seed: Optional[int] = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Create a synthetic graph for link prediction.

    Returns a dict with keys ``graph``, ``train_edges``, ``val_edges``,
    ``test_edges`` (as LongTensor[2, E]).
    """
    from tgraphx import Graph

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    dev = torch.device(device)
    x = torch.randn(num_nodes, num_features, generator=gen)
    all_edges = torch.randint(0, num_nodes, (2, num_edges), generator=gen)
    split = [int(num_edges * 0.8), int(num_edges * 0.1)]
    split.append(num_edges - split[0] - split[1])
    parts = torch.split(all_edges, split, dim=1)
    graph = Graph(node_features=x.to(dev), edge_index=parts[0].to(dev))
    return {
        "graph": graph,
        "train_edges": parts[0].to(dev),
        "val_edges": parts[1].to(dev),
        "test_edges": parts[2].to(dev),
    }


def synthetic_graph_classification(
    num_graphs: int = 100,
    num_nodes_per_graph: int = 20,
    num_features: int = 16,
    num_classes: int = 4,
    num_edges_per_graph: int = 40,
    seed: Optional[int] = 42,
) -> List[Any]:
    """Create a list of synthetic graphs for graph classification.

    Each graph has a ``graph_label`` tensor.

    Returns:
        List of :class:`~tgraphx.Graph` objects.
    """
    from tgraphx import Graph

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    graphs = []
    for i in range(num_graphs):
        N = num_nodes_per_graph
        x = torch.randn(N, num_features, generator=gen)
        ei = torch.randint(0, N, (2, num_edges_per_graph), generator=gen)
        label = torch.tensor(i % num_classes, dtype=torch.long)
        graphs.append(Graph(node_features=x, edge_index=ei, graph_label=label))
    return graphs


# ── Model creation ────────────────────────────────────────────────────────────


def make_tensor_node_classifier(
    in_shape: Tuple[int, ...],
    num_classes: int,
    hidden_channels: int = 16,
) -> nn.Module:
    """Create a simple tensor-aware node classifier.

    The model uses two :class:`~tgraphx.ConvMessagePassing` layers followed
    by global average pooling over spatial dimensions and a linear head.

    Args:
        in_shape: Per-node feature shape ``(C, H, W)`` or ``(C, D, H, W)``.
        num_classes: Number of output classes.
        hidden_channels: Hidden channel count for intermediate layers.

    Returns:
        ``nn.Module`` with ``forward(x, edge_index) -> logits``.

    Example::

        model = tgx.easy.make_tensor_node_classifier(
            in_shape=(8, 6, 6), num_classes=4, hidden_channels=16,
        )
    """
    from tgraphx import ConvMessagePassing

    if len(in_shape) < 2:
        raise TGraphXShapeError(
            f"in_shape must be at least (C, H) for ConvMessagePassing, "
            f"got {in_shape}.  For vector features, use make_vector_node_classifier."
        )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = ConvMessagePassing(in_shape, (hidden_channels, *in_shape[1:]))
            self.conv2 = ConvMessagePassing(
                (hidden_channels, *in_shape[1:]),
                (hidden_channels, *in_shape[1:]),
            )
            spatial_dims = (1,) * (len(in_shape) - 1)
            if len(in_shape) == 3:
                self.pool = nn.AdaptiveAvgPool2d(spatial_dims)
            else:
                self.pool = nn.AdaptiveAvgPool3d(spatial_dims)
            self.head = nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            z = self.conv1(x, edge_index).relu()
            z = self.conv2(z, edge_index).relu()
            z = self.pool(z).flatten(1)
            return self.head(z)

    return _Model()


def make_vector_node_classifier(
    in_features: int,
    num_classes: int,
    hidden_channels: int = 64,
) -> nn.Module:
    """Create a vector-feature node classifier using GCNConv.

    Args:
        in_features: Number of input features per node.
        num_classes: Number of output classes.
        hidden_channels: Hidden dimension.

    Returns:
        ``nn.Module`` with ``forward(x, edge_index) -> logits``.
    """
    from tgraphx import GCNConv

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, x, edge_index):
            z = self.conv1(x, edge_index).relu()
            return self.conv2(z, edge_index)

    return _Model()


# ── Training ──────────────────────────────────────────────────────────────────


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_model_name(model: Optional[str], node_shape: Tuple[int, ...]) -> str:
    if model is not None and model != "auto":
        return model
    if len(node_shape) >= 2:
        return "tensor_gcn"
    return "vector_gcn"


def _build_model(
    model_name: str,
    node_shape: Tuple[int, ...],
    num_classes: int,
    hidden_channels: int,
) -> nn.Module:
    if model_name in ("tensor_gcn", "tensor_sage"):
        return make_tensor_node_classifier(
            in_shape=node_shape,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )
    elif model_name in ("vector_gcn", "linear"):
        in_features = node_shape[0] if node_shape else 1
        return make_vector_node_classifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
        )
    else:
        available = list(_MODELS.get("node_classification", {}).keys())
        raise TGraphXUnknownNameError(
            f"Unknown model '{model_name}' for node classification. "
            f"Available: {available}.\n"
            f"Use list_models('node_classification') for descriptions."
        )


def train_node_classifier(
    graph: Any,
    model: Optional[Union[str, nn.Module]] = "tensor_gcn",
    sampler: str = "neighbor",
    fanouts: Optional[List[int]] = None,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-3,
    hidden_channels: int = 16,
    device: str = "auto",
    seed: Optional[int] = None,
    mask: Optional[Any] = None,
    verbose: bool = True,
    config: Optional[EasyConfig] = None,
) -> EasyResult:
    """Train a node classifier on a graph.

    This is the canonical easy-mode entry point for node classification.
    No direct ``import torch`` is required for common use.

    Args:
        graph: A :class:`~tgraphx.Graph` with ``y`` (node labels) set.
            Create one with :func:`synthetic_tensor_node_classification` or::

                from tgraphx import Graph
                g = Graph(node_features=x, edge_index=edge_index, y=y)

        model: ``"tensor_gcn"`` (for [N,C,H,W] features), ``"vector_gcn"``
            (for [N,D] features), ``"auto"`` (infer from data), or a custom
            ``nn.Module`` with ``forward(x, edge_index)``.
        sampler: Sampling strategy.  ``"neighbor"`` (default), ``"full"``.
        fanouts: Neighbor fanouts per hop (default: ``[15, 10]``).
        batch_size: Seed nodes per batch.
        epochs: Training epochs.
        lr: Learning rate for Adam optimizer.
        hidden_channels: Hidden channels in auto-built models.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        seed: Random seed for reproducibility.
        mask: Optional ``BoolTensor[N]`` selecting which nodes to train on.
        verbose: Print epoch progress.
        config: Optional :class:`EasyConfig` that overrides keyword args.

    Returns:
        :class:`EasyResult` with metrics, history, model, graph, config, loader.

    Example::

        import tgraphx as tgx

        data = tgx.easy.synthetic_tensor_node_classification(
            num_nodes=1000, node_shape=(16, 8, 8), num_classes=10, seed=42,
        )
        result = tgx.easy.train_node_classifier(
            data, model="tensor_gcn", epochs=5, seed=42,
        )
        print(result.metrics)

    Error recovery:
        - Missing labels: create graph with ``Graph(..., y=labels)``
        - Unknown model: check ``list_models('node_classification')``
        - Device mismatch: set ``device='auto'`` or move graph manually
    """
    if config is not None:
        model = model if model != "tensor_gcn" else config.model or model
        sampler = config.sampler
        fanouts = fanouts or config.fanouts
        batch_size = config.batch_size
        epochs = config.epochs
        lr = config.lr
        hidden_channels = config.hidden_channels
        device = config.device
        seed = config.seed if config.seed is not None else seed
        verbose = config.verbose

    from tgraphx import Graph, NeighborLoader

    if not isinstance(graph, Graph):
        raise TGraphXConfigError(
            f"'graph' must be a tgraphx.Graph, got {type(graph).__name__}.\n"
            f"Create one with:\n"
            f"    from tgraphx import Graph\n"
            f"    g = Graph(node_features=x, edge_index=edge_index, y=y)\n"
            f"or use tgx.easy.synthetic_tensor_node_classification(...)"
        )

    if not graph.has_labels():
        raise TGraphXLabelError(
            "Node labels are required for node classification.\n"
            "Likely cause: the Graph was created without y/labels.\n"
            "Fix:\n"
            "    g = Graph(node_features=x, edge_index=edge_index, y=y)\n"
            "or:\n"
            "    g.y = y\n"
            "See docs/graph_basics.md#labels"
        )

    if seed is not None:
        torch.manual_seed(seed)
        from tgraphx.reproducibility import set_seed
        set_seed(seed)

    dev = _resolve_device(device)
    if fanouts is None:
        fanouts = [15, 10]

    node_shape = graph.feature_shape
    num_classes = int(graph.node_labels.max().item()) + 1

    if isinstance(model, str):
        model_name = _resolve_model_name(model, node_shape)
        net = _build_model(model_name, node_shape, num_classes, hidden_channels)
    else:
        net = model
        model_name = type(net).__name__

    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # Move graph to device if needed (no clone — we don't mutate the graph).
    if graph.device != dev:
        graph = graph.clone()
        graph.to(dev)

    resolved_config = {
        "task": "node_classification",
        "model": model_name,
        "sampler": sampler,
        "optimizer": "adam",
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(dev),
        "seed": seed,
        "fanouts": fanouts,
        "hidden_channels": hidden_channels,
        "num_classes": num_classes,
        "node_shape": list(node_shape),
    }

    if sampler not in _SAMPLERS:
        raise TGraphXUnknownNameError(
            f"Unknown sampler '{sampler}'. Available: {list(_SAMPLERS)}.\n"
            f"Use list_samplers() for descriptions."
        )

    history: List[Dict[str, float]] = []
    t0 = time.time()

    if sampler == "full":
        # Full-batch mode: use all nodes.
        loader = None
        for epoch in range(1, epochs + 1):
            net.train()
            nf = graph.node_features
            ei = graph.edge_index
            logits = net(nf, ei)
            loss = F.cross_entropy(logits, graph.node_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = logits.detach().argmax(dim=-1)
            acc = float((preds == graph.node_labels).float().mean().item())
            epoch_metrics = {"loss": loss.detach().item(), "accuracy": acc}
            history.append(epoch_metrics)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  loss={float(loss):.4f}  acc={acc:.4f}")
    else:
        # Neighbor-sampling mode.
        loader = NeighborLoader(
            graph, fanouts=fanouts, mask=mask,
            batch_size=batch_size, shuffle=True, seed=seed,
        )
        for epoch in range(1, epochs + 1):
            net.train()
            total_loss = 0.0
            total_correct = 0
            total_seeds = 0

            for batch in loader:
                batch.to(dev)
                logits = net(batch.node_features, batch.edge_index)
                s_logits = batch.seed_logits(logits)
                s_y = batch.seed_y
                loss = F.cross_entropy(s_logits, s_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.detach().item() * batch.batch_size
                preds = s_logits.detach().argmax(dim=-1)
                total_correct += int((preds == s_y).sum())
                total_seeds += batch.batch_size

            avg_loss = total_loss / max(total_seeds, 1)
            acc = total_correct / max(total_seeds, 1)
            epoch_metrics = {"loss": avg_loss, "accuracy": acc}
            history.append(epoch_metrics)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    elapsed = time.time() - t0
    final_metrics = history[-1] if history else {}

    return EasyResult(
        metrics=final_metrics,
        history=history,
        model=net,
        graph=graph,
        config=resolved_config,
        artifacts={},
        loader=loader,
        optimizer=opt,
        elapsed=elapsed,
    )


# Alias.
fit_node_classifier = train_node_classifier
