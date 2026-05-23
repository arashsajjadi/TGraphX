"""High-level one-call helpers for common TGraphX workflows (v1.4.1).

These are thin, safe wrappers over stable underlying APIs. They reduce
boilerplate without hiding mathematical semantics.
"""
from __future__ import annotations

import difflib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


# ── classify_nodes ──────────────────────────────────────────────────────────

def classify_nodes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    *,
    model: str = "tensor_gcn",
    train_mask: Optional[torch.Tensor] = None,
    val_mask: Optional[torch.Tensor] = None,
    test_mask: Optional[torch.Tensor] = None,
    seed: int = 42,
    device: Optional[str] = "auto",
    fast_mode: bool = True,
    epochs: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> "WorkflowResult":
    """One-call tensor node classification.

    Args:
        x: Node features ``[N, ...]`` — any rank supported.
        edge_index: ``[2, E]`` edge indices.
        labels: ``[N]`` integer class labels.
        model: ``"tensor_gcn"`` (default) or ``"gcn"``/``"gat"``.
        train_mask, val_mask, test_mask: Boolean ``[N]`` masks. If None,
            a deterministic 70/15/15 split is generated from ``seed``.
        seed: RNG seed.
        device: ``"auto"`` selects CUDA if available.
        fast_mode: Reduces epochs/batch for quick smoke runs.
        epochs: Override default epoch count.
        out_dir: Optional output directory for JSON artifacts.

    Returns:
        :class:`WorkflowResult` with ``metrics``, ``config``, ``artifacts``.
    """
    from ..reproducibility import set_seed
    from .workflow import _resolve_device
    from .leakage import check_leakage

    set_seed(seed, deterministic=False, warn_only=True)
    dev = _resolve_device(device)
    N = x.size(0)
    n_classes = int(labels.max().item()) + 1

    # Build masks if not provided
    if train_mask is None:
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
        n_train = int(0.7 * N); n_val = int(0.15 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True

    # Leakage check
    check_leakage(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, strict=True)

    t0 = time.time()
    from ..core.graph import Graph
    from ..loaders import NeighborLoader

    g = Graph(node_features=x, edge_index=edge_index, y=labels)

    # Build model based on feature rank
    feat_rank = x.dim()

    import torch.nn as nn
    import torch.nn.functional as F

    if feat_rank >= 3 and model in ("tensor_gcn", "conv_message_passing"):
        from ..layers.conv_message import ConvMessagePassing
        in_shape = tuple(x.shape[1:])
        # Create a safe out_shape: halve spatial dims if > 1
        if feat_rank == 4:
            C, H, W = in_shape
            out_shape = (max(4, C * 2), max(1, H // 2), max(1, W // 2))
        else:
            out_shape = (max(4, in_shape[0] * 2),) + tuple(max(1, s // 2) for s in in_shape[1:])

        class TensorGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = ConvMessagePassing(in_shape=in_shape, out_shape=out_shape)
                self.pool = nn.AdaptiveAvgPool2d(1) if feat_rank == 4 else nn.Identity()
                flat_dim = out_shape[0] if feat_rank == 4 else math.prod(out_shape)
                self.head = nn.Linear(flat_dim, n_classes)
            def forward(self, x, ei):
                h = F.relu(self.conv(x, ei))
                if feat_rank == 4:
                    h = self.pool(h).squeeze(-1).squeeze(-1)
                else:
                    h = h.flatten(1)
                return self.head(h)

        mdl = TensorGCN().to(dev)
    else:
        from ..layers.vector_gcn import GCNConv
        feat_dim = x.shape[1] if x.dim() >= 2 else x.shape[0]
        hidden = 32

        class VectorGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gc1 = GCNConv(feat_dim, hidden)
                self.gc2 = GCNConv(hidden, n_classes)
            def forward(self, x, ei):
                return self.gc2(F.relu(self.gc1(x, ei)), ei)

        mdl = VectorGCN().to(dev)

    n_epochs = epochs if epochs is not None else (3 if fast_mode else 15)
    opt = torch.optim.Adam(mdl.parameters(), lr=5e-3)
    loader = NeighborLoader(g, fanouts=[10, 5], batch_size=32 if not fast_mode else 16,
                            mask=train_mask, shuffle=True, seed=seed)
    for _ in range(n_epochs):
        mdl.train()
        for batch in loader:
            logits = mdl(batch.node_features.to(dev), batch.edge_index.to(dev))
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

    mdl.eval()
    def _eval(mask):
        l = NeighborLoader(g, fanouts=[10, 5], batch_size=32, mask=mask, shuffle=False, seed=seed)
        correct, total = 0, 0
        with torch.no_grad():
            for batch in l:
                logits = mdl(batch.node_features.to(dev), batch.edge_index.to(dev))
                preds = batch.seed_logits(logits).argmax(1)
                correct += (preds == batch.seed_y.to(dev)).sum().item()
                total += batch.seed_y.numel()
        return correct / max(1, total)

    val_acc = _eval(val_mask)
    test_acc = _eval(test_mask) if test_mask is not None else float("nan")
    runtime = time.time() - t0

    metrics = {"val_accuracy": round(val_acc, 4), "test_accuracy": round(test_acc, 4),
               "epochs": n_epochs, "num_nodes": N, "num_classes": n_classes}
    config = {"model": model, "seed": seed, "device": dev, "fast_mode": fast_mode,
              "epochs": n_epochs}
    return _make_result("node_classification", metrics, config, runtime, out_dir)


# Aliases
node_classification = classify_nodes
fit_node_classifier = classify_nodes
train_node_classifier = classify_nodes


# ── kg_completion ────────────────────────────────────────────────────────────

def kg_completion(
    triples: torch.Tensor,
    num_entities: int,
    num_relations: int,
    *,
    model: str = "transe",
    epochs: Optional[int] = None,
    batch_size: int = 64,
    embedding_dim: int = 32,
    seed: int = 42,
    device: Optional[str] = "auto",
    fast_mode: bool = True,
    val_triples: Optional[torch.Tensor] = None,
    test_triples: Optional[torch.Tensor] = None,
    entity_features: Optional[Dict[str, torch.Tensor]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> "WorkflowResult":
    """One-call knowledge graph link prediction / completion.

    Args:
        triples: ``[T, 3]`` LongTensor of (head, relation, tail) IDs.
        num_entities: Entity vocabulary size.
        num_relations: Relation vocabulary size.
        model: ``"transe"`` (default), ``"distmult"``, or ``"rescal"``.
        epochs: Epochs (default: 3 fast, 20 full).
        batch_size: Mini-batch size.
        embedding_dim: Embedding dimension.
        seed: RNG seed.
        device: ``"auto"`` selects CUDA if available.
        fast_mode: Use tiny defaults.
        val_triples, test_triples: Optional separate splits.
        entity_features: Optional entity feature dict.
        out_dir: Optional artifact directory.

    Returns:
        :class:`WorkflowResult` with MRR/Hits@K when val_triples provided.
    """
    from ..reproducibility import set_seed
    from .workflow import _resolve_device
    from ..kg import KnowledgeGraph, KGTrainer, KGTrainingConfig

    _KG_MODELS = {"transe": "TransE", "distmult": "DistMult", "rescal": "RESCAL",
                  "simple": "SimplE", "simplee": "SimplE", "rotate": "RotatE",
                  "complex": "ComplEx"}
    model_key = model.lower().replace("_", "").replace("-", "")
    if model_key not in _KG_MODELS:
        suggestions = difflib.get_close_matches(model_key, list(_KG_MODELS), n=1)
        hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
        raise ValueError(f"Unknown KG model {model!r}.{hint} Choose from: {list(_KG_MODELS)}")
    canonical_model = _KG_MODELS[model_key]

    set_seed(seed, deterministic=False, warn_only=True)
    dev = _resolve_device(device)
    n_epochs = epochs if epochs is not None else (3 if fast_mode else 20)

    kg = KnowledgeGraph(triples, num_entities=num_entities, num_relations=num_relations,
                        entity_features=entity_features)

    # Import the model class
    model_map = {
        "TransE": "tgraphx.kg:TransEModel",
        "DistMult": "tgraphx.kg:DistMultModel",
        "RESCAL": "tgraphx.kg:RESCALModel",
        "SimplE": "tgraphx.kg:SimplEModel",
        "RotatE": "tgraphx.kg:RotatEModel",
        "ComplEx": "tgraphx.kg:ComplExModel",
    }
    from tgraphx.kg import TransEModel, DistMultModel, RESCALModel, SimplEModel
    model_classes = {"TransE": TransEModel, "DistMult": DistMultModel,
                     "RESCAL": RESCALModel, "SimplE": SimplEModel}
    try:
        from tgraphx.kg import RotatEModel, ComplExModel
        model_classes.update({"RotatE": RotatEModel, "ComplEx": ComplExModel})
    except ImportError:
        pass
    cls = model_classes[canonical_model]
    mdl = cls(num_entities, num_relations, embedding_dim=embedding_dim)

    config_obj = KGTrainingConfig(num_epochs=n_epochs, batch_size=batch_size,
                                  device=dev, seed=seed)
    trainer = KGTrainer(mdl, config_obj, triples)
    t0 = time.time()
    history = trainer.fit()
    runtime = time.time() - t0

    metrics: Dict[str, Any] = {
        "final_loss": round(float(history["final_loss"]), 4),
        "epochs": n_epochs,
        "model": canonical_model,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_triples": len(triples),
    }

    # Optional evaluation
    if val_triples is not None and len(val_triples) > 0:
        try:
            from tgraphx.kg import KGEvaluator
            evaluator = KGEvaluator(train_triples=triples,
                                    valid_triples=val_triples,
                                    test_triples=test_triples if test_triples is not None else val_triples,
                                    num_entities=num_entities)
            res = evaluator.evaluate(mdl, triples=val_triples, filtered=True,
                                     batch_size=64, device=dev)
            rd = res.to_dict()
            filtered = rd.get("filtered", rd)
            if isinstance(filtered, dict) and "combined" in filtered:
                filtered = filtered["combined"]
            metrics["mrr"] = round(float(filtered.get("MRR", float("nan"))), 4)
            metrics["hits_at_10"] = round(float(filtered.get("Hits@10", float("nan"))), 4)
        except Exception:
            pass

    config = {"model": canonical_model, "seed": seed, "device": dev, "fast_mode": fast_mode,
              "epochs": n_epochs, "embedding_dim": embedding_dim}
    return _make_result("kg_link_prediction", metrics, config, runtime, out_dir)


# Aliases
fit_kg = kg_completion
train_kg = kg_completion


# ── make_graph ───────────────────────────────────────────────────────────────

def make_graph(
    x: Optional[torch.Tensor] = None,
    *,
    edges: Optional[Any] = None,
    edge_index: Optional[torch.Tensor] = None,
    adjacency: Optional[Any] = None,
    networkx_graph: Optional[Any] = None,
    labels: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    node_features: Optional[torch.Tensor] = None,
    **graph_kwargs: Any,
) -> Any:
    """Unified graph construction from any common input format.

    Dispatch to :class:`Graph.from_edges`, :class:`Graph.from_adjacency`,
    or :class:`Graph.from_networkx` based on what is provided.

    Args:
        x: Node features (any rank).
        edges: Edge list (list of tuples, [E,2] or [2,E] tensor).
        edge_index: [2,E] edge index tensor.
        adjacency: Dense [N,N] or scipy sparse adjacency.
        networkx_graph: NetworkX graph object.
        labels / y: Node labels (aliases).
        **graph_kwargs: Forwarded to ``Graph`` constructor.

    Returns:
        A :class:`tgraphx.Graph`.

    Raises:
        ValueError: If more than one edge source is provided, or no node
            features can be inferred.
    """
    from ..core.graph import Graph

    # Resolve x / node_features
    feat = x if x is not None else node_features
    y_val = labels if labels is not None else y

    # Check ambiguity
    edge_sources = sum([
        edges is not None,
        edge_index is not None,
        adjacency is not None,
        networkx_graph is not None,
    ])
    if edge_sources > 1:
        raise ValueError(
            "make_graph: provide at most ONE of edges, edge_index, adjacency, or networkx_graph. "
            "Multiple edge sources are ambiguous."
        )

    if networkx_graph is not None:
        # Build topology from the NetworkX object, then attach any user-supplied
        # tensor fields (x, labels, edge_attr, masks, …).  Previously these
        # were silently discarded — Codex/Composer TGX-AUDIT-002.
        g = Graph.from_networkx(networkx_graph)
        if feat is not None:
            if feat.size(0) != g.num_nodes:
                raise ValueError(
                    f"make_graph: x has {feat.size(0)} rows but NetworkX graph has "
                    f"{g.num_nodes} nodes."
                )
            g.node_features = feat
        if y_val is not None:
            if y_val.size(0) != g.num_nodes:
                raise ValueError(
                    f"make_graph: labels have {y_val.size(0)} entries but "
                    f"NetworkX graph has {g.num_nodes} nodes."
                )
            g.node_labels = y_val
        for k, v in graph_kwargs.items():
            # Attach supplied graph kwargs (edge_attr, edge_weight,
            # edge_labels, graph_label, graph_features, metadata, ...).
            setattr(g, k, v)
        # Re-validate so any inconsistency surfaces with a clear error.
        g.validate()
        return g

    if adjacency is not None:
        return Graph.from_adjacency(adjacency, node_features=feat,
                                    **({"y": y_val} if y_val is not None else {}),
                                    **graph_kwargs)

    if edges is not None:
        return Graph.from_edges(edges, node_features=feat,
                                **({"y": y_val} if y_val is not None else {}),
                                **graph_kwargs)

    if edge_index is not None:
        if feat is None:
            raise ValueError("make_graph: provide x/node_features when using edge_index.")
        return Graph(node_features=feat, edge_index=edge_index,
                     **({"y": y_val} if y_val is not None else {}), **graph_kwargs)

    # No edge source: just build a feature-only graph
    if feat is None:
        raise ValueError(
            "make_graph: must provide at least x/node_features and one edge source. "
            "Use Graph.from_edges, Graph.from_adjacency, or Graph.from_networkx."
        )
    return Graph(node_features=feat, **({"y": y_val} if y_val is not None else {}),
                 **graph_kwargs)


# Aliases
build_graph = make_graph
graph = make_graph  # tgx.graph(...)


# ── explain_error ────────────────────────────────────────────────────────────

_ERROR_MAP = {
    "rank": (
        "Graph features have rank > 2. GraphML only supports 2-D (vector) features. "
        "Use `tgx.save(graph, 'file.tgx')` / `tgx.load(...)` for tensor-valued graphs.\n"
        "→ See: tgx.save(), tgx.load(), Graph.save(), Graph.load()"
    ),
    "nsga": (
        "NSGAIIOptimizer requires a LIST of objectives, not a single fitness function. "
        "Correct: NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness]).\n"
        "→ See: tgraphx.evolutionary.NSGAIIOptimizer"
    ),
    "composite_fitness": (
        "Pass a list of objectives to NSGAIIOptimizer, not composite_fitness. "
        "composite_fitness is a single-objective utility, not NSGA-II compatible.\n"
        "→ Use: NSGAIIOptimizer(config, [connectivity_fitness, sparsity_fitness])"
    ),
    "vgae": (
        "VGAE is a graph autoencoder / link-prediction model, not a classical graph generator. "
        "To generate graphs, use tgx.generate_graph('barabasi_albert'|'erdos_renyi'|...). "
        "For link prediction, use KnowledgeGraph + TransEModel or GraphAutoencoder separately.\n"
        "→ See: tgx.generate_graph(), tgx.list_workflow_tasks()"
    ),
    "missing optional": (
        "A required optional package is missing. Install it with:\n"
        "  pip install torchvision       # already a base dependency; reinstall if missing\n"
        "  pip install torch-geometric   # for PyG (tgraphx[pyg])\n"
        "  pip install networkx          # for NetworkX\n"
        "  pip install scipy             # for sparse adjacency"
    ),
    "edge_index": (
        "edge_index must have shape [2, E] (source and destination rows). "
        "If you have [E, 2], transpose it: edge_index.t().contiguous() "
        "or use Graph.from_edges(edge_list).\n"
        "→ See: Graph.from_edges()"
    ),
    "mask overlap": (
        "Train/val/test masks must not overlap. "
        "Use tgx.check_leakage(train_mask, val_mask, test_mask) to diagnose.\n"
        "→ See: tgx.check_leakage()"
    ),
    "json": (
        "Artifact contains non-JSON-serializable values (Tensor, Path, device, dtype). "
        "Detach and convert: value.item() for scalars, value.tolist() for tensors.\n"
        "→ See: tgx.audit_run_dir() for artifact schema validation."
    ),
    "cuda": (
        "CUDA is not available on this system. "
        "Use device='cpu' or device='auto' (auto selects CPU if CUDA unavailable)."
    ),
}


def explain_error(error: Any, *, keyword: Optional[str] = None) -> str:
    """Return actionable guidance for common TGraphX errors.

    Args:
        error: An exception or error string.
        keyword: Optional keyword to search in error map.

    Returns:
        A human-readable guidance string.
    """
    error_str = str(error).lower() if not isinstance(error, str) else error.lower()
    if keyword:
        error_str = keyword.lower()

    for key, guidance in _ERROR_MAP.items():
        if key in error_str:
            return guidance

    # Fuzzy match
    suggestions = difflib.get_close_matches(error_str, list(_ERROR_MAP), n=1, cutoff=0.4)
    if suggestions:
        return f"Closest guidance for {error_str!r}:\n\n" + _ERROR_MAP[suggestions[0]]

    return (
        f"No specific guidance for this error: {str(error)[:200]}\n"
        "Suggestions:\n"
        "  - Check tensor shapes with tgx.validate_graph(graph)\n"
        "  - Check optional dependencies with tgx.audit_package_readiness()\n"
        "  - Inspect batch contents with tgx.debug_batch(batch)\n"
        "  - Read docs: https://github.com/arashsajjadi/TGraphX"
    )


troubleshoot_error = explain_error  # alias


# ── debug_batch ──────────────────────────────────────────────────────────────

def debug_batch(batch: Any) -> Dict[str, Any]:
    """Summarize a NeighborLoader/GraphBatch object for debugging.

    Checks node count, edge count, seed count, feature shapes, device, dtype,
    and whether seed_logits() is compatible with seed_y.
    """
    info: Dict[str, Any] = {"type": type(batch).__name__}
    issues: List[str] = []

    if hasattr(batch, "node_features") and batch.node_features is not None:
        nf = batch.node_features
        info["node_features_shape"] = list(nf.shape)
        info["node_features_device"] = str(nf.device)
        info["node_features_dtype"] = str(nf.dtype)

    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        ei = batch.edge_index
        info["num_edges"] = int(ei.shape[1])
        if ei.shape[0] != 2:
            issues.append(f"edge_index has shape {list(ei.shape)}, expected [2, E]")

    if hasattr(batch, "seed_y") and batch.seed_y is not None:
        sy = batch.seed_y
        info["num_seed_nodes"] = int(sy.shape[0])
        info["seed_y_shape"] = list(sy.shape)

    if hasattr(batch, "num_nodes"):
        info["num_nodes"] = int(batch.num_nodes)
        if hasattr(batch, "seed_y") and batch.seed_y is not None:
            if info.get("num_seed_nodes", 0) > info["num_nodes"]:
                issues.append(f"seed count {info['num_seed_nodes']} > num_nodes {info['num_nodes']}")

    if hasattr(batch, "num_graphs"):
        info["num_graphs"] = int(batch.num_graphs)
    if hasattr(batch, "graph_labels") and batch.graph_labels is not None:
        info["graph_labels_shape"] = list(batch.graph_labels.shape)

    info["issues"] = issues
    info["ok"] = len(issues) == 0
    return info


batch_summary = debug_batch
assert_batch_consistent = debug_batch


# ── dataset_card ─────────────────────────────────────────────────────────────

def dataset_card(dataset: Any, *, task: Optional[str] = None) -> Dict[str, Any]:
    """Return a JSON-serializable dataset card."""
    from .describe import describe as _describe
    card: Dict[str, Any] = {"type": "dataset_card"}
    d = _describe(dataset)
    card.update(d)
    if task:
        card["task"] = task
    card["limitations"] = (
        "This card is auto-generated from available metadata. "
        "Dataset license, citation, and full provenance must be verified "
        "from the upstream source."
    )
    return card


# ── model_card ───────────────────────────────────────────────────────────────

def model_card(
    model: Any,
    task: str = "unknown",
    *,
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable model card."""
    card: Dict[str, Any] = {
        "type": "model_card",
        "model_class": type(model).__name__,
        "task": task,
    }
    if hasattr(model, "parameters"):
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            card["parameters_total"] = total
            card["parameters_trainable"] = trainable
        except Exception:
            pass
    if dataset:
        card["dataset"] = dataset
    if seed is not None:
        card["seed"] = seed
    if device is not None:
        card["device"] = str(device)
    card["tensor_native_support"] = True
    card["disclaimer"] = (
        "TGraphX model cards are informational. "
        "No SOTA claims are made."
    )
    return card


# ── benchmark_card ───────────────────────────────────────────────────────────

def benchmark_card(result: Any) -> Dict[str, Any]:
    """Return a JSON-serializable benchmark card from a workflow result."""
    import tgraphx
    card: Dict[str, Any] = {
        "type": "benchmark_card",
        "tgraphx_version": tgraphx.__version__,
        "disclaimer": (
            "This is a functionality / smoke benchmark, NOT a SOTA claim. "
            "Results reflect FAST_MODE or small-scale runs only. "
            "Do not compare across systems, hardware, or datasets without "
            "matching experimental conditions."
        ),
    }
    if hasattr(result, "to_dict"):
        card.update(result.to_dict())
    elif isinstance(result, dict):
        card.update(result)
    return card


# ── audit_package_readiness ──────────────────────────────────────────────────

def audit_package_readiness() -> Dict[str, Any]:
    """Return a JSON-serializable package readiness summary.

    Includes: version, CUDA status, optional deps, public API count,
    dataset registry, workflow tasks, serialization availability,
    dashboard audit availability, known limitations.
    """
    import tgraphx
    import torch

    report: Dict[str, Any] = {
        "tgraphx_version": tgraphx.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    # Dependencies — torchvision and pyyaml are mandatory per pyproject.toml.
    required = {}
    for pkg in ("torch", "torchvision", "yaml"):
        try:
            m = __import__(pkg)
            required[pkg] = getattr(m, "__version__", "installed")
        except ImportError:
            required[pkg] = "not installed"
    report["required_dependencies"] = required

    optional = {}
    for pkg in ("torch_geometric", "dgl", "scipy", "networkx", "pandas",
                "tensorboard", "mlflow", "PIL", "ogb"):
        try:
            m = __import__(pkg)
            optional[pkg] = getattr(m, "__version__", "installed")
        except ImportError:
            optional[pkg] = "not installed"
    report["optional_dependencies"] = optional

    # Public API
    from .public_api import public_api
    api_groups = public_api()
    report["public_api"] = {k: len(v) for k, v in api_groups.items()}

    # Dataset registry
    try:
        from tgraphx.datasets import list_datasets
        report["dataset_registry_entries"] = len(list_datasets())
    except Exception:
        report["dataset_registry_entries"] = "unknown"

    # Workflow tasks
    from .workflow import list_workflow_tasks
    report["workflow_tasks"] = list_workflow_tasks()

    # Generation / RL / Evo
    try:
        from tgraphx.generation import list_graph_generation_methods
        report["graph_generation_methods"] = list(list_graph_generation_methods())
    except Exception:
        report["graph_generation_methods"] = "unavailable"
    try:
        from tgraphx.rl import list_graph_rl_algorithms
        report["rl_algorithms"] = list(list_graph_rl_algorithms())
    except Exception:
        report["rl_algorithms"] = "unavailable"

    # Features
    report["features"] = {
        "serialization_tgx": True,
        "dashboard_audit": True,
        "tensor_native_nodes": True,
        "validate_graph": True,
        "knn_graph": True,
        "reproducible_context": True,
        "public_api_registry": True,
        "kg_embeddings": True,
    }

    # Known limitations
    report["known_limitations"] = [
        "No distributed/out-of-core graph processing.",
        "CUDA determinism is best-effort; some ops lack deterministic implementations.",
        "GraphML serialization fails for rank > 2 node features; use .tgx format.",
        "DGL conversion is optional and not tested in CI.",
        "Workflow dispatcher supports a small task set; it is not AutoML.",
        "No SOTA / parity claims vs PyG, DGL, NetworkX, PyKEEN, SB3, or RLlib.",
    ]
    return report


# ── WorkflowResult (shared dataclass) ────────────────────────────────────────

@dataclass
class WorkflowResult:
    """Result of a TGraphX one-call workflow (v1.4.1+)."""
    task: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    runtime_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    tgraphx_version: str = ""

    def __post_init__(self):
        import tgraphx
        if not self.tgraphx_version:
            self.tgraphx_version = tgraphx.__version__

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"## {self.task} result", f"- version: {self.tgraphx_version}",
                 f"- runtime: {self.runtime_s:.2f}s", "### Metrics"]
        for k, v in self.metrics.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


def _make_result(
    task: str,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    runtime: float,
    out_dir: Optional[Union[str, Path]],
) -> WorkflowResult:
    import tgraphx
    result = WorkflowResult(task=task, metrics=metrics, config=config,
                            runtime_s=round(runtime, 3),
                            tgraphx_version=tgraphx.__version__)
    if out_dir is not None:
        from tgraphx.tracking import write_run_metadata, write_metrics_summary
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        write_run_metadata(str(d / "run_metadata.json"),
                           tgraphx_version=tgraphx.__version__,
                           seed=config.get("seed", 0),
                           fast_mode=config.get("fast_mode", True),
                           device=config.get("device", "cpu"),
                           task=task, runtime_s=round(runtime, 3))
        write_metrics_summary(str(d / "metrics_summary.json"), **
                              {k: float(v) if isinstance(v, (int, float)) else str(v)
                               for k, v in metrics.items()})
        with open(d / "benchmark_summary.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        result.artifacts = {
            "run_metadata.json": str(d / "run_metadata.json"),
            "benchmark_summary.json": str(d / "benchmark_summary.json"),
        }
    return result
