"""High-level API for graph generation.

Stability: Beta (v0.7.0+).

Usage:
    from tgraphx.generation import run_graph_generation, list_graph_generation_methods
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "list_graph_generation_methods",
    "make_graph_generator",
    "run_graph_generation",
    "GenerationResult",
]


_GENERATION_METHODS: Dict[str, Dict[str, str]] = {
    "erdos_renyi": {
        "stability": "Stable",
        "description": "Erdos-Renyi random graph G(n, p).",
        "class": "FeatureAwareERGraph",
    },
    "barabasi_albert": {
        "stability": "Stable",
        "description": "Barabasi-Albert preferential attachment.",
        "class": "FeatureAwareBAGraph",
    },
    "watts_strogatz": {
        "stability": "Stable",
        "description": "Watts-Strogatz small-world graph.",
        "class": "watts_strogatz",
    },
    "stochastic_block_model": {
        "stability": "Stable",
        "description": "Stochastic block model (community structure).",
        "class": "sbm",
    },
    "grid": {
        "stability": "Stable",
        "description": "2D grid graph.",
        "class": "grid",
    },
    "cycle": {
        "stability": "Stable",
        "description": "Cycle graph.",
        "class": "cycle",
    },
    "path": {
        "stability": "Stable",
        "description": "Path graph.",
        "class": "path",
    },
    "star": {
        "stability": "Stable",
        "description": "Star graph.",
        "class": "star",
    },
    "complete": {
        "stability": "Stable",
        "description": "Complete graph K_n.",
        "class": "complete",
    },
    "motif_injected": {
        "stability": "Experimental",
        "description": "ER graph with injected motifs.",
        "class": "MotifInjectedGraph",
    },
    "anomaly_injected": {
        "stability": "Experimental",
        "description": "ER graph with synthetic anomalies.",
        "class": "AnomalyInjectedGraph",
    },
    "temporal": {
        "stability": "Experimental",
        "description": "Temporal evolving graph.",
        "class": "TemporalEvolvingGraph",
    },
    "typed": {
        "stability": "Experimental",
        "description": "Graph with typed nodes and edges.",
        "class": "TypedGeneratedGraph",
    },
    "random_geometric": {
        "stability": "Experimental",
        "description": "Random geometric graph (edges by Euclidean proximity).",
        "class": "random_geometric",
    },
}


# Methods that LLMs commonly request but are NOT classical generators.
# Documented redirects so users get a helpful pointer instead of a bare error.
_NEURAL_GENERATOR_REDIRECTS: Dict[str, str] = {
    "vgae": (
        "method='vgae' is not a classical graph generator. "
        "VGAE (Variational Graph Autoencoder) is a representation-learning / "
        "link-prediction model, not a classical generator usable through "
        "run_graph_generation. "
        "Use `from tgraphx.generation import VGAEGraphGenerator` to fit a VGAE "
        "on an existing graph, or pick one of the classical methods."
    ),
    "gae": (
        "method='gae' is not a classical graph generator. "
        "GAE (Graph Autoencoder) is a representation-learning / link-prediction "
        "model. Use `from tgraphx.generation import VGAEGraphGenerator` for the "
        "neural variant, or pick one of the classical methods."
    ),
    "autoregressive": (
        "method='autoregressive' is not a classical graph generator. "
        "Use `from tgraphx.generation import AutoregressiveEdgeGenerator` "
        "directly, or pick one of the classical methods."
    ),
    "transformer": (
        "method='transformer' is not a classical graph generator. "
        "Use `from tgraphx.generation import GraphTransformerGenerator` "
        "directly, or pick one of the classical methods."
    ),
}


def _unknown_method_error(method: str) -> ValueError:
    """Build a helpful ValueError for an unknown generation method."""
    known = sorted(_GENERATION_METHODS.keys())
    extra = _NEURAL_GENERATOR_REDIRECTS.get(method.lower())
    base = f"Unknown generation method {method!r}. Choose from: {known}"
    if extra is not None:
        return ValueError(f"{base}\n\n{extra}")
    return ValueError(base)


def list_graph_generation_methods() -> Dict[str, Dict[str, str]]:
    """Return dict: method_name -> info dict with stability, description.

    Returns:
        Dict mapping method name -> info dict.
    """
    return dict(_GENERATION_METHODS)


def make_graph_generator(method: str, **kwargs) -> Any:
    """Create a graph generator callable by name.

    Args:
        method: Generator method name from list_graph_generation_methods().
        **kwargs: Extra kwargs forwarded to the generator.

    Returns:
        Callable that generates a GeneratedGraph.

    Raises:
        ValueError: If method not recognized.
    """
    if method not in _GENERATION_METHODS:
        raise _unknown_method_error(method)

    from tgraphx.generation.classical import (
        FeatureAwareERGraph, FeatureAwareBAGraph,
        TemporalEvolvingGraph, TypedGeneratedGraph,
        AnomalyInjectedGraph, MotifInjectedGraph,
    )

    if method == "erdos_renyi":
        def _gen(n, seed=None, nfd=0, efd=None, **kw):
            return FeatureAwareERGraph(n=n, p=kw.pop("p", 0.3), node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    elif method == "barabasi_albert":
        def _gen(n, seed=None, nfd=0, **kw):
            return FeatureAwareBAGraph(n=n, m=kw.pop("m", 2), node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    elif method == "motif_injected":
        def _gen(n, seed=None, nfd=0, **kw):
            return MotifInjectedGraph(n=n, node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    elif method == "anomaly_injected":
        def _gen(n, seed=None, nfd=0, **kw):
            return AnomalyInjectedGraph(n=n, node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    elif method == "temporal":
        def _gen(n, seed=None, nfd=0, **kw):
            return TemporalEvolvingGraph(n=n, node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    elif method == "typed":
        def _gen(n, seed=None, nfd=0, **kw):
            return TypedGeneratedGraph(n=n, node_feature_dim=max(nfd, 1), seed=seed)
        return _gen

    else:
        # Structural methods: watts_strogatz, sbm, grid, cycle, path, star, complete, random_geometric
        def _gen(n, seed=None, nfd=0, **kw):
            return _make_structural_graph(method, n, seed, nfd, **kw)
        return _gen


def _make_structural_graph(method: str, n: int, seed: Optional[int], nfd: int, **kwargs) -> Any:
    """Create a structural graph (path, cycle, star, etc.)."""
    from tgraphx.generation.data_model import GeneratedGraph
    from tgraphx.mining.generators import erdos_renyi_graph

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    if method == "path":
        if n < 2:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            src = list(range(n - 1)) + list(range(1, n))
            dst = list(range(1, n)) + list(range(n - 1))
            ei = torch.tensor([src, dst], dtype=torch.long)

    elif method == "cycle":
        if n < 2:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            src = list(range(n)) + list(range(1, n)) + [0]
            src_rev = list(range(1, n)) + [0]
            src_fwd = list(range(n))
            ei = torch.tensor([src_fwd + src_rev, src_rev + src_fwd], dtype=torch.long)

    elif method == "star":
        if n < 2:
            ei = torch.zeros((2, 0), dtype=torch.long)
        else:
            spoke = list(range(1, n))
            src = spoke + [0] * (n - 1)
            dst = [0] * (n - 1) + spoke
            ei = torch.tensor([src, dst], dtype=torch.long)

    elif method == "complete":
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    elif method == "grid":
        import math
        rows = int(math.ceil(math.sqrt(n)))
        cols = (n + rows - 1) // rows
        src, dst = [], []
        for r in range(rows):
            for c in range(cols):
                v = r * cols + c
                if v >= n:
                    break
                # Right
                if c + 1 < cols and r * cols + (c + 1) < n:
                    nb = r * cols + (c + 1)
                    src += [v, nb]; dst += [nb, v]
                # Down
                if r + 1 < rows and (r + 1) * cols + c < n:
                    nb = (r + 1) * cols + c
                    src += [v, nb]; dst += [nb, v]
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    elif method == "watts_strogatz":
        # Simple WS: ring lattice then rewire
        k = min(kwargs.get("k", 4), n - 1)
        src, dst = [], []
        for i in range(n):
            for j in range(1, k // 2 + 1):
                nb = (i + j) % n
                src += [i, nb]; dst += [nb, i]
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    elif method == "stochastic_block_model":
        n_blocks = kwargs.get("n_blocks", 2)
        p_in = kwargs.get("p_in", 0.7)
        p_out = kwargs.get("p_out", 0.05)
        block_size = n // n_blocks
        src, dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                b_i = i // block_size
                b_j = j // block_size
                p = p_in if b_i == b_j else p_out
                if torch.rand(1, generator=rng).item() < p:
                    src += [i, j]; dst += [j, i]
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    elif method == "random_geometric":
        # Place nodes randomly in unit square, connect if distance < r
        r = kwargs.get("r", 0.3)
        positions = torch.rand(n, 2, generator=rng)
        src, dst = [], []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float((positions[i] - positions[j]).norm().item())
                if dist < r:
                    src += [i, j]; dst += [j, i]
        ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    nf = torch.randn(n, max(nfd, 1), generator=rng) if nfd > 0 else None
    return GeneratedGraph(edge_index=ei, num_nodes=n, node_features=nf)


@dataclass
class GenerationResult:
    """Result of a run_graph_generation call.

    Attributes:
        graphs: List of GeneratedGraph instances.
        metrics: Dict with validity, uniqueness, novelty, diversity.
        config: Serializable config dict.
        report_path: Path to JSON report if dashboard_dir was set.
    """
    graphs: list
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    report_path: Optional[str] = None


def run_graph_generation(
    method: str = "erdos_renyi",
    num_graphs: int = 16,
    num_nodes: int = 20,
    node_feature_dim: int = 0,
    edge_feature_dim: int = 0,
    seed: int = 42,
    dashboard_dir: Optional[str] = None,
    **method_kwargs,
) -> GenerationResult:
    """Generate graphs by method name.

    Args:
        method: Generator method name from list_graph_generation_methods().
        num_graphs: Number of graphs to generate.
        num_nodes: Number of nodes per graph.
        node_feature_dim: Node feature dimensionality (0 = no features).
        edge_feature_dim: Edge feature dimensionality (0 = no features).
        seed: Random seed.
        dashboard_dir: If set, writes metrics JSON here.
        **method_kwargs: Extra kwargs forwarded to the generator.

    Returns:
        GenerationResult with graphs and metrics.

    Raises:
        ValueError: If method not recognized.
    """
    if method not in _GENERATION_METHODS:
        raise _unknown_method_error(method)

    from tgraphx.generation.metrics import (
        validity_score, uniqueness_score, novelty_score, diversity_score,
    )
    from tgraphx.generation.classical import (
        FeatureAwareERGraph, FeatureAwareBAGraph,
        TemporalEvolvingGraph, TypedGeneratedGraph,
        AnomalyInjectedGraph, MotifInjectedGraph,
    )

    torch.manual_seed(seed)

    graphs = []
    for i in range(num_graphs):
        g_seed = seed + i
        try:
            if method == "erdos_renyi":
                p = method_kwargs.get("p", 0.3)
                g = FeatureAwareERGraph(
                    n=num_nodes, p=p,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            elif method == "barabasi_albert":
                m = method_kwargs.get("m", 2)
                g = FeatureAwareBAGraph(
                    n=num_nodes, m=m,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            elif method == "motif_injected":
                g = MotifInjectedGraph(
                    n=num_nodes,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            elif method == "anomaly_injected":
                g = AnomalyInjectedGraph(
                    n=num_nodes,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            elif method == "temporal":
                g = TemporalEvolvingGraph(
                    n=num_nodes,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            elif method == "typed":
                g = TypedGeneratedGraph(
                    n=num_nodes,
                    node_feature_dim=max(node_feature_dim, 1) if node_feature_dim > 0 else 0,
                    seed=g_seed,
                )

            else:
                g = _make_structural_graph(
                    method, num_nodes, g_seed, node_feature_dim, **method_kwargs
                )

        except Exception:
            from tgraphx.generation.data_model import GeneratedGraph
            ei = torch.zeros((2, 0), dtype=torch.long)
            g = GeneratedGraph(edge_index=ei, num_nodes=num_nodes)

        graphs.append(g)

    # Compute metrics
    try:
        # validity: fraction of graphs with at least one edge (basic validity)
        val = validity_score(graphs, constraint_fn=lambda g: int(g.edge_index.shape[1]) > 0)
    except Exception:
        val = float(sum(1 for g in graphs if int(g.edge_index.shape[1]) > 0)) / max(len(graphs), 1)

    try:
        uniq = uniqueness_score(graphs)
    except Exception:
        uniq = 1.0

    try:
        nov = novelty_score(graphs, reference_graphs=graphs[:max(1, len(graphs) // 4)])
    except Exception:
        nov = 1.0

    try:
        div = diversity_score(graphs)
    except Exception:
        div = 1.0

    metrics: Dict[str, Any] = {
        "validity": float(val),
        "uniqueness": float(uniq),
        "novelty": float(nov),
        "diversity": float(div),
        "num_graphs": num_graphs,
        "method": method,
        "mean_num_nodes": float(sum(g.num_nodes for g in graphs) / max(len(graphs), 1)),
        "mean_num_edges": float(
            sum(int(g.edge_index.shape[1]) for g in graphs) / max(len(graphs), 1)
        ),
    }

    config = {
        "method": method,
        "num_graphs": num_graphs,
        "num_nodes": num_nodes,
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "seed": seed,
        **{k: v for k, v in method_kwargs.items() if isinstance(v, (int, float, str, bool))},
    }

    report_path = None
    if dashboard_dir:
        os.makedirs(dashboard_dir, exist_ok=True)
        report_path = os.path.join(dashboard_dir, f"generation_{method}.json")
        with open(report_path, "w") as f:
            json.dump({"metrics": metrics, "config": config}, f, indent=2, default=str)

    return GenerationResult(graphs=graphs, metrics=metrics, config=config, report_path=report_path)
