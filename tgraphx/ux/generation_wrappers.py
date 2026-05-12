"""User-friendly wrappers for graph generation, evolutionary optimization,
and graph RL (v1.4.1).

These are thin, safe aliases over existing stable APIs with:
- method/algorithm name normalization
- tensor-native node feature support
- helpful errors for unsupported methods
- artifact writing
- reproducibility state
"""
from __future__ import annotations

import difflib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch


# ── Method / algorithm alias maps ────────────────────────────────────────────

_GENERATION_ALIASES: Dict[str, str] = {
    "ba": "barabasi_albert", "barabasi": "barabasi_albert",
    "barabasi-albert": "barabasi_albert", "preferential_attachment": "barabasi_albert",
    "er": "erdos_renyi", "erdos": "erdos_renyi", "erdos-renyi": "erdos_renyi",
    "gnp": "erdos_renyi", "random_graph": "erdos_renyi",
    "ws": "watts_strogatz", "small_world": "watts_strogatz",
    "watts-strogatz": "watts_strogatz",
    "sbm": "stochastic_block_model", "block_model": "stochastic_block_model",
    "community": "stochastic_block_model",
    "cycle": "cycle", "path": "path", "star": "star", "complete": "complete",
    "grid": "grid", "random_geometric": "random_geometric",
    "motif_injected": "motif_injected", "anomaly_injected": "anomaly_injected",
    "temporal": "temporal", "typed": "typed",
    # Explicitly unsupported neural methods
    "vgae": None, "gae": None, "graphvae": None, "gran": None,
}

_EVO_ALIASES: Dict[str, str] = {
    "ga": "ga", "genetic": "ga", "genetic_algorithm": "ga",
    "nsga2": "nsga2", "nsga-ii": "nsga2", "nsga-2": "nsga2", "nsga_ii": "nsga2",
    "hill_climb": "hill_climbing", "hill_climbing": "hill_climbing",
    "sa": "sa", "simulated_annealing": "sa",
    "random_search": "random_search",
}

_RL_ENV_ALIASES: Dict[str, str] = {
    "maxcut": "max_cut", "max_cut": "max_cut", "graph_maxcut": "max_cut",
    "coloring": "graph_coloring", "graph_coloring": "graph_coloring",
    "navigation": "graph_navigation", "graph_navigation": "graph_navigation",
    "shortest_path": "shortest_path",
    "vertex_cover": "vertex_cover",
    "generation": "graph_generation",
    "kg_reasoning": "kg_reasoning",
}

_RL_ALGO_ALIASES: Dict[str, str] = {
    "random": "random", "greedy": "greedy",
    "reinforce": "reinforce", "REINFORCE": "reinforce",
    "a2c": "a2c", "A2C": "a2c",
    "dqn": "dqn", "DQN": "dqn",
    "double_dqn": "double_dqn",
    "dueling_dqn": "dueling_dqn",
    "ppo": "ppo", "PPO": "ppo",
}


def _normalize_generation_method(method: str) -> str:
    key = method.lower().replace("-", "_").strip()
    if key in _GENERATION_ALIASES:
        canonical = _GENERATION_ALIASES[key]
        if canonical is None:
            raise ValueError(
                f"'{method}' is not a classical graph generator in TGraphX. "
                f"VGAE / GAE are graph autoencoder (link prediction) models, "
                f"not structure generators. "
                f"To generate graphs, use one of: "
                f"barabasi_albert, erdos_renyi, watts_strogatz, stochastic_block_model, "
                f"grid, cycle, path, star, complete. "
                f"See tgx.generate_graph('ba', ...) for a quick example."
            )
        return canonical
    try:
        from tgraphx.generation import list_graph_generation_methods
        valid = list(list_graph_generation_methods())
    except Exception:
        valid = list(_GENERATION_ALIASES.values())
    valid = [v for v in valid if v is not None]
    suggestions = difflib.get_close_matches(key, valid, n=1)
    hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
    raise ValueError(f"Unknown generation method {method!r}.{hint} Valid: {valid}")


def _normalize_evo_algorithm(algorithm: str) -> str:
    key = algorithm.lower().replace("-", "_")
    if key in _EVO_ALIASES:
        return _EVO_ALIASES[key]
    valid = sorted(set(_EVO_ALIASES.values()))
    suggestions = difflib.get_close_matches(key, valid, n=1)
    hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
    raise ValueError(f"Unknown optimization algorithm {algorithm!r}.{hint} Valid: {valid}")


def _normalize_rl_env(env_name: str) -> str:
    key = env_name.lower().replace("-", "_")
    if key in _RL_ENV_ALIASES:
        return _RL_ENV_ALIASES[key]
    valid = sorted(set(_RL_ENV_ALIASES.values()))
    suggestions = difflib.get_close_matches(key, valid, n=1)
    hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
    raise ValueError(
        f"Unknown graph RL environment {env_name!r}.{hint} "
        f"Valid environments: {valid}"
    )


def _normalize_rl_algorithm(algorithm: str) -> str:
    key = algorithm.lower().replace("-", "_")
    if key in _RL_ALGO_ALIASES:
        return _RL_ALGO_ALIASES[key]
    valid = sorted(set(_RL_ALGO_ALIASES.values()))
    suggestions = difflib.get_close_matches(key, valid, n=1)
    hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
    raise ValueError(f"Unknown RL algorithm {algorithm!r}.{hint} Valid: {valid}")


# ── generate_graph ───────────────────────────────────────────────────────────

def generate_graph(
    method: str = "barabasi_albert",
    *,
    num_nodes: int = 50,
    node_shape: Optional[tuple] = None,
    node_feature_dim: int = 0,
    seed: int = 42,
    out_dir: Optional[Union[str, Path]] = None,
    return_result: bool = False,
    **method_kwargs: Any,
) -> Any:
    """Generate a single graph with a classical structure model.

    Args:
        method: Generation method. Aliases accepted: ``"ba"``, ``"er"``,
            ``"ws"``, ``"sbm"``, ``"grid"``, ``"cycle"``, ``"path"``,
            ``"star"``, ``"complete"``.
        num_nodes: Number of nodes.
        node_shape: If given, generate tensor-valued node features of this shape
            (e.g. ``(3, 8, 8)`` for RGB patch nodes). Preserves tensor-native semantics.
        node_feature_dim: Number of scalar features per node if ``node_shape`` not given.
        seed: RNG seed.
        out_dir: Optional artifact directory.
        return_result: If True, return the full :class:`GenerationResult`
            instead of just the Graph.
        **method_kwargs: Method-specific parameters forwarded to the generator.

    Returns:
        A :class:`tgraphx.Graph` (or :class:`GenerationResult` if ``return_result=True``).
    """
    canonical = _normalize_generation_method(method)

    from tgraphx.generation import run_graph_generation
    from tgraphx.mining import graph_summary

    # If tensor node features requested, we generate the graph first then attach features.
    feat_dim = node_feature_dim
    if node_shape is not None:
        feat_dim = 1  # placeholder; we'll overwrite below

    gen_result = run_graph_generation(
        method=canonical,
        num_graphs=1,
        num_nodes=num_nodes,
        node_feature_dim=feat_dim if node_shape is None else 0,
        seed=seed,
        dashboard_dir=str(out_dir) if out_dir else None,
        **method_kwargs,
    )

    gen_graph = gen_result.graphs[0] if gen_result.graphs else None
    if gen_graph is None:
        raise RuntimeError(f"generate_graph('{method}'): no graph was produced")

    # Convert GeneratedGraph → tgraphx.Graph
    from tgraphx.core.graph import Graph as TGXGraph
    n = gen_graph.num_nodes
    ei = gen_graph.edge_index
    nf = gen_graph.node_features

    # Attach tensor-valued node features if requested
    if node_shape is not None:
        gen = torch.Generator().manual_seed(seed)
        nf = torch.randn(n, *node_shape, generator=gen)
    elif nf is None or nf.numel() == 0:
        nf = torch.zeros(n, 1)

    graph = TGXGraph(node_features=nf, edge_index=ei,
                     edge_weight=getattr(gen_graph, "edge_weight", None),
                     edge_features=getattr(gen_graph, "edge_features", None))

    # Write artifacts
    if out_dir is not None:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        import tgraphx
        config_data = {
            "method": canonical, "num_nodes": num_nodes,
            "node_shape": list(node_shape) if node_shape else None,
            "seed": seed, "method_kwargs": {k: str(v) for k, v in method_kwargs.items()},
            "tgraphx_version": tgraphx.__version__,
        }
        with open(d / "generation_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        s = graph_summary(graph.edge_index, num_nodes=graph.num_nodes, directed=False)
        with open(d / "graph_summary.json", "w") as f:
            json.dump({k: v for k, v in s.items() if isinstance(v, (int, float, bool, str, list))}, f, indent=2)

        gen_metrics = {
            "num_nodes": graph.num_nodes, "num_edges": graph.num_edges,
            "density": s.get("density", 0), "method": canonical,
        }
        with open(d / "generation_metrics.json", "w") as f:
            json.dump(gen_metrics, f, indent=2)

    if return_result:
        return gen_result
    return graph


# Aliases
graph_generator = generate_graph
generate = generate_graph


# ── evaluate_generated_graphs ─────────────────────────────────────────────────

def evaluate_generated_graphs(
    graphs: Sequence[Any],
    *,
    reference_graphs: Optional[Sequence[Any]] = None,
    metrics: Optional[List[str]] = None,
    strict: bool = False,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate structural properties of a list of generated graphs.

    Args:
        graphs: Sequence of :class:`tgraphx.Graph` or edge_index tensors.
        reference_graphs: Optional reference set for novelty/diversity metrics.
        metrics: Optional list of metric names to compute. Available:
            ``"validity"``, ``"degree"``, ``"density"``, ``"clustering"``,
            ``"motifs"``, ``"diversity"``. Default: all available.
        strict: If True, raise on unsupported metrics. Otherwise warn.
        out_dir: Optional artifact directory.

    Returns:
        JSON-serializable dict with structural evaluation results.
    """
    from tgraphx.mining import graph_summary, degree_statistics, motif_profile
    import math

    if metrics is None:
        metrics = ["validity", "degree", "density", "motifs"]

    _supported = {"validity", "degree", "density", "motifs", "clustering", "diversity"}
    unsupported = set(metrics) - _supported
    if unsupported:
        msg = f"Unsupported metrics: {unsupported}. Available: {_supported}"
        if strict:
            raise ValueError(msg)
        import warnings
        warnings.warn(msg)

    report: Dict[str, Any] = {
        "num_graphs": len(graphs),
        "disclaimer": "Structural summary only. No SOTA claims.",
    }
    all_densities, all_mean_degrees, all_num_edges = [], [], []
    triangle_counts = []
    valid_count = 0

    for g in graphs:
        # Extract edge_index and num_nodes
        if hasattr(g, "edge_index") and g.edge_index is not None:
            ei = g.edge_index
            n = g.num_nodes
        elif isinstance(g, torch.Tensor):
            ei = g; n = int(ei.max().item()) + 1 if ei.numel() > 0 else 0
        else:
            continue

        if ei.numel() == 0 or n == 0:
            continue
        valid_count += 1

        s = graph_summary(ei, num_nodes=n, directed=False)
        all_densities.append(s.get("density", 0))
        all_mean_degrees.append(s.get("mean_degree", 0))
        all_num_edges.append(s.get("num_edges", 0))

        if "motifs" in metrics:
            mp = motif_profile(ei, num_nodes=n, directed=False)
            triangle_counts.append(mp.get("triangles", 0))

    def safe_mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    if "validity" in metrics:
        report["validity"] = round(valid_count / max(1, len(graphs)), 4)
    if "density" in metrics:
        report["mean_density"] = safe_mean(all_densities)
        report["max_density"] = round(max(all_densities), 4) if all_densities else 0.0
    if "degree" in metrics:
        report["mean_degree"] = safe_mean(all_mean_degrees)
    if "motifs" in metrics and triangle_counts:
        report["mean_triangles"] = safe_mean(triangle_counts)

    report["mean_edges"] = safe_mean(all_num_edges)

    if out_dir is not None:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        with open(d / "generation_eval_summary.json", "w") as f:
            json.dump(report, f, indent=2)

    return report


# Aliases
graph_generation_report = evaluate_generated_graphs
compare_generated_graphs = evaluate_generated_graphs
generation_metrics = evaluate_generated_graphs


# ── optimize_graph ────────────────────────────────────────────────────────────

def optimize_graph(
    objective: Union[str, List[str]] = "connectivity",
    *,
    algorithm: str = "ga",
    num_nodes: int = 20,
    generations: int = 10,
    population_size: int = 10,
    seed: int = 42,
    fast_mode: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
    **algo_kwargs: Any,
) -> Any:
    """One-call evolutionary graph structure optimization.

    Args:
        objective: Fitness objective name or list of names for multi-objective
            (NSGA-II). Aliases: ``"connectivity"``, ``"sparsity"``, ``"density"``.
        algorithm: Optimizer alias: ``"ga"`` (default), ``"nsga2"``,
            ``"hill_climb"``, ``"sa"``.
        num_nodes: Graph size.
        generations: Number of generations.
        population_size: Population size.
        seed: RNG seed.
        fast_mode: Use minimal defaults.
        out_dir: Optional artifact directory.
        **algo_kwargs: Forwarded to the underlying optimizer.

    Returns:
        :class:`tgraphx.evolutionary.OptimizationResult`.
    """
    from tgraphx.evolutionary import run_evolutionary_optimization

    canonical_algo = _normalize_evo_algorithm(algorithm)

    if fast_mode:
        generations = min(generations, 3)
        population_size = min(population_size, 6)

    result = run_evolutionary_optimization(
        algorithm=canonical_algo,
        objective=objective,
        population_size=population_size,
        generations=generations,
        num_nodes=num_nodes,
        seed=seed,
        dashboard_dir=str(out_dir) if out_dir else None,
        **algo_kwargs,
    )

    if out_dir is not None:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        import tgraphx
        config_data = {
            "algorithm": canonical_algo, "objective": objective if isinstance(objective, list) else [objective],
            "num_nodes": num_nodes, "generations": generations, "population_size": population_size,
            "seed": seed, "fast_mode": fast_mode, "tgraphx_version": tgraphx.__version__,
        }
        with open(d / "evolution_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        hist = getattr(result, "history", [])
        history_data = hist if isinstance(hist, list) else list(hist)
        with open(d / "evolution_history.json", "w") as f:
            def _to_jsonable(v):
                if isinstance(v, (int, float, bool, str, type(None))): return v
                if isinstance(v, (list, tuple)): return [_to_jsonable(x) for x in v]
                if isinstance(v, dict): return {str(kk): _to_jsonable(vv) for kk, vv in v.items()}
                if hasattr(v, "item"): return v.item()
                return str(v)
            json.dump(_to_jsonable(history_data), f, indent=2)

        best_fitness = getattr(result, "best_fitness", None)
        if hasattr(best_fitness, "item"):
            best_fitness = best_fitness.item()
        with open(d / "benchmark_summary.json", "w") as f:
            json.dump({
                "algorithm": canonical_algo, "objective": str(objective),
                "best_fitness": str(best_fitness), "generations": generations,
                "tgraphx_version": tgraphx.__version__,
                "disclaimer": "Research utility for graph structure optimization. Not a production optimizer.",
            }, f, indent=2)

    return result


# Aliases
evolve_graph = optimize_graph
graph_evolution = optimize_graph
run_evolution = optimize_graph


# ── train_graph_rl ────────────────────────────────────────────────────────────

def train_graph_rl(
    env: Union[str, Any] = "max_cut",
    *,
    algorithm: str = "random",
    episodes: int = 10,
    max_steps: int = 50,
    seed: int = 42,
    device: Optional[str] = "auto",
    fast_mode: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
    **env_kwargs: Any,
) -> Any:
    """One-call graph reinforcement learning.

    Args:
        env: Environment name (alias-resolved) or pre-built env object.
            Aliases: ``"maxcut"`` / ``"max_cut"``, ``"coloring"``, ``"navigation"``,
            ``"vertex_cover"``, ``"kg_reasoning"``, ``"shortest_path"``.
        algorithm: Algorithm alias: ``"random"``, ``"dqn"``, ``"ppo"``, etc.
        episodes: Number of training episodes.
        max_steps: Max steps per episode.
        seed: RNG seed.
        device: ``"auto"`` selects CUDA if available.
        fast_mode: Use tiny defaults.
        out_dir: Optional artifact directory.
        **env_kwargs: Forwarded to make_graph_env.

    Returns:
        :class:`tgraphx.rl.RLResult`.
    """
    from tgraphx.rl import run_graph_rl

    if fast_mode:
        episodes = min(episodes, 5)
        max_steps = min(max_steps, 20)

    if isinstance(env, str):
        canonical_env = _normalize_rl_env(env)
    else:
        canonical_env = env

    canonical_algo = _normalize_rl_algorithm(algorithm)

    from .workflow import _resolve_device
    dev = _resolve_device(device)

    # Only pass kwargs that run_graph_rl / make_graph_env know about
    safe_kwargs = {k: v for k, v in env_kwargs.items()
                   if k not in ("episodes", "algorithm", "seed", "device",
                                "fast_mode", "out_dir", "max_steps",
                                "edge_density", "num_nodes")}
    # max_steps → pass as hidden_dim-like param only if run_graph_rl accepts it
    try:
        result = run_graph_rl(
            env=canonical_env,
            algorithm=canonical_algo,
            episodes=episodes,
            seed=seed,
            device=dev,
            dashboard_dir=str(out_dir) if out_dir else None,
            **safe_kwargs,
        )
    except TypeError:
        # If extra kwargs cause issues, retry without them
        result = run_graph_rl(
            env=canonical_env,
            algorithm=canonical_algo,
            episodes=episodes,
            seed=seed,
            device=dev,
            dashboard_dir=str(out_dir) if out_dir else None,
        )

    if out_dir is not None:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        import tgraphx
        config_data = {
            "env": str(canonical_env), "algorithm": canonical_algo,
            "episodes": episodes, "max_steps": max_steps,
            "seed": seed, "device": dev, "fast_mode": fast_mode,
            "tgraphx_version": tgraphx.__version__,
        }
        with open(d / "rl_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        final_reward = getattr(result, "final_reward", None)
        if hasattr(final_reward, "item"):
            final_reward = final_reward.item()
        with open(d / "rl_metrics_summary.json", "w") as f:
            json.dump({
                "final_reward": float(final_reward) if final_reward is not None else None,
                "algorithm": canonical_algo, "env": str(canonical_env),
                "episodes": episodes, "seed": seed,
                "disclaimer": "Graph RL results are stochastic. Same seed on CPU is approximately reproducible.",
            }, f, indent=2)

    return result


# Aliases
graph_rl = train_graph_rl
run_rl = train_graph_rl


# ── audit extensions ──────────────────────────────────────────────────────────

_WORKFLOW_REQUIRED: Dict[str, List[str]] = {
    "generation": ["generation_config.json", "graph_summary.json", "generation_metrics.json"],
    "evolution": ["evolution_config.json", "evolution_history.json", "best_graph_summary.json"],
    "graph_rl": ["rl_config.json", "rl_training_history.json", "rl_metrics_summary.json"],
}

_WORKFLOW_RECOMMENDED: Dict[str, List[str]] = {
    "generation": ["generation_eval_summary.json", "benchmark_summary.json"],
    "evolution": ["pareto_front.json", "benchmark_summary.json"],
    "graph_rl": ["env_summary.json", "benchmark_summary.json"],
}


def audit_generation_run(path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Audit a graph generation artifact directory (v1.4.1)."""
    from .dashboard_audit import audit_run_dir
    return audit_run_dir(path, workflow="generation", **kwargs)


def audit_evolution_run(path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Audit an evolutionary optimization artifact directory."""
    from .dashboard_audit import audit_run_dir
    return audit_run_dir(path, workflow="evolution", **kwargs)


def audit_rl_run(path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Audit a graph RL artifact directory."""
    from .dashboard_audit import audit_run_dir
    return audit_run_dir(path, workflow="graph_rl", **kwargs)
