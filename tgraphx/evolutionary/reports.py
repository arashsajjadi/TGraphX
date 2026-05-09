"""Report writers for evolutionary graph optimization.

Stability: Experimental (v0.7.0+).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from .algorithms import EvolutionResult
from .config import EvolutionConfig

__all__ = ["write_evolution_report"]


def _atomic_write(path: str, payload: Dict[str, Any]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(p))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(p)


def write_evolution_report(
    path: str,
    result: EvolutionResult,
    config: EvolutionConfig,
) -> str:
    """Write evolution result to a JSON artifact.

    Includes:
        - best_fitness, mean_fitness, final_diversity
        - pareto_front (IDs + fitness vectors only, no raw genomes or tensors)
        - fitness_history, diversity_history (capped at 500 rows)
        - config dict

    Args:
        path: Output file path.
        result: EvolutionResult.
        config: EvolutionConfig.

    Returns:
        Absolute path to written file.
    """
    _MAX_ROWS = 500

    def _cap(lst: list) -> list:
        if len(lst) > _MAX_ROWS:
            return lst[:_MAX_ROWS] + [f"... ({len(lst) - _MAX_ROWS} more)"]
        return lst

    pareto_data = None
    if result.pareto_front is not None:
        pareto_data = {
            "size": len(result.pareto_front),
            "fitness_vectors": _cap(result.pareto_front.fitness_vectors),
            "crowding_distances": _cap([
                float("inf") if d == float("inf") else d
                for d in result.pareto_front.crowding_distances
            ]),
        }

    best_genome_stats = None
    if result.best_genome is not None:
        g = result.best_genome
        best_genome_stats = {
            "num_nodes": g.num_nodes,
            "num_edges": g.num_edges,
            "has_node_features": g.node_features is not None,
            "node_features_shape": list(g.node_features.shape) if g.node_features is not None else None,
        }

    mean_fitness = (
        sum(result.fitness_history) / len(result.fitness_history)
        if result.fitness_history else 0.0
    )
    final_diversity = result.diversity_history[-1] if result.diversity_history else 0.0

    payload: Dict[str, Any] = {
        "report_type": "evolution",
        "best_fitness": result.best_fitness,
        "mean_fitness": mean_fitness,
        "final_diversity": final_diversity,
        "best_genome_stats": best_genome_stats,
        "fitness_history": _cap(result.fitness_history),
        "diversity_history": _cap(result.diversity_history),
        "pareto_front": pareto_data,
        "config": config.to_dict(),
    }

    return _atomic_write(path, payload)
