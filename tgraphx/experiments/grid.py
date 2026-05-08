"""Multi-seed and grid-search runner."""
from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import ExperimentConfig, _validate
from .runner import Runner


__all__ = ["GridRunner", "expand_grid"]


def _walk_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``d['a']['b']['c']`` from a dotted path ``"a.b.c"``."""
    parts = dotted.split(".")
    cur = d
    for k in parts[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot descend into non-dict at {k!r}")
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value


def expand_grid(base_config: Dict[str, Any], grid: Dict[str, Sequence]) -> List[Dict[str, Any]]:
    """Cartesian-product expansion of a base config + a grid spec.

    Args:
        base_config: A plain dict (already-loaded YAML/JSON).
        grid: ``{"training.lr": [1e-3, 5e-3], "model.num_layers": [2, 4]}``.

    Returns:
        List of new config dicts with the grid values applied.
    """
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_config)
        for k, v in zip(keys, combo):
            _walk_set(cfg, k, v)
        out.append(cfg)
    return out


# ── Grid runner ──────────────────────────────────────────────────────────────


class GridRunner:
    """Run an experiment across a cartesian grid + multiple seeds.

    The grid spec lives **alongside** the base config in the same YAML/JSON file:

    .. code-block:: yaml

        seed: 0
        run_name: sweep_lr
        dataset: { name: synthetic:patch_graph }
        model: { task: graph_classification, layer: conv, ... }
        training: { epochs: 5, lr: 0.001 }
        grid:
          training.lr: [0.001, 0.005]
          training.epochs: [3, 5]
        seeds: [0, 1, 2]
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        grid: Optional[Dict[str, Sequence]] = None,
        seeds: Optional[Sequence[int]] = None,
        out_dir: Optional[str | Path] = None,
    ) -> None:
        self.base_config = base_config
        self.grid = grid or {}
        self.seeds = list(seeds) if seeds is not None else [int(base_config.get("seed", 0))]
        self.out_dir = Path(out_dir) if out_dir else Path("runs") / base_config.get("run_name", "grid")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GridRunner":
        """Split a combined config dict into base + grid."""
        grid = raw.pop("grid", {}) or {}
        seeds = raw.pop("seeds", None)
        return cls(base_config=raw, grid=grid, seeds=seeds)

    def run(self) -> List[Dict[str, Any]]:
        """Run every (config, seed) combo; return per-run summaries."""
        configs = expand_grid(self.base_config, self.grid) if self.grid else [self.base_config]
        results: List[Dict[str, Any]] = []
        for i, cfg_dict in enumerate(configs):
            for seed in self.seeds:
                this = copy.deepcopy(cfg_dict)
                this["seed"] = int(seed)
                run_name = f"{this.get('run_name', 'run')}_cfg{i}_seed{seed}"
                this["run_name"] = run_name
                run_dir = self.out_dir / run_name
                this["run_dir"] = str(run_dir)
                cfg = _validate(this)
                runner = Runner(cfg, run_dir=run_dir)
                history = runner.fit()
                summary = {
                    "config_index": i,
                    "seed": int(seed),
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "epochs": len(history),
                    "final_train_loss": history[-1].get("train_loss") if history else None,
                }
                results.append(summary)
        # Persist a top-level summary.
        (self.out_dir / "grid_summary.json").write_text(
            json.dumps({
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_runs": len(results),
                "results": results,
            }, indent=2),
        )
        return results
