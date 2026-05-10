"""Lightweight KG hyperparameter search (grid or random).

No heavy dependencies.  Works on any :class:`KnowledgeGraph` with any
model that implements ``score_triples``.

Usage::

    from tgraphx.kg import KnowledgeGraph, run_kg_hpo

    kg = KnowledgeGraph.from_hrt(heads, relations, tails,
                                  num_entities=50, num_relations=4)

    result = run_kg_hpo(
        kg,
        model_names=["TransE", "DistMult", "SimplE"],
        search_space={
            "embedding_dim": [16, 32],
            "lr":            [1e-2, 1e-3],
        },
        metric="mrr",
        strategy="grid",
        max_trials=8,
        epochs=5,
        seed=42,
    )

    print(result.best_config)
    print(result.best_metrics)
    result.summary()

Stability: Beta (v1.3).
"""
from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = [
    "run_kg_hpo",
    "KGSearchResult",
    "KGTrialResult",
]

_AVAILABLE_METRICS = ("mrr", "mr", "hits@1", "hits@3", "hits@10")


# ── Result objects ────────────────────────────────────────────────────────────


@dataclass
class KGTrialResult:
    """Result from a single HPO trial."""
    trial_index: int
    model_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    elapsed_s: float
    status: str = "ok"  # "ok" or "failed"
    error: Optional[str] = None


@dataclass
class KGSearchResult:
    """Aggregated result from run_kg_hpo.

    Attributes:
        trials: All trial results in order.
        best_config: Config of the best trial (by ``metric``).
        best_metrics: Metrics of the best trial.
        best_model_name: Model class name of the best trial.
        best_model: The trained model from the best trial.
        config: HPO configuration used.
        artifacts: Dict of written artifact paths.
    """

    trials: List[KGTrialResult]
    best_config: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_model_name: str
    best_model: Optional[nn.Module]
    config: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Print and return a human-readable summary."""
        lines = [
            "=" * 55,
            "TGraphX KG HPO — Search Summary",
            "=" * 55,
            f"Trials run:    {len(self.trials)}",
            f"Best model:    {self.best_model_name}",
            "Best config:",
        ]
        for k, v in self.best_config.items():
            lines.append(f"  {k}: {v}")
        lines.append("Best metrics:")
        for k, v in self.best_metrics.items():
            v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"  {k}: {v_str}")
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes non-serialisable objects)."""
        return {
            "trials": [
                {
                    "trial_index": t.trial_index,
                    "model_name": t.model_name,
                    "config": t.config,
                    "metrics": t.metrics,
                    "elapsed_s": t.elapsed_s,
                    "status": t.status,
                    "error": t.error,
                }
                for t in self.trials
            ],
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "best_model_name": self.best_model_name,
            "config": self.config,
            "artifacts": self.artifacts,
        }

    def write_dashboard_artifacts(self, run_dir: Union[str, Path]) -> Dict[str, str]:
        """Write dashboard-compatible JSON and CSV artifacts.

        Args:
            run_dir: Directory to write artifacts.

        Returns:
            Dict of artifact name → absolute path.
        """
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        # HPO results JSON.
        results_path = run_path / "kg_hpo_results.json"
        results_path.write_text(json.dumps(self.to_dict(), indent=2))

        # Best metrics JSON (dashboard-readable as benchmark summary).
        summary_path = run_path / "metrics_summary.json"
        summary_path.write_text(json.dumps({
            "best_model": self.best_model_name,
            **self.best_metrics,
            "trials": len(self.trials),
        }, indent=2))

        artifacts = {
            "kg_hpo_results.json": str(results_path),
            "metrics_summary.json": str(summary_path),
        }
        self.artifacts.update(artifacts)
        return artifacts


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_model(model_name: str, num_entities: int, num_relations: int, config: Dict) -> nn.Module:
    """Instantiate a KG model by name with the given config."""
    from tgraphx.kg.models import (
        TransEModel, DistMultModel, ComplExModel, RotatEModel, RESCALModel, SimplEModel,
    )
    _MODEL_MAP = {
        "TransE":   TransEModel,
        "DistMult": DistMultModel,
        "ComplEx":  ComplExModel,
        "RotatE":   RotatEModel,
        "RESCAL":   RESCALModel,
        "SimplE":   SimplEModel,
    }
    if model_name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown KG model '{model_name}'. Available: {sorted(_MODEL_MAP)}."
        )
    cls = _MODEL_MAP[model_name]
    dim = config.get("embedding_dim", 32)
    if model_name in ("RotatE", "TransE"):
        return cls(num_entities=num_entities, num_relations=num_relations, embedding_dim=dim)
    return cls(num_entities=num_entities, num_relations=num_relations, embedding_dim=dim)


def _train_and_eval(model, triples, config, epochs, metric, seed, device) -> Dict[str, float]:
    """Train model briefly and evaluate on held-out triples."""
    from tgraphx.kg.evaluation import evaluate_filtered_ranking

    torch.manual_seed(seed)
    n = triples.size(0)
    # Simple 80/20 train/test split.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = max(1, int(0.8 * n))
    train = triples[perm[:n_train]].to(device)
    test = triples[perm[n_train:]].to(device)
    if test.size(0) == 0:
        test = train[:1]

    model = model.to(device)
    lr = config.get("lr", 1e-2)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    margin = config.get("margin", 1.0)

    for _ in range(epochs):
        neg = train.clone()
        neg[:, 2] = torch.randint(0, model.entity_head.num_embeddings
                                  if hasattr(model, "entity_head") else
                                  model.entity_emb.num_embeddings,
                                  (n_train,), device=device)
        pos_score = model.score_triples(train)
        neg_score = model.score_triples(neg)
        loss = (float(margin) + neg_score - pos_score).clamp(min=0).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate.
    all_pos = set(map(tuple, triples.tolist()))
    Ne = (model.entity_head.num_embeddings if hasattr(model, "entity_head")
          else model.entity_emb.num_embeddings)
    res = evaluate_filtered_ranking(
        model, test, all_pos, num_entities=Ne,
        filtered=True, hits_at=(1, 3, 10),
    )
    return {
        "mrr":      float(res.filt_mrr),
        "mr":       float(res.filt_mr),
        "hits@1":   float(res.filt_hits.get(1, 0.0)),
        "hits@3":   float(res.filt_hits.get(3, 0.0)),
        "hits@10":  float(res.filt_hits.get(10, 0.0)),
    }


# ── Public API ────────────────────────────────────────────────────────────────


def run_kg_hpo(
    kg,
    model_names: Sequence[str] = ("TransE", "DistMult"),
    search_space: Optional[Dict[str, Sequence]] = None,
    metric: str = "mrr",
    strategy: str = "grid",
    max_trials: Optional[int] = None,
    epochs: int = 5,
    seed: int = 42,
    device: str = "auto",
    dashboard_dir: Optional[Union[str, Path]] = None,
) -> KGSearchResult:
    """Lightweight KG hyperparameter search (grid or random strategy).

    Trains each (model, config) combination briefly and ranks by ``metric``.
    No heavy dependency; no hidden network download; deterministic.

    Args:
        kg: :class:`~tgraphx.kg.KnowledgeGraph` instance.
        model_names: Sequence of model name strings.  Available:
            ``"TransE"``, ``"DistMult"``, ``"ComplEx"``, ``"RotatE"``,
            ``"RESCAL"``, ``"SimplE"``.
        search_space: Dict mapping hyperparameter name to a list of candidate
            values.  Supported keys: ``"embedding_dim"``, ``"lr"``,
            ``"margin"``.
        metric: Metric to optimise.  One of ``"mrr"``, ``"mr"``,
            ``"hits@1"``, ``"hits@3"``, ``"hits@10"``.
        strategy: ``"grid"`` (exhaustive cartesian product) or ``"random"``
            (random permutation of grid up to ``max_trials``).
        max_trials: Cap on number of trials.  ``None`` = run all.
        epochs: Training epochs per trial.
        seed: Global random seed.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        dashboard_dir: If set, writes dashboard-compatible artifacts here.

    Returns:
        :class:`KGSearchResult` with best config, all trial results, and
        optional artifacts.

    Raises:
        ValueError: Unknown model name, metric, or strategy.
    """
    if metric not in _AVAILABLE_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {list(_AVAILABLE_METRICS)}."
        )
    if strategy not in ("grid", "random"):
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'grid' or 'random'.")

    from tgraphx.kg import list_kg_models
    known_models = list(list_kg_models().keys())
    for m in model_names:
        if m not in known_models:
            raise ValueError(
                f"Unknown KG model '{m}'. Available: {known_models}."
            )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if search_space is None:
        search_space = {"embedding_dim": [32], "lr": [1e-2]}

    # Build all combos.
    keys = list(search_space.keys())
    values = [list(search_space[k]) for k in keys]
    all_combos: List[Dict[str, Any]] = [
        dict(zip(keys, combo)) for combo in itertools.product(*values)
    ]

    # Cross with model names.
    trials_plan: List[Dict[str, Any]] = [
        {**combo, "_model": m}
        for combo in all_combos
        for m in model_names
    ]

    if strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(trials_plan)

    if max_trials is not None:
        trials_plan = trials_plan[:max_trials]

    triples = kg.triples  # [N_t, 3]
    Ne = kg.num_entities
    Nr = kg.num_relations

    # Score direction: higher is better except for "mr".
    higher_is_better = metric != "mr"

    trial_results: List[KGTrialResult] = []
    best_score: Optional[float] = None
    best_trial: Optional[KGTrialResult] = None
    best_model: Optional[nn.Module] = None

    for i, plan in enumerate(trials_plan):
        model_name = plan.pop("_model")
        config = {k: v for k, v in plan.items()}
        t0 = time.perf_counter()
        try:
            model = _build_model(model_name, Ne, Nr, config)
            metrics = _train_and_eval(model, triples, config, epochs, metric, seed + i, device)
            elapsed = time.perf_counter() - t0
            tr = KGTrialResult(
                trial_index=i, model_name=model_name,
                config=config, metrics=metrics,
                elapsed_s=round(elapsed, 4), status="ok",
            )
            score = metrics[metric]
            if best_score is None or (higher_is_better and score > best_score) or (
                    not higher_is_better and score < best_score):
                best_score = score
                best_trial = tr
                best_model = model
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            tr = KGTrialResult(
                trial_index=i, model_name=model_name,
                config=config, metrics={}, elapsed_s=round(elapsed, 4),
                status="failed", error=str(exc),
            )
        trial_results.append(tr)

    if best_trial is None:
        raise RuntimeError("All HPO trials failed. Check model configuration and data.")

    hpo_config = {
        "model_names": list(model_names),
        "search_space": {k: list(v) for k, v in search_space.items()},
        "metric": metric,
        "strategy": strategy,
        "max_trials": max_trials,
        "epochs": epochs,
        "seed": seed,
        "device": device,
        "num_trials_run": len(trial_results),
        "num_trials_ok": sum(1 for t in trial_results if t.status == "ok"),
    }

    result = KGSearchResult(
        trials=trial_results,
        best_config={**best_trial.config, "_model": best_trial.model_name},
        best_metrics=best_trial.metrics,
        best_model_name=best_trial.model_name,
        best_model=best_model,
        config=hpo_config,
    )

    if dashboard_dir is not None:
        result.write_dashboard_artifacts(dashboard_dir)

    return result
