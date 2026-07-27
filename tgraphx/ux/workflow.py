"""Unified workflow dispatcher for common TGraphX tasks.

This is intentionally small — it does NOT try to be AutoML. It dispatches
to existing stable APIs and gives helpful errors for unsupported tasks.
"""
from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch


_TASK_ALIASES: Dict[str, str] = {
    "node_classification": "node_classification",
    "node-classification": "node_classification",
    "node_cls": "node_classification",
    "tensor_node_classification": "node_classification",
    "graph_classification": "graph_classification",
    "graph-classification": "graph_classification",
    "graph_cls": "graph_classification",
    "kg_link_prediction": "kg_link_prediction",
    "kg-link-prediction": "kg_link_prediction",
    "kg_completion": "kg_link_prediction",
    "link_prediction": "kg_link_prediction",
    "graph_mining": "graph_mining",
    "mining": "graph_mining",
    "graph_generation": "graph_generation",
    "generation": "graph_generation",
}


@dataclass
class WorkflowResult:
    """Result of a unified-workflow call."""
    task: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    runtime_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_task(task: str) -> str:
    if task in _TASK_ALIASES:
        return _TASK_ALIASES[task]
    canonical = sorted(set(_TASK_ALIASES.values()))
    suggestion = difflib.get_close_matches(task, list(_TASK_ALIASES.keys()) + canonical, n=1)
    hint = f" Closest match: {suggestion[0]!r}." if suggestion else ""
    raise ValueError(
        f"Unknown task {task!r}. Available: {canonical}.{hint}"
    )


def list_workflow_tasks() -> List[str]:
    """Return the canonical list of workflow task names."""
    return sorted(set(_TASK_ALIASES.values()))


def _resolve_device(device: Optional[str]) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def workflow(
    task: str,
    *,
    dataset: Optional[Any] = None,
    model: Optional[Any] = None,
    fast_mode: bool = True,
    seed: int = 42,
    device: Optional[str] = "auto",
    epochs: Optional[int] = None,
    out_dir: Optional[Union[str, Path]] = None,
    **task_kwargs: Any,
) -> WorkflowResult:
    """Run a small TGraphX workflow end-to-end.

    Supported tasks:
      - ``"node_classification"`` (synthetic tensor node classification)
      - ``"kg_link_prediction"`` (synthetic small KG with TransE)
      - ``"graph_mining"`` (summary + motif profile)

    Notes:
      This dispatcher is a high-level convenience for simple, deterministic
      smoke runs. For real research, call the underlying TGraphX APIs directly.

    Args:
        task: One of the workflow task names. Aliases supported (see
            :func:`list_workflow_tasks`).
        dataset: Optional dataset identifier or object (str name resolved via
            :func:`tgraphx.datasets.load_dataset` when supported).
        model: Optional model name or instance.
        fast_mode: Use a tiny configuration.
        seed: RNG seed.
        device: ``"auto"``, ``"cpu"``, or ``"cuda"``.
        epochs: Override the default per-task epoch count.
        out_dir: If provided, write run_metadata.json + benchmark_summary.json.

    Returns:
        :class:`WorkflowResult`.
    """
    from ..reproducibility import set_seed
    canonical = _normalize_task(task)
    set_seed(seed, deterministic=False, warn_only=True)
    dev = _resolve_device(device)
    t0 = time.time()

    if canonical == "node_classification":
        result = _wf_node_classification(seed=seed, device=dev,
                                          fast_mode=fast_mode,
                                          epochs=epochs, **task_kwargs)
    elif canonical == "kg_link_prediction":
        result = _wf_kg(seed=seed, device=dev, fast_mode=fast_mode,
                        epochs=epochs, **task_kwargs)
    elif canonical == "graph_mining":
        result = _wf_mining(seed=seed, **task_kwargs)
    else:
        raise NotImplementedError(
            f"Workflow {canonical!r} is reserved but not implemented in v1.4.0. "
            "Open an issue / use the canonical APIs directly."
        )

    elapsed = time.time() - t0
    config = {
        "task": canonical,
        "fast_mode": fast_mode,
        "seed": seed,
        "device": dev,
        "epochs": epochs,
    }
    wfr = WorkflowResult(task=canonical, metrics=result, config=config, runtime_s=elapsed)

    if out_dir is not None:
        from ..tracking import write_run_metadata, write_metrics_summary
        from .. import __version__
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        write_run_metadata(
            str(d / "run_metadata.json"),
            tgraphx_version=__version__, seed=seed, fast_mode=fast_mode,
            device=dev, task=canonical, runtime_s=round(elapsed, 3),
        )
        write_metrics_summary(
            str(d / "metrics_summary.json"),
            task=canonical, **{k: float(v) if isinstance(v, (int, float)) else str(v)
                                for k, v in result.items()},
        )
        with open(d / "benchmark_summary.json", "w") as f:
            json.dump({**config, "metrics": result,
                       "tgraphx_version": __version__,
                       "runtime_s": round(elapsed, 3)}, f, indent=2)
        wfr.artifacts = {
            "run_metadata.json": str(d / "run_metadata.json"),
            "metrics_summary.json": str(d / "metrics_summary.json"),
            "benchmark_summary.json": str(d / "benchmark_summary.json"),
        }
    return wfr


def run_workflow(*args, **kwargs) -> WorkflowResult:
    """Alias for :func:`workflow`."""
    return workflow(*args, **kwargs)


# ── Task implementations ──────────────────────────────────────────────────


def _wf_node_classification(
    *, seed: int, device: str, fast_mode: bool,
    epochs: Optional[int], **kw,
) -> Dict[str, Any]:
    """Tiny synthetic tensor node classification (smoke workflow)."""
    import torch.nn as nn
    import torch.nn.functional as F
    from ..core.graph import Graph
    from ..layers.conv_message import ConvMessagePassing
    from ..loaders import NeighborLoader
    from ..training import count_parameters

    gen = torch.Generator().manual_seed(seed)
    N = 100 if fast_mode else 500
    NC = 3
    x = torch.randn(N, 1, 14, 14, generator=gen)
    y = torch.randint(0, NC, (N,))
    src = torch.arange(N).unsqueeze(1).expand(-1, 3).reshape(-1)
    dst = torch.randint(0, N, (3 * N,), generator=gen)
    ei = torch.unique(torch.stack([src, dst], 0), dim=1)
    g = Graph(node_features=x, edge_index=ei, y=y)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.7 * N)]] = True
    vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.7 * N):int(0.85 * N)]] = True

    class TinyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = ConvMessagePassing(
                in_shape=(1, 14, 14), out_shape=(4, 7, 7), dropout_prob=0.0
            )
            self.head = nn.Linear(4 * 7 * 7, NC)
        def forward(self, x, ei):
            return self.head(F.relu(self.c(x, ei)).flatten(1))

    mdl = TinyGNN().to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=5e-3)
    n_epochs = epochs if epochs is not None else (2 if fast_mode else 5)
    loader = NeighborLoader(g, fanouts=[5, 3], batch_size=16, mask=tm,
                            shuffle=True, seed=seed)
    for _ in range(n_epochs):
        for batch in loader:
            logits = mdl(batch.node_features.to(device), batch.edge_index.to(device))
            loss = F.cross_entropy(batch.seed_logits(logits), batch.seed_y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    mdl.eval()
    vloader = NeighborLoader(g, fanouts=[5, 3], batch_size=16, mask=vm,
                              shuffle=False, seed=seed)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in vloader:
            logits = mdl(batch.node_features.to(device), batch.edge_index.to(device))
            preds = batch.seed_logits(logits).argmax(1)
            correct += (preds == batch.seed_y.to(device)).sum().item()
            total += batch.seed_y.numel()
    val_acc = correct / max(1, total)
    return {
        "val_accuracy": round(val_acc, 4),
        "num_nodes": N, "num_classes": NC,
        "params": count_parameters(mdl),
        "epochs": n_epochs,
    }


def _wf_kg(*, seed: int, device: str, fast_mode: bool,
            epochs: Optional[int], **kw) -> Dict[str, Any]:
    """Tiny synthetic KG with TransE (smoke workflow)."""
    from ..kg import KnowledgeGraph, KGTrainer, KGTrainingConfig, TransEModel
    NUM_ENTITIES = 20 if fast_mode else 50
    NUM_RELATIONS = 3
    triples = torch.zeros((NUM_ENTITIES * 4, 3), dtype=torch.long)
    for i in range(NUM_ENTITIES * 4):
        triples[i] = torch.tensor([i % NUM_ENTITIES, i % NUM_RELATIONS,
                                    (i + 1) % NUM_ENTITIES], dtype=torch.long)
    kg = KnowledgeGraph(triples, num_entities=NUM_ENTITIES, num_relations=NUM_RELATIONS)
    model = TransEModel(NUM_ENTITIES, NUM_RELATIONS, embedding_dim=8)
    n_epochs = epochs if epochs is not None else (2 if fast_mode else 5)
    config = KGTrainingConfig(num_epochs=n_epochs, batch_size=16, device=device, seed=seed)
    trainer = KGTrainer(model, config, triples)
    h = trainer.fit()
    return {"final_loss": round(float(h["final_loss"]), 4), "epochs": n_epochs}


def _wf_mining(*, seed: int, **kw) -> Dict[str, Any]:
    """Tiny graph-mining smoke."""
    from ..mining import graph_summary, motif_profile
    gen = torch.Generator().manual_seed(seed)
    N = 30
    src = torch.randint(0, N, (N * 3,), generator=gen)
    dst = torch.randint(0, N, (N * 3,), generator=gen)
    m = src != dst
    src, dst = src[m], dst[m]
    ei = torch.unique(
        torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0), dim=1
    )
    s = graph_summary(ei, num_nodes=N, directed=False)
    mp = motif_profile(ei, num_nodes=N, directed=False)
    return {"density": s["density"], "triangles": mp.get("triangles", 0),
            "num_nodes": N, "num_edges": int(ei.shape[1])}
