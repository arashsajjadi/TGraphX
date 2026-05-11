"""KG training pipeline.

KGTrainer runs training and optional validation evaluation.  It is
designed to be:
  - Reproducible with a seed.
  - Dashboard-compatible via loss history and validation metrics.
  - Honest about what is and is not validated.

Stability: Experimental.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .sampling import UniformNegativeSampler, _BaseNegativeSampler
from .losses import MarginRankingLoss, BCEKGLoss, SoftplusKGLoss
from .evaluation import KGEvaluator

__all__ = ["KGTrainer", "KGTrainingConfig"]


@dataclass
class KGTrainingConfig:
    """Configuration for KGTrainer.

    Attributes:
        num_epochs: Training epochs.
        batch_size: Positive triples per batch.
        num_negatives: Negatives per positive.
        loss_type: One of ``"margin"``, ``"bce"``, ``"softplus"``.
        lr: Learning rate.
        weight_decay: L2 weight decay (AdamW).
        margin: Margin for margin ranking loss.
        grad_clip_norm: Max gradient norm (None = no clipping).
        valid_every: Evaluate on validation set every N epochs (0 = never).
        seed: RNG seed.
        device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
    """

    num_epochs: int = 100
    batch_size: int = 256
    num_negatives: int = 1
    loss_type: str = "softplus"
    lr: float = 1e-3
    weight_decay: float = 0.0
    margin: float = 1.0
    grad_clip_norm: Optional[float] = None
    valid_every: int = 10
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.loss_type not in ("margin", "bce", "softplus"):
            raise ValueError(f"loss_type must be 'margin'/'bce'/'softplus'; got {self.loss_type!r}")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")


class KGTrainer:
    """Reproducible KG embedding trainer.

    Canonical form (v0.6+):
        ``KGTrainer(model, config: KGTrainingConfig, train_triples: Tensor[T, 3])``

    LLM-friendly form (v1.3.6+):
        ``KGTrainer(model, kg_or_triples, lr=..., num_epochs=..., batch_size=..., ...)``
        where ``kg_or_triples`` may be a ``KnowledgeGraph`` (its ``.triples`` is used)
        or a ``LongTensor[T, 3]``. Extra kwargs are forwarded to ``KGTrainingConfig``.

    Args:
        model: A :class:`KGScoringModel` implementing ``score_triples``.
        config: :class:`KGTrainingConfig` **or** a :class:`KnowledgeGraph`/tensor
            (LLM-friendly form). When a KG/tensor is passed here, a default
            :class:`KGTrainingConfig` is built from ``**kwargs``.
        train_triples: ``LongTensor[T, 3]`` (canonical form). Omit when using
            the LLM-friendly form (the KG/tensor is taken from ``config``).
        sampler: Negative sampler.  When None, uses
            :class:`UniformNegativeSampler`.
        evaluator: Optional :class:`KGEvaluator` for validation.
        on_epoch_end: Optional callback ``(epoch, loss, metrics) -> None``.
        **kwargs: When using the LLM-friendly form, forwarded to
            :class:`KGTrainingConfig` (e.g. ``lr``, ``num_epochs``, ``batch_size``).

    Methods:
        - ``train()`` — full canonical training loop (returns history dict).
        - ``fit(epochs=None, batch_size=None)`` — LLM-friendly alias; optionally
          overrides config fields then runs ``train()``.
        - ``evaluate(triples=None)`` — returns evaluator metrics dict, or a
          simple final-loss dict when no evaluator is configured.

    Stability: Experimental.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[KGTrainingConfig, "KnowledgeGraph", torch.Tensor, None] = None,
        train_triples: Optional[torch.Tensor] = None,
        sampler: Optional[_BaseNegativeSampler] = None,
        evaluator: Optional[KGEvaluator] = None,
        on_epoch_end: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        # LLM-friendly form: detect when ``config`` is actually a KG/tensor.
        if not isinstance(config, KGTrainingConfig):
            kg_or_triples = config
            if hasattr(kg_or_triples, "triples"):
                triples_from_kg = kg_or_triples.triples
            elif isinstance(kg_or_triples, torch.Tensor):
                triples_from_kg = kg_or_triples
            elif kg_or_triples is None and train_triples is not None:
                triples_from_kg = None
            else:
                raise TypeError(
                    "KGTrainer expects `config` to be a KGTrainingConfig, a "
                    "KnowledgeGraph, or a LongTensor[T, 3] of triples. "
                    f"Got {type(config).__name__!r}."
                )
            if triples_from_kg is not None:
                if train_triples is None:
                    train_triples = triples_from_kg
                else:
                    raise TypeError(
                        "KGTrainer received both a KG/tensor in `config` and a "
                        "separate `train_triples`; pass only one."
                    )
            # Build a default KGTrainingConfig from kwargs.
            config = KGTrainingConfig(**kwargs)
        elif kwargs:
            raise TypeError(
                f"KGTrainer received both a KGTrainingConfig and extra kwargs "
                f"{list(kwargs)}; pass config fields either through KGTrainingConfig "
                f"or through kwargs, not both."
            )

        if train_triples is None:
            raise TypeError(
                "KGTrainer requires `train_triples` (canonical form) or a "
                "KnowledgeGraph/tensor in `config` (LLM-friendly form)."
            )

        self.model = model
        self.config = config
        self.train_triples = train_triples
        self.evaluator = evaluator
        self.on_epoch_end = on_epoch_end
        # Device.
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        # Sampler.
        N_e = getattr(model, "entity_emb", None)
        num_e = N_e.num_embeddings if N_e is not None else 1
        self.sampler = sampler or UniformNegativeSampler(num_e, config.num_negatives)
        # Loss.
        self.loss_fn: nn.Module
        if config.loss_type == "margin":
            self.loss_fn = MarginRankingLoss(margin=config.margin)
        elif config.loss_type == "bce":
            self.loss_fn = BCEKGLoss()
        else:
            self.loss_fn = SoftplusKGLoss()
        # RNG.
        self._gen = torch.Generator()
        self._gen.manual_seed(int(config.seed))
        # History.
        self.loss_history: List[float] = []
        self.valid_history: List[Dict[str, Any]] = []
        self._epoch: int = 0

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with ``loss_history``, ``valid_history``,
            ``final_loss``, and (if validating) ``best_valid_mrr``.
        """
        cfg = self.config
        dev = self.device
        model = self.model.to(dev)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        triples = self.train_triples.to(dev)
        T = triples.size(0)
        torch.manual_seed(int(cfg.seed))
        t_start = time.perf_counter()

        for epoch in range(1, cfg.num_epochs + 1):
            self._epoch = epoch
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            # Shuffle.
            perm = torch.randperm(T, generator=self._gen, device=dev)
            shuffled = triples[perm]

            for start in range(0, T, cfg.batch_size):
                end = min(start + cfg.batch_size, T)
                pos_batch = shuffled[start:end]  # [B, 3]
                # Generate negatives on CPU then move.
                neg_raw = self.sampler.sample(pos_batch.cpu(), generator=self._gen)
                neg_batch = neg_raw.view(-1, 3).to(dev)  # [B*K, 3]
                K = cfg.num_negatives
                B = pos_batch.size(0)
                # Score.
                optimizer.zero_grad()
                pos_scores = model.score_triples(pos_batch)
                neg_scores = model.score_triples(neg_batch)
                loss = self.loss_fn(pos_scores, neg_scores)
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()
                epoch_loss += float(loss.detach().item())
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.loss_history.append(avg_loss)

            # Validation.
            val_metrics: Dict[str, Any] = {}
            if self.evaluator is not None and cfg.valid_every > 0 and epoch % cfg.valid_every == 0:
                result = self.evaluator.evaluate(
                    model, triples=self.evaluator.valid_triples, device=dev
                )
                val_metrics = result.to_dict()
                self.valid_history.append({"epoch": epoch, **val_metrics})

            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_loss, val_metrics)

        runtime = time.perf_counter() - t_start
        best_mrr = max(
            (v["filtered"]["combined"]["MRR"] for v in self.valid_history), default=None
        )
        return {
            "loss_history": self.loss_history,
            "valid_history": self.valid_history,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "best_valid_mrr": best_mrr,
            "runtime_s": round(runtime, 3),
            "num_epochs": cfg.num_epochs,
            "seed": cfg.seed,
            "device": str(dev),
        }

    # ------------------------------------------------------------------ #
    # LLM-friendly aliases (v1.3.6+).                                     #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """LLM-friendly alias for :meth:`train`.

        Optionally overrides ``num_epochs`` / ``batch_size`` (and any other
        :class:`KGTrainingConfig` field via kwargs) before running the canonical
        training loop.

        Args:
            epochs: Override ``num_epochs`` if not None.
            batch_size: Override ``batch_size`` if not None.
            **kwargs: Any other KGTrainingConfig field override.

        Returns:
            Same dict as :meth:`train`.
        """
        if epochs is not None:
            self.config.num_epochs = int(epochs)
        if batch_size is not None:
            self.config.batch_size = int(batch_size)
        for k, v in kwargs.items():
            if not hasattr(self.config, k):
                raise TypeError(f"Unknown KGTrainingConfig field: {k!r}")
            setattr(self.config, k, v)
        return self.train()

    def evaluate(
        self,
        triples: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """LLM-friendly evaluation entrypoint.

        When an evaluator is configured, runs it on ``triples`` (or the
        evaluator's ``valid_triples`` by default) and returns its metric dict.
        When no evaluator is configured, returns the most-recent training
        summary (``{"final_loss": ..., "num_epochs": ..., "seed": ...}``).

        Args:
            triples: Optional override for the triples to evaluate on.

        Returns:
            Metric dict (evaluator format) or a small training-summary dict.
        """
        if self.evaluator is not None:
            eval_triples = triples if triples is not None else self.evaluator.valid_triples
            result = self.evaluator.evaluate(self.model, triples=eval_triples, device=self.device)
            return result.to_dict()
        return {
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "num_epochs": self.config.num_epochs,
            "seed": self.config.seed,
            "note": "No evaluator configured; pass `evaluator=KGEvaluator(...)` for ranking metrics.",
        }
