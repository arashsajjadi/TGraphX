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

    Args:
        model: A :class:`KGScoringModel` implementing ``score_triples``.
        config: :class:`KGTrainingConfig`.
        train_triples: ``LongTensor[T, 3]``.
        sampler: Negative sampler.  When None, uses
            :class:`UniformNegativeSampler`.
        evaluator: Optional :class:`KGEvaluator` for validation.
        on_epoch_end: Optional callback ``(epoch, loss, metrics) -> None``.

    Stability: Experimental.
    """

    def __init__(
        self,
        model: nn.Module,
        config: KGTrainingConfig,
        train_triples: torch.Tensor,
        sampler: Optional[_BaseNegativeSampler] = None,
        evaluator: Optional[KGEvaluator] = None,
        on_epoch_end: Optional[Callable] = None,
    ) -> None:
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
