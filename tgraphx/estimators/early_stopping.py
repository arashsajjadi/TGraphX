"""Early-stopping helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["EarlyStopping"]


@dataclass
class EarlyStopping:
    """Track validation metric; signal when patience runs out.

    Args:
        patience: Number of non-improving steps before stopping.
        mode: ``"max"`` (default; higher is better) or ``"min"``.
        min_delta: Minimum delta to count as improvement.

    Usage::

        es = EarlyStopping(patience=5, mode="max")
        for epoch in ...:
            score = evaluate(...)
            if es.step(score):
                break

    Stability: Beta.
    """

    patience: int = 10
    mode: str = "max"
    min_delta: float = 0.0
    counter: int = 0
    best_score: Optional[float] = None
    best_step: int = 0
    _step: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {self.mode!r}")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def _improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float) -> bool:
        """Update with ``score``; return ``True`` when training should stop."""
        self._step += 1
        if self._improved(score):
            self.best_score = float(score)
            self.best_step = self._step
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.best_step = 0
        self._step = 0
