"""Curriculum schedule: controls which sampler is active based on training progress."""
from __future__ import annotations

from typing import List, Tuple

from .samplers import BearOffSampler, RacingSampler, FullGameSampler


class CurriculumSchedule:
    """Linear curriculum: promote to next stage when win rate threshold is met.

    stages: list of (sampler, promote_win_rate_threshold)
    """

    def __init__(
        self,
        stages: List[Tuple[object, float]] = None,
    ) -> None:
        if stages is None:
            stages = [
                (BearOffSampler(), 0.90),
                (RacingSampler(),  0.80),
                (FullGameSampler(), 1.0),
            ]
        self._stages  = stages
        self._current = 0

    @property
    def sampler(self):
        return self._stages[self._current][0]

    @property
    def stage_idx(self) -> int:
        return self._current

    def maybe_promote(self, win_rate: float) -> bool:
        if self._current >= len(self._stages) - 1:
            return False
        threshold = self._stages[self._current][1]
        if win_rate >= threshold:
            self._current += 1
            return True
        return False

    def is_final_stage(self) -> bool:
        return self._current == len(self._stages) - 1
