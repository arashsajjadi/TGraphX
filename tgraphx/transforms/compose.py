"""Transform composition primitives."""
from __future__ import annotations

from typing import Any, Callable, List, Sequence

import torch


class Compose:
    """Apply a sequence of transforms left-to-right.

    Each transform must be a callable ``Graph -> Graph`` (or whatever
    graph-like type the inner transform supports).
    """

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms: List[Callable[[Any], Any]] = list(transforms)

    def __call__(self, item: Any) -> Any:
        for t in self.transforms:
            item = t(item)
        return item

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"Compose([{', '.join(names)}])"


class RandomApply:
    """Apply a transform with probability ``p``."""

    def __init__(
        self,
        transform: Callable[[Any], Any],
        p: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        self.transform = transform
        self.p = float(p)
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def __call__(self, item: Any) -> Any:
        if torch.rand((), generator=self._gen).item() < self.p:
            return self.transform(item)
        return item


class LambdaTransform:
    """Wrap an arbitrary user callable.

    The callable is expected to take a single graph-like argument and
    return a graph-like result.  Use sparingly — explicit transform
    classes are preferred for reproducibility.
    """

    def __init__(self, fn: Callable[[Any], Any]) -> None:
        if not callable(fn):
            raise TypeError(f"fn must be callable; got {type(fn)}")
        self.fn = fn

    def __call__(self, item: Any) -> Any:
        return self.fn(item)
