"""Lightweight callbacks for graph RL training loops.

A minimal callback system inspired by the SL ``tgraphx.experiments.callbacks``
API.  Designed for graph RL agents that emit per-episode and per-update
events.  Pure-PyTorch; no heavy dependencies.

Usage::

    from tgraphx.rl.callbacks import (
        Callback, CallbackList,
        EarlyStoppingCallback,
        CSVLoggerCallback,
    )

    csv_log = CSVLoggerCallback("runs/rl/episodes.csv")
    early   = EarlyStoppingCallback(monitor="reward", patience=20, mode="max")
    cb_list = CallbackList([csv_log, early])

    for ep_idx in range(num_episodes):
        cb_list.on_episode_start(episode=ep_idx)
        # ... rollout / update ...
        cb_list.on_episode_end(episode=ep_idx, reward=ep_return, steps=ep_len)
        if cb_list.should_stop():
            break
    cb_list.on_train_end()

Stability: Beta — fully tested with deterministic toy events.
"""
from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStoppingCallback",
    "CSVLoggerCallback",
]


# ── Base interface ────────────────────────────────────────────────────────────


class Callback:
    """Base callback interface.

    Subclasses override the events they care about.  All hooks accept
    ``**kwargs`` for forward-compatible extension.
    """

    def on_train_start(self, **kwargs: Any) -> None: ...
    def on_episode_start(self, episode: int, **kwargs: Any) -> None: ...
    def on_episode_end(
        self,
        episode: int,
        reward: float,
        steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None: ...
    def on_update_end(self, update: int, loss: Optional[float] = None, **kwargs: Any) -> None: ...
    def on_train_end(self, **kwargs: Any) -> None: ...


# ── Container ─────────────────────────────────────────────────────────────────


class CallbackList:
    """Aggregator that fan-outs events to a list of callbacks.

    Tracks a single ``should_stop()`` flag so any callback (e.g.
    ``EarlyStoppingCallback``) can request training termination.
    """

    def __init__(self, callbacks: Sequence[Callback] = ()) -> None:
        self._callbacks: List[Callback] = list(callbacks)
        self._stop = False

    def append(self, cb: Callback) -> None:
        self._callbacks.append(cb)

    def should_stop(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True

    # Event fan-out
    def on_train_start(self, **kw: Any) -> None:
        for c in self._callbacks:
            c.on_train_start(**kw)

    def on_episode_start(self, episode: int, **kw: Any) -> None:
        for c in self._callbacks:
            c.on_episode_start(episode=episode, **kw)

    def on_episode_end(self, episode: int, reward: float,
                       steps: Optional[int] = None, **kw: Any) -> None:
        for c in self._callbacks:
            c.on_episode_end(episode=episode, reward=reward, steps=steps, **kw)
        # Honour stop requests from individual callbacks.
        for c in self._callbacks:
            if isinstance(c, EarlyStoppingCallback) and c.requested_stop:
                self._stop = True

    def on_update_end(self, update: int, loss: Optional[float] = None, **kw: Any) -> None:
        for c in self._callbacks:
            c.on_update_end(update=update, loss=loss, **kw)

    def on_train_end(self, **kw: Any) -> None:
        for c in self._callbacks:
            c.on_train_end(**kw)


# ── Early stopping ────────────────────────────────────────────────────────────


class EarlyStoppingCallback(Callback):
    """Stop training when the monitored metric stops improving.

    Args:
        monitor: Metric name to watch (looked up in ``on_episode_end`` kwargs;
            defaults to ``"reward"``).
        patience: Number of episodes without improvement before stopping.
        mode: ``"max"`` (improvement = larger value) or ``"min"`` (smaller).
        min_delta: Minimum change to count as an improvement.

    Side effect:
        Sets ``self.requested_stop = True`` when the patience window expires.
        :class:`CallbackList` propagates this to ``should_stop()``.
    """

    def __init__(
        self,
        monitor: str = "reward",
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
        self.monitor = monitor
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.bad_episodes = 0
        self.requested_stop = False

    def _is_better(self, new: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return new > self.best + self.min_delta
        return new < self.best - self.min_delta

    def on_episode_end(self, episode: int, reward: float, steps=None, **kw: Any) -> None:
        # Pull monitored metric from reward (default) or kwargs.
        if self.monitor == "reward":
            value = float(reward)
        elif self.monitor in kw:
            value = float(kw[self.monitor])
        else:
            return  # monitor not observed this episode; skip silently.
        if self._is_better(value):
            self.best = value
            self.bad_episodes = 0
        else:
            self.bad_episodes += 1
            if self.bad_episodes >= self.patience:
                self.requested_stop = True


# ── CSV logger ────────────────────────────────────────────────────────────────


class CSVLoggerCallback(Callback):
    """Append per-episode rows to a CSV file.

    Header is written on the first ``on_episode_end`` based on the seen
    keys (``episode``, ``reward``, ``steps``, plus any extra ``**kw`` keys
    that are scalar-typed: int/float/bool/str).

    Args:
        path: Output CSV path (created if missing; parent dirs created).

    The file is opened lazily on first event so test-mode usage that never
    fires events does not create empty files.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._fp = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None

    def _open(self, fieldnames: List[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", newline="")
        self._writer = _csv.DictWriter(self._fp, fieldnames=fieldnames)
        self._writer.writeheader()
        self._fieldnames = fieldnames

    def _scalar(self, v: Any) -> bool:
        return isinstance(v, (int, float, bool, str))

    def on_episode_end(self, episode: int, reward: float,
                       steps: Optional[int] = None, **kw: Any) -> None:
        row = {"episode": int(episode), "reward": float(reward)}
        if steps is not None:
            row["steps"] = int(steps)
        for k, v in kw.items():
            if self._scalar(v):
                row[k] = v
        if self._writer is None:
            self._open(list(row.keys()))
        # If a new key appears later, ignore it to keep header stable.
        filtered = {k: row.get(k) for k in self._fieldnames}
        self._writer.writerow(filtered)
        self._fp.flush()

    def on_train_end(self, **kw: Any) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
            self._writer = None
