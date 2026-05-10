"""RL callback system tests (v1.2)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tgraphx.rl import (
    Callback, CallbackList,
    EarlyStoppingCallback,
    CSVLoggerCallback,
)


class TestCallbackBase:
    def test_default_callback_is_a_no_op(self):
        cb = Callback()
        # No raises on any event with arbitrary kwargs.
        cb.on_train_start()
        cb.on_episode_start(episode=0)
        cb.on_episode_end(episode=0, reward=1.0, steps=5, custom=42)
        cb.on_update_end(update=0, loss=0.1)
        cb.on_train_end()


class TestCallbackList:
    def test_empty_list_does_not_stop(self):
        cl = CallbackList()
        assert not cl.should_stop()
        cl.on_episode_end(episode=0, reward=1.0)
        assert not cl.should_stop()

    def test_request_stop(self):
        cl = CallbackList()
        cl.request_stop()
        assert cl.should_stop()

    def test_fan_out(self):
        events = []

        class Recorder(Callback):
            def on_episode_end(self, episode, reward, steps=None, **kw):
                events.append(("end", episode, reward))

            def on_update_end(self, update, loss=None, **kw):
                events.append(("update", update, loss))

        cl = CallbackList([Recorder(), Recorder()])
        cl.on_episode_end(episode=3, reward=0.5)
        cl.on_update_end(update=0, loss=0.1)
        # Each event fans out to 2 recorders.
        assert events.count(("end", 3, 0.5)) == 2
        assert events.count(("update", 0, 0.1)) == 2

    def test_append(self):
        cl = CallbackList()
        events = []

        class R(Callback):
            def on_train_start(self, **kw): events.append("start")

        cl.append(R())
        cl.on_train_start()
        assert events == ["start"]


class TestEarlyStopping:
    def test_max_mode_triggers_after_patience(self):
        cb = EarlyStoppingCallback(monitor="reward", patience=3, mode="max")
        # Reward sequence: 1, 0.5, 0.5, 0.5 (3 bad episodes after first).
        cb.on_episode_end(episode=0, reward=1.0)   # best = 1.0
        assert not cb.requested_stop
        cb.on_episode_end(episode=1, reward=0.5)   # bad 1
        assert not cb.requested_stop
        cb.on_episode_end(episode=2, reward=0.5)   # bad 2
        assert not cb.requested_stop
        cb.on_episode_end(episode=3, reward=0.5)   # bad 3 → stop
        assert cb.requested_stop

    def test_max_mode_does_not_trigger_when_improving(self):
        cb = EarlyStoppingCallback(monitor="reward", patience=2, mode="max")
        cb.on_episode_end(episode=0, reward=0.0)
        cb.on_episode_end(episode=1, reward=1.0)   # improves
        cb.on_episode_end(episode=2, reward=2.0)   # improves
        cb.on_episode_end(episode=3, reward=3.0)   # improves
        assert not cb.requested_stop

    def test_min_mode_triggers_when_loss_stops_decreasing(self):
        cb = EarlyStoppingCallback(monitor="loss", patience=2, mode="min")
        cb.on_episode_end(episode=0, reward=0.0, loss=2.0)
        cb.on_episode_end(episode=1, reward=0.0, loss=1.0)  # improves
        cb.on_episode_end(episode=2, reward=0.0, loss=1.5)  # bad 1
        cb.on_episode_end(episode=3, reward=0.0, loss=1.5)  # bad 2 → stop
        assert cb.requested_stop

    def test_min_delta_blocks_marginal_improvement(self):
        cb = EarlyStoppingCallback(monitor="reward", patience=2, mode="max", min_delta=0.5)
        cb.on_episode_end(episode=0, reward=1.0)
        # Tiny improvement (0.4 < min_delta) → counts as bad.
        cb.on_episode_end(episode=1, reward=1.4)  # bad 1
        cb.on_episode_end(episode=2, reward=1.4)  # bad 2 → stop
        assert cb.requested_stop

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must"):
            EarlyStoppingCallback(mode="bogus")

    def test_unknown_monitor_silently_skipped(self):
        cb = EarlyStoppingCallback(monitor="unobserved", patience=1, mode="max")
        cb.on_episode_end(episode=0, reward=1.0)  # 'unobserved' not in kwargs
        cb.on_episode_end(episode=1, reward=0.0)
        # No stop since the monitored key never appeared.
        assert not cb.requested_stop


class TestEarlyStoppingThroughCallbackList:
    def test_callback_list_propagates_stop_flag(self):
        cb = EarlyStoppingCallback(monitor="reward", patience=2, mode="max")
        cl = CallbackList([cb])
        cl.on_episode_end(episode=0, reward=1.0)
        assert not cl.should_stop()
        cl.on_episode_end(episode=1, reward=0.5)  # bad 1
        cl.on_episode_end(episode=2, reward=0.5)  # bad 2 → stop
        assert cl.should_stop()


class TestCSVLogger:
    def test_lazy_file_creation(self, tmp_path):
        path = tmp_path / "no_events.csv"
        cb = CSVLoggerCallback(path)
        # No events fired → file should NOT be created.
        cb.on_train_end()
        assert not path.exists()

    def test_writes_episode_rows(self, tmp_path):
        path = tmp_path / "ep.csv"
        cb = CSVLoggerCallback(path)
        cb.on_episode_end(episode=0, reward=1.0, steps=10, custom=42)
        cb.on_episode_end(episode=1, reward=2.0, steps=20, custom=43)
        cb.on_train_end()

        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["episode"] == "0"
        assert rows[0]["reward"] == "1.0"
        assert rows[0]["steps"] == "10"
        assert rows[0]["custom"] == "42"
        assert rows[1]["episode"] == "1"

    def test_header_stable_across_episodes(self, tmp_path):
        """Schema is fixed at the first episode_end; later extra keys ignored."""
        path = tmp_path / "stable.csv"
        cb = CSVLoggerCallback(path)
        cb.on_episode_end(episode=0, reward=1.0, steps=5)  # establishes header
        cb.on_episode_end(episode=1, reward=2.0, steps=6, new_key="ignored")
        cb.on_train_end()
        with path.open() as f:
            reader = csv.DictReader(f)
            assert "new_key" not in reader.fieldnames
            assert "episode" in reader.fieldnames
            assert "reward" in reader.fieldnames
            assert "steps" in reader.fieldnames

    def test_path_creates_parent_dir(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "log.csv"
        cb = CSVLoggerCallback(path)
        cb.on_episode_end(episode=0, reward=0.0)
        cb.on_train_end()
        assert path.exists()
