"""Training logger — writes JSONL and optionally TensorBoard."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """Logs training metrics to console and a JSONL file."""

    def __init__(self, log_dir: Path, tb: bool = False) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.log_dir / "metrics.jsonl", "a")
        self._t0    = time.time()
        self._writer: Optional[Any] = None
        if tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.log_dir / "tb")
            except ImportError:
                print("[logger] TensorBoard not available", file=sys.stderr)

    def log(self, record: Dict[str, Any]) -> None:
        record["wall_time"] = time.time() - self._t0
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        step = record.get("update", 0)
        # console summary
        parts = [f"upd={step:6d}",
                 f"games={record.get('games', 0):8d}",
                 f"pi={record.get('policy_loss', 0):.4f}",
                 f"vf={record.get('value_loss',  0):.4f}",
                 f"ent={record.get('entropy',      0):.3f}",
                 f"len={record.get('mean_length', 0):.1f}",
                 f"gps={record.get('gps', 0):.1f}"]
        print("  ".join(parts))

        if self._writer is not None:
            for k, v in record.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        self._jsonl.close()
        if self._writer is not None:
            self._writer.close()

    def __del__(self) -> None:
        try:
            self._jsonl.close()
        except Exception:
            pass
