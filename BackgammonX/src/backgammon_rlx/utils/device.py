"""Device selection utility."""
from __future__ import annotations

import torch


def get_device(cfg_device: str = "auto") -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(cfg_device)
