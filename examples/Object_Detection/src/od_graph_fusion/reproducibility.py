"""Reproducibility utilities."""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Seed Python, NumPy, and PyTorch. Returns the state record."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    return {
        "seed": seed,
        "deterministic": deterministic,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
