"""Environment reporting + package availability."""
from __future__ import annotations

import importlib
import platform
import sys
from typing import Any, Dict


def env_report() -> Dict[str, Any]:
    """Return a JSON-serializable environment report."""
    report: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    pkgs = ("torch", "torchvision", "tgraphx", "ultralytics",
            "transformers", "timm", "pycocotools",
            "cv2", "PIL", "matplotlib", "pandas", "numpy", "sklearn", "yaml")
    for pkg in pkgs:
        try:
            mod_name = pkg if pkg != "PIL" else "PIL"
            m = importlib.import_module(mod_name)
            v = getattr(m, "__version__", "installed")
            report[pkg] = v
        except ImportError:
            report[pkg] = "not_installed"

    try:
        import torch
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        report["cuda_available"] = False
    return report


def package_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False
