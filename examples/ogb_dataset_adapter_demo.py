"""ogb_dataset_adapter_demo.py — wrap a tiny mock OGB evaluator.

Skips cleanly if OGB is not installed.  No network — uses a fake
evaluator class that mimics the OGB API for the demo.
"""
from __future__ import annotations

import sys

import torch

try:
    import ogb  # type: ignore[import]  # noqa: F401
except ImportError as exc:
    # Even without OGB, we can demonstrate ``OGBEvaluatorWrapper`` with a
    # mock evaluator class — the wrapper itself does not import OGB.
    print(f"Note: ogb is not installed ({exc}); using a fake evaluator class")

from tgraphx.datasets import OGBEvaluatorWrapper


class _FakeEvaluator:
    """Stand-in for ``ogb.nodeproppred.Evaluator`` in this offline demo."""

    expected_input_format = {"y_true": "[N, 1]", "y_pred": "[N, 1]"}
    expected_output_format = {"acc": "float"}

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, input_dict):
        y_true = input_dict["y_true"].view(-1).long()
        y_pred = input_dict["y_pred"].view(-1).long()
        return {"acc": float((y_true == y_pred).float().mean())}


def main() -> None:
    wrapper = OGBEvaluatorWrapper(_FakeEvaluator, name="ogbn-demo")
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 0])
    out = wrapper.eval({"y_true": y_true, "y_pred": y_pred})
    print(f"Fake OGB evaluator score: {out}")


if __name__ == "__main__":
    main()
