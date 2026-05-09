"""Tests for calibration utilities."""
from __future__ import annotations

import torch
import pytest

from tgraphx.calibration import (
    expected_calibration_error,
    reliability_diagram_data,
    TemperatureScaler,
    calibrate_temperature,
)


def test_ece_perfect_calibration():
    """Perfectly calibrated model: ECE = 0."""
    torch.manual_seed(0)
    N, C = 200, 2
    # Create perfectly calibrated: prediction = label, confidence always 1.
    logits = torch.zeros(N, C)
    y = torch.randint(0, C, (N,))
    logits[torch.arange(N), y] = 100.0
    ece = expected_calibration_error(logits, y)
    assert ece < 0.01, f"ECE should be near 0 for perfect logits; got {ece}"


def test_ece_worst_case():
    """Confidently wrong model has high ECE."""
    N, C = 100, 2
    logits = torch.zeros(N, C)
    y = torch.ones(N, dtype=torch.long)
    # Always predict class 0 with high confidence when true class is 1.
    logits[:, 0] = 100.0
    ece = expected_calibration_error(logits, y)
    assert ece > 0.5, f"ECE should be high for confidently-wrong model; got {ece}"


def test_ece_bounds():
    torch.manual_seed(42)
    logits = torch.randn(200, 4)
    y = torch.randint(0, 4, (200,))
    ece = expected_calibration_error(logits, y)
    assert 0.0 <= ece <= 1.0, f"ECE out of [0, 1]: {ece}"


def test_ece_invalid_inputs():
    with pytest.raises(ValueError, match="logits must be"):
        expected_calibration_error(torch.randn(10), torch.zeros(10, dtype=torch.long))
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(torch.randn(10, 2), torch.zeros(10, dtype=torch.long), n_bins=1)


def test_reliability_diagram_json_serializable():
    logits = torch.randn(100, 3)
    y = torch.randint(0, 3, (100,))
    result = reliability_diagram_data(logits, y, n_bins=5)
    for key in ("bin_confidences", "bin_accuracies", "bin_fractions", "ece"):
        assert key in result
    assert len(result["bin_confidences"]) == 5
    import json
    json.dumps(result)  # must be serializable


def test_temperature_scaler_scales():
    scaler = TemperatureScaler(temperature=2.0)
    logits = torch.randn(10, 3)
    scaled = scaler(logits)
    assert torch.allclose(scaled, logits / 2.0, atol=1e-5)


def test_temperature_scaler_positive():
    with pytest.raises(ValueError):
        TemperatureScaler(temperature=0.0)
    with pytest.raises(ValueError):
        TemperatureScaler(temperature=-1.0)


def test_calibrate_temperature_reduces_nll():
    torch.manual_seed(0)
    N, C = 100, 4
    # Overconfident logits.
    logits = torch.randn(N, C) * 10.0
    y = torch.randint(0, C, (N,))
    import torch.nn.functional as F_
    nll_before = float(F_.cross_entropy(logits, y).item())
    scaler = calibrate_temperature(logits, y)
    nll_after = float(F_.cross_entropy(scaler(logits), y).item())
    assert nll_after <= nll_before + 0.1, \
        f"Temperature scaling increased NLL: {nll_before:.3f}→{nll_after:.3f}"
    assert float(scaler.temperature.item()) > 0.0


def test_calibrate_no_autograd_retained():
    """ECE / reliability computations do not retain autograd graphs."""
    logits = torch.randn(20, 3, requires_grad=True)
    y = torch.randint(0, 3, (20,))
    ece = expected_calibration_error(logits, y)
    # ece is a plain float; no grad_fn.
    assert isinstance(ece, float)
