"""Benchmark GPU inference throughput for BackgammonPolicyValueNet."""
from __future__ import annotations

import time
from typing import Dict

import torch
import numpy as np

from ..env.encoding import OBS_DIM, ACT_DIM
from ..models.policy_value_net import BackgammonPolicyValueNet


def benchmark_inference_throughput(
    model:        BackgammonPolicyValueNet,
    device_str:   str   = "cuda",
    batch_sizes:  tuple = (1, 8, 32, 128, 512, 2048),
    n_actions:    int   = 20,
    n_warmup:     int   = 5,
    n_iters:      int   = 50,
    use_amp:      bool  = True,
) -> Dict:
    """Measure throughput for various batch sizes."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    results = {}

    for B in batch_sizes:
        obs_t  = torch.randn(B, OBS_DIM, device=device, dtype=torch.float32)
        act_t  = torch.randn(B, n_actions, ACT_DIM, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    model(obs_t, act_t)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    logits, values, _ = model(obs_t, act_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_inferences = B * n_iters
        results[B] = {
            "batch_size":      B,
            "n_actions":       n_actions,
            "elapsed_s":       elapsed,
            "states_per_s":    total_inferences / elapsed,
            "actions_per_s":   total_inferences * n_actions / elapsed,
            "ms_per_batch":    elapsed / n_iters * 1000,
        }

    return {"device": str(device), "use_amp": use_amp, "results": results}


def benchmark_latency(
    model:      BackgammonPolicyValueNet,
    device_str: str = "cuda",
    n_actions:  int = 20,
    n_iters:    int = 200,
) -> Dict:
    """Measure single-sample latency (single game inference call)."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    obs_t = torch.randn(1, OBS_DIM, device=device)
    act_t = torch.randn(1, n_actions, ACT_DIM, device=device)

    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(obs_t, act_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    import statistics
    return {
        "device":       str(device),
        "mean_ms":      statistics.mean(times) * 1000,
        "median_ms":    statistics.median(times) * 1000,
        "p99_ms":       sorted(times)[int(0.99 * len(times))] * 1000,
        "min_ms":       min(times) * 1000,
    }
