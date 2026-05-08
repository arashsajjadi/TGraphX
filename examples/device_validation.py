"""Device validation: run a tiny smoke across CPU, CUDA, MPS, and AMP.

The script never claims a device works it has not actually tested.  It
emits a JSON report so README / docs claims can be cross-checked
against reality.

Usage::

    python examples/device_validation.py --device cpu --quick
    python examples/device_validation.py --device cuda --amp --output-json cuda.json
    python examples/device_validation.py --device mps --quick
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def _resolve(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _run_layer(name: str, layer, x, ei, **call) -> Dict[str, Any]:
    t0 = time.perf_counter()
    out = layer(x, ei, **call)
    elapsed = time.perf_counter() - t0
    finite = bool(torch.isfinite(out).all().item())
    # Backward smoke.
    grad_finite = True
    if x.requires_grad:
        loss = out.sum()
        loss.backward()
        grad_finite = bool(torch.isfinite(x.grad).all().item())
        x.grad = None
    return {
        "layer": name,
        "out_shape": list(out.shape),
        "finite_forward": finite,
        "finite_backward": grad_finite,
        "elapsed_s": elapsed,
    }


def _vector_smoke(device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
    from tgraphx import (
        APPNP, GATv2Conv, GCNConv, LinearMessagePassing,
    )

    N, D = 8, 4
    x = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                       [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long, device=device)
    out: List[Dict[str, Any]] = []
    out.append(_run_layer("LinearMessagePassing",
                          LinearMessagePassing(in_shape=(D,), out_shape=(D,)).to(device).to(dtype),
                          x, ei))
    out.append(_run_layer("GCNConv",
                          GCNConv(D, D).to(device).to(dtype), x, ei))
    out.append(_run_layer("GATv2Conv",
                          GATv2Conv(D, D, num_heads=2).to(device).to(dtype), x, ei))
    out.append(_run_layer("APPNP",
                          APPNP(K=2, alpha=0.2).to(device), x, ei))
    return out


def _spatial_smoke(device: torch.device, dtype: torch.dtype) -> List[Dict[str, Any]]:
    from tgraphx import (
        ConvMessagePassing, TensorGATLayer, TensorGINLayer, TensorGraphSAGELayer,
    )
    N, C, H, W = 4, 3, 4, 4
    x = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=True)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long, device=device)
    out: List[Dict[str, Any]] = []
    out.append(_run_layer("ConvMessagePassing",
                          ConvMessagePassing(in_shape=(C, H, W), out_shape=(4, H, W),
                                             aggr="sum").to(device).to(dtype),
                          x, ei))
    out.append(_run_layer("TensorGATLayer",
                          TensorGATLayer(in_channels=C, out_channels=4, num_heads=2)
                          .to(device).to(dtype), x, ei))
    out.append(_run_layer("TensorGraphSAGELayer",
                          TensorGraphSAGELayer(in_channels=C, out_channels=4)
                          .to(device).to(dtype), x, ei))
    out.append(_run_layer("TensorGINLayer",
                          TensorGINLayer(in_channels=C, out_channels=4)
                          .to(device).to(dtype), x, ei))
    return out


def _dataset_metric_smoke(device: torch.device) -> Dict[str, Any]:
    from tgraphx.datasets import SyntheticPatchGraphDataset
    from tgraphx.metrics import accuracy
    ds = SyntheticPatchGraphDataset(num_graphs=2, image_size=8, patch_size=4, seed=0)
    g = ds[0].to(device)
    finite = bool(torch.isfinite(g.node_features).all().item())
    acc = accuracy(torch.tensor([0, 1, 1]), torch.tensor([0, 1, 0]))
    return {"dataset_features_finite": finite, "metric_smoke_accuracy": acc}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--amp", action="store_true",
                   help="Run an autocast pass on top of the CUDA / CPU smoke (CUDA: float16; CPU: bfloat16).")
    p.add_argument("--quick", action="store_true",
                   help="Skip the spatial smoke for a faster run.")
    p.add_argument("--strict", action="store_true",
                   help="Fail (exit 2) when the requested device is unavailable.")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args(argv)

    requested = args.device
    device = _resolve(requested)
    if requested in ("cuda", "mps") and device.type != requested:
        msg = f"requested {requested} but it is not available; running on cpu"
        if args.strict:
            print(msg)
            return 2
        print(msg)

    import tgraphx
    report: Dict[str, Any] = {
        "tgraphx_version": tgraphx.__version__,
        "torch_version": torch.__version__,
        "requested_device": requested,
        "actual_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "amp": bool(args.amp),
        "results": {},
    }

    # Vector smoke (always).
    report["results"]["vector"] = _vector_smoke(device, torch.float32)

    # Spatial smoke unless --quick.
    if not args.quick:
        try:
            report["results"]["spatial"] = _spatial_smoke(device, torch.float32)
        except RuntimeError as exc:
            report["results"]["spatial"] = [{"error": str(exc)}]

    # Dataset + metric.
    report["results"]["dataset_metric"] = _dataset_metric_smoke(device)

    # AMP autocast pass.
    if args.amp:
        amp_results: Dict[str, Any] = {}
        if device.type == "cuda":
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    amp_results["cuda_float16"] = _vector_smoke(device, torch.float32)
            except RuntimeError as exc:
                amp_results["cuda_float16"] = [{"error": str(exc)}]
        elif device.type == "cpu":
            try:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    amp_results["cpu_bfloat16"] = _vector_smoke(device, torch.float32)
            except RuntimeError as exc:
                amp_results["cpu_bfloat16"] = [{"error": str(exc)}]
        report["results"]["amp"] = amp_results

    # Verdict: every "finite_forward" / "finite_backward" should be True.
    failures = []
    for group, rows in report["results"].items():
        if isinstance(rows, list):
            for r in rows:
                if "error" in r:
                    failures.append(f"{group}/{r.get('layer', '?')}: {r['error']}")
                if r.get("finite_forward") is False:
                    failures.append(f"{group}/{r.get('layer','?')}: forward not finite")
                if r.get("finite_backward") is False:
                    failures.append(f"{group}/{r.get('layer','?')}: backward not finite")
    report["failures"] = failures
    report["all_passed"] = not failures

    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).expanduser().write_text(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
