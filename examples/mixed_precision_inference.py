"""mixed_precision_inference.py — AMP (autocast) forward-pass demonstration.

Shows how to run TGraphX layers under torch.autocast for reduced memory
and potentially faster inference on CUDA.

* CUDA : float16 autocast
* CPU  : bfloat16 autocast (PyTorch 1.13+ on supported hardware)
* MPS  : float16 autocast if available; otherwise full precision
* No GPU? The script still runs full-precision on CPU and prints what would
  happen on CUDA.

No training loop is performed — this is an inference / forward-only demo.
No file writes.
"""
import torch
import torch.nn as nn

from tgraphx import Graph, GraphBatch, build_grid_graph, build_model
from tgraphx.performance import env_report


def _sizeof_mb(t: torch.Tensor) -> float:
    return t.nelement() * t.element_size() / 1024**2


def _run(model: nn.Module, x: torch.Tensor, ei: torch.Tensor,
         batch: torch.Tensor, label: str, ctx=None) -> None:
    """Run one forward pass, optionally inside autocast."""
    model.eval()
    ctx = ctx or _NullCtx()
    try:
        with torch.no_grad(), ctx:
            out = model(x, ei, batch=batch)
        act_dtype = out.dtype
        finite = torch.isfinite(out).all().item()
        print(f"  [{label}]  output {tuple(out.shape)}  dtype={act_dtype}  "
              f"finite={'yes' if finite else 'WARN: non-finite'}  "
              f"size={_sizeof_mb(out):.3f} MB")
    except RuntimeError as e:
        if "scalar type" in str(e).lower() or "dtype" in str(e).lower():
            # Rare dtype mismatch — should not occur after v0.2.2 dtype fixes
            # (broadcast_edge_weight casts weight; GAT casts attn to activation dtype).
            # If seen, it indicates a new op combination not yet handled.
            print(f"  [{label}]  skipped — unexpected dtype mismatch: {e}")
        else:
            raise


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def main() -> None:
    info = env_report()
    device = torch.device(info["recommended_device"])
    cuda  = info["cuda_available"]
    mps   = info["mps_available"]

    print(f"\nDevice             : {device}")
    print(f"PyTorch            : {info['torch']}")
    print(f"CUDA available     : {cuda}")
    print(f"MPS available      : {mps}")

    # ── Build a small patch-graph classification model ────────────────────────
    B, C, ph, pw = 2, 4, 4, 4   # 2 images, 4-channel 4×4 patches
    n_h = n_w = 2                # 2×2 patch grid → 4 nodes per graph

    model = build_model(
        task="graph_classification",
        layer="gat",
        in_shape=(C, ph, pw),
        hidden_shape=(8, ph, pw),
        num_layers=2,
        num_classes=3,
        heads=2,
        pooling="mean",
    ).to(device)

    # Synthetic node features: [B*P, C, ph, pw] = [8, 4, 4, 4]
    x = torch.randn(B * n_h * n_w, C, ph, pw, device=device)
    ei = build_grid_graph(n_h, n_w, directed=False, self_loops=True).to(device)
    # Two graphs of 4 nodes each
    batch = torch.cat([torch.full((n_h * n_w,), i, dtype=torch.long) for i in range(B)]).to(device)

    print(f"\nNode features      : {tuple(x.shape)}  ({_sizeof_mb(x):.3f} MB float32)")

    print("\n─── Precision comparison ───────────────────────────────────")

    # 1. Full precision baseline
    _run(model, x, ei, batch, "float32 (baseline)")

    # 2. CUDA float16 autocast
    if cuda:
        ctx = torch.autocast("cuda", dtype=torch.float16)
        _run(model.cuda(), x.cuda(), ei.cuda(), batch.cuda(), "CUDA float16 autocast", ctx)
    else:
        print("  [CUDA float16]    skipped — no CUDA device")

    # 3. CPU bfloat16 autocast
    try:
        ctx = torch.autocast("cpu", dtype=torch.bfloat16)
        _run(model.cpu(), x.cpu(), ei.cpu(), batch.cpu(), "CPU bfloat16 autocast", ctx)
    except RuntimeError as e:
        print(f"  [CPU bfloat16]    skipped — {e}")

    # 4. MPS float16 (if available)
    if mps:
        try:
            mps_dev = torch.device("mps")
            ctx = torch.autocast("mps", dtype=torch.float16)
            _run(model.to(mps_dev), x.to(mps_dev), ei.to(mps_dev),
                 batch.to(mps_dev), "MPS float16 autocast", ctx)
        except Exception as e:
            print(f"  [MPS float16]     skipped — {e}")

    print("\nNote: autocast may keep some ops in float32 for numerical stability.")
    print("      Output tensor dtypes depend on the layer's internal ops.")
    print()


if __name__ == "__main__":
    main()
