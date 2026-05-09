"""Minimal two-process distributed smoke: gloo/CPU, world_size=2.

Intended for:
  1. Direct invocation via ``torchrun``:
       torchrun --nproc_per_node=2 examples/distributed_smoke.py

  2. Subprocess-pair invocation (CI-friendly, no torchrun required):
       python examples/distributed_smoke.py --subprocess-pair \
              --output-dir /tmp/dist_smoke

  3. Single-process no-op (always passes, for import-level validation):
       python examples/distributed_smoke.py --world-size 1

Each rank:
- Initialises torch.distributed with the gloo backend.
- Uses rank_seed to set a per-rank RNG.
- Constructs a tiny 2-layer MLP.
- Wraps the model with maybe_wrap_ddp.
- Runs one forward/backward/optimizer step on random data.
- Rank-0 writes distributed_run_summary.json.
- Calls cleanup_distributed and exits cleanly.

No GPU is required.  No network access beyond localhost.
No hanging: a 60-second timeout guard is applied.

Stability: Experimental — distributed multi-process validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


def _worker_main() -> None:
    """Per-rank worker logic; called after the process group is live."""
    from tgraphx.distributed import (
        get_rank, get_world_size, is_rank_zero,
        rank_seed, distributed_device, maybe_wrap_ddp,
        write_distributed_run_summary,
    )
    rank = get_rank()
    world_size = get_world_size()
    seed = rank_seed(42)
    torch.manual_seed(seed)

    device = distributed_device(int(os.environ.get("LOCAL_RANK", 0)))

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    model = maybe_wrap_ddp(model)

    x = torch.randn(16, 4, device=device)
    y = torch.randint(0, 2, (16,), device=device)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    opt.step()

    loss_val = float(loss.detach().item())
    loss_finite = bool(torch.isfinite(torch.tensor(loss_val)).item())

    if is_rank_zero():
        out_dir = os.environ.get("DIST_SMOKE_OUTPUT_DIR", "/tmp/dist_smoke")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out = str(Path(out_dir) / "distributed_run_summary.json")
        write_distributed_run_summary(
            out,
            world_size=world_size,
            backend="gloo",
            seed=seed,
            device=str(device),
            step_completed=True,
            loss=loss_val,
            loss_finite=loss_finite,
            rank_zero_artifact_written=True,
        )
        print(f"[rank 0] smoke ok — loss={loss_val:.4f}, wrote {out}", file=sys.stderr)
    else:
        print(f"[rank {rank}] smoke ok", file=sys.stderr)


def run_subprocess_pair(output_dir: str, world_size: int, timeout: int = 60) -> bool:
    """Launch world_size sub-processes, each running this file.

    Returns True if all ranks exit 0 within ``timeout`` seconds.
    """
    import subprocess

    port = int(os.environ.get("DIST_SMOKE_PORT", "29510"))
    env = {
        **os.environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "WORLD_SIZE": str(world_size),
        "DIST_SMOKE_OUTPUT_DIR": output_dir,
        "DIST_SMOKE_WORKER": "1",
    }
    procs = []
    for r in range(world_size):
        e = {**env, "RANK": str(r), "LOCAL_RANK": str(r)}
        p = subprocess.Popen([sys.executable, __file__], env=e)
        procs.append(p)

    deadline = time.time() + timeout
    ok = True
    for p in procs:
        remaining = max(0.5, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
            ok = False
        if p.returncode != 0:
            ok = False
    return ok


def main() -> None:
    if os.environ.get("DIST_SMOKE_WORKER") == "1":
        # We are a worker spawned by run_subprocess_pair.
        import torch.distributed as dist
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            init_method="env://",
        )
        try:
            _worker_main()
        finally:
            dist.destroy_process_group()
        return

    parser = argparse.ArgumentParser(description="TGraphX distributed smoke test")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-dir", default="/tmp/dist_smoke")
    parser.add_argument("--subprocess-pair", action="store_true",
                        help="Launch via subprocess (CI-friendly, no torchrun needed)")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if "RANK" in os.environ or "TORCHELASTIC_RESTART_COUNT" in os.environ:
        # Invoked by torchrun — worker path.
        import torch.distributed as dist
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        try:
            _worker_main()
        finally:
            dist.destroy_process_group()
        return

    if args.subprocess_pair or args.world_size > 1:
        ok = run_subprocess_pair(
            args.output_dir, args.world_size, timeout=args.timeout
        )
        result = {
            "world_size": args.world_size,
            "backend": "gloo",
            "passed": ok,
            "mode": "subprocess_pair",
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"distributed smoke: {'PASSED' if ok else 'FAILED'} (world_size={args.world_size})")
        sys.exit(0 if ok else 1)

    # Single-process no-op path (world_size=1).
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29510")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["DIST_SMOKE_OUTPUT_DIR"] = args.output_dir
        _worker_main()
    finally:
        dist.destroy_process_group()
    result = {"world_size": 1, "backend": "gloo", "passed": True, "mode": "single_process"}
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
