# Distributed training

TGraphX provides **distributed-training helpers** but does not ship a
full distributed framework.  All multi-process orchestration is the
caller's responsibility (e.g. `torchrun`, `srun`, manual MPI launch);
the helpers in `tgraphx.distributed` make the *single-process* and
*rank-aware* paths work cleanly.

## Helpers

| Function | Purpose |
|---|---|
| `detect_distributed_environment()` | Inspect `RANK`, `WORLD_SIZE`, SLURM equivalents; never calls `init_process_group`. |
| `get_rank()` / `get_world_size()` / `is_rank_zero()` | Safe getters with single-process defaults. |
| `rank_zero_only` | Decorator: function runs only on rank 0. |
| `rank_zero_print` | `print` only on rank 0. |
| `barrier()` | `dist.barrier()` when initialised, no-op otherwise. |
| `rank_seed(base_seed)` | Deterministic per-rank seed derived from a shared base. |
| `distributed_device(local_rank)` | `cuda:local_rank` when CUDA is available, else CPU. |
| `maybe_wrap_ddp(model)` | Wraps in `DistributedDataParallel` only when a process group is up. |
| `shard_indices(idx, rank, world_size)` | Contiguous shard of an index tensor. |
| `write_distributed_run_summary(path, **fields)` | Rank-0-only writer for `distributed_run_summary.json`. |

## Honesty

Multi-process distributed graph training is **Experimental**.  This
release does not include a multi-process integration test on CI; the
helpers are unit-tested in single-process mode and against environment
variables.  A future v0.5.x will extend coverage with a 2-process
subprocess smoke once the test runner allows it.

## Example

```python
import os
import torch
from tgraphx.distributed import (
    detect_distributed_environment, rank_seed, distributed_device,
    maybe_wrap_ddp, write_distributed_run_summary,
)

env = detect_distributed_environment()
torch.manual_seed(rank_seed(42))
device = distributed_device(env["local_rank"])

model = build_model().to(device)
model = maybe_wrap_ddp(model, device_ids=[device])

write_distributed_run_summary(
    "logs/distributed_run_summary.json",
    base_seed=42, model="GCN", dataset="ogbn-arxiv",
)
```
