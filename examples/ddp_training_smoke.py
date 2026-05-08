"""ddp_training_smoke.py — distributed-helper smoke (single-process safe).

This script exercises ``tgraphx.distributed`` helpers in a single-process
CPU-safe way.  It does **not** start a multi-process job — that is the
user's responsibility (see the docstring at the bottom for a launch
command).

CPU-safe.  No torch.distributed.init_process_group is called.
"""
import torch

from tgraphx.distributed import (
    barrier,
    get_rank,
    get_world_size,
    is_distributed_available_and_initialized,
    is_rank_zero,
    rank_zero_only,
    rank_zero_print,
)

print("--- TGraphX distributed helpers (single-process smoke) ---")
print(f"is_distributed_available_and_initialized: {is_distributed_available_and_initialized()}")
print(f"world_size = {get_world_size()}")
print(f"rank       = {get_rank()}")
print(f"is_rank_zero = {is_rank_zero()}")

@rank_zero_only
def expensive_logging():
    return "logged on rank 0"

result = expensive_logging()
print(f"rank_zero_only result: {result!r}")

barrier()  # no-op outside DDP
rank_zero_print("rank_zero_print: visible on this rank")

print("\nddp_training_smoke: PASSED")
print()
print("Multi-process launch (advanced; user-controlled):")
print("    torchrun --nproc_per_node=2 examples/ddp_training_smoke.py")
print("Inside DDP, get_world_size()/get_rank() reflect the actual layout")
print("and rank_zero_only / rank_zero_print suppress on non-zero ranks.")
