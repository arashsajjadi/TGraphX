"""memory_report.py — environment and memory estimate report.

Prints the runtime environment (Python/PyTorch/hardware) and estimates
the peak message-buffer memory for various graph configurations.
No file writes. No GPU required.
"""
import torch
from tgraphx.performance import env_report, estimate_message_memory


def _section(title: str) -> None:
    print(f"\n{'─'*56}")
    print(f"  {title}")
    print(f"{'─'*56}")


def main() -> None:
    # ── Environment ───────────────────────────────────────────────────────────
    _section("Runtime Environment")
    info = env_report(include_hardware=True, include_sensors=False)
    for key, val in info.items():
        if val is not None:
            print(f"  {key:<26}: {val}")

    # ── Message memory estimates ──────────────────────────────────────────────
    _section("Message Buffer Memory Estimates  [E, *out_shape]")

    configs = [
        ("Small grid (9 nodes, 33 edges)",    33,   (8, 4, 4)),
        ("Medium grid (256 nodes, 1024 edges)",1024, (32, 8, 8)),
        ("Large grid (1024 nodes, 4096 edges)",4096, (64, 8, 8)),
        ("Volumetric small (8 nodes, 32 edges)", 32, (4, 4, 4, 4)),
        ("Vector layer (N=1024, E=8192)",       8192, (128,)),
    ]

    print(f"\n  {'Description':<40} {'Edges':>6}  {'Est. MB':>8}")
    print(f"  {'─'*40} {'─'*6}  {'─'*8}")
    for desc, E, shape in configs:
        m = estimate_message_memory(E, shape)
        print(f"  {desc:<40} {E:>6}  {m['total_mb']:>8.3f}")

    print()
    print("  Note: actual peak usage is typically 2–3× the estimate above")
    print("        due to intermediate conv outputs inside the message step.")

    # ── float16 vs float32 ────────────────────────────────────────────────────
    _section("float16 vs float32 (large grid 4096 edges, shape 64,8,8)")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        m = estimate_message_memory(4096, (64, 8, 8), dtype=dt)
        print(f"  {str(dt):<22}: {m['total_mb']:.3f} MB")

    print()


if __name__ == "__main__":
    main()
