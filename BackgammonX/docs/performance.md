# BackgammonX Performance Notes

## Hardware Target

- GPU: NVIDIA RTX 5080 (16 GB VRAM, CUDA 13)
- CPU: Intel Core i7-14700F (20 cores / 28 threads)
- RAM: 64 GB
- OS: Linux

## Baseline Benchmarks (RTX 5080, default config)

| Metric | Value |
|--------|-------|
| Legal move generation | ~4,000 positions/s |
| Random game simulation | ~36 games/s (~3,350 steps/s) |
| Neural inference (B=1) | ~1,300 states/s, 0.65ms latency |
| Neural inference (B=512) | ~385,000 states/s, 7.7M actions/s |
| Single-process self-play | ~7 games/s, ~580 steps/s |
| Multiprocess self-play (4 workers) | ~5 games/s (tested) |

## Bottleneck Analysis

At small batch sizes (B=1), the bottleneck is **Python overhead** in env simulation.
At production batch sizes (B=512), GPU is fully utilized.
The self-play throughput gap vs raw inference indicates **IPC queue latency** is significant.

## Tuning for RTX 5080

```yaml
# configs/rtx5080.yaml key settings
rollout_mode: multiprocess
num_self_play_workers: 12    # i7-14700F has 20 cores
inference_batch_size: 512    # saturates GPU without excess queue wait
minibatch_size: 4096         # large PPO batches for GPU efficiency
mixed_precision: true        # halves VRAM and bandwidth
pin_memory: true             # faster CPU→GPU transfer
```

## Numba Backend (Planned)

The move generator is pure Python. A Numba or Cython backend could provide
2-5× speedup for legal move generation. The backend interface is defined in
`env/movegen.py` but the accelerated implementation is a TODO.

Current correctness is the priority; acceleration is post-MVP.

## Memory Usage

- Model (state_dim=256, default): ~123K parameters, ~0.5 MB
- Model (state_dim=512, rtx5080): ~5M parameters, ~20 MB
- PPO buffer (128 games × 100 steps): ~32K transitions, ~100 MB numpy
- GPU VRAM peak during update: <2 GB at minibatch_size=4096

## Profiling

Run the benchmark suite to identify bottlenecks:
```bash
python -m backgammon_rlx.benchmarks.benchmark_all --config configs/rtx5080.yaml
```

Use `torch.profiler` for GPU kernel analysis:
```python
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(obs_t, act_t)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```
