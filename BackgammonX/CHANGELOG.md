# Changelog

## [Unreleased]

### Added
- Multiprocess self-play rollout (`MultiprocessRolloutCollector`) with GPU inference thread
- 10 new golden legal-move fixtures covering edge cases
- `tests/test_mandatory_dice_usage.py` — larger-die rule, mandatory dice usage
- `tests/test_bar_entry.py` — bar priority, entry points, hit on entry
- `tests/test_random_stress.py` — 1000+ random games, invariant checking
- `explain_illegal_action()` in `validation/explain_illegal.py`
- Transition tracing via `env.step(action, trace=True)`
- `run_metadata.py` — full reproducibility metadata saved per run
- `data/generate_expert_dataset.py` — expert dataset generation
- `benchmarks/benchmark_inference.py` — GPU throughput at various batch sizes
- `Makefile` with common commands
- `LICENSE`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.gitignore`
- `.github/workflows/ci.yml` — GitHub Actions CI
- `examples/` — runnable example scripts

### Changed
- `self_play.py` — supports `rollout_mode: "single" | "multiprocess"`
- `rtx5080.yaml` — updated with multiprocess and larger model settings
- `benchmark_all.py` — includes inference throughput and latency benchmarks
- `train_ppo.py` — saves run metadata on start

## [0.1.0] — Initial release

- Complete backgammon environment with exact rule enforcement
- 117 unit tests, all passing
- PPO self-play training (single process)
- BackgammonPolicyValueNet (candidate-action-scoring architecture)
- Evaluation with confidence intervals
- Move notation, position I/O, invariant checker
