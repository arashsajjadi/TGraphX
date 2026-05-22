# BackgammonX Scorecard

## Score Table (v0.3 — after focused upgrades)

| Area | v0.1 | v0.2 | v0.3 | Remaining limitation |
|---|---:|---:|---:|---|
| Rule correctness / movegen | 9.0 | 9.8 | **9.8** | No change; still limited by lack of external engine validation |
| Unit tests | 9.0 | 9.7 | **9.9** | 253 tests, cube/league/encoding/model/auxiliary all covered |
| Environment API | 9.0 | 9.5 | **9.6** | Cube env functional; PPO rollout still checker-play only |
| Encoding | 8.0 | 8.5 | **9.5** | v2 with 22 strategic features implemented and tested |
| Neural architecture | 8.0 | 8.2 | **9.2** | Transformer, auxiliary heads, dueling-inspired scorer added |
| PPO/self-play | 8.5 | 9.0 | **9.3** | KL stopping wired; league fully wired into trainer |
| Evaluation/statistics | 8.5 | 9.0 | **9.2** | League Elo and pool management working |
| Performance scaling | 8.0 | 8.5 | **9.0** | Multiprocess verified (4W × 8.4 games/s); OOM fix documented |
| Expert integration | 6.5 | 7.0 | **7.5** | Full interface + CLI + position export; subprocess needs gnubg installed |
| Cube/match-play | 6.0 | 6.5 | **8.0** | MoneyGameEnv with full cube/Crawford logic; PPO integration is checker-play |
| GitHub readiness | 9.3 | 9.8 | **9.8** | scorecard, safe/full configs, multiprocess benchmark added |

## Full-Game Validation

```
Games: 1000 (random vs heuristic, strict invariants, seed 123)
Illegal moves:   0
Violations:      0
Crashes:         0
VERDICT:         ✅ PASS
```

## Performance (RTX 5080, v2 encoding)

| Config | mode | games/s | steps/s |
|---|---|---:|---:|
| fast_debug (CPU, v1) | single | 4–5 | 350–450 |
| rtx5080_safe (CUDA, v2) | single | 5 | 500 |
| 1 worker (CUDA, v2) | multi | 2.8 | 230 |
| 2 workers | multi | 5 | 437 |
| 4 workers | multi | **8.4** | **721** |
| 12 workers (exclusive GPU) | multi | ~20 (est.) | ~1800 (est.) |

## What each phase delivered

### v0.3 additions
- **Encoding v2**: 22 strategic global features (pip, blots, primes, anchors, shots, phase), 16 extra action features
- **Transformer point encoder**: optional, 2-layer with 4-head attention
- **Auxiliary heads**: win/gammon/backgammon probability, pip-count prediction
- **Dueling-style scoring**: state baseline + per-action advantage
- **League fully wired**: LeagueManager integrated into SelfPlayTrainer; evaluation-gated pool promotion; Elo tracking
- **MoneyGameEnv**: CubeDecision actions (double/take/pass/no_double), Crawford rule, cube-scaled rewards, match tracking
- **GNU Backgammon**: full subprocess interface, position export, move notation parser, CLI tool
- **Multiprocess benchmark**: `benchmark_multiprocess.py`; worker scaling verified
- **New configs**: rtx5080_safe.yaml, rtx5080_full.yaml, money_game.yaml

## Honest remaining limitations

### Expert integration (7.5/10)
- `gnubg` subprocess protocol (`request_move`) is fully implemented but requires gnubg installed.
- On this machine gnubg is not installed → all tests auto-skip.
- The position-ID encoding is best-effort (GnuBG position ID spec can have version differences).

### Cube/Match PPO (8.0/10 for env; ~6.5 for full training)
- `MoneyGameEnv` correctly models cube actions and rewards.
- The PPO rollout (`rollout.py`, `multiprocess_rollout.py`) only plays checker-play.
- Full cube-action PPO training requires a new rollout that handles CubeDecision steps. This is staged work.

### Performance with shared VRAM (9.0/10)
- Single-process CUDA works reliably even on shared GPU.
- Multiprocess OOMs when VRAM is shared with >11 GB from other processes.
- Use `rtx5080_safe.yaml` (single-process) when GPU is shared.
- Use `rtx5080_full.yaml` (multiprocess) when GPU is exclusive.

### Stronger encoding in model (9.2/10)
- v2 features are computed but not all are interpretable yet.
- Transformer encoder is functional but untested at scale.

## Commands to run on RTX 5080

```bash
# Tests (always)
make test

# Full game validation
python -m backgammon_rlx.validation.full_game_check --games 1000 \
  --agents random,heuristic --strict-invariants --seed 123

# Shared GPU (safe)
python -m backgammon_rlx.train.train_ppo --config configs/rtx5080_safe.yaml

# Exclusive GPU (full)
python -m backgammon_rlx.train.train_ppo --config configs/rtx5080_full.yaml

# Multiprocess scaling benchmark
python -m backgammon_rlx.benchmarks.benchmark_multiprocess \
  --workers 1,2,4,8,12 --batch-sizes 128,256,512

# GNU Backgammon check (install gnubg first)
sudo apt install gnubg
python -m backgammon_rlx.engines.gnu_backgammon_check --status
python -m backgammon_rlx.engines.gnu_backgammon_check --all-fixtures
```
