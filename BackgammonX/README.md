# BackgammonX

Research-grade self-play reinforcement learning for standard international backgammon.

## Project Goal

BackgammonX trains strong backgammon agents using PPO self-play with a candidate-action-scoring policy/value network.  The architecture is designed to scale on modern hardware (RTX 5080 + i7-14700F) while remaining clean enough to maintain and extend.

---

## Installation

```bash
git clone <repo>
cd BackgammonX
pip install -e ".[dev]"     # includes pytest
```

Python ≥ 3.10 required.  PyTorch ≥ 2.1 required for mixed-precision and `torch.compile` support.

---

## Quick Start

### Run tests (Stage 1 verification)
```bash
pytest
```

### Fast debug training (CPU, 200 games)
```bash
python -m backgammon_rlx.train.train_ppo --config configs/fast_debug.yaml
```

### Full training on RTX 5080
```bash
python -m backgammon_rlx.train.train_ppo --config configs/rtx5080.yaml
```

### Evaluate a checkpoint
```bash
python -m backgammon_rlx.train.evaluate \
    --checkpoint runs/<run_id>/checkpoints/latest.pt \
    --games 500
```

### Benchmark
```bash
python -m backgammon_rlx.benchmarks.benchmark_all --quick
```

### Imitation pretraining (optional)
```bash
python -m backgammon_rlx.train.pretrain_imitation \
    --config configs/imitation.yaml \
    --dataset data/expert_positions.jsonl
```

---

## Rules Enforcement

`src/backgammon_rlx/env/` implements complete standard backgammon rules:

| Rule | File | Function |
|---|---|---|
| Initial position | `state.py` | `GameState.initial()` |
| Point open/blocked | `rules.py` | `is_point_open()` |
| Bar priority | `movegen.py` | `get_legal_atomic_moves()` |
| Hitting blots | `movegen.py` | `AtomicMove.hit` |
| Mandatory dice usage | `movegen.py` | `get_legal_turns()` |
| Larger-die rule | `movegen.py` | post-filtering in `get_legal_turns()` |
| Bearing-off phase | `rules.py` | `can_bear_off()` |
| Exact + larger-die bear-off | `rules.py` | `can_bear_off_checker()` |
| Gammon / backgammon scoring | `rules.py` | `score_value()` |
| Strict invariant checker | `validation/invariants.py` | `check_state_invariants()` |

The legal move generator is exhaustive.  All mandatory-dice-usage branches, the larger-die rule, and bearing-off edge cases have unit tests.

---

## Architecture

### Observation encoding (298-d flat vector)
- **24 × 12** per-point features: own/opponent counts, blot/prime/occupied flags, position normalisation, board-zone indicators.
- **10** global features: bar counts, borne-off counts, dice values, pip counts.

### Action encoding (36-d flat vector per legal turn)
- Up to 4 atomic moves (padded), each with 7 features (src, dst, die, hit, bar-entry, bear-off, normal).
- 8 turn-level features: move count, hits, bear-offs, blots created, points made, pip gain, opponent bar after.

### Neural network (`BackgammonPolicyValueNet`)
```
Observation → PointEncoder (24-point residual MLP) + GlobalMLP
            → state_embedding [D]

Legal action i → ActionMLP
              → action_embedding [D]

Policy:  MLP(cat(state, action_i, state * action_i)) → logit_i
         softmax over all legal actions

Value:   MLP(state_embedding) → scalar
```

### PPO self-play
- Symmetric self-play: single shared network, board canonicalised for current player.
- GAE with alternating-perspective discounting (zero-sum sign flip between consecutive turns).
- Mixed precision (`torch.amp`), gradient clipping, AdamW.
- Terminal rewards: ±1/2/3 for normal/gammon/backgammon.

---

## Hardware Optimisation (RTX 5080 + i7-14700F)

| Component | Role |
|---|---|
| CPU | Game simulation, legal move generation, self-play rollout |
| GPU | Batched neural inference and PPO update |
| AMP | `torch.amp.autocast("cuda")` halves VRAM/bandwidth |
| `torch.compile` | Optional (set `compile_model: true` in config) |
| Minibatch size | 2048 in `rtx5080.yaml` to saturate GPU |
| `pinned_memory` | Can be added to DataLoader for faster CPU→GPU |

Benchmarks on a debug run (CPU): ~4000 move-gen positions/s, ~219K neural actions/s (GPU).

---

## Advanced Features

| Feature | Module |
|---|---|
| Expectimax search | `search/expectimax.py` |
| Checkpoint league + Elo | `rl/league.py` |
| Imitation pretraining | `train/pretrain_imitation.py` |
| Curriculum learning | `curriculum/` |
| Match state + doubling cube | `match/` (extensible stub) |
| External engine interface | `engines/external_engine.py` |
| GNU Backgammon adapter | `engines/gnu_backgammon.py` (stub) |
| Position JSON export/import | `notation/position_io.py` |
| Move notation | `notation/move_notation.py` |
| Statistical evaluation (CI) | `train/evaluate.py` |
| Performance benchmarks | `benchmarks/` |

---

## Project Status

**Implemented and tested:**
- Complete rule enforcement with 117 unit tests
- Exhaustive legal move generator (all mandatory-dice-usage branches)
- BackgammonEnv with strict invariant checker
- Observation and action encoders
- Random, greedy-pip, and heuristic baseline agents
- BackgammonPolicyValueNet (residual MLP, mixed precision)
- PPO self-play training loop
- Evaluation with binomial + bootstrap confidence intervals
- Checkpoint save/load with full reproducibility metadata
- Move notation and position I/O
- Match state and doubling cube (extensible stubs)
- Expectimax search (depth-1 and depth-2)
- Checkpoint league + Elo
- Curriculum learning scaffolding
- External engine adapter interface
- Benchmark suite

**Planned extensions:**
- GNU Backgammon adapter (requires gnubg)
- Expert dataset generation and imitation pretraining
- Multiprocess CPU self-play workers
- Numba/Rust accelerated move generator
- Full doubling cube and match-play training

---

## Running Tests

```bash
pytest                          # all 117 tests
pytest tests/test_movegen.py    # move-generation tests only
pytest tests/test_bearing_off.py
pytest -v --tb=long             # verbose with full tracebacks
```

---

## Limitations

- The self-play rollout is synchronous (single process).  Multiprocess workers would significantly increase throughput on the i7-14700F's 20 cores.
- The doubling cube API is stubbed; training ignores cube decisions.
- GNU Backgammon integration requires manual implementation of the position-ID protocol.
- No MCTS; the search module implements value-guided expectimax only.
