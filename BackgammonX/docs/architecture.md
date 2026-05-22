# BackgammonX Architecture

## Overview

```
backgammon_rlx/
├── env/           # Game environment (rules, movegen, API)
├── models/        # Neural network (policy/value)
├── rl/            # PPO, rollout, self-play, league
├── train/         # Training and evaluation entry points
├── agents/        # Baseline agents (random, heuristic, neural)
├── notation/      # Move notation and position I/O
├── validation/    # Invariant checker, golden tests, full-game check
├── engines/       # External engine adapters (GNU Backgammon)
├── search/        # Expectimax search
├── curriculum/    # Curriculum samplers and schedule
├── benchmarks/    # Performance measurement
├── data/          # Expert dataset generation
├── match/         # Cube state and match-play (staged)
└── utils/         # Seeds, logging, checkpointing, metadata
```

## Board Representation

- 24-element integer list: `board[i]` = checkers on point `i+1`
- Positive = Player 0 checkers; negative = Player 1 checkers
- `bar = [p0_count, p1_count]`
- `borne_off = [p0_count, p1_count]`
- Player 0 moves 24 → 1 → off; Player 1 moves 1 → 24 → off

## Move Representation

```
AtomicMove(src, dst, die, hit)
  src: 1-24 for board, BAR=0
  dst: 1-24 for board, OFF=25

Turn(moves: tuple[AtomicMove])
```

## Legal Move Generation

1. For each die in remaining dice (skip tried values)
2. Enumerate atomic moves (`get_legal_atomic_moves`)
3. Recurse with remaining dice on updated state
4. Post-process: filter to max moves, apply larger-die rule, deduplicate by final board state

Complexity: O(24 × dice) per level, depth ≤ 4. Fast in practice.

## Observation Encoding (v1)

- 24 × 12 per-point features + 10 global = 298-dimensional flat vector
- Always presented from current player's canonical perspective (board mirrored for P1)

## Action Encoding (v1)

- 4 × 7 atomic move features + 8 turn-level = 36-dimensional vector per legal action

## Neural Network (BackgammonPolicyValueNet)

```
Observation → PointEncoder (24 points, residual MLP) + GlobalMLP → state_emb [D]
Legal action_i → ActionMLP (residual) → act_emb_i [D]

Policy logit_i = MLP(cat(state_emb, act_emb_i, state_emb * act_emb_i))
Value = MLP(state_emb)
```

- SiLU activations, LayerNorm, orthogonal init
- AMP mixed precision (`torch.amp.autocast("cuda")`)

## PPO Self-Play

- Symmetric self-play: single shared model, board always canonical
- Alternating-perspective GAE: `R[t] = r[t] - γ * R[t+1]` (zero-sum sign flip)
- Terminal rewards: ±1/2/3 for normal/gammon/backgammon
- KL early stopping: configurable via `target_kl`

## Multiprocess Architecture (Phase 2)

```
Workers (N processes, CPU):
  game simulation → encode obs/actions → send to req_queue

Inference Thread (main process, GPU):
  batch inference → send responses

Main Process:
  collect trajectories → PPO update
```

## Key Files for Correctness

- `env/rules.py` — all rule checks
- `env/movegen.py` — exhaustive legal move generation (**do not modify without full test suite**)
- `env/state.py` — data structures
