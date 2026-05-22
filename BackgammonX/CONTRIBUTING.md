# Contributing to BackgammonX

## Priorities

1. **Rule correctness first** — any change to `env/movegen.py` or `env/rules.py` must be accompanied by new tests. Run `pytest tests/test_golden_legal_moves.py tests/test_movegen.py tests/test_bearing_off.py` before submitting.
2. **Do not break invariants** — `total_checkers(state, p) == 15` must hold after every move. Enable `strict_invariants=True` when debugging.
3. **Performance changes must be benchmarked** — run `make benchmark` before and after.

## Development setup

```bash
git clone <repo>
cd BackgammonX
pip install -e ".[dev]"
pytest -q   # must pass before any PR
```

## Adding golden test fixtures

Add JSON files to `tests/fixtures/legal_moves/` using the schema in `tests/fixtures/legal_moves/README.md`.

## Running the full test suite

```bash
make test          # 117+ unit tests
make test-stress   # random-game property tests
```

## Code style

- Python ≥ 3.10, type hints where useful.
- `ruff format` for formatting.
- No comments explaining *what* the code does; only *why* for non-obvious constraints.

## Submitting changes

1. Fork and create a branch.
2. Run `make test` — must pass.
3. Add tests for any new behavior.
4. Open a PR with a clear description.
