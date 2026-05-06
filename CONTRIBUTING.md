# Contributing to TGraphX

Thank you for your interest in TGraphX!  Contributions of all kinds are
welcome — bug reports, documentation improvements, new examples, and code.

## Before you start

- Read [docs/limitations.md](docs/limitations.md) to understand what is and
  is not in scope for the current release.
- Search [existing issues](https://github.com/arashsajjadi/TGraphX/issues)
  to avoid duplicates.
- For large changes, open a discussion or issue first.

## Development setup

```bash
git clone https://github.com/arashsajjadi/TGraphX.git
cd TGraphX

# Create environment
conda env create -f environment.yml   # or use pip
conda activate tgraphx

# Install in editable mode with dev dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                          # full suite
pytest -q                       # quiet mode
pytest tests/test_graph.py      # specific file
```

All tests must pass on CPU. GPU tests are skipped automatically on
CPU-only environments.

## Code style

- Python 3.9+, no f-strings with complex walrus operators.
- Type hints on all public functions.
- No bare `except:` — catch specific exceptions.
- Docstrings on all public functions and classes.
- No hardcoded device strings; use `device=` parameters.

## Adding a new feature

1. Open an issue to discuss the design.
2. Fork and create a feature branch: `git checkout -b feat/my-feature`.
3. Write tests in `tests/` that fail before your change.
4. Implement the feature.
5. Ensure all existing tests still pass: `pytest -q`.
6. Add a docstring and update `docs/` if appropriate.
7. Open a pull request referencing the issue.

## Adding an example

- Place new examples in `examples/`.
- Use synthetic data only (no real datasets, no internet).
- Must run fast on CPU (< 30 seconds) or skip gracefully when GPU is absent.
- No permanent file writes unless inside a `tempfile.TemporaryDirectory`.
- Add the example to `examples/run_all_fast_examples.py`.

## Claim control

Please do not add documentation or comments that overclaim:

- Do not claim `train_epoch / evaluate / fit` exist — they don't.
- Do not claim `TensorBoardLogger / MLflowLogger` exist — they don't.
- Do not claim universal `torch.compile` speedup.
- Do not claim full AMP support for all devices and layers.
- Do not claim GAT / SAGE / GIN chunking unless implemented.

## Pull request checklist

- [ ] All tests pass locally (`pytest -q`).
- [ ] New tests cover the change.
- [ ] Docstrings updated.
- [ ] `docs/` updated if the API changes.
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`.
- [ ] No overclaims in docstrings or docs.
- [ ] No hardcoded paths or device strings.

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).
Include the output of `python -c "from tgraphx.performance import env_report; print(env_report())"`.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
