## Summary

Briefly describe what this PR changes and why.

Fixes # (issue number, if applicable)

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation / example
- [ ] Refactor / cleanup
- [ ] CI / tooling

## Checklist

- [ ] `pytest -q` passes locally.
- [ ] New tests cover the change.
- [ ] Docstrings are updated for any changed public API.
- [ ] `docs/` is updated if the user-facing API changes.
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`.
- [ ] No overclaims in docstrings or docs (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- [ ] New examples use synthetic data and run fast on CPU.
- [ ] No permanent file writes in examples or tests without explicit paths.

## Test results

Paste `pytest -q` tail:

```
N passed, M skipped in X.Xs
```

## Claim control (if docs/examples were added)

Confirm that the following are **not** claimed to exist (unless they do):
- [ ] `train_epoch`, `evaluate`, `fit`
- [ ] `TensorBoardLogger`, `MLflowLogger`
- [ ] Universal `torch.compile` speedup
- [ ] Full AMP support across all devices and layers
- [ ] GAT / SAGE / GIN chunked forward
