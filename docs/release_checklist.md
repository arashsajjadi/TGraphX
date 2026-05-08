# Release checklist

Use this checklist before tagging a new TGraphX release.

## Pre-flight

- [ ] `git status` — confirm changes are intentional and no data files are staged
- [ ] `grep -n 'version = ' pyproject.toml` and `grep __version__ tgraphx/__init__.py` — both match intended version
- [ ] `git tag --list | grep v<MAJOR>.<MINOR>` — confirm the new tag does not already exist

## Local tests

```bash
python -m pip install -e .
pytest -q                                           # must pass
python examples/run_all_fast_examples.py            # must all pass
```

## Build and packaging

```bash
rm -rf dist build *.egg-info
python -m build
twine check dist/*                                  # both must say PASSED
# Data-safety check (no .npy/.csv/.h5 etc. in wheel):
python - <<'PY'
from pathlib import Path
import zipfile
w = sorted(Path("dist").glob("*.whl"))[-1]
bad = {".npy",".npz",".h5",".hdf5",".csv",".zip",".tar",".gz"}
with zipfile.ZipFile(w) as z:
    oops = [n for n in z.namelist() if Path(n).suffix.lower() in bad and "dashboard/static" not in n]
    assert not oops, oops
print("Wheel data safety OK:", w)
PY
```

## Wheel-install smoke (outside repo)

```bash
python -m venv /tmp/tgx_release_smoke --clear
/tmp/tgx_release_smoke/bin/pip install --quiet torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
/tmp/tgx_release_smoke/bin/pip install --quiet dist/tgraphx-*.whl
cd /tmp && /tmp/tgx_release_smoke/bin/python - <<'PY'
import tgraphx, tgraphx.datasets, tgraphx.transforms, tgraphx.metrics
import tgraphx.experiments, tgraphx.explain
import sys
for m in ["torch_geometric","dgl","ogb"]:
    assert m not in sys.modules, m
from tgraphx.datasets import get_dataset
ds = get_dataset("synthetic:patch_graph", num_graphs=2, seed=0)
print("wheel smoke OK, version:", tgraphx.__version__)
PY
cd -
rm -rf /tmp/tgx_release_smoke
```

## Core validation scripts

```bash
# All must exit 0
python examples/dashboard_artifact_validation.py
python examples/experiment_end_to_end_validation.py
python examples/explainability_end_to_end_validation.py
python examples/device_validation.py --device cpu --quick \
    --output-json /tmp/tgraphx_cpu_validation.json
```

## Optional: CUDA local validation (requires GPU)

```bash
python examples/device_validation.py --device cuda --amp \
    --output-json cuda_validation.json
# Attach cuda_validation.json to GitHub release notes.
```

## Optional: FakeData public dataset smoke (CI-safe, no download)

```bash
python examples/public_datasets/fake_torchvision_patch_smoke.py --epochs 2
```

## Optional: real public dataset validation (requires --download, manual)

```bash
python examples/public_datasets/mnist_patch_smoke.py \
    --download --max-samples 100 --epochs 3 \
    --output-run-dir runs/validation_mnist_patch
python examples/public_datasets/pyg_cora_smoke.py \
    --download --epochs 3 \
    --output-run-dir runs/validation_pyg_cora
# etc.
```

## README honesty check

```bash
python - <<'PY'
from pathlib import Path
text = Path("README.md").read_text(encoding="utf-8")
bad = ["⚠️","❌","⛔","⏳","🧪","🚫"]
assert not [s for s in bad if s in text], "Scary symbols in README"
text_lower = text.lower()
assert "cuda ci" not in text_lower
assert "full mps" not in text_lower
assert "SOTA" not in text
assert "state-of-the-art" not in text_lower
print("README claim smoke OK")
PY
```

## Git operations

Only perform these AFTER CI is green.

```bash
export GIT_PAGER=cat GH_PAGER=cat PAGER=cat

# Stage all intentional v0.3.0 changes
git add tgraphx/ tests/ examples/ docs/ \
        README.md CHANGELOG.md pyproject.toml

git commit -m "Prepare TGraphX v0.3.0"
git push origin main

# Wait for CI without blocking a pager
gh run watch --exit-status      # blocks until green or fails; Ctrl-C to abort

# Tag only after CI green
git tag -a v0.3.0 -m "TGraphX v0.3.0"
git push origin v0.3.0

# Publish
twine upload dist/tgraphx-0.3.0*
```

## Post-release

- [ ] Verify the PyPI page renders correctly (`pip install tgraphx==0.3.0`).
- [ ] Test that a fresh install imports correctly.
- [ ] Close any `v0.3.0` milestone issues.
- [ ] Open `v0.3.1` milestone for bugfixes.
