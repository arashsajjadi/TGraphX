"""Public-dataset benchmark scripts for TGraphX (v0.3.2 foundation).

Each script in this package follows a uniform CLI:

    --root             cache directory for the upstream dataset
    --download         allow network download (default: off)
    --max-samples      cap on graph-level samples
    --max-nodes        cap on node-level slices
    --epochs           number of training epochs for the smoke run
    --device           cpu / cuda / mps / auto
    --output-dir       where to write benchmark JSON artefacts
    --seed             RNG seed
    --json             write a machine-readable summary alongside artefacts
    --strict           hard-fail (exit 2) when an optional dependency is missing

Each script writes (at minimum):

    benchmark_results.json
    run_metadata.json
    dataset_metadata.json
    metrics_summary.json

These artefacts are read by the local TGraphX dashboard.

Honesty constraints
-------------------
- These scripts are **not** leaderboard runs.  They report small-data
  engineering metrics (loss decrease, training-time, node/edge counts).
- No script downloads anything unless ``--download`` is passed
  explicitly.
- Optional dependencies (PyG, OGB, DGL) are imported lazily; missing
  dependencies produce an actionable skip (or an error under
  ``--strict``).
- No SOTA, no superiority, no benchmark wins.
"""
