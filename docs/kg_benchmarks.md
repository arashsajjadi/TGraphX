# KG Benchmarks

Benchmark scripts for the KG subsystem live under `benchmarks/kg/`.
All support `--small` (fast CI mode) and `--json` (machine-readable output).

**No network access required.**  All scripts use synthetic data.

## Available benchmarks

### Negative sampling throughput

```bash
python benchmarks/kg/benchmark_kg_negative_sampling.py --small --json
```

Reports per-sampler: average runtime (ms), sampler type (uniform/bernoulli/filtered).

### Filtered ranking evaluation

```bash
python benchmarks/kg/benchmark_kg_filtered_eval.py --small --json
```

Reports: MR/MRR/Hits@K for a randomly-initialised DistMult model, chunk_size, runtime.

### TransE training + evaluation

```bash
python benchmarks/kg/benchmark_kg_transe.py --small --json
python benchmarks/kg/benchmark_kg_transe.py --num-entities 500 --num-triples 2000 --epochs 100
```

Reports: num_entities, num_triples, runtime, final loss, filtered MRR/Hits@K.

### DistMult training + evaluation

```bash
python benchmarks/kg/benchmark_kg_distmult.py --small --json
```

### ComplEx training + evaluation

```bash
python benchmarks/kg/benchmark_kg_complex.py --small --json
```

### RotatE training + evaluation

```bash
python benchmarks/kg/benchmark_kg_rotate.py --small --json
```

## Benchmark JSON schema

All scripts produce a JSON report with fields:

```json
{
  "task": "link_prediction",
  "model": "TransE",
  "package_version": "0.6.0",
  "seed": 0,
  "device": "cpu",
  "num_entities": 200,
  "num_train_triples": 350,
  "num_test_triples": 75,
  "epochs": 50,
  "runtime_s": 3.2,
  "final_loss": 0.85,
  "evaluation": {"filtered": {"combined": {"MRR": 0.15, "Hits@10": 0.42}}},
  "limitation_notes": [...]
}
```

## Limitations

- All benchmarks use synthetic data; no real-dataset performance claims are made.
- TransE/DistMult/ComplEx/RotatE are **not** benchmarked against reference implementations.
- Single-machine CPU benchmarks only; CUDA is available via `--device cuda`.
