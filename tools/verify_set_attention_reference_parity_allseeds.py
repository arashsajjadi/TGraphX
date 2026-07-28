"""Maintenance tool (requires the external evidence tree; not a CI test).

5-seed extension of the 1.5.1 parity gate: for every frozen-base seed,
strict-load the best-val checkpoint into BOTH implementations, evaluate the
FULL validation split, and require identical predictions + exact match with
the recorded per-seed raw macro-F1.  Also reproduces the 5-seed mean."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

SCRATCH = Path.cwd()  # JSON report written to the invocation directory
WRITE_ROOT = Path("/home/arash/PycharmProjects/_families/TGraphX/TGraphX_revised")
PKG_ROOT = Path("/home/arash/PycharmProjects/_families/TGraphX/TGraphX")
sys.path.insert(0, str(PKG_ROOT))
for p in (WRITE_ROOT / "shared/src", WRITE_ROOT / "shared/statistics"):
    sys.path.insert(1, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tgraphx  # noqa: E402
from tgraphx import TGraphXSetAttention  # noqa: E402
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset  # noqa: E402
from models.set_transformer import SetTransformer  # noqa: E402
from data_prep import split_frames, build_graph_dataset  # noqa: E402
from stats import macro_f1  # noqa: E402

IN_SHAPE, NUM_CLASSES = (13, 32, 32), 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

frames = split_frames()
val_ds = build_graph_dataset(frames["val"])

rows = []
for seed in range(5):
    ck = torch.load(WRITE_ROOT / f"checkpoints/frozen_base/set_transformer_s{seed}.pt",
                    map_location="cpu", weights_only=False)
    state = ck["best"]["state"]
    exp = SetTransformer(IN_SHAPE[0], NUM_CLASSES)
    r = exp.load_state_dict(state, strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    new = TGraphXSetAttention.from_reference_state_dict(
        state, in_shape=IN_SHAPE, num_classes=NUM_CLASSES)
    exp.eval().to(DEVICE); new.eval().to(DEVICE)

    preds_e, preds_n, labels = [], [], []
    loader = GraphDataLoader(GraphDataset(val_ds), batch_size=64,
                             shuffle=False, num_workers=8)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gb in loader:
            xb = gb.node_features.to(DEVICE)
            bb = gb.batch.to(DEVICE)
            preds_e.append(exp(xb, gb.edge_index.to(DEVICE), batch=bb).argmax(-1).cpu())
            preds_n.append(new(xb, edge_index=None, batch=bb).argmax(-1).cpu())
            labels.append(gb.graph_labels)
    pe = torch.cat(preds_e).numpy(); pn = torch.cat(preds_n).numpy()
    ys = torch.cat(labels).numpy()
    f1_e = macro_f1(pe, ys, NUM_CLASSES)
    f1_n = macro_f1(pn, ys, NUM_CLASSES)
    recorded = json.loads(
        (WRITE_ROOT / f"01_frozen_base_revised/raw_results/runs/set_transformer_s{seed}.json"
         ).read_text())["best_val"]["macro_f1"]
    rows.append({"seed": seed, "predictions_identical": bool((pe == pn).all()),
                 "macro_f1_experiment": f1_e, "macro_f1_packaged": f1_n,
                 "macro_f1_recorded": recorded,
                 "exact_match": f1_e == recorded == f1_n})
    print(rows[-1], flush=True)

f1s = [r["macro_f1_packaged"] for r in rows]
summary = {"tgraphx_version": tgraphx.__version__, "device": DEVICE,
           "per_seed": rows,
           "mean_macro_f1_packaged": float(np.mean(f1s)),
           "std_macro_f1_packaged": float(np.std(f1s, ddof=1)),
           "all_seeds_identical_predictions": all(r["predictions_identical"] for r in rows),
           "all_seeds_exact_recorded_match": all(r["exact_match"] for r in rows)}
(SCRATCH / "parity_gate_151_allseeds.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
