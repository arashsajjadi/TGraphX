"""Maintenance tool (requires the external evidence tree; not a CI test).

1.5.1 release-gate parity check: evaluated experiment SetTransformer vs
packaged TGraphXSetAttention under the reference configuration.

Loads the completed frozen-base checkpoint (seed 0, best-val state) into BOTH
implementations, runs them in eval mode on one fixed real PASTIS-R validation
batch, and compares every stage: encoder output, per-block attention outputs,
pooled representation, logits, predicted labels.  Then evaluates BOTH models
on the FULL validation split (GPU) and compares predictions + macro-F1 against
the recorded raw result.

Read-only with respect to TGraphX_revised and 'TGraphX_final exp'.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRATCH = Path.cwd()  # JSON report written to the invocation directory
WRITE_ROOT = Path("/home/arash/PycharmProjects/_families/TGraphX/TGraphX_revised")
PKG_ROOT = Path("/home/arash/PycharmProjects/_families/TGraphX/TGraphX")

# Working-tree tgraphx (1.5.1 candidate) must win over any installed copy.
sys.path.insert(0, str(PKG_ROOT))
for p in (WRITE_ROOT / "shared/src", WRITE_ROOT / "shared/statistics"):
    sys.path.insert(1, str(p))

import torch  # noqa: E402
import tgraphx  # noqa: E402
from tgraphx import TGraphXSetAttention, SetTransformerModel  # noqa: E402
from tgraphx.core.dataloader import GraphDataLoader, GraphDataset  # noqa: E402
from models.set_transformer import SetTransformer  # experiment class  # noqa: E402
from data_prep import split_frames, build_graph_dataset  # noqa: E402
from stats import macro_f1  # noqa: E402

CKPT = WRITE_ROOT / "checkpoints/frozen_base/set_transformer_s0.pt"
RAW = WRITE_ROOT / "01_frozen_base_revised/raw_results/runs/set_transformer_s0.json"
IN_SHAPE, NUM_CLASSES = (13, 32, 32), 18

report: dict = {"tgraphx_path": str(Path(tgraphx.__file__).resolve()),
                "tgraphx_version": tgraphx.__version__,
                "torch_version": torch.__version__,
                "checkpoint": str(CKPT), "state_used": "best.state (epoch 8)"}


def maxmean(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = (a.double() - b.double()).abs()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item()}


def main() -> None:
    assert SetTransformerModel is TGraphXSetAttention
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck["best"]["state"]

    exp = SetTransformer(IN_SHAPE[0], NUM_CLASSES)
    r = exp.load_state_dict(state, strict=True)
    assert not r.missing_keys and not r.unexpected_keys
    new = TGraphXSetAttention.from_reference_state_dict(
        state, in_shape=IN_SHAPE, num_classes=NUM_CLASSES)
    exp.eval(); new.eval()

    n_exp = sum(p.numel() for p in exp.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    report["param_count"] = {"experiment": n_exp, "packaged_reference": n_new,
                             "equal": n_exp == n_new}

    # every mapped parameter/buffer bitwise identical after load
    mapped = TGraphXSetAttention.map_reference_state_dict(state)
    new_sd = new.state_dict()
    exact = all(torch.equal(new_sd[k], v) for k, v in mapped.items())
    report["state_dict"] = {"n_tensors": len(mapped),
                            "all_mapped_tensors_bitwise_equal": exact}

    # ---- fixed real validation batch --------------------------------------
    frames = split_frames()
    val_ds = build_graph_dataset(frames["val"])
    loader = GraphDataLoader(GraphDataset(val_ds), batch_size=64,
                             shuffle=False, num_workers=0)
    gb = next(iter(loader))
    x, batch = gb.node_features, gb.batch
    report["fixed_batch"] = {"num_graphs": int(batch.max()) + 1,
                             "num_nodes": int(x.shape[0]),
                             "node_shape": list(x.shape[1:])}

    with torch.no_grad():
        # stage 1: encoder
        h_exp = exp.encoder(x)
        h_new = new.encoder(x)
        report["encoder_output"] = maxmean(h_exp, h_new)

        # stage 2+: identical dense batching then blocks
        from models.common import to_dense_batch
        num_graphs = int(batch.max().item()) + 1
        padded, mask = to_dense_batch(h_exp, batch, num_graphs)
        enc_exp = exp.self_attn(padded, src_key_padding_mask=~mask)
        tokens = padded
        per_block = []
        for blk in new.blocks:
            tokens = blk(tokens, key_padding_mask=~mask)
            per_block.append(tokens)
        report["attention_output"] = maxmean(enc_exp, tokens)

        # stage 3: pooled representation
        pooled_exp = exp.pma(enc_exp, key_padding_mask=~mask)
        pooled_new = new.pool(tokens, key_padding_mask=~mask)
        report["pooled_output"] = maxmean(pooled_exp, pooled_new)

        # stage 4: full-model logits + predictions on the fixed batch
        logits_exp = exp(x, gb.edge_index, batch=batch)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # TopologyIgnoredWarning expected
            logits_new = new(x, edge_index=gb.edge_index, batch=batch)
        report["logits"] = maxmean(logits_exp, logits_new)
        report["logits_bitwise_equal"] = bool(torch.equal(logits_exp, logits_new))
        pred_exp = logits_exp.argmax(-1)
        pred_new = logits_new.argmax(-1)
        report["predictions_identical_fixed_batch"] = bool(
            torch.equal(pred_exp, pred_new))

    # ---- full validation split, GPU ---------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp.to(device); new.to(device)
    preds_e, preds_n, labels = [], [], []
    loader = GraphDataLoader(GraphDataset(val_ds), batch_size=64,
                             shuffle=False, num_workers=8)
    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gb in loader:
            xb = gb.node_features.to(device)
            bb = gb.batch.to(device)
            le = exp(xb, gb.edge_index.to(device), batch=bb)
            ln = new(xb, edge_index=None, batch=bb)
            preds_e.append(le.argmax(-1).cpu())
            preds_n.append(ln.argmax(-1).cpu())
            labels.append(gb.graph_labels)
    pe = torch.cat(preds_e).numpy(); pn = torch.cat(preds_n).numpy()
    ys = torch.cat(labels).numpy()
    f1_e = macro_f1(pe, ys, NUM_CLASSES)
    f1_n = macro_f1(pn, ys, NUM_CLASSES)
    recorded = json.loads(RAW.read_text())["best_val"]["macro_f1"]
    report["full_val"] = {
        "device": device, "n_samples": int(len(ys)),
        "predictions_identical": bool((pe == pn).all()),
        "macro_f1_experiment": f1_e, "macro_f1_packaged": f1_n,
        "macro_f1_recorded_raw_result": recorded,
        "matches_recorded": abs(f1_e - recorded) < 1e-9 and abs(f1_n - recorded) < 1e-9,
    }

    out = SCRATCH / "parity_gate_151.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
