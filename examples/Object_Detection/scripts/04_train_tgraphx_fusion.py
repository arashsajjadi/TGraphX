"""Step 04: Train TGraphX model only.

Reads:  {run_dir}/graphs.pt
Writes: {run_dir}/checkpoint.pt
        {run_dir}/training_history.json

Does NOT run detectors, build graphs, or evaluate.
Does NOT call run_pipeline.
Skips if checkpoint.pt exists unless --force.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description="Step 04: train TGraphX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch, random
    from od_graph_fusion.config import load_config, resolve_device, device_audit
    from od_graph_fusion.training import train_fusion_model

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{cfg.get('run_name', 'exp')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"checkpoint_seed{args.seed}.pt"
    graphs_path = run_dir / "graphs.pt"

    if not graphs_path.exists():
        raise FileNotFoundError(f"[04] Missing graphs.pt — run step 03 first: {graphs_path}")
    if ckpt_path.exists() and not args.force:
        print(f"[04] Checkpoint exists: {ckpt_path}  (--force to rerun)")
        return

    device_spec = args.device or cfg.get("device", "auto")
    device = resolve_device(device_spec)
    audit = device_audit(device_spec, device)
    print(f"[04] Device: {device}  GPU: {audit.get('gpu_name','N/A')}")

    torch.manual_seed(args.seed); random.seed(args.seed)
    all_graphs = torch.load(graphs_path, weights_only=False)
    # all_graphs entries are (graph, meta, image_id, split)
    # Fall back to 3-tuple if old format
    def _split_of(entry):
        return entry[3] if len(entry) == 4 else "train"

    # Use split_manifest if available, else split by record.split field
    manifest_path = run_dir / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        detector_names = manifest["detector_names"]
        class_names = manifest.get("class_names", [])
        n_classes = manifest.get("num_classes", len(class_names))
        train_ids = set(manifest["split_ids"]["train"])
        val_ids   = set(manifest["split_ids"]["val"])
        train_gs = [(g, meta) for g, meta, iid, *_ in all_graphs if iid in train_ids]
        val_gs   = [(g, meta) for g, meta, iid, *_ in all_graphs if iid in val_ids]
    else:
        # Fallback: use split tag from step 03 or naive index slice
        detector_names = list({k for g,_,*_ in all_graphs for k in (g.metadata.get("detector_names") or [])})
        n_classes = cfg.get("dataset", {}).get("num_classes", 20)
        train_gs = [(g, meta) for g, meta, iid, *rest in all_graphs if _split_of((g,meta,iid)+(tuple(rest))) == "train"]
        val_gs   = [(g, meta) for g, meta, iid, *rest in all_graphs if _split_of((g,meta,iid)+(tuple(rest))) == "val"]
        if not train_gs:  # old format: no split tag
            n = len(all_graphs); n_train = int(n*0.75); n_val = int(n*0.10)
            train_gs = [(g, meta) for g, meta, *_ in all_graphs[:n_train]]
            val_gs   = [(g, meta) for g, meta, *_ in all_graphs[n_train:n_train+n_val]]
        detector_names = detector_names or []

    num_detectors = len(detector_names)
    print(f"[04] Train: {len(train_gs)} | Val: {len(val_gs)} | detectors: {detector_names}")
    if num_detectors == 0:
        raise RuntimeError("[04] num_detectors=0 — detector_names missing from graphs. Run step 03 again with --force.")

    cfg_t = cfg.get("training", {}); cfg_m = cfg.get("model", {})
    crop_size = cfg.get("graph", {}).get("crop_size", 64)
    crop_channels = cfg_m.get("crop_channels", 16)
    hidden_dim = cfg_m.get("hidden_dim", 64)
    num_message_passing = cfg_m.get("num_message_passing", 2)

    is_multiclass = n_classes > 2
    utility_mode = cfg_t.get("utility_mode", "ap50")
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", not is_multiclass))
    t0 = time.time()
    model, history = train_fusion_model(
        train_gs, val_gs,
        num_classes=n_classes,
        num_detectors=num_detectors,
        crop_size=crop_size,
        crop_channels=crop_channels,
        hidden_dim=hidden_dim,
        num_message_passing=num_message_passing,
        epochs=cfg_t.get("epochs", 50),
        lr=float(cfg_t.get("lr", 5e-4)),
        weight_decay=float(cfg_t.get("weight_decay", 1e-4)),
        device=device,
        use_source_router=cfg.get("model", {}).get("use_source_router", True),
        detector_names=detector_names,
        utility_mode=utility_mode,
        class_agnostic=class_agnostic,
        strict_source_router=bool(cfg_t.get("strict_source_router", True)),
    )
    elapsed = time.time() - t0
    # Save model_config for exact reconstruction in Step 05
    model_config = {
        "num_classes": n_classes,
        "num_detectors": num_detectors,
        "detector_names": detector_names,
        "crop_size": crop_size,
        "crop_channels": crop_channels,
        "hidden_dim": hidden_dim,
        "num_message_passing": num_message_passing,
        "use_source_router": cfg.get("model", {}).get("use_source_router", True),
    }
    torch.save({"model_state": model.state_dict(), "model_config": model_config,
                "history": history, "seed": args.seed,
                "device": device, "elapsed_s": elapsed}, ckpt_path)
    history["elapsed_s"] = elapsed
    (run_dir / f"training_history_seed{args.seed}.json").write_text(json.dumps(history, indent=2, default=str))
    print(f"[04] Done. Checkpoint → {ckpt_path}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
