"""Run TGraphXLearnedBoxFusion multi-seed against the existing graphs.pt."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir-with-graphs", required=True,
                    help="Run dir containing graphs.pt + source_labels.pt + split_manifest.json "
                         "(typically runs/real_voc_car_v2)")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda | auto (default cpu — model is small)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fusion-mode", default="residual",
                    choices=["residual", "weighted", "hybrid"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    from od_graph_fusion.multi_seed_learned_fusion import run_multi_seed_learned_fusion
    run_multi_seed_learned_fusion(
        args.config, seeds=args.seeds, out_dir=args.out_dir,
        run_dir_with_graphs=args.run_dir_with_graphs,
        fusion_mode=args.fusion_mode, epochs=args.epochs, device=args.device,
    )


if __name__ == "__main__":
    main()
