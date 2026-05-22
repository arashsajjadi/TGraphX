"""Run a fast debug training session (200 games, CPU only)."""
import sys
sys.path.insert(0, "src")

from backgammon_rlx.train.train_ppo import main
sys.argv = ["train_ppo.py", "--config", "configs/fast_debug.yaml"]
main()
