"""Evaluate a frozen unevaluated run, or display existing final results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_loader import local_path
from src.evaluator import evaluate_frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=".", help="Project-relative run directory; first run is .")
    args = parser.parse_args()
    base = local_path(args.run_dir)
    complete = base / "models/model1_static/test_evaluation_complete.json"
    if complete.exists():
        print("Final TEST was already evaluated. Displaying saved metrics without rescoring.")
        for split in ("validation", "test"):
            print(pd.read_csv(base / f"results/metrics/model1_{split}_metrics.csv").to_string(index=False))
    else:
        print(evaluate_frozen(base).to_string(index=False))
