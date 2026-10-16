"""Evaluate a frozen Model 2 run once, or display its existing descriptive results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_loader import local_path
from src.model_comparison import evaluate_model2, discover_model2_run
from src.model2_training import configure_model2_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Project-relative run; default is the latest successfully frozen Model 2 run.")
    args = parser.parse_args()
    base = local_path(args.run_dir) if args.run_dir else discover_model2_run(local_path("experiments/model2_dynamic_v1"))
    print(f"Model 2 run: {base}")
    if (base / "models/test_evaluation_complete.json").exists():
        print("Previously inspected project TEST: displaying saved results without rescoring.")
        print(pd.read_csv(base / "metrics/model1_vs_model2_metrics.csv").to_string(index=False))
    else:
        configure_model2_logging(base)
        print(evaluate_model2(base).to_string(index=False))
