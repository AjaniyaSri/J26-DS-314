"""Once-only descriptive assessment of a frozen v2 run; otherwise show saved results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_loader import configure_logging, local_path
from src.model1_diagnostics import DEFAULT_RUN, evaluate_revised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    args = parser.parse_args()
    base = local_path(args.run_dir)
    if (base / "models/test_evaluation_complete.json").exists():
        print("Previously inspected TEST: displaying saved descriptive results without rescoring.")
        print(pd.read_csv(base / "metrics/model1_test_descriptive_comparison.csv").to_string(index=False))
    else:
        configure_logging(base)
        print(evaluate_revised(base).to_string(index=False))
