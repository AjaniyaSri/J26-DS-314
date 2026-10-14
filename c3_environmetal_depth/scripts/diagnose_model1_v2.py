"""Run the approved eight-fit experiment and freeze selection without scoring TEST."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model1_diagnostics import DEFAULT_CONFIG, run_diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(f"Frozen experiment (TEST not evaluated): {run_diagnostics(args.config)}")
