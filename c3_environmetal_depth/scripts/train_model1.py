"""Audit, fit on TRAIN, select with VALIDATION, freeze, then evaluate TEST once."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trainer import train_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model1_config.yaml")
    args = parser.parse_args()
    print(f"Completed run: {train_experiment(args.config)}")
