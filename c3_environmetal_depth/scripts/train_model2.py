"""Fit Model 2 on TRAIN, select using VALIDATION, freeze; never score TEST here."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model2_training import train_model2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model2_config.yaml")
    args = parser.parse_args()
    print(f"Frozen Model 2 run (TEST not scored): {train_model2(args.config)}")
