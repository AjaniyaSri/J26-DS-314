"""Audit the exact Model 2 CSV without fitting models or changing input data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model2_data import load_model2_config, audit_to_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model2_config.yaml")
    args = parser.parse_args()
    path = audit_to_file(load_model2_config(args.config))
    print(path.read_text(encoding="utf-8"))
    print(f"Saved: {path}")
