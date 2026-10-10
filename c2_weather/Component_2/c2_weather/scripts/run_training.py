import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.preprocess import prepare_pipeline_data
from src.train_xgboost import train_all


def main():
    parser = argparse.ArgumentParser(description="Train the three C2 XGBoost experiments.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    _, splits, _, summary, _ = prepare_pipeline_data(config)
    results = train_all(splits, config)
    print(f"Trained variants: {', '.join(results)}")
    print(f"Training dataset summary: {summary['split_counts']}")


if __name__ == "__main__":
    main()