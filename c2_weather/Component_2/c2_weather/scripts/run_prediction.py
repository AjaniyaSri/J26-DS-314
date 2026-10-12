import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.predict import predict_latest


def main():
    parser = argparse.ArgumentParser(description="Predict the next-day level for the latest available date.")
    parser.add_argument("variant", choices=("raw", "engineered", "hybrid"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    predictions = predict_latest(args.variant, load_config(args.config))
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()