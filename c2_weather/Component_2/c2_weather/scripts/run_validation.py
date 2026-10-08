import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.preprocess import prepare_pipeline_data


def main():
    parser = argparse.ArgumentParser(description="Validate C2 source datasets and prepare aligned samples.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    frames, splits, reports, summary, report_path = prepare_pipeline_data(config)
    print(f"Data quality report: {report_path}")
    print(f"Common eligible samples: {summary['common_eligible_samples']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Variants: {', '.join(frames)}")
    print(f"Validation reports: {len(reports)}; split groups: {len(splits)}")


if __name__ == "__main__":
    main()