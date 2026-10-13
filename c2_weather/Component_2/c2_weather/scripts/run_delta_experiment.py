import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.delta_experiment import run_delta_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run the isolated, train/validation-only XGBoost Delta Regression experiment."
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    output_dir, summary = run_delta_experiment(load_config(args.config))
    print(f"Delta experiment outputs: {output_dir}")
    print(f"Selected model: {summary['selected_delta_model']}")
    print(
        "Persistence validation MAE: "
        f"{summary['persistence_validation_mae_m']:.6f} m; "
        "best Delta validation MAE: "
        f"{summary['best_delta_validation_mae_m']:.6f} m; "
        "improvement: "
        f"{summary['delta_improvement_vs_persistence_mae_m']:.6f} m"
    )
    print("Held-out test rows used: 0")


if __name__ == "__main__":
    main()