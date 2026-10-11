import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evaluate import run_evaluation
from src.explain_shap import explain_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate test predictions and generate XGBoost explanations.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics, _ = run_evaluation(config)
    explain_all(config)
    for result in metrics:
        print(
            f"{result['model']}: MAE={result['mae_m']:.4f} m, "
            f"RMSE={result['rmse_m']:.4f} m, R2={result['r2']}"
        )


if __name__ == "__main__":
    main()