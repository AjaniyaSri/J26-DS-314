import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_path


def _architecture_diagram(output_path):
    figure, axis = plt.subplots(figsize=(12, 3.6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.3, "DMC levels\n+ CHIRPS / ERA5-Land", "#dcefe6"),
        (2.8, "Validate keys\n+ lag-safe features", "#e8edf3"),
        (5.3, "Chronological\ntrain / validation / test", "#fff0d8"),
        (7.8, "One-hot station\n+ XGBoost regressor", "#f3e2e2"),
        (10.3, "Metrics, plots\n+ SHAP evidence", "#e5e8f8"),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch((x, 1.3), 1.6, 1.3, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#39434d")
        axis.add_patch(patch)
        axis.text(x + 0.8, 1.95, label, ha="center", va="center", fontsize=9)
    for start in (1.95, 4.45, 6.95, 9.45):
        axis.annotate("", xy=(start + 0.75, 1.95), xytext=(start, 1.95), arrowprops={"arrowstyle": "->", "lw": 1.4})
    axis.text(6, 0.55, "LSTM: future work only", ha="center", fontsize=10, color="#5a2b2b")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate_evidence(config):
    evidence_dir = project_path(config, config["outputs"]["evidence_dir"])
    evidence_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = project_path(config, config["outputs"]["metrics_dir"])
    summary_path = metrics_dir / "training_dataset_summary.json"
    comparison_path = metrics_dir / "model_comparison.csv"
    quality_path = metrics_dir / "data_quality_report.json"
    if not all(path.is_file() for path in (summary_path, comparison_path, quality_path)):
        raise FileNotFoundError("Run validation, training, and evaluation before generating PP1 evidence")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(comparison_path)
    (evidence_dir / "feature_comparison.csv").write_text(
        pd.DataFrame(
            [
                {"variant": variant, "feature_columns": ", ".join(columns)}
                for variant, columns in summary["feature_columns"].items()
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )
    metrics.to_csv(evidence_dir / "model_performance_comparison.csv", index=False)
    _architecture_diagram(evidence_dir / "model_architecture.png")
    (evidence_dir / "reproducibility.md").write_text(
        """# Reproducibility\n\nFrom `c2_weather` in PowerShell:\n\n```powershell\npython -m pip install -r requirements.txt\npython scripts/run_validation.py\npython scripts/run_training.py\npython scripts/run_evaluation.py\npython scripts/generate_pp1_evidence.py\n```\n\nThe fixed random state is 42. Splits are chronological; validation selects the configured XGBoost candidate, and the test set is used only for final evaluation. Original CSVs remain outside the generated output directories.\n""",
        encoding="utf-8",
    )
    metric_lines = [
        f"- {row.model}: MAE {row.mae_m:.4f} m; RMSE {row.rmse_m:.4f} m; R2 {row.r2}; "
        f"beats persistence by MAE: {row.beats_persistence_mae}"
        for row in metrics.itertuples()
    ]
    (evidence_dir / "pp1_progress_summary.md").write_text(
        "# C2 PP1 Progress Summary\n\n"
        f"Common eligible station-date samples: {summary['common_eligible_samples']}.\n\n"
        f"Chronological split counts: {summary['split_counts']}.\n\n"
        "## Test results\n\n" + "\n".join(metric_lines) + "\n\n"
        "Results are limited to a short 39-day observation period and do not establish strong out-of-period or cross-station generalization.\n",
        encoding="utf-8",
    )
    (evidence_dir / "limitations_and_future_work.md").write_text(
        """# Limitations and Future Work\n\n- The source covers only 39 calendar days; generalization claims must remain modest.\n- 71 water-level cells are missing and are excluded whenever they affect current or next-day levels. No target interpolation is performed.\n- Weather observation and publication times are not present. Daily weather inputs are lagged, but source delivery latency and timezone still need verification.\n- The `water_level_9am_m` name declares metres; an authoritative unit/datum reference is still required.\n- CHIRPS/ERA5-Land provenance and exact rainfall aggregation conventions should be confirmed with source metadata.\n- LSTM is future work only and is not implemented or trained in this study.\n""",
        encoding="utf-8",
    )
    return evidence_dir


def main():
    parser = argparse.ArgumentParser(description="Generate the PP1 evidence bundle.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(generate_evidence(load_config(args.config)))


if __name__ == "__main__":
    main()