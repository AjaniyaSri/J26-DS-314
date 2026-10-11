from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import project_path


KEY_COLUMNS = ["station", "date", "target_date"]


def validate_identical_evaluation_samples(predictions):
    reference = None
    for name, frame in predictions.items():
        keys = pd.MultiIndex.from_frame(frame[KEY_COLUMNS]).sort_values()
        if reference is None:
            reference = keys
        elif not reference.equals(keys):
            raise AssertionError(f"{name} uses different test samples")


def prediction_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Actual and predicted arrays must be matching one-dimensional shapes")
    result = {
        "n": int(len(actual)),
        "mae_m": float(mean_absolute_error(actual, predicted)),
        "rmse_m": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 and np.var(actual) > 0 else None,
    }
    return result


def evaluate_prediction_frames(predictions, config):
    validate_identical_evaluation_samples(predictions)
    baseline = predictions["persistence"]["predicted_water_level_m"].to_numpy()
    actual = predictions["persistence"]["actual_water_level_m"].to_numpy()
    metrics = []
    station_rows = []
    for name, frame in predictions.items():
        values = prediction_metrics(actual, frame["predicted_water_level_m"])
        values["model"] = name
        values["beats_persistence_mae"] = (
            values["mae_m"] < prediction_metrics(actual, baseline)["mae_m"]
            if name != "persistence"
            else None
        )
        metrics.append(values)
        for station, group in frame.groupby("station"):
            station_rows.append(
                {
                    "model": name,
                    "station": station,
                    "n": int(len(group)),
                    "mae_m": float(
                        mean_absolute_error(
                            group["actual_water_level_m"], group["predicted_water_level_m"]
                        )
                    ),
                }
            )

    metrics_dir = project_path(config, config["outputs"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_dir / "model_comparison.csv", index=False)
    pd.DataFrame(station_rows).to_csv(metrics_dir / "per_station_mae.csv", index=False)
    (metrics_dir / "model_comparison.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, pd.DataFrame(station_rows)


def load_prediction_frames(config):
    prediction_dir = project_path(config, config["outputs"]["predictions_dir"])
    frames = {
        name: pd.read_csv(prediction_dir / f"xgboost_{name}_test_predictions.csv", parse_dates=["date", "target_date"])
        for name in ("raw", "engineered", "hybrid")
    }
    sample = frames["raw"]
    baseline = sample[
        [*KEY_COLUMNS, "actual_water_level_m", "water_level_current_m"]
    ].copy()
    baseline["predicted_water_level_m"] = baseline["water_level_current_m"]
    frames["persistence"] = baseline
    baseline.to_csv(prediction_dir / "persistence_test_predictions.csv", index=False)
    return frames


def make_evaluation_figures(predictions, config):
    figures_dir = project_path(config, config["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in predictions.items():
        actual = frame["actual_water_level_m"].to_numpy()
        predicted = frame["predicted_water_level_m"].to_numpy()
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(actual, predicted, alpha=0.65, edgecolor="none")
        minimum = min(actual.min(), predicted.min())
        maximum = max(actual.max(), predicted.max())
        axis.plot([minimum, maximum], [minimum, maximum], color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Actual water level (m)", ylabel="Predicted water level (m)", title=f"{name}: actual vs predicted")
        figure.tight_layout()
        figure.savefig(figures_dir / f"{name}_actual_vs_predicted.png", dpi=160)
        plt.close(figure)

        residuals = predicted - actual
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(predicted, residuals, alpha=0.65, edgecolor="none")
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="Predicted water level (m)", ylabel="Residual (predicted - actual, m)", title=f"{name}: residuals")
        figure.tight_layout()
        figure.savefig(figures_dir / f"{name}_residuals.png", dpi=160)
        plt.close(figure)


def run_evaluation(config):
    predictions = load_prediction_frames(config)
    metrics, station_metrics = evaluate_prediction_frames(predictions, config)
    make_evaluation_figures(predictions, config)
    return metrics, station_metrics