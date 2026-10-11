"""Report figures with explicit units and a noninteractive local backend."""
import os
from pathlib import Path

from src.data_loader import ROOT, write_csv_new

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET


def save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_figures(base, test_predictions, table, station_table, model) -> None:
    directory = base / "results/figures"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 6))
    y = test_predictions[TARGET]
    prediction = test_predictions.predicted_stage_next_day_m
    ax.scatter(y, prediction, s=18, alpha=0.6)
    limits = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
    ax.plot(limits, limits, color="black", linestyle="--", label="Perfect prediction")
    ax.set(xlabel="Observed next-day gauge stage (m)", ylabel="Predicted next-day gauge stage (m)",
           title=f"Model 1: final TEST ({len(y)} station-days)")
    ax.legend()
    save(fig, directory / "model1_actual_vs_predicted.png")

    subset = station_table[station_table.chronological_split.eq("TEST")]
    pivot = subset.pivot(index="station_name", columns="model", values="mae_m").sort_values("lightgbm_model1")
    fig, ax = plt.subplots(figsize=(10, 12))
    pivot.rename(columns={"lightgbm_model1": "LightGBM", "persistence": "Persistence"}).plot.barh(ax=ax)
    ax.set(xlabel="Station MAE (m)", ylabel="Gauge", title="Final TEST: per-station comparison")
    save(fig, directory / "model1_station_mae.png")

    importance = pd.DataFrame({"feature": FEATURES,
                               "gain": model.booster_.feature_importance(importance_type="gain"),
                               "split_count": model.booster_.feature_importance(importance_type="split")})
    write_csv_new(base / "results/metrics/model1_feature_importance.csv", importance)
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = importance.sort_values("gain")
    ax.barh(ordered.feature, ordered.gain)
    ax.set(xlabel="Total training split gain (association, not causation)",
           title="Model 1: LightGBM feature importance")
    save(fig, directory / "model1_feature_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("mae_m", "rmse_m"), ("MAE", "RMSE")):
        values = table.pivot(index="split", columns="model", values=metric).reindex(["VALIDATION", "TEST"])
        values.rename(columns={"lightgbm_model1": "LightGBM", "persistence": "Persistence"}).plot.bar(ax=ax, rot=0)
        ax.set(xlabel="Split", ylabel=f"{title} (m)", title=f"{title}: lower is better")
    fig.tight_layout()
    save(fig, directory / "model1_persistence_comparison.png")
