from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

from .config import project_path


def explain_variant(variant, config, max_rows=500):
    model_dir = project_path(config, config["outputs"]["model_dir"]) / variant
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model = XGBRegressor()
    model.load_model(model_dir / "model.json")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    feature_path = project_path(config, config["outputs"]["interim_dir"]) / f"{variant}_test_features.csv"
    raw_features = pd.read_csv(feature_path)
    matrix = preprocessor.transform(raw_features[["station"] + metadata["feature_columns"]])
    transformed = pd.DataFrame(matrix, columns=metadata["transformed_feature_columns"])
    sample = transformed.sample(n=min(max_rows, len(transformed)), random_state=config["project"]["random_state"])

    figures_dir = project_path(config, config["outputs"]["figures_dir"])
    metrics_dir = project_path(config, config["outputs"]["metrics_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{variant}_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{variant}_shap_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    importance = model.get_booster().get_score(importance_type="gain")
    importance_frame = pd.DataFrame(
        [
            {"feature": name, "gain": float(value)}
            for name, value in sorted(importance.items(), key=lambda item: item[1], reverse=True)
        ]
    )
    importance_frame.to_csv(metrics_dir / f"{variant}_feature_importance.csv", index=False)
    return importance_frame


def explain_all(config):
    return {variant: explain_variant(variant, config) for variant in ("raw", "engineered", "hybrid")}