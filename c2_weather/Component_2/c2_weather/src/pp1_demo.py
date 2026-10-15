from __future__ import annotations

import argparse
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import DEFAULT_CONFIG_PATH, load_config, project_path


VARIANTS = ("raw", "engineered", "hybrid")
KEY_COLUMNS = ["station", "date", "target_date"]


def _test_bounds(config):
    return pd.Timestamp(config["splits"]["test_start"]), pd.Timestamp(config["splits"]["test_end"])


@lru_cache(maxsize=4)
def load_demo_records(config_path=None):
    config = load_config(config_path or DEFAULT_CONFIG_PATH)
    lower, upper = _test_bounds(config)
    records = {}
    for variant in VARIANTS:
        sample_path = project_path(config, config["outputs"]["processed_dir"]) / f"{variant}_samples.csv"
        prediction_path = (
            project_path(config, config["outputs"]["predictions_dir"])
            / f"xgboost_{variant}_test_predictions.csv"
        )
        samples = pd.read_csv(sample_path, parse_dates=["date", "target_date"])
        predictions = pd.read_csv(prediction_path, parse_dates=["date", "target_date"])
        samples = samples.loc[samples["date"].between(lower, upper)].copy()
        if not samples[KEY_COLUMNS].equals(predictions[KEY_COLUMNS]):
            samples = samples.sort_values(KEY_COLUMNS).reset_index(drop=True)
            predictions = predictions.sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not samples[KEY_COLUMNS].equals(predictions[KEY_COLUMNS]):
            raise AssertionError(f"{variant} processed test rows do not align with saved predictions")
        if not np.allclose(
            samples["target_water_level_m"].to_numpy(float),
            predictions["actual_water_level_m"].to_numpy(float),
        ):
            raise AssertionError(f"{variant} test actuals do not align with saved predictions")
        selected = samples.copy().loc[:, lambda frame: ~frame.columns.duplicated()]
        selected[f"saved_{variant}_prediction_m"] = predictions[
            "predicted_water_level_m"
        ].to_numpy(float)
        records[variant] = selected

    reference = records["raw"].sort_values(KEY_COLUMNS).reset_index(drop=True)
    merged = reference.rename(columns={"target_water_level_m": "actual_water_level_m"})
    for variant in ("engineered", "hybrid"):
        other = records[variant].sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not other[KEY_COLUMNS].equals(reference[KEY_COLUMNS]):
            raise AssertionError("The original models do not share identical held-out test records")
        if not np.allclose(other["target_water_level_m"], merged["actual_water_level_m"]):
            raise AssertionError("The original models disagree on test actual values")
        if not np.allclose(other["water_level_current_m"], merged["water_level_current_m"]):
            raise AssertionError("The original models disagree on current test readings")
        model_dir = project_path(load_config(), "outputs")
        metadata_path = (
            model_dir.parent / "models" / "xgboost" / variant / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for feature in metadata["feature_columns"]:
            values = other[feature].to_numpy(dtype=float)
            if feature in merged.columns:
                existing = merged[feature].to_numpy(dtype=float)
                if not np.allclose(existing, values, equal_nan=True):
                    raise AssertionError(f"Shared {feature} values differ in {variant} test rows")
            else:
                merged[feature] = values
        merged[f"saved_{variant}_prediction_m"] = other[f"saved_{variant}_prediction_m"].to_numpy(float)
    merged["persistence_prediction_m"] = merged["water_level_current_m"].to_numpy(float)
    merged["date_label"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged.sort_values(["station", "date"]).reset_index(drop=True)


@lru_cache(maxsize=3)
def load_model_bundle(variant, config_path=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    config = load_config(config_path or DEFAULT_CONFIG_PATH)
    model_dir = project_path(config, config["outputs"]["model_dir"]) / variant
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("target") != "target_water_level_m":
        raise AssertionError(f"Unexpected target schema for {variant}")
    feature_columns = metadata["feature_columns"]
    if "target_water_level_m" in feature_columns or "target_date" in feature_columns:
        raise AssertionError(f"Future target information is included in {variant} model features")
    model = XGBRegressor()
    model.load_model(model_dir / "model.json")
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    return model, preprocessor, metadata


def predict_one(row, variant, config_path=None):
    """Infer one held-out record using features only; actual target is never passed to preprocessing."""
    model, preprocessor, metadata = load_model_bundle(variant, config_path)
    feature_columns = metadata["feature_columns"]
    missing = sorted(set(feature_columns).difference(row.index))
    if missing:
        raise ValueError(f"Processed row is missing {variant} features: {missing}")
    if "target_water_level_m" in feature_columns or "actual_water_level_m" in feature_columns:
        raise AssertionError("The next-day actual is prohibited from model inputs")
    feature_input = pd.DataFrame(
        [{"station": str(row["station"]), **{name: row[name] for name in feature_columns}}]
    )
    matrix = preprocessor.transform(feature_input[["station"] + feature_columns])
    matrix = pd.DataFrame(matrix, columns=metadata["transformed_feature_columns"])
    prediction = float(model.predict(matrix)[0])
    return prediction


def evaluate_record(record, config_path=None):
    result = {
        "station": str(record["station"]),
        "forecast_origin": pd.Timestamp(record["date"]),
        "target_date": pd.Timestamp(record["target_date"]),
        "current_water_level_m": float(record["water_level_current_m"]),
        "actual_next_day_water_level_m": float(record["actual_water_level_m"]),
        "predictions": {},
    }
    if result["target_date"] != result["forecast_origin"] + pd.Timedelta(days=1):
        raise AssertionError("Selected demo record is not exactly next-day aligned")
    actual = result["actual_next_day_water_level_m"]
    for variant in VARIANTS:
        prediction = predict_one(record, variant, config_path)
        saved = float(record[f"saved_{variant}_prediction_m"])
        result["predictions"][variant] = {
            "predicted_water_level_m": prediction,
            "absolute_error_m": abs(prediction - actual),
            "saved_prediction_m": saved,
            "difference_from_saved_m": prediction - saved,
            "matches_saved_prediction": bool(np.isclose(prediction, saved, rtol=1e-6, atol=1e-6)),
        }
    persistence = float(record["persistence_prediction_m"])
    result["predictions"]["persistence"] = {
        "predicted_water_level_m": persistence,
        "absolute_error_m": abs(persistence - actual),
        "saved_prediction_m": persistence,
        "difference_from_saved_m": 0.0,
        "matches_saved_prediction": True,
    }
    return result


def verify_saved_predictions(record_count=3, config_path=None):
    records = load_demo_records(config_path)
    if record_count < 1:
        raise ValueError("record_count must be positive")
    selected = records.head(record_count)
    if len(selected) < record_count:
        raise ValueError(f"Requested {record_count} rows but only {len(selected)} held-out records exist")
    checks = []
    for _, row in selected.iterrows():
        result = evaluate_record(row, config_path)
        for variant in VARIANTS:
            check = result["predictions"][variant]
            checks.append(
                {
                    "station": result["station"],
                    "forecast_origin": result["forecast_origin"].date().isoformat(),
                    "target_date": result["target_date"].date().isoformat(),
                    "model": variant,
                    "recomputed_prediction_m": check["predicted_water_level_m"],
                    "saved_prediction_m": check["saved_prediction_m"],
                    "absolute_difference_m": abs(check["difference_from_saved_m"]),
                    "within_tolerance": check["matches_saved_prediction"],
                }
            )
    return pd.DataFrame(checks)


def _main():
    parser = argparse.ArgumentParser(description="Verify saved XGBoost predictions on historical held-out records.")
    parser.add_argument("--verify", type=int, metavar="N", help="Recompute N held-out rows and compare with saved CSVs")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.verify is None:
        parser.error("Pass --verify N; launch the interactive app with: streamlit run scripts/run_pp1_live_demo.py")
    checks = verify_saved_predictions(args.verify, args.config)
    print(checks.to_string(index=False, float_format=lambda value: f"{value:.10f}"))
    failures = checks.loc[~checks["within_tolerance"]]
    print(f"\nCompared {len(checks)} model predictions across {args.verify} held-out records.")
    print(f"Mismatches beyond tolerance: {len(failures)}")
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()