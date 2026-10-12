from __future__ import annotations

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor

from .config import project_path
from .validate_data import DATASET_FILES, load_dataset


def _latest_features(frame, variant, config):
    station_col = config["data"]["station_column"]
    date_col = config["data"]["date_column"]
    level_col = config["data"]["level_column"]
    raw_weather = config["features"]["raw_daily_weather"]
    windows = config["features"]["rainfall_windows"]
    lag_days = int(config["features"]["daily_weather_lag_days"])
    frame = frame.sort_values([station_col, date_col]).copy()
    grouped = frame.groupby(station_col, sort=False)
    frame["water_level_current_m"] = frame[level_col]
    for days in config["features"]["water_level_lags_days"]:
        lagged_level = grouped[level_col].shift(days)
        source_date = grouped[date_col].shift(days)
        frame[f"water_level_lag_{days}d_m"] = lagged_level.where(
            source_date.eq(frame[date_col] - pd.Timedelta(days=days))
        )
    daily_columns = raw_weather if variant in ("raw", "hybrid") else raw_weather[1:]
    for column in daily_columns:
        lagged = grouped[column].shift(lag_days)
        source_date = grouped[date_col].shift(lag_days)
        frame[f"{column}_lag_{lag_days}d"] = lagged.where(
            source_date.eq(frame[date_col] - pd.Timedelta(days=lag_days))
        )
    if variant in ("engineered", "hybrid"):
        for column in config["features"]["rainfall_windows"]:
            lagged = grouped[column].shift(lag_days)
            source_date = grouped[date_col].shift(lag_days)
            frame[f"{column}_lag_{lag_days}d"] = lagged.where(
                source_date.eq(frame[date_col] - pd.Timedelta(days=lag_days))
            )
    frame["station"] = frame[station_col].astype("string")
    frame["date"] = frame[date_col]
    latest_date = frame[date_col].max()
    return frame.loc[frame[date_col].eq(latest_date) & frame[level_col].notna()].copy()


def predict_latest(variant, config):
    if variant not in ("raw", "engineered", "hybrid"):
        raise ValueError(f"Unknown experiment variant: {variant}")
    datasets = {
        name: load_dataset(project_path(config, config["data"][config_key]), config)
        for name, config_key in DATASET_FILES.items()
    }
    rows = _latest_features(datasets[variant], variant, config)
    if variant in ("engineered", "hybrid"):
        expected_windows = {
            f"{column}_lag_{config['features']['daily_weather_lag_days']}d"
            for column in config["features"]["rainfall_windows"]
        }
        missing = expected_windows - set(rows.columns)
        if missing:
            raise ValueError(f"{variant} data missing engineered features: {sorted(missing)}")
    if rows.empty:
        raise ValueError("No station has a usable current 9 AM water level on the latest date")

    model_dir = project_path(config, config["outputs"]["model_dir"]) / variant
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = XGBRegressor()
    model.load_model(model_dir / "model.json")
    input_frame = rows[["station"] + metadata["feature_columns"]]
    matrix = preprocessor.transform(input_frame)
    matrix = pd.DataFrame(matrix, columns=metadata["transformed_feature_columns"])
    predictions = rows[["station", "date", "water_level_current_m"]].copy()
    predictions["target_date"] = predictions["date"] + pd.Timedelta(days=1)
    predictions["predicted_water_level_m"] = model.predict(matrix)
    predictions["model"] = f"xgboost_{variant}"
    output_dir = project_path(config, config["outputs"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / f"next_day_{variant}_predictions.csv", index=False)
    return predictions