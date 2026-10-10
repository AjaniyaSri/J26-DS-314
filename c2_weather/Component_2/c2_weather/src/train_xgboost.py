from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import project_path


IDENTIFIER_COLUMNS = {"station", "date", "target_date", "target_water_level_m"}


def validate_prediction_shape(predictions, expected_rows):
    if predictions.shape != (expected_rows,):
        raise AssertionError(
            f"Expected prediction shape {(expected_rows,)}, received {predictions.shape}"
        )


def _matrix(preprocessor, frame, fit=False):
    feature_columns = [column for column in frame.columns if column not in IDENTIFIER_COLUMNS]
    if "station" not in frame.columns:
        raise ValueError("Feature frame is missing station for safe one-hot encoding")
    numeric_columns = [column for column in feature_columns if column != "station"]
    if fit:
        preprocessor.fit(frame[["station"] + numeric_columns])
    values = preprocessor.transform(frame[["station"] + numeric_columns])
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame(values, columns=names, index=frame.index), feature_columns


def _new_preprocessor(numeric_columns):
    return ColumnTransformer(
        [
            ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station"]),
            ("numeric", "passthrough", numeric_columns),
        ],
        verbose_feature_names_out=False,
    )


def _base_params(config, candidate, n_estimators, early_stopping=False):
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": int(config["project"]["random_state"]),
        "n_estimators": int(n_estimators),
        "max_depth": int(candidate["max_depth"]),
        "learning_rate": float(candidate["learning_rate"]),
        "min_child_weight": float(candidate["min_child_weight"]),
        "reg_lambda": float(candidate["reg_lambda"]),
        "n_jobs": int(config["model"]["n_jobs"]),
        "eval_metric": "rmse",
    }
    if early_stopping:
        params["early_stopping_rounds"] = int(config["model"]["early_stopping_rounds"])
    return params


def train_variant(variant, splits, config):
    train = splits[variant]["train"].reset_index(drop=True)
    validation = splits[variant]["validation"].reset_index(drop=True)
    test = splits[variant]["test"].reset_index(drop=True)
    numeric_columns = [
        column
        for column in train.columns
        if column not in IDENTIFIER_COLUMNS and column != "station"
    ]
    preprocessor = _new_preprocessor(numeric_columns)
    train_x, feature_columns = _matrix(preprocessor, train, fit=True)
    validation_x, _ = _matrix(preprocessor, validation)
    test_x, _ = _matrix(preprocessor, test)
    train_y = train["target_water_level_m"].astype(float)
    validation_y = validation["target_water_level_m"].astype(float)

    best = None
    candidate_results = []
    for candidate in config["model"]["candidates"]:
        model = XGBRegressor(
            **_base_params(
                config,
                candidate,
                config["model"]["n_estimators"],
                early_stopping=True,
            )
        )
        model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        validation_prediction = model.predict(validation_x)
        rmse = float(((validation_prediction - validation_y.to_numpy()) ** 2).mean() ** 0.5)
        result = {
            "candidate": candidate,
            "validation_rmse_m": rmse,
            "best_iteration": int(getattr(model, "best_iteration", config["model"]["n_estimators"] - 1)),
        }
        candidate_results.append(result)
        if best is None or rmse < best["validation_rmse_m"]:
            best = result

    selected = best["candidate"]
    selected_iterations = best["best_iteration"] + 1
    combined = pd.concat([train, validation], ignore_index=True)
    combined_x, _ = _matrix(preprocessor, combined)
    final_model = XGBRegressor(
        **_base_params(config, selected, selected_iterations, early_stopping=False)
    )
    final_model.fit(combined_x, combined["target_water_level_m"].astype(float), verbose=False)
    test_prediction = final_model.predict(test_x)
    validate_prediction_shape(test_prediction, len(test))

    output_dir = project_path(config, config["outputs"]["model_dir"]) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(output_dir / "model.json")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    metadata = {
        "variant": variant,
        "target": "target_water_level_m",
        "target_unit": config["data"]["expected_level_unit"],
        "feature_columns": feature_columns,
        "transformed_feature_columns": list(train_x.columns),
        "station_encoding": "OneHotEncoder(handle_unknown='ignore') fitted on training data",
        "selected_parameters": selected,
        "selected_n_estimators": selected_iterations,
        "validation_candidates": candidate_results,
        "random_state": config["project"]["random_state"],
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    predictions = test[
        ["station", "date", "target_date", "water_level_current_m", "target_water_level_m"]
    ].copy()
    predictions = predictions.rename(columns={"target_water_level_m": "actual_water_level_m"})
    predictions["predicted_water_level_m"] = test_prediction
    predictions["model"] = f"xgboost_{variant}"
    prediction_path = project_path(config, config["outputs"]["predictions_dir"])
    prediction_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path / f"xgboost_{variant}_test_predictions.csv", index=False)
    interim_dir = project_path(config, config["outputs"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    test[["station"] + feature_columns].to_csv(interim_dir / f"{variant}_test_features.csv", index=False)
    return predictions, metadata


def train_all(splits, config):
    results = {}
    for variant in ("raw", "engineered", "hybrid"):
        results[variant] = train_variant(variant, splits, config)
    return results