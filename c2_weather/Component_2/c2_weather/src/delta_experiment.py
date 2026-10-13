from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import project_path


GROUPS = {
    "hybrid_base_station": {"trends": False, "station_encoding": True},
    "hybrid_base_pooled": {"trends": False, "station_encoding": False},
    "hybrid_trends_station": {"trends": True, "station_encoding": True},
    "hybrid_trends_pooled": {"trends": True, "station_encoding": False},
}

PARAMETER_CANDIDATES = (
    {"max_depth": 1, "learning_rate": 0.03, "min_child_weight": 10, "reg_lambda": 10.0},
    {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 5, "reg_lambda": 10.0},
    {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 10, "reg_lambda": 20.0},
)

TREND_FEATURES = [
    "water_level_change_1d_m",
    "water_level_change_3d_m",
    "water_level_change_7d_m",
    "water_level_acceleration_3d_m_per_day",
    "water_level_change_std_7d_m",
]


def _number(value):
    return np.nan if pd.isna(value) else float(value)


def build_delta_samples(levels, weather, config, maximum_origin_date):
    """Build only exact next-day targets through the allowed forecast-origin cutoff."""
    station_column = config["data"]["station_column"]
    date_column = config["data"]["date_column"]
    level_column = config["data"]["level_column"]
    maximum_origin_date = pd.Timestamp(maximum_origin_date)
    maximum_target_date = maximum_origin_date + pd.Timedelta(days=1)

    levels = levels.loc[levels[date_column].le(maximum_target_date)].copy()
    weather = weather.loc[weather[date_column].le(maximum_origin_date)].copy()
    level_lookup = {
        (str(row[station_column]), row[date_column]): _number(row[level_column])
        for _, row in levels.iterrows()
    }
    weather_lookup = {
        (str(row[station_column]), row[date_column]): row.to_dict()
        for _, row in weather.iterrows()
    }
    weather_columns = list(config["features"]["raw_daily_weather"]) + list(
        config["features"]["rainfall_windows"]
    )
    output = []
    for _, source_row in levels.iterrows():
        station = str(source_row[station_column])
        origin = source_row[date_column]
        if origin > maximum_origin_date:
            continue
        target_date = origin + pd.Timedelta(days=1)
        current = level_lookup.get((station, origin), np.nan)
        target = level_lookup.get((station, target_date), np.nan)
        if pd.isna(current) or pd.isna(target):
            continue

        row = {
            "station": station,
            "date": origin,
            "target_date": target_date,
            "water_level_current_m": current,
            "target_water_level_m": target,
            "delta_water_level_m": target - current,
        }
        for lag_days in config["features"]["water_level_lags_days"]:
            lagged = level_lookup.get((station, origin - pd.Timedelta(days=lag_days)), np.nan)
            row[f"water_level_lag_{lag_days}d_m"] = lagged
        previous_weather = weather_lookup.get((station, origin - pd.Timedelta(days=1)), {})
        for column in weather_columns:
            row[f"{column}_lag_1d"] = _number(previous_weather.get(column, np.nan))

        levels_7d = [
            level_lookup.get((station, origin - pd.Timedelta(days=days)), np.nan)
            for days in range(7, -1, -1)
        ]
        if all(not pd.isna(value) for value in levels_7d):
            daily_changes = np.diff(np.asarray(levels_7d, dtype=float))
            row[TREND_FEATURES[0]] = float(daily_changes[-1])
            row[TREND_FEATURES[1]] = float(levels_7d[-1] - levels_7d[-4])
            row[TREND_FEATURES[2]] = float(levels_7d[-1] - levels_7d[0])
            row[TREND_FEATURES[3]] = float(daily_changes[-1] - daily_changes[-4:].mean())
            row[TREND_FEATURES[4]] = float(daily_changes.std(ddof=0))
        else:
            row.update({feature: np.nan for feature in TREND_FEATURES})
        output.append(row)

    frame = pd.DataFrame(output).sort_values(["station", "date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No eligible exact-next-day samples were built before the cutoff")
    if frame["date"].max() > maximum_origin_date or frame["target_date"].max() > maximum_target_date:
        raise AssertionError("Delta dataset contains rows beyond the train/validation cutoff")
    if not ((frame["target_date"] - frame["date"]).dt.days == 1).all():
        raise AssertionError("Delta targets are not all exactly one calendar day ahead")
    return frame


def reconstruct_level(current_level, predicted_delta):
    return np.asarray(current_level, dtype=float) + np.asarray(predicted_delta, dtype=float)


def _score(frame, predicted_level, threshold=0.10, rise_cutoff=None):
    actual = frame["target_water_level_m"].to_numpy(dtype=float)
    current = frame["water_level_current_m"].to_numpy(dtype=float)
    predicted_level = np.asarray(predicted_level, dtype=float)
    residual = predicted_level - actual
    observed_change = actual - current
    predicted_change = predicted_level - current
    rising = observed_change >= threshold
    if rise_cutoff is None:
        positive = observed_change[observed_change > 0]
        rise_cutoff = float(np.quantile(positive, 0.90)) if len(positive) else float("inf")
    rapid_rise = observed_change >= rise_cutoff
    return {
        "n": int(len(frame)),
        "mae_m": float(mean_absolute_error(actual, predicted_level)),
        "rmse_m": float(mean_squared_error(actual, predicted_level) ** 0.5),
        "r2": float(r2_score(actual, predicted_level)) if len(actual) > 1 and np.var(actual) > 0 else None,
        "rising_threshold_m_per_day": float(threshold),
        "rising_n": int(rising.sum()),
        "rising_mae_m": float(mean_absolute_error(actual[rising], predicted_level[rising])) if rising.any() else None,
        "rising_bias_m": float(residual[rising].mean()) if rising.any() else None,
        "rising_underprediction_share": float((residual[rising] < 0).mean()) if rising.any() else None,
        "rise_cutoff_m_per_day": float(rise_cutoff),
        "rapid_rise_n": int(rapid_rise.sum()),
        "rapid_rise_mae_m": float(mean_absolute_error(actual[rapid_rise], predicted_level[rapid_rise])) if rapid_rise.any() else None,
        "rapid_rise_bias_m": float(residual[rapid_rise].mean()) if rapid_rise.any() else None,
        "rapid_rise_underprediction_share": float((residual[rapid_rise] < 0).mean()) if rapid_rise.any() else None,
        "rapid_rise_change_underprediction_share": float(
            ((predicted_change[rapid_rise] - observed_change[rapid_rise]) < 0).mean()
        ) if rapid_rise.any() else None,
        "mean_observed_rising_change_m": float(observed_change[rising].mean()) if rising.any() else None,
        "mean_predicted_rising_change_m": float(predicted_change[rising].mean()) if rising.any() else None,
    }


def _make_preprocessor(feature_columns, station_encoding):
    if station_encoding:
        return ColumnTransformer(
            [
                ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station"]),
                ("numeric", "passthrough", feature_columns),
            ],
            verbose_feature_names_out=False,
        )
    return ColumnTransformer(
        [("numeric", "passthrough", feature_columns)],
        verbose_feature_names_out=False,
    )


def _input_frame(frame, feature_columns, station_encoding):
    columns = (["station"] if station_encoding else []) + feature_columns
    return frame[columns]


def _fit_group(name, options, train, validation, common_features, output_root, config):
    feature_columns = common_features + (TREND_FEATURES if options["trends"] else [])
    preprocessor = _make_preprocessor(feature_columns, options["station_encoding"])
    train_x = preprocessor.fit_transform(_input_frame(train, feature_columns, options["station_encoding"]))
    validation_x = preprocessor.transform(
        _input_frame(validation, feature_columns, options["station_encoding"])
    )
    validation_rows = []
    best = None
    for candidate_index, candidate in enumerate(PARAMETER_CANDIDATES, start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=int(config["project"]["random_state"]),
            n_estimators=400,
            max_depth=candidate["max_depth"],
            learning_rate=candidate["learning_rate"],
            min_child_weight=candidate["min_child_weight"],
            reg_lambda=candidate["reg_lambda"],
            n_jobs=int(config["model"]["n_jobs"]),
            eval_metric="rmse",
            early_stopping_rounds=int(config["model"]["early_stopping_rounds"]),
        )
        model.fit(
            train_x,
            train["delta_water_level_m"].to_numpy(dtype=float),
            eval_set=[(validation_x, validation["delta_water_level_m"].to_numpy(dtype=float))],
            verbose=False,
        )
        predicted_delta = model.predict(validation_x)
        predicted_level = reconstruct_level(validation["water_level_current_m"], predicted_delta)
        scores = _score(validation, predicted_level)
        record = {
            "group": name,
            "candidate_index": candidate_index,
            "parameters": candidate,
            "selected_n_estimators": int(getattr(model, "best_iteration", 399)) + 1,
            **scores,
        }
        validation_rows.append(record)
        rank = (scores["mae_m"], scores["rmse_m"])
        if best is None or rank < best["rank"]:
            best = {
                "rank": rank,
                "model": model,
                "candidate": candidate,
                "record": record,
                "predicted_delta": predicted_delta,
                "predicted_level": predicted_level,
            }

    group_dir = output_root / "models" / name
    group_dir.mkdir(parents=True, exist_ok=True)
    best["model"].save_model(group_dir / "model.json")
    joblib.dump(preprocessor, group_dir / "preprocessor.joblib")
    metadata = {
        "experiment": name,
        "target": "delta_water_level_m = next_day_water_level_m - current_water_level_m",
        "reconstruction": "predicted_next_day_level_m = current_water_level_m + predicted_delta_water_level_m",
        "station_encoding": "one-hot fitted on train only" if options["station_encoding"] else "pooled numeric model without station identifier",
        "features": feature_columns,
        "trends_enabled": options["trends"],
        "selected_parameters": best["candidate"],
        "selected_n_estimators": best["record"]["selected_n_estimators"],
        "selection_metric": "validation reconstructed-level MAE, RMSE tie-breaker; rising/extreme performance descriptive only",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_origin_rows_used": 0,
        "fit_scope": "training period only",
    }
    (group_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best, validation_rows, metadata


def _legacy_validation_scores(validation, output_root, config):
    model_root = project_path(config, config["outputs"]["model_dir"])
    scores = []
    predictions = {}
    for variant in ("raw", "engineered", "hybrid"):
        model_dir = model_root / variant
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        model = XGBRegressor()
        model.load_model(model_dir / "model.json")
        preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        matrix = preprocessor.transform(
            validation[["station"] + metadata["feature_columns"]]
        )
        level_prediction = model.predict(matrix)
        predictions[variant] = level_prediction
        scores.append(
            {
                "model": variant,
                "model_family": "existing_level_xgboost",
                "fit_status": "in_sample_train_plus_validation; not a fair validation comparison",
                **_score(validation, level_prediction),
            }
        )
    return scores, predictions


def _save_predictions(frame, prediction_map, output_root):
    results = []
    for name, level in prediction_map.items():
        selected = frame[["station", "date", "target_date", "water_level_current_m", "target_water_level_m"]].copy()
        selected["observed_delta_water_level_m"] = selected["target_water_level_m"] - selected["water_level_current_m"]
        selected["predicted_next_day_water_level_m"] = np.asarray(level, dtype=float)
        selected["predicted_delta_water_level_m"] = selected["predicted_next_day_water_level_m"] - selected["water_level_current_m"]
        selected["model"] = name
        selected.to_csv(output_root / f"validation_predictions_{name}.csv", index=False)
        results.append(selected)
    return results


def _make_plots(validation, prediction_map, output_root, threshold, rise_cutoff):
    actual = validation["target_water_level_m"].to_numpy(dtype=float)
    rising = (actual - validation["water_level_current_m"].to_numpy(dtype=float)) >= threshold
    labels, overall_mae, rising_mae = [], [], []
    for name, prediction in prediction_map.items():
        prediction = np.asarray(prediction, dtype=float)
        labels.append(name.replace("delta_", ""))
        overall_mae.append(mean_absolute_error(actual, prediction))
        rising_mae.append(mean_absolute_error(actual[rising], prediction[rising]))
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions - 0.18, overall_mae, width=0.36, label="All validation rows")
    axis.bar(positions + 0.18, rising_mae, width=0.36, label=f"Rising >= {threshold:.2f} m/day")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylabel("Reconstructed level MAE (m)")
    axis.set_title("Delta experiment and existing baselines on validation")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_root / "validation_overall_and_rising_mae.png", dpi=160)
    plt.close(figure)

    names = [name for name in prediction_map if name.startswith("delta_")]
    figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]
    observed_change = actual - validation["water_level_current_m"].to_numpy(dtype=float)
    for axis, name in zip(axes, names):
        prediction_change = np.asarray(prediction_map[name]) - validation["water_level_current_m"].to_numpy(dtype=float)
        axis.scatter(observed_change, prediction_change, alpha=0.65, s=20)
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.axvline(rise_cutoff, color="red", linestyle="--", linewidth=0.9)
        limits = [min(observed_change.min(), prediction_change.min()), max(observed_change.max(), prediction_change.max())]
        axis.plot(limits, limits, color="black", linestyle=":", linewidth=0.9)
        axis.set_title(name.replace("delta_", ""))
        axis.set_xlabel("Observed level change (m)")
    axes[0].set_ylabel("Predicted level change (m)")
    figure.suptitle("Delta XGBoost change calibration (red line: validation top-decile rise)")
    figure.tight_layout()
    figure.savefig(output_root / "delta_change_calibration.png", dpi=160)
    plt.close(figure)


def _write_report(output_root, comparison, candidates, best_group, train_count, validation_count, rising_count, rise_cutoff):
    persistence = comparison.loc[comparison["model"].eq("persistence")].iloc[0]
    delta_rows = comparison.loc[comparison["model"].str.startswith("delta_")]
    best = delta_rows.sort_values(["mae_m", "rmse_m"]).iloc[0]
    improvement = float(persistence["mae_m"] - best["mae_m"])
    if improvement > 0:
        conclusion = f"Best Delta variant improves validation MAE over persistence by {improvement:.4f} m. This is validation-selected and requires a future untouched holdout before any generalization claim."
    else:
        conclusion = f"Best Delta variant does not improve validation MAE over persistence; it is worse by {-improvement:.4f} m. Treat this as a negative validation result."
    comparison.to_csv(output_root / "validation_model_comparison.csv", index=False)
    candidates.to_csv(output_root / "validation_candidate_metrics.csv", index=False)
    report = [
        "# XGBoost Delta Regression Experiment",
        "",
        "## Scope and Data Protection",
        "",
        "This is a separate validation-only experiment. Source CSV frames were read and filtered to forecast origins through 2025-12-15 and labels through 2025-12-16 before feature construction. No test-origin examples (2025-12-16 through 2025-12-24), saved test-prediction CSVs, or test metrics were used. The Dec 16 reading is only the known next-day label for the final Dec 15 validation origin. Existing models/results and source datasets were not changed; outputs are confined to this experiment directory.",
        "",
        f"Training samples: {train_count}. Validation samples: {validation_count}. Validation rising samples (observed change >= 0.10 m/day): {rising_count}. Top-decile positive-rise cutoff: {rise_cutoff:.4f} m/day.",
        "",
        "## Target and Features",
        "",
        "The model predicts `delta_water_level_m = next_day_water_level_m - current_water_level_m`; the reported level is reconstructed as current level plus predicted delta. Only exact next-calendar-day targets with nonmissing current and target observations are included.",
        "",
        "Hybrid weather fields are sourced from the previous calendar date. Water-level history uses the current 9 AM reading and exact 1/3/7-day station-local lags. Trend ablations add 1/3/7-day level changes, 3-day acceleration, and 7-day change standard deviation only when all underlying daily observations exist. No gap is bridged. CSVs do not provide weather publication timestamps/timezone, so prior-day lagging is conservative but delivery at the cutoff remains unverified.",
        "",
        "Station representation is compared as train-fitted one-hot station indicators versus a pooled model with no station identifier. Ordinal station IDs were not used.",
        "",
        "## Validation Result",
        "",
        conclusion,
        "",
        f"Best Delta configuration by reconstructed validation MAE: `{best['model']}` (trend group `{best_group}`). MAE={best['mae_m']:.4f} m; RMSE={best['rmse_m']:.4f} m; rising MAE={best['rising_mae_m']:.4f} m over {int(best['rising_n'])} rising rows; rapid-rise underprediction share={best['rapid_rise_underprediction_share']:.1%} over {int(best['rapid_rise_n'])} top-decile rises.",
        "",
        "Candidate selection minimizes overall validation MAE, with RMSE as tie-breaker. Rising and rare rapid-rise metrics are descriptive, not tuning objectives. The validation set is reused for candidate and feature-representation selection, so the winning score is selection-biased and needs a new future holdout.",
        "",
        "## Comparator Caveat",
        "",
        "Persistence uses zero predicted delta and is a fair validation baseline. Raw, Engineered, and Hybrid are scored by loading their saved models without retraining; however, those persisted models were refit on train plus validation. Their validation metrics are in-sample and are shown only as context, not fair model-selection comparisons. Stored train-only candidate validation RMSEs are also retained in model metadata, but no train-only legacy model/prediction was saved for validation MAE or rising-case comparison.",
        "",
        "## Test Set",
        "",
        "No held-out test-origin examples, saved test-prediction files, or test metrics were used for fitting or selection. This experiment makes no test-performance claim. A future evaluation should first freeze the selected Delta configuration, then run one final test comparison without tuning on those results.",
        "",
        "## Artifacts",
        "",
        "- `validation_model_comparison.csv`: persistence, saved existing models (in-sample caveat), and four Delta configurations.",
        "- `validation_candidate_metrics.csv`: compact candidate results for all Delta groups.",
        "- `validation_predictions_*.csv`: validation predictions and reconstructed deltas/levels.",
        "- `models/`: new train-only Delta XGBoost models and preprocessing metadata.",
        "- `validation_overall_and_rising_mae.png` and `delta_change_calibration.png`: overall/rising trade-off and rise calibration.",
        "- `experiment_summary.json`: numerical summary, data contract, chosen model, and validation-only boundary.",
        "",
        "## Limitations",
        "",
        "Only 303 validation records cover eight forecast-origin dates across 39 heterogeneous stations; rapid-rise cases are sparse. Station one-hot indicators can capture fixed offsets but cannot prove stable station-specific dynamics. The short series limits feature-ablation confidence, and source weather timing and level datum remain unverified. No rare-event weighting or oversampling was used.",
        "",
    ]
    (output_root / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")


def run_delta_experiment(config):
    output_root = project_path(config, "outputs/experiments/delta_xgboost")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing Delta experiment evidence: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    split_config = config["splits"]
    train_start = pd.Timestamp(split_config["train_start"])
    train_end = pd.Timestamp(split_config["train_end"])
    validation_start = pd.Timestamp(split_config["validation_start"])
    validation_end = pd.Timestamp(split_config["validation_end"])
    raw_path = project_path(config, config["data"]["raw_csv"])
    hybrid_path = project_path(config, config["data"]["hybrid_csv"])
    date_format = config["data"]["date_format"]
    raw_levels = pd.read_csv(raw_path)
    hybrid_weather = pd.read_csv(hybrid_path)
    for frame in (raw_levels, hybrid_weather):
        frame[config["data"]["date_column"]] = pd.to_datetime(
            frame[config["data"]["date_column"]], format=date_format, errors="raise"
        )
    samples = build_delta_samples(raw_levels, hybrid_weather, config, validation_end)
    train = samples.loc[samples["date"].between(train_start, train_end)].copy().reset_index(drop=True)
    validation = samples.loc[samples["date"].between(validation_start, validation_end)].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Configured train or validation interval has no eligible samples")
    if samples["date"].max() > validation_end or samples["target_date"].max() > validation_end + pd.Timedelta(days=1):
        raise AssertionError("The Delta experiment must not include test forecast origins or labels")
    if train["target_date"].max() >= validation["target_date"].min():
        raise AssertionError("Train and validation target dates overlap")

    feature_columns = [
        "water_level_current_m",
        *[f"water_level_lag_{days}d_m" for days in config["features"]["water_level_lags_days"]],
        *[
            f"{column}_lag_1d"
            for column in (
                list(config["features"]["raw_daily_weather"])
                + list(config["features"]["rainfall_windows"])
            )
        ],
    ]
    observed_change = validation["target_water_level_m"] - validation["water_level_current_m"]
    positive_rises = observed_change[observed_change > 0]
    rise_cutoff = float(positive_rises.quantile(0.90)) if not positive_rises.empty else float("inf")

    predictions = {
        "persistence": validation["water_level_current_m"].to_numpy(dtype=float)
    }
    comparison_rows = [
        {
            "model": "persistence",
            "model_family": "persistence",
            "fit_status": "zero_delta_baseline",
            **_score(validation, predictions["persistence"], rise_cutoff=rise_cutoff),
        }
    ]
    legacy_scores, legacy_predictions = _legacy_validation_scores(validation, output_root, config)
    comparison_rows.extend(legacy_scores)
    predictions.update(legacy_predictions)

    candidate_records = []
    selected_models = {}
    for name, options in GROUPS.items():
        best, records, metadata = _fit_group(
            name, options, train, validation, feature_columns, output_root, config
        )
        for record in records:
            candidate_records.append(record)
        predictions[f"delta_{name}"] = best["predicted_level"]
        comparison_rows.append(
            {
                "model": f"delta_{name}",
                "model_family": "xgboost_delta",
                "fit_status": "train_only; selected using validation",
                "selected_n_estimators": metadata["selected_n_estimators"],
                **best["record"],
            }
        )
        selected_models[name] = {
            "validation_mae_m": best["record"]["mae_m"],
            "validation_rmse_m": best["record"]["rmse_m"],
            "selected_parameters": best["candidate"],
            "selected_n_estimators": metadata["selected_n_estimators"],
        }

    comparison = pd.DataFrame(comparison_rows)
    candidates = pd.DataFrame(candidate_records)
    delta_comparison = comparison.loc[comparison["model"].str.startswith("delta_")]
    champion_name = str(delta_comparison.sort_values(["mae_m", "rmse_m"]).iloc[0]["model"])
    champion_group = champion_name.removeprefix("delta_")
    prediction_frames = []
    for name, level in predictions.items():
        frame = validation[["station", "date", "target_date", "water_level_current_m", "target_water_level_m"]].copy()
        frame["predicted_next_day_water_level_m"] = np.asarray(level, dtype=float)
        frame["observed_delta_water_level_m"] = frame["target_water_level_m"] - frame["water_level_current_m"]
        frame["predicted_delta_water_level_m"] = frame["predicted_next_day_water_level_m"] - frame["water_level_current_m"]
        frame["model"] = name
        frame.to_csv(output_root / f"validation_predictions_{name}.csv", index=False)
        prediction_frames.append(frame)
    comparison.to_csv(output_root / "validation_model_comparison.csv", index=False)
    candidates.to_csv(output_root / "validation_candidate_metrics.csv", index=False)

    _make_plots(validation, predictions, output_root, threshold=0.10, rise_cutoff=rise_cutoff)
    summary = {
        "experiment": "XGBoost Delta Regression",
        "target_formula": "delta_water_level_m = next_day_water_level_m - current_water_level_m",
        "level_reconstruction": "predicted_next_day_water_level_m = current_water_level_m + predicted_delta_water_level_m",
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_origin_min": train["date"].min().date().isoformat(),
        "train_origin_max": train["date"].max().date().isoformat(),
        "validation_origin_min": validation["date"].min().date().isoformat(),
        "validation_origin_max": validation["date"].max().date().isoformat(),
        "maximum_target_date_used": samples["target_date"].max().date().isoformat(),
        "validation_rising_threshold_m_per_day": 0.10,
        "validation_rising_rows": int((observed_change >= 0.10).sum()),
        "validation_top_decile_positive_rise_cutoff_m_per_day": rise_cutoff,
        "validation_top_decile_positive_rise_rows": int((observed_change >= rise_cutoff).sum()),
        "weather_lag_days": 1,
        "test_rows_loaded_or_used": 0,
        "existing_models_retrained_or_overwritten": False,
        "existing_results_overwritten": False,
        "selected_delta_model": champion_name,
        "selected_delta_group": champion_group,
        "best_delta_validation_mae_m": float(delta_comparison.set_index("model").loc[champion_name, "mae_m"]),
        "persistence_validation_mae_m": float(comparison.set_index("model").loc["persistence", "mae_m"]),
        "delta_improvement_vs_persistence_mae_m": float(
            comparison.set_index("model").loc["persistence", "mae_m"]
            - delta_comparison.set_index("model").loc[champion_name, "mae_m"]
        ),
        "delta_groups": selected_models,
        "existing_xgboost_validation_status": "in_sample because saved final models were refit on train plus validation; not a fair comparator",
        "selection_metric": "overall validation reconstructed-level MAE; RMSE tie-breaker; rising/extreme scores descriptive only",
        "all_validation_predictions": int(len(validation)),
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_report(
        output_root,
        comparison,
        candidates,
        champion_group,
        len(train),
        len(validation),
        int((observed_change >= 0.10).sum()),
        rise_cutoff,
    )
    return output_root, summary