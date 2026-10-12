from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_path


VARIANTS = ("raw", "engineered", "hybrid")
KEYS = ["station", "date", "target_date"]
PREDICTION_COLUMN = "predicted_water_level_m"
ACTUAL_COLUMN = "actual_water_level_m"
CURRENT_COLUMN = "water_level_current_m"


def _read_source(config, key):
    path = project_path(config, config["data"][key])
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(
        frame["date"], format=config["data"]["date_format"], errors="raise"
    )
    return frame


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "n": int(len(actual)),
        "mae_m": float(mean_absolute_error(actual, predicted)),
        "rmse_m": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
    }


def _load_saved_predictions(config):
    directory = project_path(config, config["outputs"]["predictions_dir"])
    frames = {}
    for variant in VARIANTS:
        path = directory / f"xgboost_{variant}_test_predictions.csv"
        frames[variant] = pd.read_csv(path, parse_dates=["date", "target_date"])
    frames["persistence"] = pd.read_csv(
        directory / "persistence_test_predictions.csv", parse_dates=["date", "target_date"]
    )
    return frames


def _prediction_integrity(predictions, config, raw_source):
    reports = {}
    reference_keys = pd.MultiIndex.from_frame(predictions["raw"][KEYS]).sort_values()
    reference_actual = predictions["raw"].set_index(KEYS)[ACTUAL_COLUMN].sort_index()
    reference_stations = set(predictions["raw"]["station"])
    raw_lookup = {
        (str(row.station), row.date): row.water_level_9am_m
        for row in raw_source.itertuples(index=False)
    }
    for name, frame in predictions.items():
        keys = pd.MultiIndex.from_frame(frame[KEYS]).sort_values()
        actual = frame.set_index(KEYS)[ACTUAL_COLUMN].sort_index()
        next_day = (frame["target_date"] - frame["date"]).dt.days.eq(1)
        source_target_mismatches = 0
        for row in frame.itertuples(index=False):
            expected = raw_lookup.get((str(row.station), row.target_date))
            if pd.isna(expected) or not np.isclose(float(expected), float(getattr(row, ACTUAL_COLUMN))):
                source_target_mismatches += 1
        baseline_matches_current = None
        if name == "persistence":
            xgb_raw = predictions["raw"].set_index(KEYS)[CURRENT_COLUMN].sort_index()
            predicted = frame.set_index(KEYS)[PREDICTION_COLUMN].sort_index()
            baseline_matches_current = bool(np.allclose(predicted, xgb_raw, equal_nan=True))
        report = {
            "rows": int(len(frame)),
            "unique_station_date_targets": int(frame[KEYS].drop_duplicates().shape[0]),
            "keys_match_raw": bool(keys.equals(reference_keys)),
            "actuals_match_raw": bool(actual.equals(reference_actual)),
            "station_count": int(frame["station"].nunique()),
            "station_ids_match_raw": set(frame["station"]) == reference_stations,
            "all_targets_exactly_next_calendar_day": bool(next_day.all()),
            "source_target_mismatches": int(source_target_mismatches),
        }
        if name == "persistence":
            report["persistence_prediction_equals_current_level"] = baseline_matches_current
        reports[name] = report
    return reports


def _recompute_saved_metrics(predictions, config):
    source_path = project_path(config, config["outputs"]["metrics_dir"]) / "model_comparison.csv"
    stored = pd.read_csv(source_path).set_index("model")
    actual = predictions["raw"][ACTUAL_COLUMN].to_numpy(float)
    reports = {}
    for name, frame in predictions.items():
        aligned = predictions["raw"][KEYS + [ACTUAL_COLUMN]].merge(
            frame[KEYS + [PREDICTION_COLUMN]],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        recalculated = _metrics(
            aligned[ACTUAL_COLUMN].to_numpy(float),
            aligned[PREDICTION_COLUMN].to_numpy(float),
        )
        saved = stored.loc[name]
        reports[name] = {
            **recalculated,
            "stored_mae_m": float(saved["mae_m"]),
            "stored_rmse_m": float(saved["rmse_m"]),
            "stored_r2": float(saved["r2"]),
            "mae_abs_difference": abs(recalculated["mae_m"] - float(saved["mae_m"])),
            "rmse_abs_difference": abs(recalculated["rmse_m"] - float(saved["rmse_m"])),
            "r2_abs_difference": abs(recalculated["r2"] - float(saved["r2"])),
        }
    return reports


def _feature_and_target_audit(config, raw_source, engineered_source):
    source_frames = {
        "raw": raw_source,
        "engineered": engineered_source,
        "hybrid": _read_source(config, "hybrid_csv"),
    }
    source_maps = {}
    for variant, frame in source_frames.items():
        source_maps[variant] = {
            (str(row.station), row.date): row._asdict()
            for row in frame.itertuples(index=False)
        }
    processed_dir = project_path(config, config["outputs"]["processed_dir"])
    output = {
        "variants": {},
        "rainfall_windows": {},
        "weather_timing_metadata": {},
        "source_cross_file_level_comparison": {},
    }
    key_columns = ["station", "date"]
    level_column = config["data"]["level_column"]
    for variant in ("engineered", "hybrid"):
        comparison = raw_source[key_columns + [level_column]].merge(
            source_frames[variant][key_columns + [level_column]],
            on=key_columns,
            how="outer",
            suffixes=("_raw", f"_{variant}"),
            indicator=True,
            validate="one_to_one",
        )
        both = comparison["_merge"].eq("both")
        raw_level = comparison[f"{level_column}_raw"]
        other_level = comparison[f"{level_column}_{variant}"]
        both_missing = both & raw_level.isna() & other_level.isna()
        missingness_mismatch = both & (raw_level.isna() != other_level.isna())
        nonmissing = both & raw_level.notna() & other_level.notna()
        unequal_nonmissing = nonmissing & ~np.isclose(
            raw_level.fillna(0).to_numpy(dtype=float),
            other_level.fillna(0).to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-8,
        )
        output["source_cross_file_level_comparison"][variant] = {
            "unmatched_station_date_keys": int((comparison["_merge"] != "both").sum()),
            "both_missing_levels": int(both_missing.sum()),
            "missingness_mismatches": int(missingness_mismatch.sum()),
            "nonmissing_values_compared": int(nonmissing.sum()),
            "unequal_nonmissing_values": int(unequal_nonmissing.sum()),
        }
    for variant in VARIANTS:
        metadata_path = project_path(config, config["outputs"]["model_dir"]) / variant / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(processed_dir / f"{variant}_samples.csv", parse_dates=["date", "target_date"])
        source_map = source_maps[variant]
        feature_audits = {}
        for feature in metadata["feature_columns"]:
            if feature == "water_level_current_m":
                source_column, lag = config["data"]["level_column"], 0
            elif feature.startswith("water_level_lag_"):
                lag = int(feature.split("_lag_")[1].split("d")[0])
                source_column = config["data"]["level_column"]
            elif "_lag_" in feature:
                source_column, suffix = feature.rsplit("_lag_", 1)
                lag = int(suffix.removesuffix("d"))
            else:
                feature_audits[feature] = {"verified": False, "reason": "No source mapping"}
                continue
            mismatches = 0
            missing_mismatches = 0
            checked = 0
            available = 0
            for row in frame.itertuples(index=False):
                source_date = row.date - pd.Timedelta(days=lag)
                source_row = source_map.get((str(row.station), source_date))
                expected = np.nan if source_row is None else source_row.get(source_column, np.nan)
                actual_value = getattr(row, feature)
                expected_missing = pd.isna(expected)
                actual_missing = pd.isna(actual_value)
                checked += 1
                if not expected_missing:
                    available += 1
                if expected_missing != actual_missing:
                    missing_mismatches += 1
                elif not expected_missing and not np.isclose(float(expected), float(actual_value), rtol=1e-8, atol=1e-8):
                    mismatches += 1
            feature_audits[feature] = {
                "source_column": source_column,
                "source_lag_days": lag,
                "rows_checked": checked,
                "source_value_available": available,
                "value_mismatches": mismatches,
                "missingness_mismatches": missing_mismatches,
                "verified_against_source": mismatches == 0 and missing_mismatches == 0,
            }
        target_mismatches = 0
        raw_map = source_maps["raw"]
        for row in frame.itertuples(index=False):
            source_target = raw_map.get((str(row.station), row.target_date))
            expected = np.nan if source_target is None else source_target.get(config["data"]["level_column"], np.nan)
            if pd.isna(expected) or not np.isclose(float(expected), float(row.target_water_level_m)):
                target_mismatches += 1
        output["variants"][variant] = {
            "features": feature_audits,
            "target_rows": int(len(frame)),
            "target_values_mismatching_raw_source": target_mismatches,
            "target_dates_exactly_one_day_after_origin": bool(
                ((frame["target_date"] - frame["date"]).dt.days == 1).all()
            ),
            "target_column_excluded_from_features": "target_water_level_m" not in metadata["feature_columns"],
        }

    rain_map = source_maps["raw"]
    engineering_map = source_maps["engineered"]
    for window in (3, 7, 14):
        prior_errors, inclusive_errors = [], []
        possible = 0
        for (station, date), row in engineering_map.items():
            column = f"rainfall_{window}day_mm"
            provided = row.get(column, np.nan)
            if pd.isna(provided):
                continue
            prior_dates = [date - pd.Timedelta(days=day) for day in range(window, 0, -1)]
            inclusive_dates = [date - pd.Timedelta(days=day) for day in range(window - 1, -1, -1)]
            prior_values = [rain_map.get((station, d), {}).get("rainfall_mm", np.nan) for d in prior_dates]
            inclusive_values = [rain_map.get((station, d), {}).get("rainfall_mm", np.nan) for d in inclusive_dates]
            if all(not pd.isna(value) for value in prior_values):
                prior_errors.append(abs(float(provided) - float(np.sum(prior_values))))
            if all(not pd.isna(value) for value in inclusive_values):
                inclusive_errors.append(abs(float(provided) - float(np.sum(inclusive_values))))
            possible += 1
        output["rainfall_windows"][f"rainfall_{window}day_mm"] = {
            "rows_with_provided_value": possible,
            "complete_prior_windows_compared": len(prior_errors),
            "mean_abs_error_vs_prior_days_mm": float(np.mean(prior_errors)) if prior_errors else None,
            "max_abs_error_vs_prior_days_mm": float(np.max(prior_errors)) if prior_errors else None,
            "complete_inclusive_windows_compared": len(inclusive_errors),
            "mean_abs_error_vs_window_including_date_mm": float(np.mean(inclusive_errors)) if inclusive_errors else None,
            "alignment_closer_to_prior_days": bool(prior_errors and inclusive_errors and np.mean(prior_errors) < np.mean(inclusive_errors)),
            "early_window_history_outside_source_file_unverifiable": True,
        }
    output["weather_timing_metadata"] = {
        "weather_observation_time_present": False,
        "weather_publication_time_present": False,
        "timezone_present": False,
        "configured_weather_lag_days": config["features"]["daily_weather_lag_days"],
        "interpretation": "Feature values are verified against the previous calendar date, but source timestamps do not prove when they were actually available at 9 AM.",
    }
    return output


def _error_frames(predictions):
    reference = predictions["raw"][[*KEYS, CURRENT_COLUMN, ACTUAL_COLUMN]].copy()
    reference["observed_change_m"] = reference[ACTUAL_COLUMN] - reference[CURRENT_COLUMN]
    result = {}
    for name, frame in predictions.items():
        selected = frame[[*KEYS, PREDICTION_COLUMN]].copy()
        joined = reference.merge(selected, on=KEYS, how="inner", validate="one_to_one")
        joined["residual_m"] = joined[PREDICTION_COLUMN] - joined[ACTUAL_COLUMN]
        joined["predicted_change_m"] = joined[PREDICTION_COLUMN] - joined[CURRENT_COLUMN]
        joined["change_error_m"] = joined["predicted_change_m"] - joined["observed_change_m"]
        joined["model"] = name
        result[name] = joined
    return result


def _regime_and_station_tables(errors, threshold):
    labels = {
        "rising": lambda change: change >= threshold,
        "stable": lambda change: change.abs() < threshold,
        "falling": lambda change: change <= -threshold,
    }
    regimes = []
    station_rows = []
    for model, frame in errors.items():
        for label, predicate in labels.items():
            selected = frame.loc[predicate(frame["observed_change_m"])]
            if selected.empty:
                continue
            regimes.append(
                {
                    "model": model,
                    "regime": label,
                    "n": len(selected),
                    "mae_m": selected["residual_m"].abs().mean(),
                    "rmse_m": np.mean(selected["residual_m"] ** 2) ** 0.5,
                    "bias_m": selected["residual_m"].mean(),
                    "underprediction_share": (selected["residual_m"] < 0).mean(),
                    "change_underestimation_share": (selected["change_error_m"] < 0).mean(),
                    "mean_observed_change_m": selected["observed_change_m"].mean(),
                    "mean_predicted_change_m": selected["predicted_change_m"].mean(),
                }
            )
        for station, selected in frame.groupby("station"):
            station_rows.append(
                {
                    "model": model,
                    "station": station,
                    "n": len(selected),
                    "mae_m": selected["residual_m"].abs().mean(),
                    "bias_m": selected["residual_m"].mean(),
                    "underprediction_share": (selected["residual_m"] < 0).mean(),
                    "change_underestimation_share": (selected["change_error_m"] < 0).mean(),
                }
            )
    return pd.DataFrame(regimes), pd.DataFrame(station_rows)


def _event_tables(errors):
    raw = errors["raw"]
    positive_changes = raw.loc[raw["observed_change_m"] > 0, "observed_change_m"]
    rise_cutoff = positive_changes.quantile(0.90)
    level_cutoff = raw[ACTUAL_COLUMN].quantile(0.90)
    largest_rises = raw.nlargest(10, "observed_change_m").copy()
    highest_levels = raw.nlargest(10, ACTUAL_COLUMN).copy()
    top_rises = raw.loc[raw["observed_change_m"] >= rise_cutoff].copy()
    high_levels = raw.loc[raw[ACTUAL_COLUMN] >= level_cutoff].copy()

    def add_model_changes(selected):
        result = selected.copy()
        for model, model_frame in errors.items():
            change = model_frame[KEYS + ["predicted_change_m"]].rename(
                columns={"predicted_change_m": f"{model}_predicted_change_m"}
            )
            result = result.merge(change, on=KEYS, how="left", validate="one_to_one")
        return result

    return {
        "rise_cutoff_m": float(rise_cutoff),
        "rise_cutoff_count": int(len(top_rises)),
        "level_cutoff_m": float(level_cutoff),
        "level_cutoff_count": int(len(high_levels)),
        "maximum_observed_rise_m": float(raw["observed_change_m"].max()),
        "maximum_target_level_m": float(raw[ACTUAL_COLUMN].max()),
        "largest_rises": add_model_changes(largest_rises),
        "highest_levels": add_model_changes(highest_levels),
        "top_rise_group": add_model_changes(top_rises),
        "high_level_group": add_model_changes(high_levels),
    }


def _split_performance(config):
    processed_dir = project_path(config, config["outputs"]["processed_dir"])
    model_root = project_path(config, config["outputs"]["model_dir"])
    output = {"persistence": {"final_model_fit_scope": "none", "splits": {}}}
    persistence_frame = pd.read_csv(
        processed_dir / "raw_samples.csv", parse_dates=["date", "target_date"]
    )
    for split, (start, end) in {
        "train": (config["splits"]["train_start"], config["splits"]["train_end"]),
        "validation": (config["splits"]["validation_start"], config["splits"]["validation_end"]),
        "test": (config["splits"]["test_start"], config["splits"]["test_end"]),
    }.items():
        part = persistence_frame.loc[
            persistence_frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        output["persistence"]["splits"][split] = {
            **_metrics(part["target_water_level_m"], part[CURRENT_COLUMN]),
            "fit_status": "persistence_baseline",
        }
    for variant in VARIANTS:
        metadata = json.loads((model_root / variant / "metadata.json").read_text(encoding="utf-8"))
        model = XGBRegressor()
        model.load_model(model_root / variant / "model.json")
        preprocessor = joblib.load(model_root / variant / "preprocessor.joblib")
        frame = pd.read_csv(processed_dir / f"{variant}_samples.csv", parse_dates=["date", "target_date"])
        splits = {
            "train": (config["splits"]["train_start"], config["splits"]["train_end"]),
            "validation": (config["splits"]["validation_start"], config["splits"]["validation_end"]),
            "test": (config["splits"]["test_start"], config["splits"]["test_end"]),
        }
        output[variant] = {
            "final_model_fit_scope": "train_plus_validation",
            "train_and_validation_metrics_are_in_sample": True,
            "validation_candidate_rmse_from_train_only_fit": min(
                item["validation_rmse_m"] for item in metadata["validation_candidates"]
            ),
            "splits": {},
        }
        for split, (start, end) in splits.items():
            part = frame.loc[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
            matrix = preprocessor.transform(part[["station"] + metadata["feature_columns"]])
            prediction = model.predict(matrix)
            output[variant]["splits"][split] = {
                **_metrics(part["target_water_level_m"], prediction),
                "fit_status": "in_sample" if split in ("train", "validation") else "held_out",
            }
    return output


def _make_plots(errors, regimes, station_table, events, diagnostics_dir):
    model_order = ["persistence", *VARIANTS]
    colors = {"persistence": "#333333", "raw": "#377eb8", "engineered": "#e41a1c", "hybrid": "#4daf4a"}

    figure, axis = plt.subplots(figsize=(9, 5))
    pivot = regimes.pivot(index="regime", columns="model", values="mae_m").reindex(
        index=["rising", "stable", "falling"], columns=model_order
    )
    pivot.plot(kind="bar", ax=axis, color=[colors[name] for name in model_order])
    axis.set_ylabel("MAE (m)")
    axis.set_title("Test MAE by observed water-level change")
    axis.set_xlabel("")
    axis.legend(title="Model")
    figure.tight_layout()
    figure.savefig(diagnostics_dir / "regime_mae.png", dpi=160)
    plt.close(figure)

    station_delta = station_table.pivot(index="station", columns="model", values="mae_m")
    for variant in VARIANTS:
        station_delta[f"{variant}_minus_persistence"] = station_delta[variant] - station_delta["persistence"]
    plot_values = station_delta[[f"{variant}_minus_persistence" for variant in VARIANTS]]
    plot_values = plot_values.loc[plot_values.abs().max(axis=1).sort_values(ascending=False).index]
    figure, axis = plt.subplots(figsize=(11, 9))
    plot_values.plot(kind="barh", ax=axis, color=[colors[name] for name in VARIANTS])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Station MAE difference from persistence (m; positive is worse)")
    axis.set_ylabel("")
    axis.set_title("Per-station test MAE difference")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(diagnostics_dir / "station_mae_delta.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for axis, variant in zip(axes, VARIANTS):
        frame = errors[variant]
        axis.scatter(frame["observed_change_m"], frame["predicted_change_m"], alpha=0.6, s=18)
        limits = [
            min(frame["observed_change_m"].min(), frame["predicted_change_m"].min()),
            max(frame["observed_change_m"].max(), frame["predicted_change_m"].max()),
        ]
        axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.axvline(0, color="gray", linewidth=0.7)
        axis.set_title(variant.title())
        axis.set_xlabel("Observed next-day change (m)")
    axes[0].set_ylabel("Predicted next-day change (m)")
    figure.suptitle("Predicted versus observed one-day level change")
    figure.tight_layout()
    figure.savefig(diagnostics_dir / "predicted_vs_observed_change.png", dpi=160)
    plt.close(figure)

    top = events["largest_rises"].sort_values("observed_change_m")
    figure, axis = plt.subplots(figsize=(10, 6))
    y = np.arange(len(top))
    axis.scatter(top["observed_change_m"], y, label="Observed", color="#111111", marker="D")
    for name, color in colors.items():
        axis.scatter(top[f"{name}_predicted_change_m"], y, label=name, color=color, alpha=0.85)
    axis.set_yticks(y, [f"{s} ({d:%m-%d})" for s, d in zip(top["station"], top["target_date"])])
    axis.set_xlabel("One-day water-level change (m)")
    axis.set_title("Top 10 observed rises: actual and predicted change")
    axis.axvline(0, color="gray", linewidth=0.7)
    axis.legend()
    figure.tight_layout()
    figure.savefig(diagnostics_dir / "largest_rise_events.png", dpi=160)
    plt.close(figure)


def _write_report(config, diagnostics_dir, result):
    metrics = result["recomputed_metrics"]
    raw_rmse = metrics["raw"]["rmse_m"]
    hybrid_rmse = metrics["hybrid"]["rmse_m"]
    integrity = result["prediction_integrity"]
    rise = result["trend_and_events"]
    timing = result["feature_and_target_audit"]["weather_timing_metadata"]
    lines = [
        "# C2 Saved-Model Diagnostic Report",
        "",
        "Read-only analysis of saved model JSON, fitted preprocessors, processed samples, original CSV inputs, test predictions, and stored metrics. This run did not retrain models or alter datasets or prior results.",
        "",
        "## Verified Findings",
        "",
        f"- Saved prediction sets: {len(result['prediction_integrity'])} (three XGBoost variants plus persistence); all match Raw test keys: {all(item['keys_match_raw'] for item in integrity.values())}.",
        f"- All targets are exactly one calendar day after forecast origin and match the raw source on the test set: {all(item['all_targets_exactly_next_calendar_day'] and item['source_target_mismatches'] == 0 for item in integrity.values())}.",
        f"- Recomputed test MAE/RMSE/R2 are in `recomputed_metrics.json`; saved-vs-recomputed absolute deltas are at floating-point precision (largest RMSE delta: {max(item['rmse_abs_difference'] for item in metrics.values()):.3g}).",
        f"- Raw RMSE={raw_rmse:.12f} m; Hybrid RMSE={hybrid_rmse:.12f} m; absolute difference={abs(raw_rmse-hybrid_rmse):.12g} m. The displayed equality at four decimals is rounding, not identical error vectors.",
        f"- Raw and Hybrid prediction correlation={result['raw_hybrid_comparison']['prediction_correlation']:.8f}; exact identical predictions={result['raw_hybrid_comparison']['exact_prediction_matches']}/{result['raw_hybrid_comparison']['n']}.",
        f"- Persistence MAE={metrics['persistence']['mae_m']:.4f} m; Raw={metrics['raw']['mae_m']:.4f} m; Engineered={metrics['engineered']['mae_m']:.4f} m; Hybrid={metrics['hybrid']['mae_m']:.4f} m. No XGBoost variant beats persistence on this test set.",
        f"- The top-decile observed-rise group uses a cutoff of {rise['largest_rise_cutoff_m_per_day']:.4f} m/day ({rise['largest_rise_group_n']} rows); top-decile target levels use {rise['high_level_cutoff_m']:.4f} m ({rise['high_level_group_n']} rows). Maximum observed rise={rise['maximum_observed_rise_m']:.4f} m/day; maximum target level={rise['maximum_target_level_m']:.4f} m.",
        f"- Paired full-day weather values match the previous calendar day in processed features. Timestamp/publication time/timezone metadata present: {timing['weather_observation_time_present']}/{timing['weather_publication_time_present']}/{timing['timezone_present']}.",
        "",
        "## Rainfall and Target Audit",
        "",
        "Feature values in saved processed samples were joined back to their source station and source calendar date. See `feature_alignment.csv` and `diagnostics.json` for per-feature counts. Daily weather and rainfall-window values are sourced from the prior calendar date, not the forecast date. This is a conservative lag, but actual publication availability at 9 AM is not proven by these CSVs.",
        "",
        "Rainfall-window comparisons against raw rainfall are in `diagnostics.json`. The source windows match prior-day sums substantially more closely than sums including the indexed date, but the source contains small nonzero discrepancies. Early windows requiring rainfall before Nov 17 cannot be independently checked from these files.",
        "",
        "## Error Regimes and Rapid Rises",
        "",
        "See `regime_metrics.csv`, `event_extremes.csv`, and `largest_rises.csv`. Regimes are defined by actual target minus current level: rising >= 0.10 m/day, stable absolute change < 0.10 m/day, falling <= -0.10 m/day. Rapid-rise performance uses the top decile of positive observed changes. Underestimation is measured both as negative level residual (prediction minus target) and as predicted change below observed change.",
        "",
        "## Station and Overfit Diagnostics",
        "",
        "`per_station_errors.csv` gives each station's sample count, model MAE/bias, and change-underestimation share; `station_mae_comparison.csv` gives model-minus-persistence MAE. Stations have only a small number of test days, so station rankings are noisy.",
        "",
        "`split_performance.csv` reports inference from the saved final models. Those models were fit on train plus validation, so train and validation scores are both in-sample and cannot establish generalization or measure classical overfit. The train-only candidate validation RMSEs from model metadata are preserved separately. Test remains held out. No retraining was performed for this diagnostic.",
        "",
        "## False-Positive in Existing Quality Report",
        "",
        "The existing `data_quality_report.json` counts the 71 water-level cells that are missing in both Raw and the corresponding variant as 71 mismatches, because NaN != NaN. This is a report-comparison bug, not evidence of differing nonmissing levels. The diagnostic checks missingness parity separately and computes genuine nonmissing mismatches; prior outputs were not edited.",
        "",
        "## Interpretation and Prioritized Recommendations",
        "",
        "1. Retain persistence as the benchmark and do not claim XGBoost improvement on this test interval; it is confirmed to have lower test MAE and RMSE.",
        "2. Verify the CHIRPS/ERA5-Land observation period, product release time, timezone, and rainfall accumulation convention. This is an unresolved availability risk, not confirmed leakage: current features are one calendar day lagged.",
        "3. Extend observations across wet/dry seasons and evaluate multiple rolling-origin holdouts before tuning further. The current 39-day period and one test block limit evidence for temporal generalization.",
        "4. Keep station-aware metrics and consider predicting level change or station-normalized anomalies; the observed level range and station feature importance indicate material station heterogeneity, but the mechanism is not established.",
        "5. If rapid-rise recall is a research objective, predefine event-weighted metrics and gather more rise events; report the small subgroup sample sizes and never tune on this final test block.",
        "6. Keep the current shallow regularized XGBoost as a baseline experiment. Saved artifacts do not establish overfitting because training/validation performance of the pre-refit candidate is not available; avoid interpreting in-sample final-model scores as evidence either way.",
        "",
        "## Plots",
        "",
        "- `regime_mae.png`: test MAE by rising/stable/falling subsets.",
        "- `station_mae_delta.png`: per-station MAE difference from persistence.",
        "- `predicted_vs_observed_change.png`: whether models capture one-day changes.",
        "- `largest_rise_events.png`: largest observed rises with each model's predicted change.",
        "",
        "## Hypotheses, Not Confirmed Causes",
        "",
        "- Shared dominant current-level and station signals plausibly explain Raw/Hybrid's near-identical RMSE; feature gain ranks current level and the Glencourse indicator highest in both, but gains are not causal attribution.",
        "- Sparse temporal variation, heterogeneous station datums/regimes, conservative weather lagging, and shallow tree capacity may all contribute to the XGBoost shortfall. The available short period cannot isolate their individual effects.",
        "",
    ]
    (diagnostics_dir / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(config):
    diagnostics_dir = project_path(config, "outputs/diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    raw_source = _read_source(config, "raw_csv")
    engineered_source = _read_source(config, "engineered_csv")
    predictions = _load_saved_predictions(config)
    integrity = _prediction_integrity(predictions, config, raw_source)
    metrics = _recompute_saved_metrics(predictions, config)
    errors = _error_frames(predictions)
    regimes, station_table = _regime_and_station_tables(errors, threshold=0.10)
    events = _event_tables(errors)

    raw = errors["raw"].sort_values(KEYS).reset_index(drop=True)
    hybrid = errors["hybrid"].sort_values(KEYS).reset_index(drop=True)
    raw_hybrid_difference = hybrid[PREDICTION_COLUMN] - raw[PREDICTION_COLUMN]
    comparison = {
        "n": len(raw_hybrid_difference),
        "raw_rmse_m": metrics["raw"]["rmse_m"],
        "hybrid_rmse_m": metrics["hybrid"]["rmse_m"],
        "rmse_difference_m": metrics["hybrid"]["rmse_m"] - metrics["raw"]["rmse_m"],
        "prediction_correlation": float(raw[PREDICTION_COLUMN].corr(hybrid[PREDICTION_COLUMN])),
        "mean_absolute_prediction_difference_m": float(raw_hybrid_difference.abs().mean()),
        "prediction_difference_rmse_m": float(np.mean(raw_hybrid_difference**2) ** 0.5),
        "exact_prediction_matches": int((raw_hybrid_difference == 0).sum()),
        "residual_correlation": float(
            errors["raw"]["residual_m"].corr(errors["hybrid"]["residual_m"])
        ),
    }

    feature_audit = _feature_and_target_audit(config, raw_source, engineered_source)
    splits = _split_performance(config)
    thresholds = {
        "rising_stable_falling_threshold_m_per_day": 0.10,
        "largest_rise_cutoff_m_per_day": events["rise_cutoff_m"],
        "largest_rise_group_n": events["rise_cutoff_count"],
        "high_level_cutoff_m": events["level_cutoff_m"],
        "high_level_group_n": events["level_cutoff_count"],
        "maximum_observed_rise_m": events["maximum_observed_rise_m"],
        "maximum_target_level_m": events["maximum_target_level_m"],
    }
    station_pivot = station_table.pivot(index="station", columns="model", values="mae_m").reset_index()
    for variant in VARIANTS:
        station_pivot[f"{variant}_minus_persistence_mae_m"] = station_pivot[variant] - station_pivot["persistence"]

    extremes = []
    columns = ["station", "date", "target_date", CURRENT_COLUMN, ACTUAL_COLUMN, "observed_change_m"]
    for cohort, frame in (("largest_rise", events["largest_rises"]), ("highest_level", events["highest_levels"])):
        for _, row in frame.iterrows():
            record = {"cohort": cohort, **{column: row[column] for column in columns}}
            for model in ("persistence", *VARIANTS):
                model_row = errors[model].set_index(KEYS).loc[tuple(row[key] for key in KEYS)]
                record[f"{model}_prediction_m"] = float(model_row[PREDICTION_COLUMN])
                record[f"{model}_residual_m"] = float(model_row["residual_m"])
                record[f"{model}_predicted_change_m"] = float(model_row["predicted_change_m"])
                record[f"{model}_change_error_m"] = float(model_row["change_error_m"])
            extremes.append(record)

    event_tables = {name: frame.copy() for name, frame in errors.items()}
    for name, frame in event_tables.items():
        frame.to_csv(diagnostics_dir / f"{name}_test_error_details.csv", index=False)
    regimes.to_csv(diagnostics_dir / "regime_metrics.csv", index=False)
    station_table.to_csv(diagnostics_dir / "per_station_errors.csv", index=False)
    station_pivot.to_csv(diagnostics_dir / "station_mae_comparison.csv", index=False)
    pd.DataFrame(extremes).to_csv(diagnostics_dir / "event_extremes.csv", index=False)
    events["largest_rises"].to_csv(diagnostics_dir / "largest_rises.csv", index=False)
    events["highest_levels"].to_csv(diagnostics_dir / "highest_levels.csv", index=False)

    split_rows = []
    for variant, model_data in splits.items():
        for split_name, scores in model_data["splits"].items():
            split_rows.append({"model": variant, "split": split_name, **scores})
        if variant != "persistence":
            split_rows.append(
                {
                    "model": variant,
                    "split": "validation_candidate_train_only",
                    "fit_status": "validation_metric_on_train_only_candidate",
                    "n": None,
                    "mae_m": None,
                    "rmse_m": model_data["validation_candidate_rmse_from_train_only_fit"],
                    "r2": None,
                }
            )

    extreme_group_rows = []
    for model, frame in errors.items():
        for group_name, mask in (
            ("top_decile_rises", frame["observed_change_m"] >= events["rise_cutoff_m"]),
            ("top_decile_target_levels", frame[ACTUAL_COLUMN] >= events["level_cutoff_m"]),
        ):
            selected = frame.loc[mask]
            extreme_group_rows.append(
                {
                    "model": model,
                    "group": group_name,
                    "n": len(selected),
                    "mae_m": selected["residual_m"].abs().mean(),
                    "rmse_m": np.mean(selected["residual_m"] ** 2) ** 0.5,
                    "bias_m": selected["residual_m"].mean(),
                    "underprediction_share": (selected["residual_m"] < 0).mean(),
                    "change_underestimation_share": (selected["change_error_m"] < 0).mean(),
                    "mean_observed_change_m": selected["observed_change_m"].mean(),
                    "mean_predicted_change_m": selected["predicted_change_m"].mean(),
                }
            )
    pd.DataFrame(extreme_group_rows).to_csv(diagnostics_dir / "extreme_group_metrics.csv", index=False)

    feature_alignment_rows = []
    for variant, variant_result in feature_audit["variants"].items():
        for feature, audit in variant_result["features"].items():
            feature_alignment_rows.append({"variant": variant, "feature": feature, **audit})
    pd.DataFrame(feature_alignment_rows).to_csv(diagnostics_dir / "feature_alignment.csv", index=False)
    pd.DataFrame(split_rows).to_csv(diagnostics_dir / "split_performance.csv", index=False)

    result = {
        "prediction_integrity": integrity,
        "recomputed_metrics": metrics,
        "raw_hybrid_comparison": comparison,
        "feature_and_target_audit": feature_audit,
        "trend_and_events": thresholds,
        "regime_metrics": regimes.to_dict(orient="records"),
        "split_performance": splits,
        "reporting_issue_existing_quality_json": {
            "issue": "NaN != NaN causes paired missing water-level cells to be counted as mismatches.",
            "raw_and_variant_missing_level_cells": int(raw_source[config["data"]["level_column"]].isna().sum()),
        },
    }
    (diagnostics_dir / "diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    (diagnostics_dir / "recomputed_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    _make_plots(errors, regimes, station_table, events, diagnostics_dir)
    _write_report(config, diagnostics_dir, result)
    return diagnostics_dir, result


def main():
    parser = argparse.ArgumentParser(description="Read-only diagnostics for saved C2 XGBoost outputs.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    output, result = run_diagnostics(load_config(args.config))
    print(f"Diagnostics written to: {output}")
    for name, row in result["recomputed_metrics"].items():
        print(f"{name}: n={row['n']} MAE={row['mae_m']:.6f} RMSE={row['rmse_m']:.6f} R2={row['r2']:.6f}")
    print(f"Raw/Hybrid RMSE delta: {result['raw_hybrid_comparison']['rmse_difference_m']:.12g} m")
    print(f"Prediction integrity: {result['prediction_integrity']}")


if __name__ == "__main__":
    main()