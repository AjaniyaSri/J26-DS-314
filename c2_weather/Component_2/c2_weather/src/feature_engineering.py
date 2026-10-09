from __future__ import annotations

import numpy as np
import pandas as pd

from .validate_data import DATASET_FILES, load_dataset
from .config import project_path


VARIANTS = ("raw", "engineered", "hybrid")


def _add_target_and_level_history(frame, config):
    station_col = config["data"]["station_column"]
    date_col = config["data"]["date_column"]
    level_col = config["data"]["level_column"]
    frame = frame.sort_values([station_col, date_col]).copy()
    grouped = frame.groupby(station_col, sort=False)
    frame["target_date"] = grouped[date_col].shift(-1)
    frame["target_water_level_m"] = grouped[level_col].shift(-1)
    exact_next_day = frame["target_date"].eq(frame[date_col] + pd.Timedelta(days=1))
    frame.loc[~exact_next_day, "target_water_level_m"] = np.nan

    for lag_days in config["features"]["water_level_lags_days"]:
        frame[f"water_level_lag_{lag_days}d_m"] = _calendar_lag(
            frame, level_col, lag_days, station_col, date_col
        )
    frame["water_level_current_m"] = frame[level_col]
    return frame


def _calendar_lag(frame, column, lag_days, station_col, date_col):
    grouped = frame.groupby(station_col, sort=False)
    values = grouped[column].shift(lag_days)
    source_dates = grouped[date_col].shift(lag_days)
    return values.where(source_dates.eq(frame[date_col] - pd.Timedelta(days=lag_days)))


def _shift_weather(frame, columns, lag_days, station_col, date_col):
    frame = frame.copy()
    for column in columns:
        output_column = f"{column}_lag_{lag_days}d"
        frame[output_column] = _calendar_lag(frame, column, lag_days, station_col, date_col)
    return frame


def build_feature_frames(datasets, config):
    station_col = config["data"]["station_column"]
    date_col = config["data"]["date_column"]
    level_col = config["data"]["level_column"]
    raw_weather = config["features"]["raw_daily_weather"]
    rainfall_windows = config["features"]["rainfall_windows"]
    lag_days = int(config["features"]["daily_weather_lag_days"])

    prepared = {}
    for variant in VARIANTS:
        source = datasets[variant].copy()
        missing_columns = sorted(
            set(raw_weather if variant in ("raw", "hybrid") else raw_weather[1:])
            - set(source.columns)
        )
        if variant in ("engineered", "hybrid"):
            missing_columns += sorted(set(rainfall_windows) - set(source.columns))
        if missing_columns:
            raise ValueError(f"{variant} dataset is missing configured features: {missing_columns}")

        source = _add_target_and_level_history(source, config)
        daily_columns = raw_weather if variant in ("raw", "hybrid") else [
            column for column in raw_weather if column != "rainfall_mm" or variant == "hybrid"
        ]
        weather_columns = daily_columns + (rainfall_windows if variant in ("engineered", "hybrid") else [])
        source = _shift_weather(source, weather_columns, lag_days, station_col, date_col)

        feature_columns = ["water_level_current_m"] + [
            f"water_level_lag_{days}d_m" for days in config["features"]["water_level_lags_days"]
        ]
        feature_columns.extend(f"{column}_lag_{lag_days}d" for column in daily_columns)
        if variant in ("engineered", "hybrid"):
            feature_columns.extend(f"{column}_lag_{lag_days}d" for column in rainfall_windows)
        source["station"] = source[station_col].astype("string")
        source["date"] = source[date_col]
        prepared[variant] = (source, feature_columns)

    key_columns = [station_col, date_col]
    reference = prepared["raw"][0]
    common_keys = set(zip(reference[station_col], reference[date_col]))
    for variant in VARIANTS[1:]:
        keys = set(zip(prepared[variant][0][station_col], prepared[variant][0][date_col]))
        common_keys.intersection_update(keys)
    eligible = reference[reference.apply(lambda row: (row[station_col], row[date_col]) in common_keys, axis=1)]
    eligible = eligible[
        eligible["target_water_level_m"].notna()
        & eligible["water_level_current_m"].notna()
        & eligible["target_date"].eq(eligible[date_col] + pd.Timedelta(days=1))
    ]
    common_sample_keys = set(zip(eligible[station_col], eligible[date_col]))

    results = {}
    for variant, (source, feature_columns) in prepared.items():
        mask = source.apply(lambda row: (row[station_col], row[date_col]) in common_sample_keys, axis=1)
        selected = source.loc[mask].copy()
        output_columns = list(
            dict.fromkeys(
                [station_col, date_col, "date", "target_date", "station", "target_water_level_m"]
                + feature_columns
            )
        )
        results[variant] = selected[output_columns].reset_index(drop=True)
    assert_identical_samples(results, station_col, date_col)
    return results


def assert_identical_samples(frames, station_col="station", date_col="date"):
    reference = None
    for variant, frame in frames.items():
        keys = pd.MultiIndex.from_frame(frame[[station_col, date_col]]).sort_values()
        if reference is None:
            reference = keys
        elif not reference.equals(keys):
            raise AssertionError(f"{variant} does not have identical station-date samples")


def load_and_build_feature_frames(config):
    datasets = {
        name: load_dataset(project_path(config, config["data"][config_key]), config)
        for name, config_key in DATASET_FILES.items()
    }
    return build_feature_frames(datasets, config)


def past_rainfall_windows(frame, station_column, date_column, rainfall_column, windows):
    """Return complete sums over the N days before each row's date."""
    ordered = frame.sort_values([station_column, date_column]).copy()
    output = ordered[[station_column, date_column]].copy()
    for window in windows:
        sums = []
        for _, row in ordered.iterrows():
            start = row[date_column] - pd.Timedelta(days=window)
            prior = ordered[
                ordered[station_column].eq(row[station_column])
                & ordered[date_column].ge(start)
                & ordered[date_column].lt(row[date_column])
            ]
            values = prior[rainfall_column]
            expected_dates = pd.date_range(start, row[date_column] - pd.Timedelta(days=1), freq="D")
            complete = pd.DatetimeIndex(prior[date_column]).sort_values().equals(expected_dates)
            sums.append(
                float(values.sum())
                if complete and values.notna().all()
                else np.nan
            )
        output[f"rainfall_{window}day_mm"] = sums
    return output


def audit_rainfall_windows(raw, engineered, config):
    station_col = config["data"]["station_column"]
    date_col = config["data"]["date_column"]
    raw_rain = config["features"]["raw_daily_weather"][0]
    keys = [station_col, date_col]
    expected = past_rainfall_windows(
        raw, station_col, date_col, raw_rain, [3, 7, 14]
    )
    comparison = engineered[keys + config["features"]["rainfall_windows"]].merge(
        expected, on=keys, how="inner", suffixes=("_provided", "_recomputed")
    )
    report = {}
    for window in (3, 7, 14):
        column = f"rainfall_{window}day_mm"
        provided = comparison[f"{column}_provided"]
        recomputed = comparison[f"{column}_recomputed"]
        valid = provided.notna() & recomputed.notna()
        differences = (provided[valid] - recomputed[valid]).abs()
        report[column] = {
            "complete_windows_compared": int(valid.sum()),
            "mean_absolute_difference_mm": None if differences.empty else float(differences.mean()),
            "maximum_absolute_difference_mm": None if differences.empty else float(differences.max()),
        }
    return report