from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import project_path


DATASET_FILES = {
    "raw": "raw_csv",
    "engineered": "engineered_csv",
    "hybrid": "hybrid_csv",
}


def load_dataset(path, config):
    data_config = config["data"]
    frame = pd.read_csv(path, dtype={data_config["station_column"]: "string"})
    required = {
        data_config["station_column"],
        data_config["date_column"],
        data_config["level_column"],
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame[data_config["station_column"]] = frame[data_config["station_column"]].str.strip()
    frame[data_config["date_column"]] = pd.to_datetime(
        frame[data_config["date_column"]],
        format=data_config["date_format"],
        errors="coerce",
    )
    if frame[data_config["date_column"]].isna().any():
        count = int(frame[data_config["date_column"]].isna().sum())
        raise ValueError(f"{path} has {count} invalid or missing dates")
    if frame[data_config["station_column"]].isna().any() or frame[data_config["station_column"]].eq("").any():
        raise ValueError(f"{path} has missing station identifiers")

    key_columns = [data_config["station_column"], data_config["date_column"]]
    duplicates = frame.duplicated(key_columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, key_columns].head(5).to_dict("records")
        raise ValueError(f"{path} has duplicate station-date keys, sample: {sample}")

    for column in frame.columns:
        if column not in key_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(key_columns).reset_index(drop=True)


def inspect_dataset(name, frame, config):
    station_col = config["data"]["station_column"]
    date_col = config["data"]["date_column"]
    level_col = config["data"]["level_column"]
    numeric_columns = [column for column in frame.columns if column not in (station_col, date_col)]
    date_counts = frame.groupby(station_col)[date_col].nunique()
    all_dates = pd.date_range(frame[date_col].min(), frame[date_col].max(), freq="D")
    missing_station_dates = []
    for station, group in frame.groupby(station_col):
        present = pd.DatetimeIndex(group[date_col].unique())
        missing_station_dates.extend(
            {"station": str(station), "date": date.date().isoformat()}
            for date in all_dates.difference(present)
        )

    numeric_summary = {}
    for column in numeric_columns:
        values = frame[column]
        numeric_summary[column] = {
            "missing_or_non_numeric": int(values.isna().sum()),
            "minimum": None if values.dropna().empty else float(values.min()),
            "maximum": None if values.dropna().empty else float(values.max()),
        }

    return {
        "dataset": name,
        "rows": int(len(frame)),
        "station_count": int(frame[station_col].nunique()),
        "date_min": frame[date_col].min().date().isoformat(),
        "date_max": frame[date_col].max().date().isoformat(),
        "duplicate_station_dates": int(frame.duplicated([station_col, date_col]).sum()),
        "stations_with_daily_date_gaps": int((date_counts < len(all_dates)).sum()),
        "station_date_gaps": missing_station_dates,
        "numeric_columns": numeric_summary,
        "water_level_unit_label": config["data"]["expected_level_unit"],
        "water_level_unit_verification": "Column name indicates metres; physical unit and vertical datum require authoritative metadata.",
        "water_level_missing": int(frame[level_col].isna().sum()),
        "water_level_minimum": None if frame[level_col].dropna().empty else float(frame[level_col].min()),
        "water_level_maximum": None if frame[level_col].dropna().empty else float(frame[level_col].max()),
    }


def validate_all(config):
    loaded = {}
    reports = {}
    for name, config_key in DATASET_FILES.items():
        path = project_path(config, config["data"][config_key])
        if not path.is_file():
            raise FileNotFoundError(f"Configured {name} CSV does not exist: {path}")
        loaded[name] = load_dataset(path, config)
        reports[name] = inspect_dataset(name, loaded[name], config)

    data_config = config["data"]
    key_columns = [data_config["station_column"], data_config["date_column"]]
    reference = loaded["raw"]
    for name in ("engineered", "hybrid"):
        raw_stations = set(reference[data_config["station_column"]].dropna().astype(str))
        variant_stations = set(loaded[name][data_config["station_column"]].dropna().astype(str))
        reports[name]["unmatched_station_ids_vs_raw"] = sorted(raw_stations.symmetric_difference(variant_stations))
        comparison = reference[key_columns + [data_config["level_column"]]].merge(
            loaded[name][key_columns + [data_config["level_column"]]],
            on=key_columns,
            how="outer",
            suffixes=("_raw", f"_{name}"),
            indicator=True,
        )
        reports[name]["unmatched_station_dates_vs_raw"] = int((comparison["_merge"] != "both").sum())
        common = comparison["_merge"].eq("both")
        reports[name]["water_level_mismatches_vs_raw"] = int(
            (comparison.loc[common, f"{data_config['level_column']}_raw"]
             != comparison.loc[common, f"{data_config['level_column']}_{name}"]).sum()
        )

    return loaded, reports


def write_quality_report(config, reports):
    output_dir = project_path(config, config["outputs"]["metrics_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data_quality_report.json"
    output_path.write_text(json.dumps(reports, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def data_quality_report(config):
    _, reports = validate_all(config)
    return reports