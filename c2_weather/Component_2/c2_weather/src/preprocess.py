import json

from .config import project_path
from .feature_engineering import audit_rainfall_windows, build_feature_frames
from .split_data import split_by_forecast_origin
from .validate_data import validate_all, write_quality_report


def prepare_pipeline_data(config):
    datasets, quality = validate_all(config)
    quality["weather_availability"] = {
        "daily_weather_lag_days": int(config["features"]["daily_weather_lag_days"]),
        "observation_and_publication_timestamps_present": False,
        "note": "Daily weather features are lagged; CSVs do not establish publication time or timezone.",
    }
    quality["rainfall_window_audit"] = audit_rainfall_windows(
        datasets["raw"], datasets["engineered"], config
    )
    report_path = write_quality_report(config, quality)

    frames = build_feature_frames(datasets, config)
    processed_dir = project_path(config, config["outputs"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    for variant, frame in frames.items():
        frame.to_csv(processed_dir / f"{variant}_samples.csv", index=False)

    splits = split_by_forecast_origin(frames, config)
    summary = {
        "common_eligible_samples": len(frames["raw"]),
        "split_counts": {
            split: len(splits["raw"][split]) for split in ("train", "validation", "test")
        },
        "feature_columns": {
            variant: [
                column
                for column in frame.columns
                if column
                not in {
                    config["data"]["station_column"],
                    config["data"]["date_column"],
                    "date",
                    "target_date",
                    "target_water_level_m",
                }
            ]
            for variant, frame in frames.items()
        },
        "split_origin_dates": {
            split: {
                "minimum": splits["raw"][split]["date"].min().date().isoformat(),
                "maximum": splits["raw"][split]["date"].max().date().isoformat(),
            }
            for split in ("train", "validation", "test")
        },
        "target_dates_do_not_overlap": True,
    }
    summary_path = project_path(config, config["outputs"]["metrics_dir"]) / "training_dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frames, splits, quality, summary, report_path