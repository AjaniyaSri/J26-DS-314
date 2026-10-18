"""Offline historical inference only: never fits estimators or writes results."""
import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES, TARGET
from src.model2_data import MODEL2_FEATURES

RUN = Path("experiments/model2_dynamic_v1/runs/20261019T115140_557496Z")
STATIC = Path("C3_Static_Model_Lag2_Ready_PRELIMINARY.csv")
WEATHER = Path("C3_Model2_Weather_PreviousDay_1314_PRELIMINARY (1).csv")
REVISED = Path("experiments/model1_diagnostics_v2")
KEYS = ["station_id", "forecast_date"]
INFO = [*KEYS, "station_name", "chronological_split"]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def verify_hash(path, expected):
    path = ROOT / path
    if not path.is_file():
        raise ValueError(f"Required local file is missing: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"File differs from its frozen metadata: {path}")


def load_inputs():
    """Target is excluded at CSV parsing, including before the reveal prompt."""
    m1 = read_json("models/model1_static/model1_metadata.json")
    m2 = read_json(RUN / "models/model2_metadata.json")
    verify_hash(STATIC, m1["dataset_sha256"])
    verify_hash(WEATHER, m2["dataset_sha256"])
    first = pd.read_csv(ROOT / STATIC, usecols=INFO + list(FEATURES))
    second = pd.read_csv(ROOT / WEATHER, usecols=INFO + list(MODEL2_FEATURES))
    for frame in (first, second):
        if frame[INFO].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("Missing identifiers or duplicate station/date records.")
    shared = first[INFO].merge(second[INFO], on=KEYS, how="outer",
                               suffixes=("_m1", "_m2"), indicator=True, validate="one_to_one")
    if not shared._merge.eq("both").all():
        raise ValueError("Datasets do not have identical station/date coverage.")
    for name in ("station_name", "chronological_split"):
        if not shared[name + "_m1"].eq(shared[name + "_m2"]).all():
            raise ValueError(f"Datasets disagree on {name}.")
    return first, second


def validate_features(frame, features):
    if tuple(frame.columns) != tuple(features):
        raise ValueError("Input feature names/order differ from the frozen contract.")
    try:
        x = frame.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Model inputs must be numeric in documented units.") from exc
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Input features contain missing or nonfinite values.")
    for name in ("tree_fraction", "cropland_fraction", "builtup_fraction",
                 "c2_soil_moisture_0_7cm", "c2_soil_moisture_7_28cm"):
        if name in x and not x[name].between(0, 1).all():
            raise ValueError(f"{name} is outside its documented preliminary [0,1] screen.")
    if not x.mean_slope_deg.between(0, 90).all() or not x.catchment_area_km2.gt(0).all():
        raise ValueError("Invalid slope (degrees) or catchment area (km2).")
    for name in MODEL2_FEATURES[10:14]:
        if name in x and x[name].lt(0).any():
            raise ValueError(f"Negative accumulated rainfall: {name}")
    # No stage/elevation lower bound: gauge datums and terrain can be negative.
    return x


def select_rows(first, second, station, forecast_date):
    try:
        parsed = date.fromisoformat(forecast_date)
        if parsed.isoformat() != forecast_date:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("Date must be a valid calendar date in YYYY-MM-DD format.") from exc
    station = station.strip().casefold()
    matches = first.loc[first.station_id.str.casefold().eq(station)
                        | first.station_name.str.casefold().eq(station), "station_id"].unique()
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous station. Use --list-stations for available IDs.")
    station_id = matches[0]
    one = first.loc[first.station_id.eq(station_id) & first.forecast_date.eq(forecast_date)]
    two = second.loc[second.station_id.eq(station_id) & second.forecast_date.eq(forecast_date)]
    if len(one) != 1 or len(two) != 1:
        raise ValueError("Station/date is unavailable. Use --station ID --list-dates.")
    x1 = validate_features(one.loc[:, list(FEATURES)], FEATURES)
    x2 = validate_features(two.loc[:, list(MODEL2_FEATURES)], MODEL2_FEATURES)
    if not np.allclose(x1.to_numpy(), x2[list(FEATURES)].to_numpy(), atol=1e-10, rtol=0):
        raise ValueError("Shared Model 1/2 inputs disagree.")
    return one.iloc[0], x1, x2


def check_estimator(estimator, features):
    if not hasattr(estimator, "booster_"):
        raise ValueError("Expected an already fitted LightGBM estimator.")
    if tuple(estimator.booster_.feature_name()) != tuple(features):
        raise ValueError("Serialized estimator feature names/order are incompatible.")
    if estimator.n_features_in_ != len(features):
        raise ValueError("Serialized estimator input count is incompatible.")


@dataclass
class Models:
    model1: object
    model2: object
    variant: str
    path1: Path


def load_models(variant):
    if variant not in ("original", "revised"):
        raise ValueError("Model 1 must be original or revised.")
    meta2 = read_json(RUN / "models/model2_metadata.json")
    meta1_path = (Path("models/model1_static/model1_metadata.json") if variant == "original"
                  else REVISED / "models/frozen_experiment.json")
    meta1 = read_json(meta1_path)
    path1 = (Path("models/model1_static/lightgbm_model1.joblib") if variant == "original"
             else REVISED / "models/selected_environmental_model.joblib")
    path2 = RUN / "models/model2_selected.joblib"
    if tuple(meta1["features"]) != FEATURES or tuple(meta2["features"]) != MODEL2_FEATURES:
        raise ValueError("Metadata feature contracts are incompatible.")
    if meta1["target"] != TARGET or meta2["target"] != TARGET:
        raise ValueError("Unexpected response in model metadata.")
    if meta2["selected_model2"] != "model2_delta_mae_loss" or meta2["selected_representation"] != "delta":
        raise ValueError("This demo expects the frozen selected stage-change Model 2.")
    if variant == "revised" and (meta1["selected_environmental"] != "delta_small"
                                  or meta1["representation"] != "delta"):
        raise ValueError("This demo expects revised Model 1 delta_small.")
    for dependency in ("lightgbm", "scikit-learn", "numpy", "pandas", "joblib"):
        if version(dependency) != meta2["dependencies"][dependency]:
            raise ValueError(f"Dependency mismatch for {dependency}; use the existing RP .venv.")
    verify_hash(path1, meta1["model_sha256"])
    verify_hash(path2, meta2["frozen_artifact_sha256"][path2.as_posix()])
    # Only trusted, checksum-verified local artifacts. No fit function is called.
    model1, model2 = joblib.load(ROOT / path1), joblib.load(ROOT / path2)
    from src.model1_variants import StageVariant
    from src.model2_training import Model2Regressor
    if variant == "revised":
        if not isinstance(model1, StageVariant) or model1.representation != "delta" or model1.features != FEATURES:
            raise ValueError("Revised Model 1 wrapper is incompatible.")
    if (not isinstance(model2, Model2Regressor) or model2.representation != "delta"
            or model2.is_control or model2.features != MODEL2_FEATURES):
        raise ValueError("Selected Model 2 wrapper is incompatible.")
    check_estimator(model1 if variant == "original" else model1.estimator, FEATURES)
    check_estimator(model2.estimator, MODEL2_FEATURES)
    return Models(model1, model2, variant, path1)


def predict(models, x1, x2):
    x1 = validate_features(x1, FEATURES)
    x2 = validate_features(x2, MODEL2_FEATURES)
    p1 = (models.model1.predict(x1) if models.variant == "original"
          else models.model1.predict_stage(x1))
    p2 = models.model2.predict_stage(x2)
    values = {"Model 1": np.asarray(p1), "Model 2": np.asarray(p2),
              "Persistence": x1.stage_at_issue_m.to_numpy()}
    if any(v.shape != (len(x1),) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Model returned incompatible or nonfinite predictions.")
    return values


def reveal_actual(station_id, forecast_date):
    # This separate function is called only after predictions have been printed.
    values = []
    for path in (STATIC, WEATHER):
        frame = pd.read_csv(ROOT / path, usecols=[*KEYS, TARGET])
        found = frame.loc[frame.station_id.eq(station_id) & frame.forecast_date.eq(forecast_date), TARGET]
        if len(found) != 1 or not np.isfinite(float(found.iloc[0])):
            raise ValueError("Actual next-day observation is unavailable or invalid.")
        values.append(float(found.iloc[0]))
    if not np.isclose(*values, atol=1e-10, rtol=0):
        raise ValueError("Datasets disagree on the actual next-day stage.")
    return values[0]


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def choose(prompt, options):
    while True:
        value = ask(prompt)
        if not value or value.casefold() == "q":
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        if value.casefold() in {str(x).casefold() for x in options}:
            return next(x for x in options if str(x).casefold() == value.casefold())
        print("Choose a listed number/value, or q to quit.")


def show_prediction(models, first, second, station, day, auto_reveal=False):
    row, x1, x2 = select_rows(first, second, station, day)
    predictions = predict(models, x1, x2)
    target_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
    print(f"\n{row.station_name} ({row.station_id}) | input {day} | target {target_day} | {row.chronological_split}")
    print(f"Model 1: {'revised delta_small (stage change)' if models.variant == 'revised' else 'original (absolute stage)'}")
    print(f"Artifact: {models.path1.as_posix()}")
    print("Model 2: selected model2_delta_mae_loss (stage change)")
    print(f"Artifact: {(RUN / 'models/model2_selected.joblib').as_posix()}")
    print("Feature order, finite values, range screens and frozen compatibility checks passed.")
    print("Stage/elevation: m; slope: degrees; area: km2; rainfall: mm; cover: fraction.")
    print("Soil moisture: presumed m3/m3. Units/availability are not independently verified.")
    print("Change-model forecasts include current stage exactly once.")
    for label, value in predictions.items():
        print(f"{label:<14} {float(value[0]):.6f} m")
    if auto_reveal or ask("Reveal historical actual and signed errors? [y/N]: ").casefold() in ("y", "yes"):
        actual = reveal_actual(row.station_id, day)
        print(f"Actual next-day stage: {actual:.6f} m")
        for label, value in predictions.items():
            print(f"{label:<14} signed error {float(value[0]) - actual:+.6f} m")
    else:
        print("Actual next-day stage remains hidden.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", help="Station ID or exact name (quote names containing spaces)")
    parser.add_argument("--date", help="Forecast-input date, YYYY-MM-DD")
    parser.add_argument("--model1", choices=("revised", "original"), default="revised")
    parser.add_argument("--list-stations", action="store_true")
    parser.add_argument("--list-dates", action="store_true", help="Requires --station")
    parser.add_argument("--reveal-actual", action="store_true", help="Reveal actual only after predictions")
    args = parser.parse_args(argv)
    try:
        print("C3 offline historical inference demo | no training or saved-result writes")
        print("Previously examined evaluation periods are descriptive. This is not a real-time warning system.")
        first, second = load_inputs()
        stations = first[["station_id", "station_name"]].drop_duplicates().sort_values("station_id")
        if args.list_stations or not args.station:
            for i, row in enumerate(stations.itertuples(), 1):
                print(f"{i:2}. {row.station_id:<27} {row.station_name}")
        if args.list_stations:
            return 0
        if args.list_dates and not args.station:
            raise ValueError("--list-dates requires --station.")
        if args.date and not args.station:
            raise ValueError("--date requires --station.")
        models = None
        while True:
            station = args.station or choose("Station number/ID (q to quit): ", stations.station_id.tolist())
            if station is None:
                return 0
            available = first.loc[first.station_id.str.casefold().eq(station.casefold())
                                  | first.station_name.str.casefold().eq(station.casefold())]
            if available.empty:
                raise ValueError("Unknown station. Use --list-stations for available IDs.")
            dates = sorted(available.forecast_date.unique())
            if args.list_dates or not args.date:
                for i, day in enumerate(dates, 1):
                    split = available.loc[available.forecast_date.eq(day), "chronological_split"].iloc[0]
                    print(f"{i:2}. {day} ({split})")
            if args.list_dates:
                return 0
            day = args.date or choose("Input date number/YYYY-MM-DD (q to quit): ", dates)
            if day is None:
                return 0
            # Validate selection before loading serialized models.
            select_rows(first, second, station, day)
            if models is None:
                models = load_models(args.model1)
            show_prediction(models, first, second, station, day, args.reveal_actual)
            if args.station and args.date:
                return 0
            if ask("Predict another station/date? [y/N]: ").casefold() not in ("y", "yes"):
                return 0
    except (ValueError, OSError, KeyError, ImportError, AttributeError, TypeError) as exc:
        print(f"Demo error: {exc}", file=sys.stderr)
        print("Use the RP .venv and unchanged saved artifacts. No models were retrained.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nDemo closed.")
        raise SystemExit(0)
