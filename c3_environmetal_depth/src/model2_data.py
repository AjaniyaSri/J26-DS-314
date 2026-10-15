"""Exact Model 2 schema, paired gauge integrity and documented weather windows."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loader import ROOT, local_path, load_dataset, sha256, write_text_new, timestamp
from src.data_validator import validate_dataset
from src.features import FEATURES, SCHEMA, TARGET

DATASET = "C3_Model2_Weather_PreviousDay_1314_PRELIMINARY (1).csv"
WEATHER = ("c2_rainfall_mm", "c2_rainfall_3day_mm", "c2_rainfall_7day_mm", "c2_rainfall_14day_mm",
           "c2_soil_moisture_0_7cm", "c2_soil_moisture_7_28cm")
MODEL2_FEATURES = (*FEATURES, *WEATHER)
WEATHER_METADATA = ("c2_weather_source_date", "c2_weather_alignment", "c2_temperature_C")
MODEL2_SCHEMA = (*SCHEMA, *WEATHER_METADATA[:2], *WEATHER, WEATHER_METADATA[2])
KEYS = ["station_id", "forecast_date"]


def load_model2_config(path="configs/model2_config.yaml") -> dict:
    config = yaml.safe_load(local_path(path).read_text(encoding="utf-8"))
    if config["dataset"] != DATASET or tuple(config["features"]) != MODEL2_FEATURES or config["target"] != TARGET:
        raise ValueError("Model 2 requires the exact approved dataset, ordered 16 features and stage target.")
    if config["selection_metric"] != "reconstructed_stage_validation_mae_m":
        raise ValueError("Selection must use reconstructed-stage validation MAE.")
    if config["family_order"] != ["absolute", "delta"] or len(config["candidates"]) != 3:
        raise ValueError("The approved budget is three absolute and three secondary delta fits.")
    identifiers = [c["id"] for c in config["candidates"]]
    if len(set(identifiers)) != 3 or any(not name.replace("_", "").isalnum() for name in identifiers):
        raise ValueError("Candidate identifiers must be unique, safe filename components.")
    if config["seed"] != config["base_parameters"]["random_state"]:
        raise ValueError("Experiment and model seeds differ.")
    if config["test_policy"] != "previously_inspected_project_holdout_final_descriptive_assessment_only":
        raise ValueError("The project TEST set must be acknowledged as previously inspected.")
    for key in ("dataset", "model1_dataset", "output_root"):
        local_path(config[key])
    return config


def align_model1(model2: pd.DataFrame, model1: pd.DataFrame) -> pd.DataFrame:
    """One-to-one keys and matching gauge values; never silently take a subset."""
    for label, frame in (("Model 2", model2), ("Model 1", model1)):
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{label}: duplicate station-date keys.")
    joined = model2.merge(model1, on=KEYS, how="outer", suffixes=("_m2", "_m1"),
                          indicator=True, validate="one_to_one", sort=False)
    if not joined._merge.eq("both").all() or len(joined) != len(model2) or len(model2) != len(model1):
        bad = joined.loc[joined._merge.ne("both"), [*KEYS, "_merge"]]
        raise ValueError("Evaluation keys differ; no subset/rebuild is authorized: " + bad.to_json(orient="records"))
    failures = []
    for column in (c for c in SCHEMA if c not in KEYS):
        first, second = joined[column + "_m2"], joined[column + "_m1"]
        if column in (*FEATURES, TARGET):
            x, y = pd.to_numeric(first, errors="coerce"), pd.to_numeric(second, errors="coerce")
            # Only tolerate numerical roundoff, never metre-scale or centi-metre changes.
            matches = np.isclose(x, y, atol=1e-10, rtol=0)
        else:
            matches = first.astype(str).eq(second.astype(str))
        if not np.all(matches):
            bad = joined.loc[~matches, [*KEYS, column + "_m2", column + "_m1"]]
            failures.append(column + ": " + bad.to_json(orient="records"))
    if failures:
        raise ValueError("Model 1 correspondence failed: " + "; ".join(failures))
    return joined


def weather_windows(frame: pd.DataFrame) -> pd.DataFrame:
    source = pd.to_datetime(frame.c2_weather_source_date, format="%Y-%m-%d", errors="coerce")
    forecast = pd.to_datetime(frame.forecast_date, format="%Y-%m-%d", errors="coerce")
    out = frame[["record_id", *KEYS, "c2_weather_source_date"]].copy()
    out["source_age_days"] = (forecast - source).dt.days
    for days in (3, 7, 14):
        out[f"rain{days}_start_inclusive"] = (source - pd.Timedelta(days=days)).dt.strftime("%Y-%m-%d")
        out[f"rain{days}_end_inclusive"] = (source - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    return out


@dataclass
class Model2Audit:
    errors: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_table: list[dict] = field(default_factory=list)

    def markdown(self) -> str:
        status = "PASS WITH DOCUMENTED LIMITATIONS" if not self.errors else "FAIL: TRAINING PROHIBITED"
        text = f"# Model 2 data audit\n\nStatus: **{status}**\n\n"
        text += "\n".join("- " + message for message in self.findings)
        text += ("\n\n## Schema and missing cells\n\n"
                 "Types show the modelling parse: predictors and target become numerical; excluded metadata is preserved as text.\n\n"
                 "| Column | Parsed type | Missing cells |\n|---|---|---:|\n")
        for row in self.schema_table:
            text += f"| {row['column']} | {row['dtype']} | {row['missing']} |\n"
        text += "\n## Critical failures\n\n" + ("\n".join("- " + m for m in self.errors) or "None.")
        text += "\n\n## Warnings and verification limits\n\n" + "\n".join("- " + m for m in self.warnings)
        text += ("\n\nRainfall windows are derived from the source dates and the C2 script definition "
                 "supplied by the user. Raw C2 assets, spatial aggregation definitions, publication "
                 "records and original gauge reports are not present for independent verification. "
                 "Absence of future source dates alone does not prove operational availability. "
                 "No data was clipped, shifted, filled, replaced or dropped.\n")
        return text


def validate_model2(raw: pd.DataFrame, model1: pd.DataFrame) -> tuple[pd.DataFrame, Model2Audit]:
    frame = raw.copy(deep=True)
    audit = Model2Audit()
    audit.findings.append(f"Authoritative input: {DATASET}; shape {len(frame)} x {len(frame.columns)}.")
    if set(frame.columns) != set(MODEL2_SCHEMA) or len(frame.columns) != 30:
        audit.errors.append(f"Unexpected schema: missing={sorted(set(MODEL2_SCHEMA)-set(frame.columns))}; "
                            f"extra={sorted(set(frame.columns)-set(MODEL2_SCHEMA))}.")
        return frame, audit
    shared, gauge_audit = validate_dataset(frame[list(SCHEMA)])
    audit.errors.extend(gauge_audit.errors)
    audit.findings.extend(("Shared gauge subset: " + m if m.startswith("Shape:") else m)
                          for m in gauge_audit.findings if not m.startswith("Exactly ten"))
    for column in (*FEATURES, TARGET):
        frame[column] = shared[column]

    def problem(mask, message, columns=()):
        bad = frame.loc[mask, list(dict.fromkeys([*KEYS, *columns]))]
        if not bad.empty:
            audit.errors.append(message + ": " + bad.to_json(orient="records"))

    for column in WEATHER:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        problem(~np.isfinite(numeric), f"Invalid/missing/nonfinite weather predictor {column}", [column])
        frame[column] = numeric
    for column in WEATHER[:4]:
        problem(frame[column].lt(0), f"Negative accumulated rainfall {column}", [column])
    for column in WEATHER[4:]:
        problem(~frame[column].between(0, 1), f"Soil moisture outside presumed fraction range [0,1]: {column}", [column])
    source = pd.to_datetime(frame.c2_weather_source_date, format="%Y-%m-%d", errors="coerce")
    forecast = pd.to_datetime(frame.forecast_date, format="%Y-%m-%d", errors="coerce")
    problem(source.isna(), "Unparseable weather source date", ["c2_weather_source_date"])
    problem((forecast - source).dt.days.ne(1), "Weather source must be previous calendar day", ["c2_weather_source_date"])
    problem(frame.c2_weather_alignment.ne("previous_calendar_day"), "Unexpected weather alignment label", ["c2_weather_alignment"])
    # Totals 3/7/14 are nested under the supplied script convention. Differences
    # are reported as errors, never clipped. This check does not extend to rain1.
    for shorter, longer in ((WEATHER[1], WEATHER[2]), (WEATHER[2], WEATHER[3])):
        problem(frame[shorter].gt(frame[longer] + 1e-6), "Non-nested rolling totals under supplied C2 convention",
                [shorter, longer])
    try:
        align_model1(frame, model1)
        audit.findings.append("All 1,314 station-date records and shared gauge/static values match Model 1; no merge row multiplication.")
    except ValueError as error:
        audit.errors.append(str(error))
    missing = raw.isna() | raw.astype(str).apply(lambda column: column.str.strip().eq(""))
    for column in frame.columns:
        audit.schema_table.append({"column": column, "dtype": str(frame[column].dtype), "missing": int(missing[column].sum())})
    audit.findings.append("Predictors: exact ordered 16-column allowlist (4 stage + 6 static + 4 rainfall + 2 soil moisture). Temperature is metadata.")
    audit.findings.append("Source dates are forecast day minus 1. Rolling N-day windows cover forecast day minus (N+1) through forecast day minus 2.")
    conflicts = frame.c2_rainfall_mm.gt(frame.c2_rainfall_3day_mm + 1e-6)
    audit.warnings += [
        f"{int(conflicts.sum())} rows have 1-day rainfall larger than the 3-day total. The 1-day interval is unverified and not assumed nested; values are retained.",
        "Previous-day joining introduces one extra day of lag for 3/7/14-day totals under the supplied script definition; no alignment correction is made.",
        "Rainfall units are accumulated mm per the supplied C2 description. Soil moisture is presumed m3/m3 from names/ranges, not independently certified; [0,1] is a preliminary screening assumption.",
        "Soil-moisture aggregation periods and weather timezone are unverified. Source dates precede issuance but cannot establish all individual observation cutoffs or publication times.",
        "CHIRPS/ERA5-Land availability and gauge publication before assumed 09:30 Asia/Colombo issuance are unverified. This is retrospective/preliminary, not proven real-time forecasting.",
        "Uncheckable lag/label reference dates and independent measurement units remain limitations, as in Model 1.",
        "Project TEST outcomes were already inspected during Model 1 development. New assessment is descriptive only.",
    ]
    return frame, audit


def load_and_audit(config: dict) -> tuple[pd.DataFrame, Model2Audit]:
    raw = load_dataset(config)
    model1 = load_dataset({"dataset": config["model1_dataset"]})
    return validate_model2(raw, model1)


def audit_to_file(config: dict) -> Path:
    _, audit = load_and_audit(config)
    base = local_path(config["output_root"])
    path = base / "reports/model2_data_audit.md"
    if path.exists():
        path = base / "reports" / f"model2_data_audit_{timestamp()}.md"
    write_text_new(path, audit.markdown())
    if audit.errors:
        raise ValueError(f"Audit failed. See {path}\n" + audit.markdown())
    return path
