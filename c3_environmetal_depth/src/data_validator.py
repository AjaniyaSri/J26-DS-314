"""Fail closed on malformed data; distinguish observable checks from assumptions."""
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from src.features import FEATURES, METADATA, SCHEMA, TARGET

EXPECTED_SPLITS = {
    "TRAIN": (700, "2025-11-19", "2025-12-09"),
    "VALIDATION": (265, "2025-12-10", "2025-12-16"),
    "TEST": (349, "2025-12-17", "2025-12-25"),
}


@dataclass
class Audit:
    errors: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)

    def markdown(self) -> str:
        status = "PASS" if not self.errors else "FAIL — training prohibited"
        return (f"# Model 1 data quality audit\n\nStatus: **{status}**\n\n"
                + "\n".join(f"- {x}" for x in self.findings)
                + "\n\n## Critical failures\n\n"
                + ("\n".join(f"- {x}" for x in self.errors) or "None.")
                + "\n\n## Limits of verification\n\n"
                "Calendar checks use exact station/date keys, never preceding available rows. "
                "Unavailable earlier/later readings cannot be independently verified here. "
                "Matching values establish internal consistency, not independent source provenance. "
                "Column names specify metres; original source units and gauge datums cannot be "
                "independently certified from this CSV. Negative stages are allowed. "
                "Publication availability and candidate catchment boundaries remain unverified.\n")


def validate_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, Audit]:
    frame = raw.copy(deep=True)
    audit = Audit()
    audit.findings.append(f"Shape: {len(frame)} rows × {len(frame.columns)} columns.")
    if set(frame.columns) != set(SCHEMA) or len(frame.columns) != 21:
        audit.errors.append(f"Schema mismatch: missing={sorted(set(SCHEMA)-set(frame.columns))}; "
                            f"extra={sorted(set(frame.columns)-set(SCHEMA))}.")
        return frame, audit
    if len(frame) != 1314:
        audit.errors.append(f"Expected 1,314 rows; received {len(frame)}.")

    def problem(mask, description: str, columns=()):
        bad = frame.loc[mask, list(dict.fromkeys(("station_id", "forecast_date", *columns)))]
        if not bad.empty:
            # All offending rows are retained in the local audit, not silently truncated.
            audit.errors.append(f"{description} ({len(bad)} rows): "
                                + bad.to_json(orient="records", date_format="iso"))

    blank = frame.isna() | frame.astype(str).apply(lambda col: col.str.strip().eq(""))
    problem(blank.any(axis=1), "Missing cells", tuple(frame.columns))
    for col in (*FEATURES, TARGET):
        parsed = pd.to_numeric(frame[col], errors="coerce")
        problem(parsed.isna() | ~np.isfinite(parsed), f"Invalid/nonfinite number in {col}", (col,))
        frame[col] = parsed
    dates = pd.to_datetime(frame["forecast_date"], format="%Y-%m-%d", errors="coerce")
    problem(dates.isna(), "Unparseable forecast date")
    problem(frame.duplicated("record_id", keep=False), "Duplicate record_id", ("record_id",))
    problem(frame.duplicated(["station_id", "forecast_date"], keep=False), "Duplicate station/date")
    n_stations = frame.station_id.nunique()
    audit.findings.append(f"Unique stations: {n_stations}; no stations removed.")
    if n_stations != 39:
        audit.errors.append(f"Expected 39 stations; received {n_stations}.")
    split_counts = frame.chronological_split.value_counts().to_dict()
    audit.findings.append(f"Split counts: {json.dumps(split_counts)}.")
    if set(split_counts) != set(EXPECTED_SPLITS):
        audit.errors.append(f"Unexpected split names: {list(split_counts)}.")
    for split, (count, start, end) in EXPECTED_SPLITS.items():
        mask = frame.chronological_split.eq(split)
        subset_dates = dates[mask]
        if int(mask.sum()) != count:
            audit.errors.append(f"{split}: expected {count} rows, received {int(mask.sum())}.")
        if subset_dates.empty or subset_dates.min() != pd.Timestamp(start) or subset_dates.max() != pd.Timestamp(end):
            audit.errors.append(f"{split}: expected range {start} through {end}; "
                                f"received {subset_dates.min()} through {subset_dates.max()}.")
        problem(mask & ~dates.between(start, end), f"Rows outside {split} date window")
        audit.findings.append(f"{split}: {start} through {end}; "
                              f"{frame.loc[mask, 'station_id'].nunique()} stations.")
    for col in ("tree_fraction", "cropland_fraction", "builtup_fraction"):
        problem(~frame[col].between(0, 1), f"{col} outside [0, 1]", (col,))
    problem(frame.catchment_area_km2.le(0), "Nonpositive catchment area", ("catchment_area_km2",))
    problem(~np.isclose(frame.stage_change_1d_m,
                        frame.stage_at_issue_m - frame.stage_lag_1d_m, atol=1e-6, rtol=0),
            "Stage-change mismatch", ("stage_change_1d_m", "stage_at_issue_m", "stage_lag_1d_m"))
    # Do not construct an ambiguous lookup after invalid dates or duplicate keys.
    if dates.notna().all() and not frame.duplicated(["station_id", "forecast_date"]).any():
        lookup = dict(zip(zip(frame.station_id, dates), frame.stage_at_issue_m))
        for day_offset, col in ((-1, "stage_lag_1d_m"), (-2, "stage_lag_2d_m"), (1, TARGET)):
            reference = pd.Series([lookup.get((station, date + pd.Timedelta(days=day_offset)), np.nan)
                                   for station, date in zip(frame.station_id, dates)], index=frame.index)
            available = reference.notna()
            mismatch = available & ~np.isclose(frame[col], reference, atol=1e-6, rtol=0)
            problem(mismatch, f"Calendar/station mismatch in {col}", (col, "stage_at_issue_m"))
            audit.findings.append(f"{col}: {int(available.sum())} calendar references available; "
                                  f"{int((~available).sum())} unavailable; {int(mismatch.sum())} mismatches.")
    for col in ("catchment_qa_status", "environment_feature_status", "source_publication_time_verified"):
        audit.findings.append(f"{col}: {json.dumps(frame[col].value_counts().to_dict())}.")
    expected_issue = frame.forecast_date.astype(str) + " 09:30 Asia/Colombo"
    problem(frame.issue_time_assumed_local.ne(expected_issue), "Unexpected assumed issue timestamp",
            ("issue_time_assumed_local",))
    for col in FEATURES[4:]:
        inconsistent = frame.groupby("station_id")[col].transform("nunique").gt(1)
        problem(inconsistent, f"Static feature varies within station: {col}", (col,))
    for col in ("station_name", "basin"):
        problem(frame.groupby("station_id")[col].transform("nunique").gt(1),
                f"Inconsistent station metadata: {col}", (col,))
    counts = frame.groupby("station_id").size()
    audit.findings.append(f"Records per station: {counts.min()}–{counts.max()}.")
    audit.findings.append("Exactly ten fixed predictors; target and metadata excluded from features.")
    return frame, audit


def require_valid(raw: pd.DataFrame) -> tuple[pd.DataFrame, Audit]:
    frame, audit = validate_dataset(raw)
    if audit.errors:
        raise ValueError(audit.markdown())
    return frame, audit
