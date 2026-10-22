"""C1 validated two-band station-day feature matrix.

Reads but never modifies existing Phase 5 / 13 outputs. Uses original shared
DMC labels as single source of truth. Only pre-forecast features are used.
This is an exploratory FEATURE-BASED prototype, not a U-Net.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEY = ["station_id", "forecast_date"]
SAR_BASE = ["sar_latest_age_hours", "sar_patch_valid_fraction", "sar_vv_mean_db",
            "sar_vh_mean_db", "sar_vv_std_db", "sar_vh_std_db", "sar_vv_p10_db",
            "sar_vv_p90_db", "sar_vh_p10_db", "sar_vh_p90_db"]
SAR_HIST = ["sar_vv_mean_change_db", "sar_vh_mean_change_db"]
SWI_NOW = ["swi005_d1", "swi010_d1"]
SWI_HIST = [f"swi{b}_d{k}" for k in range(2, 8) for b in ("005", "010")]
STAGE_CORE = ["stage_at_issue_m"]
STAGE_HIST = ["stage_lag1_m", "stage_lag2_m", "stage_change_1d_m",
              "stage_change_prev_1d_m", "stage_acceleration_m"]
SAR_NOW_ENG = ["sar_vv_minus_vh_db", "sar_age_days"]
SAR_HIST_ENG = ["sar_vv_change_per_day_db", "sar_vh_change_per_day_db",
                "sar_acquisition_gap_days"]
SWI_NOW_ENG = ["swi005_minus_swi010_latest"]
SWI_HIST_ENG = [f"swi{b}_{stat}" for b in ("005", "010")
                for stat in ("mean7", "mean3", "trend7", "change7")]


def read_unique(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required project input not found: {path}")
    df = pd.read_csv(path, dtype={"station_id": str, "forecast_date": str})
    missing = set(KEY) - set(df)
    if missing:
        raise ValueError(f"Missing station-day key {missing}: {path}")
    if df.duplicated(KEY).any():
        example = df.loc[df.duplicated(KEY), KEY].head(1).to_dict("records")
        raise ValueError(f"Duplicate station-day keys in {path}: {example}")
    return df


def validate_labels(labels: pd.DataFrame) -> None:
    need = KEY + ["record_id", "chronological_split", "stage_at_issue_m",
                  "target_delta_stage_m", "target_stage_next_day_m"]
    miss = [x for x in need if x not in labels]
    if miss:
        raise ValueError(f"Missing label columns: {miss}")
    if labels["record_id"].duplicated().any():
        raise ValueError("Duplicated record_id")
    for col in ("stage_at_issue_m", "target_delta_stage_m", "target_stage_next_day_m"):
        x = pd.to_numeric(labels[col], errors="coerce")
        if not bool(np.isfinite(x).all()):
            raise ValueError(f"Non-finite verified labels: {col}")
    s = pd.to_numeric(labels["stage_at_issue_m"])
    delta = pd.to_numeric(labels["target_delta_stage_m"])
    target = pd.to_numeric(labels["target_stage_next_day_m"])
    if not bool(np.allclose(s + delta, target, atol=1e-5, rtol=0)):
        raise ValueError("Label leakage/inconsistency: next-day stage != current stage + delta")
    dates = pd.to_datetime(labels["forecast_date"], errors="raise")
    split_map = {"TRAIN": ("2025-11-17", "2025-12-09"),
                 "VALIDATION": ("2025-12-10", "2025-12-16"),
                 "TEST": ("2025-12-17", "2025-12-25")}
    for split, (lo, hi) in split_map.items():
        actual = labels["chronological_split"] == split
        expected = dates.between(pd.Timestamp(lo), pd.Timestamp(hi))
        if not actual.equals(expected):
            raise ValueError(f"Chronological split/date mismatch for {split}")
    if not labels["chronological_split"].isin(split_map).all():
        raise ValueError("Unknown split values")


def add_stage_history(labels: pd.DataFrame) -> pd.DataFrame:
    """Calendar-correct lag features from *past/current stages*, never targets.

    Uses lookup by station and exact D-1/D-2 dates; no forward fill across
    missing days and no future stage access. Same historical observation may
    legitimately be known on a later forecast date in VALIDATION.
    """
    result = labels.copy()
    result["forecast_date"] = pd.to_datetime(result["forecast_date"])
    lookup = result.set_index(["station_id", "forecast_date"])["stage_at_issue_m"]
    for k in (1, 2):
        pairs = list(zip(result["station_id"], result["forecast_date"] - pd.Timedelta(days=k)))
        result[f"stage_lag{k}_m"] = pd.to_numeric(lookup.reindex(pd.MultiIndex.from_tuples(pairs)), errors="coerce").to_numpy()
    result["stage_change_1d_m"] = result["stage_at_issue_m"] - result["stage_lag1_m"]
    result["stage_change_prev_1d_m"] = result["stage_lag1_m"] - result["stage_lag2_m"]
    result["stage_acceleration_m"] = result["stage_change_1d_m"] - result["stage_change_prev_1d_m"]
    result["forecast_date"] = result["forecast_date"].dt.strftime("%Y-%m-%d")
    return result


def engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Features from known SAR/SWI observation values, not future targets."""
    df = df.copy()
    numeric = list(set(STAGE_CORE + SAR_BASE + SAR_HIST + SWI_NOW + SWI_HIST))
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["sar_vv_minus_vh_db"] = df["sar_vv_mean_db"] - df["sar_vh_mean_db"]
    df["sar_age_days"] = df["sar_latest_age_hours"] / 24.0
    latest = pd.to_datetime(df.get("sar_latest_acquisition_utc"), errors="coerce", utc=True)
    prev = pd.to_datetime(df.get("sar_previous_acquisition_utc"), errors="coerce", utc=True)
    gap = (latest - prev).dt.total_seconds() / 86400.0
    df["sar_acquisition_gap_days"] = gap.where(gap > 0, np.nan)
    for b in ("vv", "vh"):
        df[f"sar_{b}_change_per_day_db"] = (df[f"sar_{b}_mean_change_db"] / df["sar_acquisition_gap_days"])
    df["swi005_minus_swi010_latest"] = df["swi005_d1"] - df["swi010_d1"]
    for band in ("005", "010"):
        series = [df[f"swi{band}_d{i}"] for i in range(1, 8)]
        mat = np.column_stack(series)
        df[f"swi{band}_mean7"] = np.mean(mat, axis=1)
        df[f"swi{band}_mean3"] = np.mean(mat[:, :3], axis=1)
        # Day1 is most recent, day7 is oldest. Positive trend -> wetter recently.
        df[f"swi{band}_trend7"] = (mat[:, 0] - mat[:, -1]) / 6.0
        df[f"swi{band}_change7"] = mat[:, 0] - mat[:, -1]
    return df


def check_forecast_cutoff(df: pd.DataFrame, sar_assumed_lag_hours: int = 48,
                          swi_assumed_lag_days: int = 2) -> dict:
    issue = (pd.to_datetime(df["forecast_date"]).dt.tz_localize("Asia/Colombo")
             + pd.Timedelta(hours=9, minutes=30)).dt.tz_convert("UTC")
    latest = pd.to_datetime(df["sar_latest_acquisition_utc"], errors="coerce", utc=True)
    prev = pd.to_datetime(df["sar_previous_acquisition_utc"], errors="coerce", utc=True)
    cutoff = issue - pd.Timedelta(hours=sar_assumed_lag_hours)
    violations = int(((latest.notna() & (latest > cutoff)) |
                      (prev.notna() & (prev > cutoff))).sum())
    if violations:
        raise ValueError(f"SAR scenes newer than assumed forecast cutoff: {violations} station-days")
    if "latest_eligible_swi_date" not in df:
        raise ValueError("Missing latest_eligible_swi_date; cannot verify SWI lag")
    actual = pd.to_datetime(df["latest_eligible_swi_date"], errors="coerce")
    expected = pd.to_datetime(df["forecast_date"]) - pd.Timedelta(days=swi_assumed_lag_days)
    mismatched = int((actual != expected).sum())
    if mismatched:
        raise ValueError(f"SWI reference date vs provisional {swi_assumed_lag_days}d lag differs: {mismatched}")
    return {"assumed_sar_lag_hours": sar_assumed_lag_hours,
            "assumed_swi_lag_days": swi_assumed_lag_days,
            "observed_future_cutoff_violations": violations,
            "swi_assumed_lag_mismatches": mismatched,
            "actual_publication_time_verified": False}


def load_project(root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    base = root / "local_data"
    out = root / "outputs"
    labels = read_unique(base / "FloodRisk_DMC_StationDay_LABELLED_1429.csv")
    sar = read_unique(out / "c1_sar_station_day_features.csv")
    swi = read_unique(out / "c1_swi_bandwise_7day_features.csv")
    validate_labels(labels)
    # Avoid feature tables overriding authoritative split, IDs or labels.
    sar = sar.drop(columns=["record_id", "chronological_split"], errors="ignore")
    swi = swi.drop(columns=["record_id", "chronological_split"], errors="ignore")
    df = add_stage_history(labels)
    df = df.merge(sar, on=KEY, how="left", validate="one_to_one")
    df = df.merge(swi, on=KEY, how="left", validate="one_to_one")
    needed = ["sar_multi_local_available", "swi005010_complete_7",
              "sar_latest_acquisition_utc", "sar_previous_acquisition_utc",
              "latest_eligible_swi_date"] + SAR_BASE + SAR_HIST + SWI_NOW + SWI_HIST
    missing = [x for x in needed if x not in df]
    if missing:
        raise ValueError(f"Missing columns in real Phase-5/13 outputs: {missing}")
    for c in ("sar_multi_local_available", "swi005010_complete_7"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Only genuinely complete 005/010 and genuine two same-orbit SAR acquisitions.
    keep = (df["sar_multi_local_available"].eq(1) & df["swi005010_complete_7"].eq(1))
    for c in SAR_BASE + SAR_HIST + SWI_NOW + SWI_HIST:
        keep &= pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()
    eligible = df.loc[keep].copy()
    cutoff_result = check_forecast_cutoff(eligible)
    eligible = engineering(eligible)
    eligible = eligible.sort_values(["forecast_date", "station_id"]).reset_index(drop=True)
    split_n = eligible["chronological_split"].value_counts().to_dict()
    return eligible, {"source_labels_n": int(len(labels)), "station_count_labels": int(labels.station_id.nunique()),
                      "eligible_n": int(len(eligible)), "eligible_by_split":
                      {k: int(split_n.get(k, 0)) for k in ("TRAIN", "VALIDATION", "TEST")},
                      "cutoff_audit": cutoff_result,
                      "feature_family": "two-band SWI005+010 exploratory only", "test_tuning": False}
