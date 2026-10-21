"""Read-only, validated PP1 historical prediction service for the C4 Streamlit UI.

All research data stays in the user's existing private/local repository.
No outbound calls, data mutation, training or GEE requests are performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

MODELS = tuple(f"M{i}" for i in range(1, 8))
MODEL_NAMES = {
    "M1": "M1 · Satellite",
    "M2": "M2 · Weather",
    "M3": "M3 · Environment",
    "M4": "M4 · Satellite + Weather",
    "M5": "M5 · Satellite + Environment",
    "M6": "M6 · Weather + Environment",
    "M7": "M7 · All three modalities",
    "PERSISTENCE": "Persistence · No change",
}
WARN_LEVELS = ("NORMAL", "ALERT", "MINOR FLOOD", "MAJOR FLOOD")
REQUIRED_SOURCE = (
    "record_id", "station_id", "forecast_date", "stage_at_issue_m",
    "target_stage_next_day_m", "alert_threshold_m", "minor_flood_threshold_m",
    "major_flood_threshold_m", "catchment_polygon_path",
)
REQUIRED_PRED = (
    "record_id", "station_id", "forecast_date", "split", "current_stage_m",
    "target_stage_m", "pred_stage_m",
)


class SourceDataError(RuntimeError):
    """An input file is missing, inconsistent or not safe to display."""


def repo_root_from_app() -> Path:
    # .../<repo>/c4_fusion/app/data_service.py
    return Path(__file__).resolve().parents[2]


def station_slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).casefold())).strip("_")


def load_station_registry(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise SourceDataError(
            f"Missing private station map: {path}\n"
            "Copy stations_40_PRIVATE.geojson into c4_fusion/private_data/."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as e:
        raise SourceDataError(f"Private station registry is not valid GeoJSON: {e}") from e
    features = raw.get("features", [])
    if raw.get("type") != "FeatureCollection" or len(features) != 40:
        raise SourceDataError(f"Expected 40 private gauge features; found {len(features)}")

    rows = []
    for f in features:
        props = f.get("properties") or {}
        geo = f.get("geometry") or {}
        name = props.get("station_name") or props.get("station")
        sid = props.get("station_id") or station_slug(name)
        coords = geo.get("coordinates") or []
        if geo.get("type") != "Point" or len(coords) != 2:
            raise SourceDataError(f"Invalid original gauge point for {sid}")
        lon, lat = map(float, coords)
        if not (78 <= lon <= 83 and 5 <= lat <= 11):
            raise SourceDataError(f"Unreasonable WGS84 coordinates for {sid}")
        if not name or not sid or not props.get("basin"):
            raise SourceDataError(f"Missing gauge identity or basin for {sid}")
        rows.append({
            "station_id": str(sid), "station_name": str(name),
            "basin": str(props["basin"]).strip(), "lat": lat, "lon": lon,
        })
    stations = pd.DataFrame(rows)
    if stations.station_id.duplicated().any():
        raise SourceDataError("Private gauge registry has duplicate station IDs")
    if {'ampara', 'calidonia'}.intersection(set(stations.station_id)):
        raise SourceDataError('This is the unfiltered 42-station GIS export; use stations_40_PRIVATE.geojson')
    return stations.sort_values(["basin", "station_name"]).reset_index(drop=True), raw


def load_dmc_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SourceDataError(
            f"Missing private DMC labelled source: {path}\n"
            "Keep the CSV in c4_fusion/private_data/observations/."
        )
    src = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    absent = set(REQUIRED_SOURCE).difference(src.columns)
    if absent:
        raise SourceDataError(f"DMC CSV missing required columns: {sorted(absent)}")
    if not src.record_id.is_unique or len(src) != 1429:
        raise SourceDataError(f"DMC labels must contain 1,429 unique records; got {len(src)}")
    for column in ["stage_at_issue_m", "target_stage_next_day_m", *REQUIRED_SOURCE[-4:-1]]:
        src[column] = pd.to_numeric(src[column], errors="coerce")
        if not np.isfinite(src[column]).all():
            raise SourceDataError(f"DMC CSV contains missing or nonnumeric values in {column}")
    if not ((src.alert_threshold_m < src.minor_flood_threshold_m) &
            (src.minor_flood_threshold_m < src.major_flood_threshold_m)).all():
        raise SourceDataError("Gauge-specific flood levels are unordered; source review needed")
    if not (src.groupby("station_id")[["alert_threshold_m", "minor_flood_threshold_m", "major_flood_threshold_m"]].nunique() == 1).all().all():
        raise SourceDataError("Flood thresholds are inconsistent across dates for the same gauge")
    return src


def classify_stage(stage: float, alert: float, minor: float, major: float) -> str:
    """Exclusive status derived from that particular gauge's threshold levels."""
    if float(stage) >= float(major):
        return "MAJOR FLOOD"
    if float(stage) >= float(minor):
        return "MINOR FLOOD"
    if float(stage) >= float(alert):
        return "ALERT"
    return "NORMAL"


@dataclass
class C4Workspace:
    root: Path
    stations: pd.DataFrame
    station_geojson: dict
    observations: pd.DataFrame
    predictions: pd.DataFrame  # 351 rows × available models; includes station DMC thresholds
    comparison: pd.DataFrame  # includes persistence reference
    metrics: pd.DataFrame
    available_models: tuple[str, ...]
    missing_models: tuple[str, ...]
    aligned_index: pd.DataFrame | None
    folder: Path

    def available_station_ids(self) -> set[str]:
        return set(self.predictions.station_id)

    def dates(self, station_id: str) -> list[str]:
        return sorted(self.predictions.loc[self.predictions.station_id.eq(station_id), 'forecast_date'].unique().tolist())

    def record(self, station_id: str, date: str, model: str) -> pd.Series | None:
        subset = self.comparison.loc[
            self.comparison.station_id.eq(station_id) &
            self.comparison.forecast_date.eq(date) &
            self.comparison.model.eq(model)
        ]
        return None if subset.empty else subset.iloc[0]

    def model_comparison(self, station_id: str, date: str) -> pd.DataFrame:
        frame = self.comparison.loc[
            self.comparison.station_id.eq(station_id) & self.comparison.forecast_date.eq(date)
        ].copy()
        frame['order'] = frame['model'].map({**{m: i for i, m in enumerate(MODELS)}, 'PERSISTENCE': 7})
        return frame.sort_values('order').drop(columns='order')

    def station_series(self, station_id: str, model: str) -> pd.DataFrame:
        frame = self.comparison.loc[
            self.comparison.station_id.eq(station_id) & self.comparison.model.eq(model),
            ['forecast_date', 'current_stage_m', 'target_stage_m', 'pred_stage_m']
        ].copy()
        return frame.sort_values('forecast_date')

    def provenance(self, record_id: str) -> pd.Series | None:
        if self.aligned_index is None:
            return None
        match = self.aligned_index.loc[self.aligned_index.record_id.eq(record_id)]
        return None if match.empty else match.iloc[0]


def load_workspace(root: Path | str | None = None) -> C4Workspace:
    """Strictly validate the original 7 × 351 held-out results; never fabricate predictions."""
    root = Path(root or repo_root_from_app()).resolve()
    private = root / "c4_fusion" / "private_data"
    stations, geo = load_station_registry(private / "stations_40_PRIVATE.geojson")
    source = load_dmc_source(private / "observations" / "FloodRisk_DMC_StationDay_LABELLED_1429.csv")
    if not set(source.station_id).issubset(set(stations.station_id)):
        raise SourceDataError("DMC station IDs do not match the private 40-gauge registry")

    folder = root / "results" / "pp1" / "c4_neural"
    if not folder.is_dir():
        raise SourceDataError(
            f"Model prediction folder not found: {folder}\n"
            "Run Notebook 07, or copy the saved M1–M7 test-prediction CSVs here."
        )
    predictions = []
    available = []
    shared_ids = None
    reference = None
    for model in MODELS:
        filename = folder / f"{model}_test_predictions.csv"
        if not filename.is_file():
            continue
        frame = pd.read_csv(filename, encoding="utf-8-sig", low_memory=False)
        missing = set(REQUIRED_PRED).difference(frame.columns)
        if missing:
            raise SourceDataError(f"{filename.name} missing columns: {sorted(missing)}")
        if len(frame) != 351 or not frame.record_id.is_unique or not frame.split.eq('TEST').all():
            raise SourceDataError(f"{filename.name} must contain 351 unique held-out TEST rows")
        for field in ("current_stage_m", "target_stage_m", "pred_stage_m"):
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
            if not np.isfinite(frame[field]).all():
                raise SourceDataError(f"{filename.name} has invalid {field}")
        if not (frame.record_id.astype(str) == frame.station_id.astype(str) + '_' + frame.forecast_date.astype(str)).all():
            raise SourceDataError(f"{filename.name} record IDs do not match station/date")
        frame = frame.sort_values('record_id').reset_index(drop=True)
        ids = set(frame.record_id)
        if shared_ids is None:
            shared_ids = ids
            reference = frame[['record_id', 'station_id', 'forecast_date', 'current_stage_m', 'target_stage_m']].copy()
        elif ids != shared_ids:
            raise SourceDataError(f"{filename.name} uses a different held-out record set")
        else:
            for field in ("current_stage_m", "target_stage_m"):
                if not np.allclose(frame[field].to_numpy(), reference[field].to_numpy(), rtol=0, atol=5e-5):
                    raise SourceDataError(f"{filename.name} differs from other models in actual/current stage")
        frame['model'] = model
        predictions.append(frame)
        available.append(model)

    if not predictions:
        raise SourceDataError(
            f"No M1–M7 test-prediction CSVs found inside {folder}.\n"
            "These are produced by Notebook 07. Do not use training labels as replacement predictions."
        )
    common = source.set_index("record_id")
    if not shared_ids.issubset(set(common.index)):
        raise SourceDataError("Model TEST records are missing from canonical private DMC labels")
    actual = common.loc[reference.record_id]
    for expected_column, actual_column in (("current_stage_m", "stage_at_issue_m"),
                                           ("target_stage_m", "target_stage_next_day_m")):
        if not np.allclose(reference[expected_column].to_numpy(), actual[actual_column].to_numpy(), rtol=0, atol=2e-4):
            raise SourceDataError(f"Saved TEST {expected_column} doesn't match private DMC CSV")
    if len(set(reference.station_id)) != 39 or len(set(reference.forecast_date)) != 9:
        raise SourceDataError('Expected exactly 39 labelled gauges and 9 TEST dates')
    if (reference.forecast_date.min(), reference.forecast_date.max()) != ('2025-12-17', '2025-12-25'):
        raise SourceDataError('The saved prediction files are not the frozen PP1 TEST period')

    # Enrich with station-specific thresholds and existing candidate-polygon file paths.
    enrich = ["record_id", "alert_threshold_m", "minor_flood_threshold_m", "major_flood_threshold_m", "catchment_polygon_path"]
    if 'catchment_qa_status' in source.columns:
        enrich.append('catchment_qa_status')
    predictions_df = pd.concat(predictions, ignore_index=True).merge(
        source[enrich], on='record_id', how='left', validate='many_to_one'
    )
    if predictions_df[enrich[1:]].isna().any().any():
        raise SourceDataError("At least one saved TEST prediction lacks its gauge threshold or catchment provenance")
    predictions_df['abs_error_m'] = (predictions_df.pred_stage_m - predictions_df.target_stage_m).abs()
    predictions_df['signed_error_m'] = predictions_df.pred_stage_m - predictions_df.target_stage_m
    base = predictions_df.loc[predictions_df.model.eq(available[0])].copy()
    base['model'] = 'PERSISTENCE'
    base['pred_stage_m'] = base['current_stage_m']
    base['abs_error_m'] = (base.pred_stage_m - base.target_stage_m).abs()
    base['signed_error_m'] = base.pred_stage_m - base.target_stage_m
    comparison = pd.concat([predictions_df, base], ignore_index=True)
    metrics = (comparison.groupby('model', as_index=False)
               .agg(n=('abs_error_m', 'size'), mae_m=('abs_error_m', 'mean')))
    rmses = comparison.groupby('model').apply(
        lambda f: float(np.sqrt(np.mean(f.signed_error_m.to_numpy() ** 2))),
        include_groups=False,
    ).rename('rmse_m').reset_index()
    metrics = metrics.merge(rmses, on='model', validate='one_to_one')
    aligned_file = root / "data" / "processed" / "c4" / "multimodal" / "c4_aligned_index_1429_PRIVATE.csv"
    aligned = None
    if aligned_file.is_file():
        aligned = pd.read_csv(aligned_file, encoding='utf-8-sig', low_memory=False)
        if not {'record_id', 'station_id', 'forecast_date'}.issubset(aligned.columns):
            aligned = None
        elif not aligned.record_id.is_unique:
            raise SourceDataError('Aligned index contains duplicate record IDs')
    return C4Workspace(
        root=root, stations=stations, station_geojson=geo, observations=source,
        predictions=predictions_df, comparison=comparison, metrics=metrics,
        available_models=tuple(available), missing_models=tuple(m for m in MODELS if m not in available),
        aligned_index=aligned, folder=folder,
    )
