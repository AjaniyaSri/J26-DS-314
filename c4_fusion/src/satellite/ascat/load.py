"""Read-only ASCAT SWI V4 station-day loader for C4 FloodRisk Sri Lanka.

Source: Notebook 03 exports at data/processed/c4/satellite/ascat/.
No HTTP, Copernicus credentials, geospatial extraction, label reading, or scaling.

Inputs returned are *unscaled* SWI001/005/010 percentages and separate
quality, age, provenance, and availability. Missing observations are represented
by zero-valued *batch placeholders* only when swi_available=False; never treat
those placeholders as real wetness observations.

Usage (from the repository root or with that directory on sys.path):
    from c4_fusion.src.satellite.ascat.load import ASCATSWILoader
    loader = ASCATSWILoader()
    one = loader.get_sample("holombuwa_2025-11-17")
    batch = loader.for_station_days(sample_index)  # 1429 x 3 SWI inputs
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("swi001_percent", "swi005_percent", "swi010_percent")
QUALITY_COLUMNS = ("qflag001_percent", "qflag005_percent", "qflag010_percent")
SAMPLE_FILENAME = "c4_ascat_swi_sample_index_1429_PRIVATE.csv"
DAILY_FILENAME = "c4_ascat_swi_daily_39_PRIVATE.csv"
QA_FILENAME = "c4_ascat_swi_row_QA_1429_PRIVATE.csv"
MANIFEST_FILENAME = "c4_ascat_swi_manifest_PRIVATE.json"
EXPECTED_VERSION = "C4_ASCAT_SWIV4_V1_DAY_MINUS1_NATIVE_01DEG_JOINTQA"
EXPECTED_COUNTS = {"TRAIN": 809, "VALIDATION": 269, "TEST": 351}
EXPECTED_STATIONS = 39
EXPECTED_RECORDS = 1429
MAX_AGE_DAYS = 7
MIN_QFLAG = 20.0
ALLOWED_METHODS = {"CATCHMENT_AREA_WEIGHTED", "NEAREST_NATIVE_CELL", "MISSING"}


class ASCATDataError(ValueError):
    """ASCAT files do not satisfy the frozen C4 Notebook 03 contract."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_root_from_file() -> Path:
    # <root>/c4_fusion/src/satellite/ascat/load.py
    return Path(__file__).resolve().parents[4]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ASCATDataError(message)


def _parse_dates(series: pd.Series, name: str) -> pd.Series:
    try:
        dates = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ASCATDataError(f"Invalid {name} ISO date: {exc}") from exc
    _expect(bool(dates.notna().all()), f"Missing {name} dates")
    return dates


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for field in columns:
        _expect(field in df.columns, f"{name} lacks required column {field}")
        try:
            df[field] = pd.to_numeric(df[field], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ASCATDataError(f"Non-numeric values in {name}.{field}") from exc


def _validate_boolean(column: pd.Series, name: str) -> None:
    _expect(pd.api.types.is_bool_dtype(column), f"{name} must contain true/false booleans")
    _expect(bool(column.notna().all()), f"{name} contains missing booleans")


def _same(left: pd.Series, right: pd.Series, field: str) -> None:
    _expect(left.reset_index(drop=True).equals(right.reset_index(drop=True)),
            f"Sample/QA {field} differs in values or order")


class ASCATSWILoader:
    """Verify and align Notebook 03 ASCAT SWI observations to station-day rows.

    Parameters
    ----------
    repo_root : path-like, optional
        Repository root containing c4_fusion/ and data/processed/c4/.
    data_dir : path-like, optional
        Directory containing the four Notebook 03 private exports.
    verify_provenance : bool
        Check all exported-file SHA256 digests and immutable source schema.
    verify_daily_alignment : bool
        Check that every selected observation reproduces its source daily row.
    verify_original_index : bool
        If Notebook 01's original sample index is present, require an exact
        SHA256 match to the Notebook 03 source-index fingerprint.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        verify_provenance: bool = True,
        verify_daily_alignment: bool = True,
        verify_original_index: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else _repo_root_from_file()
        self.data_dir = (Path(data_dir).expanduser().resolve() if data_dir
                         else self.repo_root / "data/processed/c4/satellite/ascat")
        paths = {name: self.data_dir / name for name in
                 (SAMPLE_FILENAME, DAILY_FILENAME, QA_FILENAME, MANIFEST_FILENAME)}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Required ASCAT Notebook 03 export missing: {path}")

        try:
            self._manifest = json.loads(paths[MANIFEST_FILENAME].read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ASCATDataError("Invalid ASCAT provenance manifest JSON") from exc
        m = self._manifest
        _expect(m.get("source_version") == EXPECTED_VERSION, "Unexpected ASCAT source version")
        _expect(m.get("feature_order") == list(FEATURE_COLUMNS), "Unexpected SWI feature order")
        _expect(m.get("qa_feature_order") == list(QUALITY_COLUMNS), "Unexpected QFLAG order")
        _expect(m.get("n_station_days") == EXPECTED_RECORDS, "Unexpected manifest record count")
        _expect(m.get("n_labelled_stations") == EXPECTED_STATIONS, "Unexpected gauge count")
        _expect(m.get("n_required_product_dates") == 45, "Unexpected historical product-date count")
        config = m.get("source_config", {})
        _expect(config.get("max_age_days") == MAX_AGE_DAYS, "Different SWI age policy")
        _expect(config.get("min_qflag_percent") == MIN_QFLAG, "Different SWI quality threshold")
        _expect(config.get("time_policy") == "product date < forecast date (T-1 or older)",
                "Different SWI source-date policy")
        if verify_provenance:
            for name in (SAMPLE_FILENAME, DAILY_FILENAME, QA_FILENAME):
                expected = m.get("output_sha256", {}).get(name)
                _expect(isinstance(expected, str) and len(expected) == 64,
                        f"Missing stored checksum for {name}")
                _expect(_sha256(paths[name]) == expected,
                        f"Checksum mismatch for {name}; files may have changed")
            # Frozen configuration hash is calculated exactly as in Notebook 03.
            config_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest()
            _expect(config_hash == m.get("config_hash"), "ASCAT source configuration hash mismatch")
            canonical = self.repo_root / "data/processed/c4/manifests/c4_sample_index_1429.csv"
            if verify_original_index and canonical.is_file():
                _expect(_sha256(canonical) == m.get("source_index_sha256"),
                        "Notebook 01 index changed since Notebook 03 extraction")

        # This loader owns no target labels. It never opens Notebook 01's PRIVATE labels.
        samples = pd.read_csv(paths[SAMPLE_FILENAME], low_memory=False)
        daily = pd.read_csv(paths[DAILY_FILENAME], low_memory=False)
        qa = pd.read_csv(paths[QA_FILENAME], low_memory=False)
        self._validate_samples(samples, m)
        self._validate_qa(samples, qa)
        if verify_daily_alignment:
            self._validate_daily(samples, daily)
        self._samples = samples.reset_index(drop=True)
        self._daily = daily.reset_index(drop=True)
        self._qa = qa.reset_index(drop=True)
        self._lookup = {record_id: pos for pos, record_id in enumerate(self._samples.record_id)}

    @staticmethod
    def _validate_samples(a: pd.DataFrame, manifest: dict) -> None:
        required = {"record_id", "station_id", "forecast_date", "split", "swi_available",
                    "swi_source_date", "swi_age_days", "swi_age_logscaled",
                    "swi_validity_flag", "extraction_method", "area_valid_fraction",
                    "nearest_distance_km", *FEATURE_COLUMNS, *QUALITY_COLUMNS}
        _expect(required.issubset(a.columns), f"Missing sample columns: {sorted(required-set(a.columns))}")
        _expect(len(a) == EXPECTED_RECORDS, f"Expected {EXPECTED_RECORDS} SWI station-days, got {len(a)}")
        _expect(a.record_id.notna().all() and a.record_id.is_unique, "Duplicate/missing record IDs")
        _expect(a.station_id.nunique() == EXPECTED_STATIONS, "Unexpected number of stations")
        _expect(a.split.value_counts().to_dict() == EXPECTED_COUNTS, "Chronological split counts changed")
        _expect((a.record_id == a.station_id.astype(str) + "_" + a.forecast_date.astype(str)).all(),
                "Record IDs do not match station ID and forecast date")
        _expect(not a[["station_id", "forecast_date"]].duplicated().any(),
                "Duplicate station-day pair")
        _validate_boolean(a.swi_available, "swi_available")
        _ensure_numeric(a, (*FEATURE_COLUMNS, *QUALITY_COLUMNS, "swi_age_days", "swi_age_logscaled",
                            "swi_validity_flag", "area_valid_fraction", "nearest_distance_km"), "sample")
        _expect(a.extraction_method.isin(ALLOWED_METHODS).all(), "Unexpected extraction method")
        flag = a.swi_available.to_numpy(dtype=bool)
        _expect(np.array_equal(a.swi_validity_flag.to_numpy(), flag.astype(int)),
                "SWI availability flag inconsistent")
        _expect(a.loc[flag, "extraction_method"].isin(ALLOWED_METHODS - {"MISSING"}).all(),
                "Available SWI must have a real extraction method")
        _expect(a.loc[~flag, "extraction_method"].eq("MISSING").all(),
                "Missing SWI must have method MISSING")
        date_forecast = _parse_dates(a.forecast_date, "forecast_date")
        if flag.any():
            sel = a.loc[flag]
            dt = date_forecast.loc[flag] - _parse_dates(sel.swi_source_date, "swi_source_date")
            days = dt.dt.days.to_numpy(dtype=float)
            _expect(np.array_equal(days, sel.swi_age_days.to_numpy(dtype=float)),
                    "SWI source dates do not match age in days")
            _expect(np.all((days >= 1) & (days <= MAX_AGE_DAYS)),
                    "Future-day or stale SWI source detected")
            scaled = np.log1p(days) / math.log1p(MAX_AGE_DAYS)
            _expect(np.allclose(scaled, sel.swi_age_logscaled.to_numpy(dtype=float), atol=1e-12, rtol=0),
                    "Wrong age log-scaling")
            values = sel.loc[:, list(FEATURE_COLUMNS + QUALITY_COLUMNS)].to_numpy(dtype=float)
            _expect(np.isfinite(values).all(), "Missing/invalid SWI or QFLAG despite availability")
            _expect(np.all((values >= 0) & (values <= 100)), "SWI or QFLAG outside 0-100%")
            _expect(np.all(sel.loc[:, list(QUALITY_COLUMNS)].to_numpy(dtype=float) >= MIN_QFLAG - 1e-9),
                    "Quality below minimum threshold")
            area = sel.area_valid_fraction.to_numpy(dtype=float)
            _expect(np.isfinite(area).all() and np.all((area >= 0) & (area <= 1 + 1e-8)),
                    "Invalid catchment coverage")
            method = sel.extraction_method.eq("CATCHMENT_AREA_WEIGHTED").to_numpy()
            _expect(np.all(area[method] >= 0.5 - 1e-9), "Catchment extraction below accepted area fraction")
            near = sel.extraction_method.eq("NEAREST_NATIVE_CELL").to_numpy()
            distance = sel.nearest_distance_km.to_numpy(dtype=float)
            _expect(np.isnan(distance[~near]).all(), "Catchment extraction unexpectedly has nearest distance")
            _expect(np.isfinite(distance[near]).all() and np.all((distance[near] >= 0) & (distance[near] <= 25)),
                    "Nearest-grid fallback outside permitted distance")
        if (~flag).any():
            missing = a.loc[~flag]
            _expect(missing.loc[:, list(FEATURE_COLUMNS + QUALITY_COLUMNS)].isna().all().all(),
                    "Missing SWI contains fabricated wetness/quality observations")
            _expect(missing.swi_source_date.isna().all(), "Missing SWI has a source date")
            _expect(missing.swi_age_days.isna().all(), "Missing SWI has a numerical age")
            _expect((missing.swi_age_logscaled == 0).all(), "Missing SWI age placeholder must be 0")
        _expect(int(flag.sum()) == manifest.get("n_available") and
                int((~flag).sum()) == manifest.get("n_missing"),
                "Availability counts differ from frozen manifest")

    @staticmethod
    def _validate_qa(a: pd.DataFrame, qa: pd.DataFrame) -> None:
        _expect(len(qa) == EXPECTED_RECORDS, "Wrong SWI row QA count")
        shared = ("record_id", "station_id", "forecast_date", "split", "swi_available",
                  "swi_source_date", "swi_age_days", "extraction_method", "area_valid_fraction", "nearest_distance_km")
        for field in shared:
            _expect(field in qa, f"QA missing {field}")
            left = a[field]
            right = qa[field]
            if pd.api.types.is_numeric_dtype(left):
                _expect(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                    equal_nan=True, rtol=0, atol=1e-10), f"QA {field} mismatches samples")
            else:
                _same(left, right, field)
        for field in ("id_match", "split_match", "validity_consistent", "pre_origin_source",
                      "feature_complete_when_available", "missing_unfabricated", "qa_pass"):
            _expect(field in qa.columns, f"QA missing {field}")
            _validate_boolean(qa[field], f"row_qa.{field}")
            _expect(qa[field].all(), f"QA failures present in {field}")

    @staticmethod
    def _validate_daily(a: pd.DataFrame, daily: pd.DataFrame) -> None:
        _expect(len(daily) == 39 * 45, "Daily table does not cover all 39 gauges x 45 product dates")
        _expect(not daily[["station_id", "product_date"]].duplicated().any(), "Duplicate daily SWI source rows")
        _expect(daily.station_id.nunique() == EXPECTED_STATIONS and daily.product_date.nunique() == 45,
                "Daily source coverage mismatch")
        _validate_boolean(daily.quality_accepted, "daily.quality_accepted")
        daily_index = daily.set_index(["station_id", "product_date"])
        for record in a.itertuples(index=False):
            # Verify Notebook 03's newest-accepted T-1 ... T-7 policy,
            # not merely that the chosen date is earlier than the forecast.
            forecast = pd.Timestamp(record.forecast_date)
            candidates = [
                (record.station_id, (forecast - pd.Timedelta(days=lag)).strftime("%Y-%m-%d"))
                for lag in range(1, MAX_AGE_DAYS + 1)
            ]
            expected_date = None
            for cand in candidates:
                _expect(cand in daily_index.index,
                        f"Required historical SWI daily row missing: {cand}")
                if bool(daily_index.loc[cand, "quality_accepted"]):
                    expected_date = cand[1]
                    break
            if expected_date is None:
                _expect(not record.swi_available,
                        f"SWI marked available without an accepted source day: {record.record_id}")
                continue
            _expect(bool(record.swi_available),
                    f"SWI marked missing despite accepted historical source: {record.record_id}")
            _expect(record.swi_source_date == expected_date,
                    f"SWI did not select the newest quality-accepted T-1..T-7 source: {record.record_id}")
            key = (record.station_id, record.swi_source_date)
            _expect(key in daily_index.index, f"Source daily SWI row missing for {record.record_id}")
            source = daily_index.loc[key]
            _expect(bool(source.quality_accepted), f"Unaccepted SWI source day: {record.record_id}")
            _expect(record.extraction_method == source.extraction_method,
                    f"SWI extraction method differs from source: {record.record_id}")
            for field in (*FEATURE_COLUMNS, *QUALITY_COLUMNS, "area_valid_fraction", "nearest_distance_km"):
                v = getattr(record, field)
                u = source[field]
                _expect(bool(np.isclose(v, u, equal_nan=True, rtol=1e-10, atol=1e-10)),
                        f"SWI source daily value mismatch for {record.record_id}, {field}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_COLUMNS

    @property
    def quality_names(self) -> tuple[str, ...]:
        return QUALITY_COLUMNS

    @property
    def station_count(self) -> int:
        return int(self._samples.station_id.nunique())

    @property
    def record_count(self) -> int:
        return len(self._samples)

    def table(self) -> pd.DataFrame:
        """Return a copy of the verified target-free sample index."""
        return self._samples.copy(deep=True)

    def daily_table(self) -> pd.DataFrame:
        """Return a copy of the source catchment-day observations."""
        return self._daily.copy(deep=True)

    def get_sample(self, record_id: str) -> dict:
        """Get SWI values, quality, age, and validity for one station-day."""
        if str(record_id) not in self._lookup:
            raise ASCATDataError(f"Unknown SWI record_id: {record_id}")
        row = self._samples.iloc[self._lookup[str(record_id)]]
        available = bool(row.swi_available)
        return {
            "record_id": str(row.record_id),
            "station_id": str(row.station_id),
            "forecast_date": str(row.forecast_date),
            "split": str(row.split),
            "swi_values": (row.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
                           if available else np.zeros(3, dtype=np.float32)),
            "swi_quality": (row.loc[list(QUALITY_COLUMNS)].to_numpy(dtype=np.float32)
                            if available else np.zeros(3, dtype=np.float32)),
            "swi_available": available,
            "swi_validity_flag": int(row.swi_validity_flag),
            "swi_source_date": str(row.swi_source_date) if available else None,
            "swi_age_days": float(row.swi_age_days) if available else math.nan,
            "swi_age_logscaled": float(row.swi_age_logscaled),
            "extraction_method": str(row.extraction_method),
            "area_valid_fraction": float(row.area_valid_fraction),
            "nearest_distance_km": float(row.nearest_distance_km),
        }

    def get_batch(self, record_ids: Iterable[str]) -> dict:
        """Return aligned arrays in the *requested* order, preserving repeats."""
        ids = [str(x) for x in record_ids]
        if not ids:
            raise ASCATDataError("Cannot create an empty ASCAT SWI batch")
        missing = [rid for rid in ids if rid not in self._lookup]
        if missing:
            raise ASCATDataError(f"Unknown SWI record IDs: {missing[:5]}")
        block = self._samples.iloc[[self._lookup[x] for x in ids]]
        available = block.swi_available.to_numpy(dtype=bool)
        values = block.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        quality = block.loc[:, list(QUALITY_COLUMNS)].to_numpy(dtype=np.float32)
        # Batch placeholders are never genuine measurements: mask them downstream.
        values[~available] = 0.0
        quality[~available] = 0.0
        _expect(np.isfinite(values).all() and np.isfinite(quality).all(), "Invalid batch SWI values")
        return {
            "record_ids": np.asarray(ids, dtype=str),
            "swi_values": values,                         # (N,3)
            "swi_quality": quality,                       # (N,3)
            "swi_available": available,                    # (N,)
            "swi_validity_flag": available.astype(np.uint8),  # (N,)
            "swi_age_days": block.swi_age_days.to_numpy(dtype=np.float32),
            "swi_age_logscaled": block.swi_age_logscaled.to_numpy(dtype=np.float32),
            "area_valid_fraction": block.area_valid_fraction.to_numpy(dtype=np.float32),
            "nearest_distance_km": block.nearest_distance_km.to_numpy(dtype=np.float32),
            "swi_source_dates": block.swi_source_date.fillna("").to_numpy(dtype=str),
            "extraction_methods": block.extraction_method.to_numpy(dtype=str),
        }

    def for_station_days(self, records: pd.DataFrame) -> dict:
        """Join target-free station-day records without reordering or dropping rows.

        Requires a record_id and station_id; also validates forecast_date/split if
        present. It intentionally never accesses next-stage labels or model targets.
        """
        _expect(isinstance(records, pd.DataFrame), "Expected a pandas DataFrame")
        _expect({"record_id", "station_id"}.issubset(records.columns),
                "Input must contain record_id and station_id")
        ids = records.record_id.astype(str).tolist()
        batch = self.get_batch(ids)
        source = self._samples.iloc[[self._lookup[x] for x in ids]].reset_index(drop=True)
        check = records.reset_index(drop=True)
        for field in ("station_id", "forecast_date", "split"):
            if field in check.columns:
                _expect((check[field].astype(str).to_numpy() == source[field].astype(str).to_numpy()).all(),
                        f"Station-day {field} mismatch: wrong ASCAT observation would be attached")
        return batch


# Optional spelling alias for projects using CamelCase style.
AscatSWILoader = ASCATSWILoader

__all__ = ["ASCATSWILoader", "AscatSWILoader", "ASCATDataError", "FEATURE_COLUMNS", "QUALITY_COLUMNS"]
