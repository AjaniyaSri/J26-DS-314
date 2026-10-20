"""Read C4's frozen ERA5-Land weather sequences produced by Notebook 04.

Install as: J26-DS-314/c4_fusion/src/weather/load.py

Local inputs (Notebook 04 Cell 10; never commit them to a public repository):
  data/processed/c4/weather/c4_weather_sequences_1429_PRIVATE.npz
  data/processed/c4/weather/c4_weather_sequence_index_1429_PRIVATE.csv
  data/processed/c4/weather/c4_weather_daily_era5land_PRIVATE.csv
  data/processed/c4/weather/c4_weather_QA_PRIVATE.csv
  data/processed/c4/weather/c4_weather_manifest_PRIVATE.json

No Earth Engine, downloads, geopandas, or feature recalculation. The loader
returns UNSTANDARDIZED NumPy float32 [14, 4] sequences. Fit any scalers on
TRAIN ONLY later during model development.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS: tuple[str, ...] = (
    "precipitation_mm",
    "temperature_C",
    "dewpoint_C",
    "surface_pressure_hPa",
)

SOURCE_ASSET = "ECMWF/ERA5_LAND/HOURLY"
SOURCE_VERSION = "C4_WEATHER_ERA5LAND_V1_0830_24H_NATIVE_WEIGHTS"
EXPECTED_SHAPE = (1429, 14, 4)
EXPECTED_GAUGES = 39
EXPECTED_GAUGE_DAYS = 2026
EXPECTED_SPLITS = {"TRAIN": 809, "VALIDATION": 269, "TEST": 351}

ARRAY_FILENAME = "c4_weather_sequences_1429_PRIVATE.npz"
INDEX_FILENAME = "c4_weather_sequence_index_1429_PRIVATE.csv"
DAILY_FILENAME = "c4_weather_daily_era5land_PRIVATE.csv"
QA_FILENAME = "c4_weather_QA_PRIVATE.csv"
MANIFEST_FILENAME = "c4_weather_manifest_PRIVATE.json"


class WeatherDataError(ValueError):
    """Saved weather data, metadata, or requested record alignment is invalid."""


def _repo_root_from_file() -> Path:
    # J26-DS-314/c4_fusion/src/weather/load.py -> J26-DS-314
    root = Path(__file__).resolve().parents[3]
    if not (root / "c4_fusion").is_dir():
        raise FileNotFoundError(
            "Cannot infer repository root. Pass repo_root=Path('.../J26-DS-314') "
            "or data_dir=Path('.../data/processed/c4/weather') explicitly."
        )
    return root


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Notebook 04 export: {path}. "
            "Complete Notebook 04 Cell 10 and keep its outputs together."
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_date(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise WeatherDataError(f"{name}: missing date")
    raw = series.astype(str)
    try:
        result = pd.to_datetime(raw, format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise WeatherDataError(f"{name}: dates must be YYYY-MM-DD") from exc
    if (result.dt.strftime("%Y-%m-%d") != raw).any():
        raise WeatherDataError(f"{name}: noncanonical calendar date")
    return result


class WeatherSequenceLoader:
    """Load the already-built 1429 x 14 x 4 retrospective weather matrix.

    Parameters
    ----------
    repo_root: Repository folder containing ``c4_fusion/`` and ``data/``.
    data_dir: Optional explicit directory holding Notebook 04's five exports.
    verify_provenance: Check manifest hashes, dataset version, daily/QA coverage,
        window dates, and reconstruct each sequence from daily CSV data.
        Recommended for Notebook 06; does not call Earth Engine.

    Public methods
    --------------
    get_sequence(record_id) -> [14,4]
    get_batch(record_ids) -> [N,14,4]
    for_station_days(records) -> [N,14,4], exact input row order;
        require record_id + station_id, cross-check forecast_date/split if given.
    index_table() -> copy of sequence index with metadata (NO targets).
    """

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        verify_provenance: bool = True,
    ) -> None:
        self.repo_root = (
            Path(repo_root).expanduser().resolve() if repo_root is not None
            else (_repo_root_from_file() if data_dir is None else None)
        )
        self.data_dir = (
            Path(data_dir).expanduser().resolve() if data_dir is not None
            else self.repo_root / "data" / "processed" / "c4" / "weather"
        )
        paths = {
            "array": _require(self.data_dir / ARRAY_FILENAME),
            "index": _require(self.data_dir / INDEX_FILENAME),
        }
        if verify_provenance:
            paths.update({
                "daily": _require(self.data_dir / DAILY_FILENAME),
                "qa": _require(self.data_dir / QA_FILENAME),
                "manifest": _require(self.data_dir / MANIFEST_FILENAME),
            })

        self.manifest: dict | None = None
        if verify_provenance:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            expected_hashes = manifest.get("outputs_sha256")
            if not isinstance(expected_hashes, dict):
                raise WeatherDataError("Manifest missing outputs_sha256")
            for key, filename in (
                ("array", ARRAY_FILENAME), ("index", INDEX_FILENAME),
                ("daily", DAILY_FILENAME), ("qa", QA_FILENAME),
            ):
                wanted = expected_hashes.get(filename)
                if not isinstance(wanted, str) or _sha256(paths[key]) != wanted:
                    raise WeatherDataError(
                        f"Manifest SHA256 mismatch for {filename}. "
                        "Do not mix outputs from different Notebook 04 runs."
                    )
            if manifest.get("weather_feature_order") != list(FEATURE_COLUMNS):
                raise WeatherDataError("Manifest weather_feature_order mismatch")
            if manifest.get("shape") != list(EXPECTED_SHAPE):
                raise WeatherDataError("Manifest expected sequence shape mismatch")
            if (manifest.get("n_sequences") != EXPECTED_SHAPE[0]
                or manifest.get("n_gauges") != EXPECTED_GAUGES
                or manifest.get("n_unique_gauge_days") != EXPECTED_GAUGE_DAYS):
                raise WeatherDataError("Manifest record/day/gauge counts are incorrect")
            config = manifest.get("source_config", {})
            if (config.get("asset") != SOURCE_ASSET
                or config.get("source_version") != SOURCE_VERSION
                or config.get("window_hours") != 24
                or config.get("window_end_utc") != "03:00"
                or config.get("precip_band") != "total_precipitation_hourly"):
                raise WeatherDataError("Unexpected ERA5-Land source/version or time convention")
            config_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if manifest.get("source_config_sha256") != config_hash:
                raise WeatherDataError("Manifest source_config_sha256 mismatch")
            self.manifest = manifest

        index = pd.read_csv(paths["index"], dtype={"record_id": "string", "station_id": "string"})
        needed = {
            "record_id", "station_id", "forecast_date", "split", "weather_available",
            "earliest_weather_day", "last_weather_day", "weather_sequence_shape",
            "weather_source", "weather_last_hour_local",
        }
        if set(index.columns) != needed:
            raise WeatherDataError(
                f"Sequence index schema mismatch: missing={sorted(needed - set(index.columns))}, "
                f"unexpected={sorted(set(index.columns) - needed)}"
            )
        if len(index) != EXPECTED_SHAPE[0]:
            raise WeatherDataError(f"Expected 1429 index rows, got {len(index)}")
        for key in ("record_id", "station_id"):
            if index[key].isna().any() or index[key].str.strip().eq("").any():
                raise WeatherDataError(f"Missing/empty {key} in weather index")
        if index["record_id"].duplicated().any():
            raise WeatherDataError("Duplicate weather record_id in sequence index")
        if index[["station_id", "forecast_date"]].duplicated().any():
            raise WeatherDataError("Duplicate (station_id, forecast_date) in sequence index")
        if index["station_id"].nunique() != EXPECTED_GAUGES:
            raise WeatherDataError("Expected 39 labelled stations in weather index")
        if index["split"].value_counts().to_dict() != EXPECTED_SPLITS:
            raise WeatherDataError("Chronological split counts changed")
        if (index["weather_available"].isna().any()
            or not index["weather_available"].astype(str).str.lower().eq("true").all()):
            raise WeatherDataError("Some sequences are not marked weather_available")
        if not index["weather_sequence_shape"].eq("14x4").all():
            raise WeatherDataError("Index contains an unexpected per-record weather shape")
        if not index["weather_source"].eq(SOURCE_ASSET).all():
            raise WeatherDataError("Index weather_source is not ERA5-Land Hourly")
        if not index["weather_last_hour_local"].eq("08:30 Asia/Colombo").all():
            raise WeatherDataError("Weather last-hour convention changed")
        forecast = _strict_date(index["forecast_date"], "forecast_date")
        earliest = _strict_date(index["earliest_weather_day"], "earliest_weather_day")
        last = _strict_date(index["last_weather_day"], "last_weather_day")
        if not (earliest.eq(forecast - pd.Timedelta(days=13)) & last.eq(forecast)).all():
            raise WeatherDataError("14-day weather window dates are not aligned to forecast date")

        # Loading the NPZ is local and safe: allow_pickle=False rejects unsafe pickled objects.
        with np.load(paths["array"], allow_pickle=False) as bundle:
            required_arrays = {"X_weather", "record_id", "station_id", "feature_names"}
            if set(bundle.files) != required_arrays:
                raise WeatherDataError("Unexpected keys in weather .npz export")
            values = bundle["X_weather"]
            npz_ids = bundle["record_id"].astype(str)
            npz_stations = bundle["station_id"].astype(str)
            names = bundle["feature_names"].astype(str).tolist()
        if values.shape != EXPECTED_SHAPE or values.dtype != np.float32:
            raise WeatherDataError(f"Expected float32 [1429,14,4], got {values.shape} {values.dtype}")
        if not np.isfinite(values).all():
            raise WeatherDataError("Weather tensor contains NaN/Inf")
        if names != list(FEATURE_COLUMNS):
            raise WeatherDataError(f"Weather .npz feature order differs from {FEATURE_COLUMNS}")
        if not np.array_equal(npz_ids, index["record_id"].to_numpy(dtype=str)):
            raise WeatherDataError("Array record IDs do NOT align with sequence index rows")
        if not np.array_equal(npz_stations, index["station_id"].to_numpy(dtype=str)):
            raise WeatherDataError("Array station IDs do NOT align with sequence index rows")
        if (values[:, :, 0] < -0.01).any():
            raise WeatherDataError("Unacceptable negative daily precipitation in weather tensor")
        if ((values[:, :, 1] < -35) | (values[:, :, 1] > 65)).any():
            raise WeatherDataError("Implausible temperature in weather tensor")
        if ((values[:, :, 3] < 400) | (values[:, :, 3] > 1100)).any():
            raise WeatherDataError("Implausible pressure in weather tensor")
        self._values = values.copy()
        self._index = index.copy()
        self._position = {str(record_id): int(i) for i, record_id in enumerate(npz_ids)}

        if verify_provenance:
            daily = pd.read_csv(paths["daily"])
            qa = pd.read_csv(paths["qa"])
            daily_needed = {"station_id", "weather_day", *FEATURE_COLUMNS,
                            "hour_count", "cache_key", "window_start_utc", "window_last_hour_utc",
                            "valid_24h_grid_coverage_fraction", "negative_rain_pixel_fraction"}
            qa_needed = {"station_id", "weather_day", "hour_count", "cache_key",
                         "window_start_utc", "window_last_hour_utc",
                         "valid_24h_grid_coverage_fraction", "negative_rain_pixel_fraction"}
            if not daily_needed.issubset(daily.columns) or not qa_needed.issubset(qa.columns):
                raise WeatherDataError("Missing daily or QA provenance columns")
            if len(daily) != EXPECTED_GAUGE_DAYS or len(qa) != EXPECTED_GAUGE_DAYS:
                raise WeatherDataError("Daily weather and QA CSVs must each have 2026 rows")
            if daily[["station_id", "weather_day"]].duplicated().any() or qa[["station_id", "weather_day"]].duplicated().any():
                raise WeatherDataError("Duplicate (station_id, weather_day) in weather/QA")
            daily_days = _strict_date(daily["weather_day"], "daily weather_day")
            _strict_date(qa["weather_day"], "QA weather_day")
            if not daily["hour_count"].eq(24).all() or not qa["hour_count"].eq(24).all():
                raise WeatherDataError("Not every weather-day has 24 ERA5-Land hourly records")
            if not daily["valid_24h_grid_coverage_fraction"].between(0.75, 1.000001).all():
                raise WeatherDataError("At least one weather day has insufficient grid coverage")
            if not daily["negative_rain_pixel_fraction"].between(0, 1).all():
                raise WeatherDataError("Invalid negative-rain diagnostic fraction")
            if not np.isfinite(daily[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
                raise WeatherDataError("Nonfinite values in daily weather CSV")
            # The two daily/QA exports should describe the same cached days and QA values.
            compare = daily.merge(qa, on=["station_id", "weather_day"], how="outer",
                                  suffixes=("_daily", "_qa"), validate="one_to_one", indicator=True)
            if not compare["_merge"].eq("both").all():
                raise WeatherDataError("Daily CSV and QA CSV do not describe identical station-days")
            for col in qa_needed - {"station_id", "weather_day"}:
                left, right = compare[f"{col}_daily"], compare[f"{col}_qa"]
                if col in ("cache_key", "window_start_utc", "window_last_hour_utc"):
                    same = left.eq(right).all()
                else:
                    same = np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), atol=1e-8, rtol=1e-9)
                if not same:
                    raise WeatherDataError(f"QA metadata differs from daily weather CSV: {col}")
            # Cross-check the UTC validity windows for all station-day rows.
            ends = daily_days.dt.strftime("%Y-%m-%d") + "T03:00:00"
            starts = (daily_days - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d") + "T04:00:00"
            if not (daily["window_last_hour_utc"].eq(ends) & daily["window_start_utc"].eq(starts)).all():
                raise WeatherDataError("Weather-day UTC 24-hour validity interval mismatch")
            # Rebuild every 14-day sequence from the source-day statistics.
            daily_sorted = daily.set_index(["station_id", "weather_day"])
            repeated_ids = np.repeat(index["station_id"].to_numpy(dtype=str), 14)
            repeated_dates = np.concatenate([
                (day - pd.to_timedelta(np.arange(13, -1, -1), unit="D")).strftime("%Y-%m-%d").to_numpy()
                for day in forecast
            ])
            key = pd.MultiIndex.from_arrays([repeated_ids, repeated_dates],
                                            names=["station_id", "weather_day"])
            if not key.isin(daily_sorted.index).all():
                raise WeatherDataError("Weather tensor references missing cached station-days")
            rebuilt = daily_sorted.loc[key, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32).reshape(EXPECTED_SHAPE)
            if not np.allclose(rebuilt, values, rtol=2e-6, atol=5e-5):
                diff = np.abs(rebuilt - values)
                where = np.unravel_index(np.argmax(diff), diff.shape)
                raise WeatherDataError(
                    f"Weather arrays differ from daily CSV reconstruction: "
                    f"sample={where[0]}, day={where[1]}, feature={FEATURE_COLUMNS[where[2]]}, "
                    f"max_abs_diff={float(diff.max())}"
                )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature order: precipitation, temperature, dewpoint, surface pressure."""
        return FEATURE_COLUMNS

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._values.shape

    def index_table(self) -> pd.DataFrame:
        """Return a fresh copy of the sequence index; never includes DMC targets."""
        return self._index.copy()

    def get_sequence(self, record_id: str) -> np.ndarray:
        """Return one float32 [14,4] array for a canonical station-day record ID."""
        key = str(record_id)
        if key not in self._position:
            raise KeyError(f"Unknown weather record_id {key!r}")
        return self._values[self._position[key]].copy()

    def get_batch(self, record_ids: Iterable[str]) -> np.ndarray:
        """Return float32 [N,14,4] in the exact requested ID order; repeats allowed."""
        keys = [str(x) for x in record_ids]
        missing = sorted(set(keys) - set(self._position))
        if missing:
            raise KeyError(f"Unknown weather record IDs: {missing[:10]}")
        if not keys:
            return np.empty((0, 14, 4), dtype=np.float32)
        return self._values[[self._position[x] for x in keys]].copy()

    def for_station_days(self, records: pd.DataFrame) -> np.ndarray:
        """Align weather to a Notebook 01 / DMC station-day table by record_id.

        Requires ``record_id`` and ``station_id``. If given, ``forecast_date``
        and ``split`` (or ``chronological_split``) are also checked against
        Notebook 04 metadata. Row order is preserved; labels are not read.
        """
        if not isinstance(records, pd.DataFrame) or not {"record_id", "station_id"}.issubset(records.columns):
            raise WeatherDataError("Input must be a DataFrame with record_id and station_id")
        if records[["record_id", "station_id"]].isna().any().any():
            raise WeatherDataError("Missing station_id or record_id in requested station-days")
        keys = records["record_id"].astype(str).tolist()
        missing = sorted(set(keys) - set(self._position))
        if missing:
            raise KeyError(f"Unknown weather record IDs: {missing[:10]}")
        selected = self._index.iloc[[self._position[key] for key in keys]].reset_index(drop=True)
        supplied_station = records["station_id"].astype(str).reset_index(drop=True)
        if not selected["station_id"].eq(supplied_station).all():
            raise WeatherDataError("Station ID mismatch for one or more weather record IDs")
        if "forecast_date" in records.columns:
            supplied = _strict_date(records["forecast_date"].reset_index(drop=True), "requested forecast_date")
            expected = _strict_date(selected["forecast_date"], "saved forecast_date")
            if not supplied.equals(expected):
                raise WeatherDataError("Forecast date mismatch for one or more weather record IDs")
        for user_col in ("split", "chronological_split"):
            if user_col in records.columns:
                supplied = records[user_col].astype(str).reset_index(drop=True)
                if not supplied.eq(selected["split"]).all():
                    raise WeatherDataError(f"{user_col} mismatch for one or more weather record IDs")
        return self.get_batch(keys)


__all__ = ["FEATURE_COLUMNS", "WeatherDataError", "WeatherSequenceLoader"]
