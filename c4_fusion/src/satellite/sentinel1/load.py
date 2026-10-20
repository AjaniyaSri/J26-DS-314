"""Read-only Sentinel-1 VV/VH loader for FloodRisk Sri Lanka C4.

Notebook 02 does the Earth Engine retrieval and pixel QA; this module ONLY reads
its exported target-free station-day index and cached, validated image patches.

Example (run with repository root on sys.path)::

    from c4_fusion.src.satellite.sentinel1.load import Sentinel1PatchLoader

    sar = Sentinel1PatchLoader()
    one = sar.get_sample("holombuwa_2025-11-27")
    # one['sar_image']: [256, 256, 2], float32, in source dB
    # one['sar_valid_mask']: [256, 256], bool
    # one['sar_available']: 0 or 1
    # one['sar_age_hours']: NaN if SAR is unavailable

    # For an aligned subset, not all 1,429 large images in memory at once:
    X = sar.get_batch(sample_index.iloc[:8])
    for batch in sar.iter_batches(sample_index, batch_size=16):
        ...

IMPORTANT MODEL RULES:
- If no SAR image exists, a zero tensor is returned ONLY as a batch placeholder.
  It is not a real radar observation. Gate the CNN output with sar_available.
- sar_age_logscaled==0 on missing samples is NOT a zero-hour-old image; the
  availability flag must accompany it. Raw sar_age_hours remains NaN.
- Valid source pixels are raw dB and have NOT been normalized. Train-only
  statistics and masking must be applied in the future model/data pipeline.
- Invalid pixels inside otherwise accepted patches are neutral-filled with
  zero for array transport; sar_valid_mask marks them as absent.
- ASCAT SWI is separate. Missing SAR does not imply missing SWI.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterator
import json

import numpy as np
import pandas as pd


class Sentinel1DataError(ValueError):
    """Notebook 02 output violates the frozen Sentinel-1 data contract."""


INDEX_FILENAME = "c4_s1_sample_index_1429_PRIVATE.csv"
MANIFEST_FILENAME = "c4_s1_manifest_PRIVATE.json"
EXPECTED_SOURCE_VERSION = "C4_S1_V1_DUALPOL_IW_UTM44_50M_NATIVE_DB_SAMPLE"
IMAGE_SHAPE = (256, 256, 2)
MASK_SHAPE = (256, 256)
EXPECTED_SAMPLES = 1429
EXPECTED_STATIONS = 39
EXPECTED_SPLITS = {"TRAIN": 809, "VALIDATION": 269, "TEST": 351}
REQUIRED_COLUMNS = (
    "record_id", "station_id", "forecast_date", "split", "sar_available",
    "sar_status", "sar_scene_id", "sar_acquisition_utc", "sar_age_hours",
    "sar_age_logscaled", "sar_validity_flag", "sar_valid_pixel_fraction",
    "sar_orbit_pass", "sar_relative_orbit", "patch_relpath", "patch_sha256",
)


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def _resolve_default_root() -> Path:
    # <repo>/c4_fusion/src/satellite/sentinel1/load.py
    return Path(__file__).resolve().parents[4]


def _require(condition: bool, explanation: str) -> None:
    if not bool(condition):
        raise Sentinel1DataError(explanation)


class Sentinel1PatchLoader:
    """Validate sample metadata once and lazily read each unique cached patch.

    Args:
        repo_root: J26-DS-314 repository root. Usually found automatically.
        data_dir: Override directory containing Notebook 02's index & manifest.
        verify_provenance: Check exported index SHA256, source configuration,
            and Notebook 01's target-free index SHA256 if the file exists.
        verify_patch_hashes: Verify cached patch SHA256 on first read (default).

    Loading a 1,429-row index does NOT load images or contact Earth Engine.
    Images are decoded only when get_sample/get_batch is called. A small
    per-loader LRU cache avoids repeated decoding of shared station-scene files.
    """

    def __init__(
        self,
        repo_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        *,
        verify_provenance: bool = True,
        verify_patch_hashes: bool = True,
    ) -> None:
        self.repo_root = (Path(repo_root) if repo_root is not None
                          else _resolve_default_root()).expanduser().resolve()
        self.data_dir = (Path(data_dir) if data_dir is not None
                         else self.repo_root / "data/processed/c4/satellite/sentinel1").expanduser().resolve()
        self.index_path = self.data_dir / INDEX_FILENAME
        self.manifest_path = self.data_dir / MANIFEST_FILENAME
        for path in (self.index_path, self.manifest_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing Notebook 02 output: {path}. Keep generated data under "
                    "data/processed/c4/satellite/sentinel1; do not rerun Earth Engine."
                )
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.verify_patch_hashes = bool(verify_patch_hashes)
        self._validate_manifest(verify_provenance)
        self._frame = pd.read_csv(self.index_path, encoding="utf-8-sig", low_memory=False)
        self._validate_index()
        self._by_id = self._frame.set_index("record_id", drop=False)

        # Cache only a few UNIQUE, already-verified scene patches (not full study tensors).
        @lru_cache(maxsize=8)
        def read_cached(relative_path: str, expected_hash: str) -> tuple[np.ndarray, np.ndarray]:
            return self._read_patch(relative_path, expected_hash)

        self._read_cached = read_cached

    def _validate_manifest(self, verify_provenance: bool) -> None:
        m = self.manifest
        _require(m.get("source_version") == EXPECTED_SOURCE_VERSION,
                 "Unexpected Sentinel-1 source version; do not mix extraction configurations.")
        cfg = m.get("source_config", {})
        _require(cfg.get("source") == "COPERNICUS/S1_GRD", "Incorrect SAR source collection.")
        _require(cfg.get("bands") == ["VV", "VH"] and cfg.get("mode") == "IW",
                 "Expected IW Sentinel-1 VV/VH bands in this order.")
        _require(cfg.get("grid_size") == IMAGE_SHAPE[0] and cfg.get("grid_crs") == "EPSG:32644",
                 "Unexpected SAR patch grid.")
        _require(float(cfg.get("pixel_metres", -1)) == 50.0 and
                 float(cfg.get("max_age_days", -1)) == 18.0 and
                 float(cfg.get("min_valid_fraction", -1)) == 0.97,
                 "Unexpected SAR 50m/18-day/97% selection policy.")
        _require(m.get("n_labelled_station_days") == EXPECTED_SAMPLES and
                 m.get("n_labelled_gauges") == EXPECTED_STATIONS,
                 "Unexpected Sentinel-1 record or station count in manifest.")
        _require(m.get("n_available_station_days") == 1408 and
                 m.get("n_unique_accepted_patches") == 261,
                 "Manifest differs from the audited Notebook 02 export (1408/261).")
        if verify_provenance:
            recorded = m.get("output_sha256", {}).get(INDEX_FILENAME)
            _require(isinstance(recorded, str) and recorded == _digest(self.index_path),
                     "Sentinel-1 sample index SHA256 differs from Notebook 02 manifest.")
            expected = m.get("notebook01_target_free_index_sha256")
            _require(isinstance(expected, str) and len(expected) == 64,
                     "Notebook 01 source-index SHA256 missing from SAR manifest.")
            original = self.repo_root / "data/processed/c4/manifests/c4_sample_index_1429.csv"
            if original.is_file():
                _require(_digest(original) == expected,
                         "Notebook 01 sample index changed after SAR extraction.")

    def _validate_index(self) -> None:
        df = self._frame
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        _require(not missing, f"Sentinel-1 index is missing columns: {missing}")
        _require(len(df) == EXPECTED_SAMPLES, f"Expected 1429 station-days, got {len(df)}.")
        _require(df["record_id"].notna().all() and df["record_id"].is_unique,
                 "Null or duplicate SAR record IDs.")
        _require(df["station_id"].nunique() == EXPECTED_STATIONS,
                 "Expected 39 labelled stations.")
        _require(df["split"].value_counts().to_dict() == EXPECTED_SPLITS,
                 "SAR chronology split counts differ from Notebook 01.")
        dates = pd.to_datetime(df["forecast_date"], format="%Y-%m-%d", errors="coerce", utc=True)
        _require(dates.notna().all(), "Invalid forecast date in SAR index.")
        _require((df["record_id"] == df["station_id"] + "_" + df["forecast_date"]).all(),
                 "Record/station/date identity mismatch.")
        _require(df["sar_available"].notna().all() and
                 df["sar_available"].map(lambda x: isinstance(x, (bool, np.bool_))).all(),
                 "sar_available must contain genuine boolean values.")
        available = df["sar_available"]
        _require(int(available.sum()) == self.manifest["n_available_station_days"],
                 "SAR available count differs from manifest.")
        _require(df["sar_validity_flag"].isin([0, 1]).all() and
                 (df["sar_validity_flag"].astype(bool) == available).all(),
                 "SAR validity flag and availability disagree.")
        _require((df.loc[available, "sar_status"] == "AVAILABLE").all(),
                 "An available SAR sample lacks AVAILABLE status.")
        _require(df.loc[~available, "sar_status"].isin(
            ["ALL_CANDIDATES_REJECTED", "NO_ELIGIBLE_SCENE"]).all(),
                 "Missing SAR must have an explicit final no-image status.")
        _require(df.loc[available, "sar_scene_id"].notna().all() and
                 df.loc[available, "patch_relpath"].notna().all() and
                 df.loc[available, "patch_sha256"].str.fullmatch(r"[0-9a-f]{64}").all(),
                 "Available SAR sample lacks scene reference, path or SHA256.")
        _require(df.loc[available, "sar_acquisition_utc"].notna().all() and
                 df.loc[available, "sar_orbit_pass"].isin(["ASCENDING", "DESCENDING"]).all() and
                 df.loc[available, "sar_relative_orbit"].notna().all(),
                 "Available SAR sample is missing acquisition or orbit metadata.")
        _require(df.loc[available, "sar_valid_pixel_fraction"].between(
            float(self.manifest["source_config"]["min_valid_fraction"]), 1.0).all(),
                 "Accepted SAR scene falls below 97% valid pixel coverage.")
        _require(df.loc[~available, ["sar_scene_id", "sar_acquisition_utc", "patch_relpath",
                 "patch_sha256", "sar_orbit_pass", "sar_relative_orbit",
                 "sar_valid_pixel_fraction", "sar_age_hours"]].isna().all().all(),
                 "Missing SAR row contains scene/age/path/pixel information.")
        _require((df.loc[~available, "sar_age_logscaled"] == 0.0).all(),
                 "Missing SAR must have age_logscaled=0 (paired with availability=0).")
        age = pd.to_numeric(df.loc[available, "sar_age_hours"], errors="coerce")
        age_scaled = pd.to_numeric(df.loc[available, "sar_age_logscaled"], errors="coerce")
        _require(age.notna().all() and np.isfinite(age).all() and age.gt(0).all() and
                 age.le(18 * 24).all(), "SAR age is missing, negative or above 18 days.")
        _require(age_scaled.notna().all() and np.isfinite(age_scaled).all(),
                 "Available SAR scaled age contains missing or nonfinite values.")
        expected_scaled = np.log1p(age.to_numpy(dtype=np.float64) / 24.0) / np.log1p(18.0)
        _require(np.allclose(age_scaled.to_numpy(dtype=np.float64), expected_scaled,
                             rtol=0, atol=1e-6), "SAR scaled-age transformation is inconsistent.")
        acquired = pd.to_datetime(df.loc[available, "sar_acquisition_utc"], utc=True, errors="coerce")
        _require(acquired.notna().all(), "Invalid SAR acquisition time.")
        # Local Sri Lanka origin 09:00 is always 03:30 UTC (UTC+05:30).
        cutoff = dates.loc[available] + pd.Timedelta(hours=3, minutes=30)
        actual_age = (cutoff - acquired).dt.total_seconds() / 3600.0
        _require((actual_age > 0).all() and np.allclose(
            actual_age.to_numpy(), age.to_numpy(), rtol=0, atol=1e-4),
            "SAR acquisition is after the origin or reported age is inconsistent.")
        unique_patches = df.loc[available, ["station_id", "sar_scene_id"]].drop_duplicates()
        _require(len(unique_patches) == self.manifest["n_unique_accepted_patches"],
                 "Number of distinct accepted SAR station-scenes changed.")
        grouped = df.loc[available].groupby(["station_id", "sar_scene_id"], dropna=False)
        for col in ("patch_relpath", "patch_sha256", "sar_valid_pixel_fraction", "sar_acquisition_utc"):
            _require(grouped[col].nunique().eq(1).all(),
                     f"One cached scene has conflicting {col} across forecast dates.")
        # Ensure all patch links remain private and inside the expected repo cache.
        for relative in df.loc[available, "patch_relpath"].drop_duplicates():
            self._resolve_patch_path(relative)
        _require(not set(df.columns).intersection(
            {"target_delta_stage_m", "target_stage_next_day_m", "stage_at_issue_m"}),
            "Target data unexpectedly appear inside the Sentinel-1 index.")

    def _resolve_patch_path(self, relative_path: str) -> Path:
        value = str(relative_path).replace("\\", "/")
        relative = Path(value)
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 f"Unsafe SAR cache path: {value}")
        expected_prefix = Path("data/processed/c4/satellite/sentinel1/cache/patches")
        _require(relative.parts[:len(expected_prefix.parts)] == expected_prefix.parts,
                 f"SAR path is outside Notebook 02's accepted patch cache: {value}")
        full = (self.repo_root / relative).resolve()
        _require(full.is_relative_to(self.repo_root), f"SAR cache path leaves repo: {value}")
        return full

    def _read_patch(self, relative_path: str, expected_hash: str) -> tuple[np.ndarray, np.ndarray]:
        path = self._resolve_patch_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Cached Sentinel-1 image not present: {path}. Copy the original Notebook 02 "
                "cache/patches directory to this machine; do not recreate missing pixels."
            )
        if self.verify_patch_hashes and _digest(path) != expected_hash:
            raise Sentinel1DataError(f"Sentinel-1 patch SHA256 changed: {path}")
        with np.load(path, allow_pickle=False) as f:
            _require("sar_db" in f.files and "valid_mask" in f.files,
                     f"Cached patch lacks sar_db/valid_mask: {path}")
            image = f["sar_db"].astype(np.float32, copy=True)
            valid = f["valid_mask"].astype(bool, copy=True)
        _require(image.shape == IMAGE_SHAPE and valid.shape == MASK_SHAPE,
                 f"Incorrect cached SAR array shapes: {path}")
        _require(valid.mean() >= float(self.manifest["source_config"]["min_valid_fraction"]),
                 f"Accepted cached patch now fails pixel-validity threshold: {path}")
        _require(np.isfinite(image[valid]).all(),
                 f"Valid SAR pixels must have finite VV/VH dB: {path}")
        _require(np.isnan(image[~valid]).all(),
                 f"Invalid SAR pixels must remain NaN in raw cached array: {path}")
        # Only neutral-fill INVALID pixels for batching. Mask + availability are mandatory.
        image[~valid] = 0.0
        image.setflags(write=False)
        valid.setflags(write=False)
        return image, valid

    @property
    def station_count(self) -> int:
        return EXPECTED_STATIONS

    @property
    def sample_count(self) -> int:
        return len(self._frame)

    def table(self) -> pd.DataFrame:
        """Return a copy of the target-free metadata index."""
        return self._frame.copy(deep=True)

    def get_sample(self, record_id: str) -> dict:
        """Retrieve one record's SAR, mask, age, status, and scene reference.

        Missing-SAR records return a zero image/mask for transport ONLY, with
        sar_available=0 and sar_age_hours=NaN. The zeros are never observations.
        """
        rid = str(record_id)
        if rid not in self._by_id.index:
            raise KeyError(f"Unknown SAR station-day record ID: {rid}")
        r = self._by_id.loc[rid]
        available = bool(r["sar_available"])
        if available:
            cached_image, cached_mask = self._read_cached(str(r["patch_relpath"]), str(r["patch_sha256"]))
            image = cached_image.copy()
            mask = cached_mask.copy()
            _require(abs(float(mask.mean()) - float(r["sar_valid_pixel_fraction"])) <= 1e-6,
                     f"Cached SAR mask coverage conflicts with index for {rid}.")
            age = np.float32(r["sar_age_hours"])
            scaled = np.float32(r["sar_age_logscaled"])
        else:
            image = np.zeros(IMAGE_SHAPE, dtype=np.float32)
            mask = np.zeros(MASK_SHAPE, dtype=bool)
            age = np.float32(np.nan)
            scaled = np.float32(0.0)
        return {
            "record_id": rid,
            "station_id": str(r["station_id"]),
            "forecast_date": str(r["forecast_date"]),
            "split": str(r["split"]),
            "sar_image": image,
            "sar_valid_mask": mask,
            "sar_age_hours": age,
            "sar_age_logscaled": scaled,
            "sar_available": np.float32(int(available)),
            "sar_validity_flag": np.float32(int(available)),
            "sar_scene_id": str(r["sar_scene_id"]) if available else None,
            "sar_acquisition_utc": str(r["sar_acquisition_utc"]) if available else None,
            "sar_status": str(r["sar_status"]),
        }

    def _checked_records(self, records: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(records, pd.DataFrame):
            raise TypeError("Pass a DataFrame with record_id (and ideally station_id/date/split).")
        if "record_id" not in records.columns:
            raise Sentinel1DataError("Station-day DataFrame must include record_id.")
        ids = records["record_id"].astype(str).tolist()
        missing = sorted(set(ids).difference(self._by_id.index))
        if missing:
            raise Sentinel1DataError(f"Unknown SAR record IDs: {missing[:5]}")
        aligned = self._by_id.loc[ids].reset_index(drop=True)
        for col in ("station_id", "forecast_date", "split"):
            if col in records.columns:
                actual = records[col].astype(str).reset_index(drop=True)
                expected = aligned[col].astype(str)
                _require(actual.equals(expected),
                         f"SAR {col} disagrees with requested record IDs or order.")
        return aligned

    def get_batch(self, records: pd.DataFrame, *, max_batch_size: int = 32) -> dict:
        """Return one modest-size batch in the exact input DataFrame row order.

        Images are channel-last [N,256,256,2] raw dB. Do not call this on all
        1429 rows at once; use iter_batches or get_sample for a training Dataset.
        """
        if not 1 <= len(records) <= max_batch_size:
            raise Sentinel1DataError(
                f"Batch has {len(records)} samples; max {max_batch_size}. "
                "Use iter_batches(records, batch_size=16) to limit RAM use."
            )
        aligned = self._checked_records(records)
        samples = [self.get_sample(rid) for rid in aligned["record_id"]]
        return {
            "record_ids": aligned["record_id"].to_numpy(copy=True),
            "sar_image": np.stack([s["sar_image"] for s in samples], axis=0),
            "sar_valid_mask": np.stack([s["sar_valid_mask"] for s in samples], axis=0),
            "sar_age_hours": np.array([s["sar_age_hours"] for s in samples], dtype=np.float32).reshape(-1, 1),
            "sar_age_logscaled": np.array([s["sar_age_logscaled"] for s in samples], dtype=np.float32).reshape(-1, 1),
            "sar_available": np.array([s["sar_available"] for s in samples], dtype=np.float32).reshape(-1, 1),
        }

    def for_station_days(self, records: pd.DataFrame, *, max_batch_size: int = 32) -> dict:
        """Same aligned-subset interface as weather/environment, bounded in RAM."""
        return self.get_batch(records, max_batch_size=max_batch_size)

    def iter_batches(self, records: pd.DataFrame, *, batch_size: int = 16) -> Iterator[dict]:
        """Yield sequential batches without materializing all 1429 images at once."""
        if not 1 <= batch_size <= 32:
            raise Sentinel1DataError("Choose 1 <= batch_size <= 32 for safe SAR loading.")
        # Validate the entire requested index before returning any arrays.
        self._checked_records(records)
        for start in range(0, len(records), batch_size):
            yield self.get_batch(records.iloc[start:start + batch_size], max_batch_size=batch_size)


__all__ = ["Sentinel1PatchLoader", "Sentinel1DataError", "INDEX_FILENAME"]
