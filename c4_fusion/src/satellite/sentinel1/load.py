"""Read-only Sentinel-1 C4 loader, with the audited Glencourse mosaic extension.

Notebook 02 owns all Earth Engine retrieval and patch creation. This module
NEVER writes files or contacts Earth Engine. It selects the separately saved
Glencourse extension, never overwrites or silently falls back to the old index.

Repository layout (default):
  data/processed/c4/satellite/sentinel1/
      c4_s1_manifest_PRIVATE.json                    (original, unchanged)
      c4_s1_sample_index_1429_PRIVATE.csv            (original, unchanged)
      c4_s1_scene_inventory_39_PRIVATE.csv           (original, unchanged)
      c4_s1_patch_QA_PRIVATE.csv                      (original, unchanged)
      glencourse_mosaic_extension/
          c4_s1_sample_index_1429_GLENC_MOSAIC_PRIVATE.csv
          c4_s1_glencourse_mosaic_QA_PRIVATE.csv
          c4_s1_glencourse_mosaic_manifest_PRIVATE.json
      cache/patches/                                 (261 original + 3 mosaics)

Usage:
    sar = Sentinel1PatchLoader()   # metadata-only, no NPZ reads
    print(sar.sample_count, sar.table().sar_available.sum())
    sar.verify_all_patches()       # optional full offline NPZ integrity test
    row = sar.get_sample('glencourse_2025-11-17')
    batches = sar.iter_batches(sar.table(), batch_size=16)

Images are raw dB [256,256,2], NOT standardized. Invalid pixels are zero-filled
ONLY for transport and marked False by sar_valid_mask. Dates use 09:00 Colombo
(03:30 UTC) forecast origins. Reused acquisitions are not independent images.
Retrospective hindcast: acquisition availability/publication latency unverified.
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


INDEX_FILENAME = "c4_s1_sample_index_1429_GLENC_MOSAIC_PRIVATE.csv"
MANIFEST_FILENAME = "c4_s1_glencourse_mosaic_manifest_PRIVATE.json"
QA_FILENAME = "c4_s1_glencourse_mosaic_QA_PRIVATE.csv"
BASE_INDEX_FILENAME = "c4_s1_sample_index_1429_PRIVATE.csv"
BASE_MANIFEST_FILENAME = "c4_s1_manifest_PRIVATE.json"
BASE_OTHER_FILES = ("c4_s1_scene_inventory_39_PRIVATE.csv", "c4_s1_patch_QA_PRIVATE.csv")
EXTENSION_DIRNAME = "glencourse_mosaic_extension"
GLEN_VERSION = "C4_S1_GLENC_SAME_PASS_SLICE_MOSAIC_V1"
EXPECTED_SOURCE_VERSION = "C4_S1_V1_DUALPOL_IW_UTM44_50M_NATIVE_DB_SAMPLE"
IMAGE_SHAPE = (256, 256, 2)
MASK_SHAPE = (256, 256)
EXPECTED_SAMPLES = 1429
EXPECTED_STATIONS = 39
EXPECTED_SPLITS = {"TRAIN": 809, "VALIDATION": 269, "TEST": 351}
EXPECTED_BASE_AVAILABLE = 1408
EXPECTED_AVAILABLE = 1429
EXPECTED_BASE_PATCHES = 261
EXPECTED_MOSAIC_PATCHES = 3
EXPECTED_PATCHES = EXPECTED_BASE_PATCHES + EXPECTED_MOSAIC_PATCHES
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
    """Validate the ORIGINAL export + Glencourse extension; load images lazily.

    Args:
        repo_root: J26-DS-314 repository root (normally auto-detected).
        data_dir: Original Notebook 02 output folder (not the extension folder).
        verify_provenance: Require SHA256 checks for all original/extension artifacts.
        verify_patch_hashes: Check NPZ SHA256 on first use (default True).

    No automatic fallback to the 1408-available baseline. Missing extension
    files raise a helpful error, rather than accidentally changing the dataset.
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
        self.extension_dir = self.data_dir / EXTENSION_DIRNAME
        self.index_path = self.extension_dir / INDEX_FILENAME
        self.manifest_path = self.extension_dir / MANIFEST_FILENAME
        self.extension_qa_path = self.extension_dir / QA_FILENAME
        self.original_index_path = self.data_dir / BASE_INDEX_FILENAME
        self.original_manifest_path = self.data_dir / BASE_MANIFEST_FILENAME
        needed = (self.index_path, self.manifest_path, self.extension_qa_path,
                  self.original_index_path, self.original_manifest_path)
        for path in needed:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing Sentinel-1 artifact: {path}. Run Notebook 02 G7 or "
                    "copy its full glencourse_mosaic_extension folder alongside the "
                    "unchanged original outputs. Do not rerun study-wide extraction."
                )
        self.original_manifest = json.loads(self.original_manifest_path.read_text(encoding="utf-8"))
        self.extension_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        # Effective contract exposed to downstream code. Original on-disk manifest
        # remains an immutable 1408/261 baseline; extension adds 21/3.
        self.manifest = dict(self.original_manifest)
        self.manifest["n_available_station_days"] = EXPECTED_AVAILABLE
        self.manifest["n_unique_accepted_patches"] = EXPECTED_PATCHES
        self.manifest["glencourse_extension_version"] = self.extension_manifest.get("version")
        self.verify_patch_hashes = bool(verify_patch_hashes)
        self._validate_manifest(bool(verify_provenance))
        self._frame = pd.read_csv(self.index_path, encoding="utf-8-sig", low_memory=False)
        self._original_frame = pd.read_csv(self.original_index_path, encoding="utf-8-sig", low_memory=False)
        self._validate_index()
        self._by_id = self._frame.set_index("record_id", drop=False)

        @lru_cache(maxsize=8)
        def read_cached(relative_path: str, expected_hash: str) -> tuple[np.ndarray, np.ndarray]:
            return self._read_patch(relative_path, expected_hash)
        self._read_cached = read_cached

    def _validate_manifest(self, verify_provenance: bool) -> None:
        base, ext = self.original_manifest, self.extension_manifest
        _require(base.get("source_version") == EXPECTED_SOURCE_VERSION,
                 "Original Sentinel-1 source version changed.")
        cfg = base.get("source_config", {})
        _require(cfg.get("source") == "COPERNICUS/S1_GRD" and
                 cfg.get("bands") == ["VV", "VH"] and cfg.get("mode") == "IW",
                 "Unexpected original Sentinel-1 source/bands/mode.")
        _require(cfg.get("grid_size") == IMAGE_SHAPE[0] and cfg.get("grid_crs") == "EPSG:32644" and
                 float(cfg.get("pixel_metres", -1)) == 50.0 and
                 float(cfg.get("max_age_days", -1)) == 18.0 and
                 float(cfg.get("min_valid_fraction", -1)) == 0.97,
                 "Original SAR 50m/18-day/97% selection policy changed.")
        _require(base.get("n_labelled_station_days") == EXPECTED_SAMPLES and
                 base.get("n_labelled_gauges") == EXPECTED_STATIONS and
                 base.get("n_available_station_days") == EXPECTED_BASE_AVAILABLE and
                 base.get("n_unique_accepted_patches") == EXPECTED_BASE_PATCHES,
                 "Original Notebook 02 1408/261 manifest is not the audited baseline.")
        _require(ext.get("version") == GLEN_VERSION and
                 ext.get("parent_source_version") == EXPECTED_SOURCE_VERSION and
                 ext.get("source_asset") == "COPERNICUS/S1_GRD" and
                 ext.get("source_config_sha256") == base.get("source_config_sha256"),
                 "Glencourse extension is incompatible with original source configuration.")
        _require(ext.get("station_id") == "glencourse" and
                 ext.get("n_original_available") == EXPECTED_BASE_AVAILABLE and
                 ext.get("n_missing_before") == 21 and
                 ext.get("n_recovered") == 21 and
                 ext.get("n_available_after") == EXPECTED_AVAILABLE and
                 ext.get("n_missing_after") == 0 and
                 ext.get("n_mosaic_patches_accepted") == EXPECTED_MOSAIC_PATCHES and
                 ext.get("original_1408_rows_unchanged") is True,
                 "Glencourse extension manifest does not describe complete 21-row recovery.")
        _require(ext.get("original_1429_row_mapping_unchanged_path") ==
                 "data/processed/c4/satellite/sentinel1/" + BASE_INDEX_FILENAME,
                 "Glencourse manifest references a different original mapping.")
        mosaics = ext.get("mosaics", [])
        _require(isinstance(mosaics, list) and len(mosaics) == EXPECTED_MOSAIC_PATCHES,
                 "Expected exactly three provenance-described Glencourse mosaics.")
        self._mosaics = {m.get("mosaic_id"): m for m in mosaics}
        _require(len(self._mosaics) == EXPECTED_MOSAIC_PATCHES and
                 all(isinstance(k, str) and k.startswith("GLEN_S1_") for k in self._mosaics),
                 "Invalid or duplicated Glencourse mosaic IDs.")
        _require(set(ext.get("accepted_mosaic_ids", [])) == set(self._mosaics),
                 "Accepted mosaic IDs disagree with provenance mosaic list.")
        for mid, m in self._mosaics.items():
            scenes = m.get("scene_ids", [])
            times = pd.to_datetime(m.get("source_utc", []), utc=True, errors="coerce")
            _require(len(scenes) == 2 and len(set(scenes)) == 2 and
                     len(times) == 2 and times.notna().all(),
                     f"{mid}: expected two distinct source slices with timestamps.")
            _require(m.get("mosaic_quality_status") == "ACCEPTED" and
                     float(m.get("valid_union_fraction", -1)) > 0.97 and
                     float(m.get("valid_union_fraction", -1)) <= 1.0 and
                     isinstance(m.get("patch_sha256"), str) and
                     len(m["patch_sha256"]) == 64,
                     f"{mid}: failed mosaic quality/hash metadata.")
            _require(m.get("orbit_pass") == "DESCENDING" and
                     m.get("platform") == "A" and
                     0 < float(m.get("time_gap_seconds", -1)) <= 90,
                     f"{mid}: unexpected platform, pass or slice gap.")
            _require(times[0] < times[1] and
                     times[0] == pd.to_datetime(m.get("acq_earliest_utc"), utc=True) and
                     times[1] == pd.to_datetime(m.get("acq_latest_utc"), utc=True) and
                     abs((times[1] - times[0]).total_seconds() - float(m["time_gap_seconds"])) < 1e-5,
                     f"{mid}: source-slice timestamps are inconsistent.")

        if verify_provenance:
            for fn in (BASE_INDEX_FILENAME, *BASE_OTHER_FILES):
                path = self.data_dir / fn
                if not path.is_file():
                    raise FileNotFoundError(f"Original Notebook 02 provenance artifact missing: {path}")
                base_hash = base.get("output_sha256", {}).get(fn)
                ext_hash = ext.get("original_input_sha256", {}).get(fn)
                _require(base_hash == ext_hash == _digest(path),
                         f"Original {fn} hash differs from both manifests.")
            _require(ext.get("original_input_sha256", {}).get(BASE_MANIFEST_FILENAME)
                     == _digest(self.original_manifest_path),
                     "Original manifest changed since the Glencourse extension was created.")
            for fn, path in ((INDEX_FILENAME, self.index_path), (QA_FILENAME, self.extension_qa_path)):
                _require(ext.get("output_sha256", {}).get(fn) == _digest(path),
                         f"Glencourse {fn} SHA256 differs from extension manifest.")
            target_hash = base.get("notebook01_target_free_index_sha256")
            _require(isinstance(target_hash, str) and len(target_hash) == 64,
                     "Missing Notebook 01 target-free source-index SHA256.")
            source_index = self.repo_root / "data/processed/c4/manifests/c4_sample_index_1429.csv"
            if source_index.is_file():
                _require(_digest(source_index) == target_hash,
                         "Notebook 01 target-free sample index changed.")

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
        self._validate_overlay()

    def _validate_overlay(self) -> None:
        """Prove that exactly the original 21 missing Glencourse rows were filled."""
        before, after = self._original_frame, self._frame
        _require(list(before.columns) == list(after.columns) == list(REQUIRED_COLUMNS),
                 "Original and Glencourse mapping schemas differ.")
        _require(len(before) == len(after) == EXPECTED_SAMPLES and
                 before["record_id"].tolist() == after["record_id"].tolist(),
                 "Original and recovered record identities/order differ.")
        _require(before["sar_available"].dtype == bool and
                 int(before["sar_available"].sum()) == EXPECTED_BASE_AVAILABLE,
                 "Original 1408-available index is missing or has changed.")
        previously_missing = ~before["sar_available"]
        _require(int(previously_missing.sum()) == 21 and
                 set(before.loc[previously_missing, "station_id"]) == {"glencourse"},
                 "Baseline does not contain exactly the 21 Glencourse gaps.")
        # Column-by-column EXPLICIT exact comparison; pandas NaNs compare equal here.
        try:
            pd.testing.assert_frame_equal(after.loc[~previously_missing],
                                          before.loc[~previously_missing],
                                          check_dtype=True, check_exact=True)
        except AssertionError as exc:
            raise Sentinel1DataError(
                "One or more of the original 1408 available station-days changed."
            ) from exc
        _require(after.loc[previously_missing, "sar_available"].eq(True).all(),
                 "Not all 21 Glencourse dates were recovered.")
        # Only those 21 rows may differ (the other 1408 were compared above).
        recovered = after.loc[previously_missing]
        _require(recovered["sar_scene_id"].isin(self._mosaics).all(),
                 "Recovered Glencourse row references a non-approved mosaic.")
        _require(recovered["sar_scene_id"].value_counts().sort_values().tolist() == [6, 6, 9],
                 "Unexpected 9+6+6 Glencourse scene-to-date assignment.")
        _require(len(after.loc[after.sar_available, ["station_id", "sar_scene_id"]].drop_duplicates())
                 == EXPECTED_PATCHES,
                 "Expected 261 original plus 3 new unique station-scene patches.")
        allowed_prefix = (
            "data/processed/c4/satellite/sentinel1/cache/patches/glencourse/mosaics/"
        )
        for mid, group in recovered.groupby("sar_scene_id", sort=False):
            m = self._mosaics[mid]
            _require(group["station_id"].eq("glencourse").all() and
                     group["patch_relpath"].str.startswith(allowed_prefix).all(),
                     f"{mid}: recovered patch is not in the dedicated mosaic cache.")
            _require(group["patch_sha256"].eq(m["patch_sha256"]).all() and
                     group["sar_acquisition_utc"].eq(m["acq_earliest_utc"]).all() and
                     group["sar_orbit_pass"].eq(m["orbit_pass"]).all() and
                     group["sar_relative_orbit"].eq(float(m["relative_orbit"])).all() and
                     np.allclose(group["sar_valid_pixel_fraction"].to_numpy(dtype=float),
                                 float(m["valid_union_fraction"]), rtol=0, atol=1e-8),
                     f"{mid}: recovered rows conflict with provenance.")
            _require(group["patch_relpath"].nunique() == 1 and
                     group["patch_relpath"].iloc[0].endswith(".npz"),
                     f"{mid}: inconsistent cached NPZ path.")
            acquired_last = pd.to_datetime(m["acq_latest_utc"], utc=True)
            cutoff = pd.to_datetime(group["forecast_date"], utc=True) + pd.Timedelta(hours=3, minutes=30)
            _require((acquired_last < cutoff).all(),
                     f"{mid}: a source slice occurs after the forecast origin.")
        qa = pd.read_csv(self.extension_qa_path, low_memory=False)
        _require(len(qa) == 21 and qa["record_id"].is_unique and
                 set(qa["record_id"]) == set(recovered["record_id"]) and
                 qa["recovered"].eq(True).all() and
                 set(qa["mosaic_quality_status"]) == {"ACCEPTED"},
                 "Glencourse QA CSV disagrees with the 21 recovered rows.")
        qa_aligned = qa.set_index("record_id").loc[recovered["record_id"]]
        _require((qa_aligned["proposed_mosaic_id"].to_numpy()
                  == recovered["sar_scene_id"].to_numpy()).all() and
                 np.allclose(qa_aligned["conservative_age_hours"].to_numpy(dtype=float),
                             recovered["sar_age_hours"].to_numpy(dtype=float), rtol=0, atol=1e-6),
                 "Glencourse QA mosaic IDs or ages do not match the recovered index.")

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
        if "/glencourse/mosaics/" in relative_path.replace("\\", "/"):
            _require(valid.mean() > 0.97,
                     f"Glencourse mosaic does not pass STRICT >97% valid pixels: {path}")
        _require(np.isfinite(image[valid]).all(),
                 f"Valid SAR pixels must have finite VV/VH dB: {path}")
        _require(np.isnan(image[~valid]).all(),
                 f"Invalid SAR pixels must remain NaN in raw cached array: {path}")
        # Only neutral-fill INVALID pixels for batching. Mask + availability are mandatory.
        image[~valid] = 0.0
        image.setflags(write=False)
        valid.setflags(write=False)
        return image, valid

    def verify_all_patches(self) -> dict:
        """Offline full cache QA: read/hash-check all 264 unique NPZ patches.

        Does not contact Earth Engine. Requires the original 261 and three new
        mosaic NPZ files on disk. Reads files sequentially; bounded RAM usage.
        Returns a summary only after every file passes.
        """
        _require(self.verify_patch_hashes,
                 "Full patch verification requires verify_patch_hashes=True.")
        unique = self._frame.loc[self._frame["sar_available"],
                                 ["patch_relpath", "patch_sha256",
                                  "sar_valid_pixel_fraction"]].drop_duplicates()
        _require(len(unique) == EXPECTED_PATCHES, "Expected exactly 264 distinct patch files.")
        counts = {"original": 0, "glencourse_mosaics": 0}
        for row in unique.itertuples(index=False):
            _, valid = self._read_patch(str(row.patch_relpath), str(row.patch_sha256))
            _require(abs(float(valid.mean()) - float(row.sar_valid_pixel_fraction)) <= 1e-6,
                     f"Cached pixel mask differs from SAR mapping: {row.patch_relpath}")
            if "/glencourse/mosaics/" in str(row.patch_relpath).replace("\\", "/"):
                _require(float(valid.mean()) > 0.97,
                         f"Glencourse mosaic now fails STRICT >97% coverage: {row.patch_relpath}")
                counts["glencourse_mosaics"] += 1
            else:
                counts["original"] += 1
        _require(counts == {"original": EXPECTED_BASE_PATCHES,
                            "glencourse_mosaics": EXPECTED_MOSAIC_PATCHES},
                 "Patch validation counts disagree with audited metadata.")
        return {"station_days": len(self._frame), "available": int(self._frame.sar_available.sum()),
                "unique_patches_verified": len(unique), **counts}

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
            "sar_valid_pixel_fraction": (np.float32(r["sar_valid_pixel_fraction"])
                                         if available else np.float32(np.nan)),
            "sar_orbit_pass": str(r["sar_orbit_pass"]) if available else None,
            "sar_relative_orbit": (int(r["sar_relative_orbit"]) if available else None),
            "patch_relpath": str(r["patch_relpath"]) if available else None,
            "patch_sha256": str(r["patch_sha256"]) if available else None,
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
            "sar_validity_flag": np.array([s["sar_validity_flag"] for s in samples], dtype=np.float32).reshape(-1, 1),
            "sar_valid_pixel_fraction": np.array([s["sar_valid_pixel_fraction"] for s in samples], dtype=np.float32).reshape(-1, 1),
            "sar_scene_id": [s["sar_scene_id"] for s in samples],
            "sar_acquisition_utc": [s["sar_acquisition_utc"] for s in samples],
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


__all__ = ["Sentinel1PatchLoader", "Sentinel1DataError", "INDEX_FILENAME",
           "MANIFEST_FILENAME", "BASE_INDEX_FILENAME", "BASE_MANIFEST_FILENAME"]
