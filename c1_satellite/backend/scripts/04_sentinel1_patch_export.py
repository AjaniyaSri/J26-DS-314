"""Phase 4: quality-check and download real Sentinel-1 VV/VH patches.

Requires Phase 3 outputs on the user's computer. Does not synthesize any
missing Sentinel-1 acquisition or substitute future observations.

Run from the J26-DS-314 repository root:
    python c1_satellite/scripts/04_sentinel1_patch_export.py --dry-run
    python c1_satellite/scripts/04_sentinel1_patch_export.py --project YOUR_PROJECT_ID --station-id badalgama --max-scenes 2

Output: GeoTIFF images and preview PNGs in c1_satellite/outputs/s1_patches/
        QA/coverage manifest in c1_satellite/outputs/c1_s1_patch_qa.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "c1_satellite" / "local_data"
OUTPUT = ROOT / "c1_satellite" / "outputs"
LABELS = LOCAL / "FloodRisk_DMC_StationDay_LABELLED_1429.csv"
SCENES = OUTPUT / "c1_s1_scene_inventory.csv"
ELIGIBLE = OUTPUT / "c1_s1_day_eligibility.csv"
PATCHES = OUTPUT / "s1_patches"
QA_CSV = OUTPUT / "c1_s1_patch_qa.csv"
CRS = "EPSG:32644"  # All selected Sri Lankan gauges are within UTM zone 44N
PIXEL_METRES = 50
SIZE = 256
HALF_M = PIXEL_METRES * SIZE / 2  # 6400 m (12.8 km square)
NODATA = -9999.0
QA_HEADERS = [
    "station_id", "forecast_date", "record_id", "chronological_split",
    "scene_id", "acquisition_utc", "orbit_pass", "relative_orbit",
    "patch_coverage_fraction_ee", "raster_valid_fraction", "quality_threshold",
    "patch_pass", "status", "geotiff_path", "preview_png_path",
    "patch_crs", "patch_width_px", "patch_height_px", "pixel_size_m",
    "note",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input {path}. Complete Phases 1 and 3 first.")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def grid_for_xy(x: float, y: float) -> dict[str, Any]:
    """Fixed 256x256 grid near the gauge center; all scenes for a gauge align."""
    cx = math.floor(x / PIXEL_METRES + 0.5) * PIXEL_METRES
    cy = math.floor(y / PIXEL_METRES + 0.5) * PIXEL_METRES
    left, top = cx - HALF_M, cy + HALF_M
    return {
        "left": left,
        "right": left + SIZE * PIXEL_METRES,
        "top": top,
        "bottom": top - SIZE * PIXEL_METRES,
        "transform": [PIXEL_METRES, 0, left, 0, -PIXEL_METRES, top],
    }


def choose_forecast_row(rows: list[dict[str, str]], station_id: str, requested_date: str | None = None) -> dict[str, str]:
    station_rows = [r for r in rows if r["station_id"].casefold() == station_id.casefold()]
    if not station_rows:
        available = ", ".join(sorted({r["station_id"] for r in rows})[:8])
        raise ValueError(f"Station '{station_id}' not in Phase 3 eligibility data. Examples: {available}")
    if requested_date:
        date.fromisoformat(requested_date)
        station_rows = [r for r in station_rows if r["forecast_date"] == requested_date]
        if not station_rows:
            raise ValueError(f"No Phase 3 eligibility for {station_id} at {requested_date}")
    ready = [r for r in station_rows if r.get("sar_multi_scene_ids", "").strip()]
    single = [r for r in station_rows if r.get("sar_latest_scene_id", "").strip()]
    choices = ready or single
    if not choices:
        raise ValueError(f"No eligible Sentinel-1 scenes found for {station_id}")
    return max(choices, key=lambda r: r["forecast_date"])


def choose_scene_rows(
    forecast: dict[str, str], inventory: list[dict[str, str]],
    max_scenes: int,
) -> list[dict[str, str]]:
    candidates = [s.strip() for s in forecast.get("sar_multi_scene_ids", "").split(";") if s.strip()]
    if not candidates:
        candidates = [forecast["sar_latest_scene_id"]]
    # These are already actual acquisitions vetted for temporal eligibility in Phase 3.
    station = forecast["station_id"]
    scene_map = {(s["station_id"], s["scene_id"]): s for s in inventory}
    ids = list(dict.fromkeys(candidates))[-max_scenes:]
    missing = [i for i in ids if (station, i) not in scene_map]
    if missing:
        raise ValueError(f"Phase 3 inventory missing {len(missing)} selected scenes for {station}")
    selected = [scene_map[(station, sid)] for sid in ids]
    if len(selected) > 1:
        geometry = {(s["orbit_pass"], s["relative_orbit"]) for s in selected}
        if len(geometry) != 1 or ("UNKNOWN", "UNKNOWN") in geometry:
            raise ValueError("Selected temporal acquisitions have inconsistent/unknown orbit metadata")
    return selected


def station_coordinates(labels: list[dict[str, str]], station: str) -> tuple[float, float]:
    matches = [r for r in labels if r["station_id"] == station]
    if not matches:
        raise ValueError(f"Station {station} not in labelled CSV")
    coords = {(float(r["gauge_lon_wgs84_arcgis"]), float(r["gauge_lat_wgs84_arcgis"])) for r in matches}
    if len(coords) != 1:
        raise ValueError(f"Station {station} coordinates are not consistent")
    lon, lat = next(iter(coords))
    if not (79 <= lon <= 82 and 5 <= lat <= 10):
        raise ValueError(f"Station {station} coordinates outside Sri Lanka range")
    return lon, lat


def filename_for_scene(station: str, scene: dict[str, str]) -> str:
    acquired = scene["acquisition_utc"].replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    uid = hashlib.sha1(scene["scene_id"].encode("utf-8")).hexdigest()[:9]
    return f"{station}_{acquired}_{uid}"


def ee_valid_fraction(ee: Any, image: Any, roi: Any, transform: list[float]) -> float:
    # Global zero image makes masked pixels and areas outside scene footprint zero.
    valid = image.select("VV").mask().gt(0).And(image.select("VH").mask().gt(0))
    indicator = ee.Image.constant(0).where(valid, 1).rename("valid")
    reducer = indicator.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        crs=CRS,
        crsTransform=transform,
        maxPixels=200000,
        bestEffort=False,
    )
    fraction = reducer.get("valid").getInfo()
    if fraction is None:
        return 0.0
    return float(fraction)


def download_geotiff(image: Any, transform: list[float], target: Path) -> None:
    import requests

    # EO calibrated SAR dB; no fake values. Missing/outside pixels flagged -9999.
    clean = image.select(["VV", "VH"]).unmask(value=NODATA, sameFootprint=False).toFloat()
    url = clean.getDownloadURL({
        "bands": ["VV", "VH"],
        "crs": CRS,
        "crs_transform": transform,
        "dimensions": [SIZE, SIZE],
        "format": "GEO_TIFF",
    })
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as f:
                total = 0
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > 35 * 1024 * 1024:
                        raise ValueError("GeoTIFF exceeds expected size; download aborted")
                    f.write(chunk)
        if temporary.stat().st_size < 1000:
            raise ValueError("Downloaded file unexpectedly small")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_and_preview(tif_path: Path, png_path: Path, make_preview: bool = True) -> float:
    import numpy as np
    import rasterio

    with rasterio.open(tif_path) as ds:
        if ds.count != 2 or ds.width != SIZE or ds.height != SIZE:
            raise ValueError(f"Unexpected bands/grid: {ds.count} bands, {ds.width}x{ds.height}")
        if ds.crs is None or ds.crs.to_string() != CRS:
            raise ValueError(f"Unexpected GeoTIFF CRS {ds.crs} instead of {CRS}")
        if abs(ds.transform.a - PIXEL_METRES) > 0.01 or abs(ds.transform.e + PIXEL_METRES) > 0.01:
            raise ValueError(f"Unexpected GeoTIFF pixel size {ds.transform}")
        values = ds.read()
    valid = np.isfinite(values).all(axis=0) & (values[0] != NODATA) & (values[1] != NODATA)
    valid_fraction = float(valid.mean())
    if make_preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.8))
        for i, band in enumerate(["VV", "VH"]):
            data = np.where(valid, values[i], np.nan)
            clean = data[np.isfinite(data)]
            if clean.size > 2:
                lo, hi = np.percentile(clean, [2, 98])
                if hi <= lo:
                    lo, hi = float(lo - 1), float(hi + 1)
            else:
                lo, hi = (-25.0, 0.0) if i == 0 else (-35.0, -5.0)
            ax[i].imshow(data, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax[i].set_title(f"Sentinel-1 {band} (dB)")
            ax[i].axis("off")
        fig.suptitle(f"{tif_path.stem}\nValid pixels: {valid_fraction:.1%}", fontsize=9)
        fig.tight_layout()
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
    return valid_fraction


def write_qa(rows: list[dict[str, Any]]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with QA_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QA_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export real Sentinel-1 VV/VH patches for the C1 prototype")
    parser.add_argument("--project", help="GEE-enabled Google Cloud project ID")
    parser.add_argument("--station-id", default="badalgama", help="Use the Phase 3 station_id, e.g. badalgama")
    parser.add_argument("--forecast-date", help="Optional YYYY-MM-DD forecast date (defaults to latest eligible)")
    parser.add_argument("--max-scenes", type=int, default=2, help="Number of real SAR acquisitions to test (default 2)")
    parser.add_argument("--min-coverage", type=float, default=0.98, help="Minimum full-patch valid fraction (default .98)")
    parser.add_argument("--authenticate", action="store_true", help="Sign into Earth Engine via browser")
    parser.add_argument("--qa-only", action="store_true", help="Check patch coverage without downloading GeoTIFFs")
    parser.add_argument("--dry-run", action="store_true", help="Check local Phase 3 inputs without Earth Engine")
    args = parser.parse_args(argv)
    if not (1 <= args.max_scenes <= 10):
        parser.error("--max-scenes must be 1 to 10")
    if not (0 < args.min_coverage <= 1):
        parser.error("--min-coverage must be between 0 and 1")

    eligible = csv_rows(ELIGIBLE)
    inventory = csv_rows(SCENES)
    labels = csv_rows(LABELS)
    forecast = choose_forecast_row(eligible, args.station_id, args.forecast_date)
    station_id = forecast["station_id"]
    lon, lat = station_coordinates(labels, station_id)
    chosen = choose_scene_rows(forecast, inventory, args.max_scenes)
    date_count = len({s["acquisition_utc"][:10] for s in chosen})
    print("=== C1 PHASE 4: SENTINEL-1 VV/VH PIXEL EXPORT ===")
    print(f"Station {station_id}; gauge WGS84 ({lon:.6f}, {lat:.6f})")
    print(f"Forecast date: {forecast['forecast_date']}, split: {forecast['chronological_split']}")
    print(f"Scenes chosen: {len(chosen)}; distinct acquisition dates: {date_count}")
    for s in chosen:
        print(f"  {s['acquisition_utc']}  orbit {s['orbit_pass']}/{s['relative_orbit']}  {s['scene_id']}")
    if len(chosen) >= 2 and date_count < 2:
        print("WARNING: Multiple files from one acquisition DATE do not establish temporal depth.")
    if args.dry_run:
        print("DRY RUN PASS: local Phase 3 inventory/eligibility and labels loaded; no GEE used.")
        return
    if not args.project:
        parser.error("Provide --project YOUR_GEE_PROJECT_ID to query Earth Engine")
    try:
        import ee
        from pyproj import Transformer
    except ImportError as exc:
        raise SystemExit("Install packages: python -m pip install earthengine-api requests pyproj rasterio numpy matplotlib") from exc
    try:
        if args.authenticate:
            ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise SystemExit(f"Earth Engine authentication/init failed: {exc}") from exc

    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    grid = grid_for_xy(x, y)
    roi = ee.Geometry.Rectangle([grid["left"], grid["bottom"], grid["right"], grid["top"]], proj=CRS, geodesic=False)
    rows = []
    PATCHES.mkdir(parents=True, exist_ok=True)
    for i, scene in enumerate(chosen, 1):
        stem = filename_for_scene(station_id, scene)
        tif = PATCHES / (stem + ".tif")
        png = PATCHES / (stem + "_preview.png")
        row = {
            "station_id": station_id, "forecast_date": forecast["forecast_date"],
            "record_id": forecast["record_id"], "chronological_split": forecast["chronological_split"],
            "scene_id": scene["scene_id"], "acquisition_utc": scene["acquisition_utc"],
            "orbit_pass": scene["orbit_pass"], "relative_orbit": scene["relative_orbit"],
            "patch_coverage_fraction_ee": "", "raster_valid_fraction": "",
            "quality_threshold": args.min_coverage, "patch_pass": 0,
            "status": "NOT_STARTED", "geotiff_path": "", "preview_png_path": "",
            "patch_crs": CRS, "patch_width_px": SIZE, "patch_height_px": SIZE,
            "pixel_size_m": PIXEL_METRES, "note": "",
        }
        try:
            image = ee.Image(scene["scene_id"])
            fraction = ee_valid_fraction(ee, image, roi, grid["transform"])
            row["patch_coverage_fraction_ee"] = round(fraction, 6)
            print(f"[{i}/{len(chosen)}] {scene['acquisition_utc'][:10]} patch valid: {fraction:.2%}")
            if fraction < args.min_coverage:
                row["status"] = "SKIPPED_PARTIAL_PATCH"
                row["note"] = "Scene intersects gauge but does not cover enough of the 12.8-km square"
            elif args.qa_only:
                row["status"] = "PATCH_QA_OK_NOT_DOWNLOADED"
                row["patch_pass"] = 1
            else:
                if not tif.exists():
                    download_geotiff(image, grid["transform"], tif)
                    row["status"] = "DOWNLOADED"
                else:
                    row["status"] = "ALREADY_PRESENT"
                raster_fraction = inspect_and_preview(tif, png)
                row["raster_valid_fraction"] = round(raster_fraction, 6)
                if raster_fraction < args.min_coverage:
                    row["status"] = "FAILED_RASTER_COVERAGE"
                    row["note"] = "Downloaded raster has more missing pixels than threshold; inspect masks"
                else:
                    row["patch_pass"] = 1
                row["geotiff_path"] = str(tif.relative_to(ROOT)).replace("\\", "/")
                row["preview_png_path"] = str(png.relative_to(ROOT)).replace("\\", "/")
                print(f"    -> {row['status']}: {row['geotiff_path']} (raster valid {raster_fraction:.2%})")
        except Exception as exc:
            row["status"] = "ERROR"
            row["note"] = str(exc)[:350]
            print(f"    ERROR: {exc}")
        rows.append(row)
        write_qa(rows)
    passed = sum(int(r["patch_pass"]) for r in rows)
    summary = {
        "station_id": station_id,
        "forecast_date": forecast["forecast_date"],
        "selected_scenes": len(rows),
        "distinct_acquisition_dates": date_count,
        "patch_QA_pass_count": passed,
        "mode": "QA_ONLY" if args.qa_only else "DOWNLOAD",
        "warnings": [
            "A point-level inventory does not establish full patch coverage; use patch_coverage_fraction_ee.",
            "Only real acquisitions are used, and no missing pixel is imputed.",
            "Provisional 48-hour product lag is not operationally validated.",
            "Only use historical scenes available at the forecast cutoff; these patches may be reused across many station-days.",
            "These patch files are a prototype QA sample, not a trained SAR model or complete 1429-row satellite dataset.",
        ],
    }
    (OUTPUT / "c1_s1_patch_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"RESULT: {passed}/{len(rows)} patch QA passed. Manifest: {QA_CSV}")
    if passed == 0:
        print("No full patch passed: inspect QA and try another station or date. Do not fabricate images.")
    elif args.qa_only:
        print("QA successful: rerun WITHOUT --qa-only to download actual pixels and previews.")
    else:
        print("Open *_preview.png in VS Code. The TIFF files retain the original VV/VH dB values.")


if __name__ == "__main__":
    run()
