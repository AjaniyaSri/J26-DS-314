"""Upgrade 2 – Catchment Upstream Watershed SWI Aggregation.

For each DMC gauge station we load the upstream catchment polygon
(from shared/spatial/catchments/<basin>/polygons/<station>_CANDIDATE.geojson),
clip the daily ASCAT SWI GeoTIFF to that polygon, then compute basin-averaged
soil-moisture statistics (mean, std, max, wet-fraction) for each day.

These *true* catchment-level saturation features replace / augment the
existing pixel-point SWI look-up that only samples a single ASCAT grid
cell at the gauge location.  Because the ASCAT grid is coarse (~25 km),
a large catchment (e.g. Manampitiya 7418 km2) can contain ~30 ASCAT
pixels whose average is far more informative than the outlet pixel alone.

Usage (as a module)
-------------------
    from c1_satellite.src.improvements.catchment_swi import build_catchment_swi_table
    df = build_catchment_swi_table()
    df.to_csv("outputs/c1_catchment_swi_features.csv", index=False)

Output columns (per station-day)
---------------------------------
    station_id, date,
    catch_swi005_mean, catch_swi005_std, catch_swi005_max, catch_swi005_wet_frac,
    catch_swi010_mean, catch_swi010_std, catch_swi010_max, catch_swi010_wet_frac,
    catch_area_km2, catch_pixel_count
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import shape, mapping

ROOT = Path(__file__).resolve().parents[3]          # J26-DS-314/
CATCHMENTS_DIR = ROOT / "shared" / "spatial" / "catchments"
ASCAT_DIR = ROOT / "c1_satellite" / "local_data" / "ascat_swi_v4" / "raw"

# Wet-soil threshold for wet-fraction computation.
# CLMS V4 SWI range: 0 = dry, 100 = field capacity, >100 = oversaturated/flooded.
# Using 80 as the threshold for "wet" and >100 as "oversaturated".
WET_THRESHOLD = 80.0
OVERSAT_THRESHOLD = 100.0
# ASCAT nodata sentinel value
ASCAT_NODATA = -9999.0

# ASCAT band index cache (auto-discovered from first TIF)
_BAND_CACHE: dict[str, int] = {}


def _discover_bands(tif_path: Path) -> dict[str, int]:
    """Return {'swi005': band_idx, 'swi010': band_idx, 'datamask': band_idx} (1-based).

    CLMS SWI Global Daily V4 layout (8 bands):
      Band 1 = SWI001, Band 2 = SWI005, Band 3 = SWI010,
      Band 4 = QFLAG001, Band 5 = QFLAG005, Band 6 = QFLAG010,
      Band 7 = SSF, Band 8 = dataMask
    Values are float32, nodata = -9999, valid SWI in [0, 100].
    """
    global _BAND_CACHE
    if _BAND_CACHE:
        return _BAND_CACHE
    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):
            desc = (src.descriptions[i - 1] or "").upper().replace(" ", "")
            if desc == "SWI005":
                _BAND_CACHE["swi005"] = i
            elif desc == "SWI010":
                _BAND_CACHE["swi010"] = i
            elif desc == "DATAMASK":
                _BAND_CACHE["datamask"] = i
    # Hardcoded fallback for CLMS V4 (known layout)
    if "swi005" not in _BAND_CACHE:
        _BAND_CACHE["swi005"] = 2
    if "swi010" not in _BAND_CACHE:
        _BAND_CACHE["swi010"] = 3
    if "datamask" not in _BAND_CACHE:
        _BAND_CACHE["datamask"] = 8
    return _BAND_CACHE


# ---------------------------------------------------------------------------
# Catchment polygon registry
# ---------------------------------------------------------------------------

def _build_station_polygon_registry() -> dict[str, Path]:
    """Scan CATCHMENTS_DIR and return {station_id_lower: geojson_path}."""
    registry: dict[str, Path] = {}
    for gj_path in CATCHMENTS_DIR.rglob("*.geojson"):
        try:
            with open(gj_path) as f:
                gj = json.load(f)
            station_name = gj["features"][0]["properties"].get("station", "")
            # Normalise to lowercase snake_case to match dataset station_id
            key = station_name.strip().lower().replace(" ", "_")
            registry[key] = gj_path
        except Exception as exc:
            warnings.warn(f"Could not read {gj_path}: {exc}")
    return registry


def load_catchment_polygon(station_id: str,
                           registry: Optional[dict] = None) -> Optional[object]:
    """Return a Shapely geometry for *station_id*, or None if not found."""
    if registry is None:
        registry = _build_station_polygon_registry()
    gj_path = registry.get(station_id)
    if gj_path is None:
        # Fuzzy match: strip parentheses / alternate tokens
        canon = station_id.replace("(", "").replace(")", "").replace(" ", "_").lower()
        for key, path in registry.items():
            if canon in key or key in canon:
                gj_path = path
                break
    if gj_path is None:
        return None
    with open(gj_path) as f:
        gj = json.load(f)
    geoms = [shape(feat["geometry"]) for feat in gj["features"]]
    if len(geoms) == 1:
        return geoms[0]
    from shapely.ops import unary_union
    return unary_union(geoms)


# ---------------------------------------------------------------------------
# Per-pixel ASCAT extraction within catchment polygon
# ---------------------------------------------------------------------------

def _extract_basin_stats(tif_path: Path, geom) -> dict[str, float]:
    """Clip *tif_path* to *geom* and return basin-average stats for SWI005/010.

    ASCAT CLMS V4 specifics:
      - Nodata = -9999.0 (float32)
      - Valid SWI in [0, 100]
      - dataMask band signals ASCAT orbit coverage, NOT land/ocean.
        When dataMask=0 it means no ascending/descending pass that day,
        so it is NOT used as a filter here – the -9999 sentinel alone
        is the authoritative nodata indicator per CLMS V4 documentation.
    """
    result: dict[str, float] = {}
    bands = _discover_bands(tif_path)
    geojson_geom = [mapping(geom)]

    # Only extract SWI bands (skip QFLAG, SSF, dataMask)
    swi_bands = {k: v for k, v in bands.items() if k in ("swi005", "swi010")}

    with rasterio.open(tif_path) as src:
        for band_name, band_idx in swi_bands.items():
            try:
                clipped, _ = rasterio_mask(src, geojson_geom,
                                           crop=True, all_touched=True,
                                           filled=True,
                                           nodata=ASCAT_NODATA,
                                           indexes=[band_idx])
                arr = clipped[0].astype(np.float32)
                # Mask only the -9999 nodata sentinel; values >100 are valid
                # CLMS V4 SWI is NOT capped at 100 – values up to ~160 occur
                # during monsoon oversaturation and are physically meaningful
                # for flood prediction (higher = wetter / more flood risk).
                arr[arr <= ASCAT_NODATA + 1] = np.nan   # -9999 nodata only
                arr[arr < 0] = np.nan                    # no negative SWI

                valid = arr[np.isfinite(arr)]
                if valid.size == 0:
                    result[f"catch_{band_name}_mean"] = np.nan
                    result[f"catch_{band_name}_std"] = np.nan
                    result[f"catch_{band_name}_max"] = np.nan
                    result[f"catch_{band_name}_wet_frac"] = np.nan
                    result[f"catch_{band_name}_oversat_frac"] = np.nan
                    result["catch_pixel_count"] = 0
                else:
                    result[f"catch_{band_name}_mean"] = float(np.nanmean(valid))
                    result[f"catch_{band_name}_std"] = float(np.nanstd(valid))
                    result[f"catch_{band_name}_max"] = float(np.nanmax(valid))
                    result[f"catch_{band_name}_wet_frac"] = float(
                        np.mean(valid > WET_THRESHOLD))
                    result[f"catch_{band_name}_oversat_frac"] = float(
                        np.mean(valid > OVERSAT_THRESHOLD))
                    result["catch_pixel_count"] = int(valid.size)
            except Exception as exc:
                warnings.warn(f"Mask error {tif_path.name} band {band_name}: {exc}")
                for suffix in ("mean", "std", "max", "wet_frac"):
                    result[f"catch_{band_name}_{suffix}"] = np.nan
                result["catch_pixel_count"] = 0
    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_catchment_swi_table(
        ascat_dir: Path = ASCAT_DIR,
        stations: Optional[list[str]] = None,
        verbose: bool = True,
) -> pd.DataFrame:
    """Build a station x date DataFrame of basin-averaged ASCAT SWI statistics.

    Parameters
    ----------
    ascat_dir : Path
        Directory containing daily ASCAT_SWI_V4_YYYY-MM-DD.tif files.
    stations : list[str] | None
        Which station_ids to process.  None means all stations in registry.
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.DataFrame with columns:
        station_id, date,
        catch_swi005_mean/std/max/wet_frac,
        catch_swi010_mean/std/max/wet_frac,
        catch_area_km2, catch_pixel_count
    """
    registry = _build_station_polygon_registry()
    if stations is None:
        stations = sorted(registry.keys())
    if verbose:
        print(f"[Upgrade 2] Registry: {len(registry)} catchments found")
        print(f"[Upgrade 2] Processing {len(stations)} stations")

    tif_files = sorted(ascat_dir.glob("ASCAT_SWI_V4_*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No ASCAT TIF files found in {ascat_dir}")
    if verbose:
        first_d = tif_files[0].stem.replace("ASCAT_SWI_V4_", "")
        last_d = tif_files[-1].stem.replace("ASCAT_SWI_V4_", "")
        print(f"[Upgrade 2] ASCAT files: {len(tif_files)} days ({first_d} to {last_d})")

    # Pre-load polygon geometries
    geom_cache: dict[str, object] = {}
    area_cache: dict[str, float] = {}
    for sid in stations:
        geom = load_catchment_polygon(sid, registry)
        geom_cache[sid] = geom
        if geom is None:
            if verbose:
                print(f"  [WARN] No catchment polygon for station '{sid}' – NaN rows")
            area_cache[sid] = np.nan
        else:
            try:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
                gdf_proj = gdf.to_crs("EPSG:5234")   # Sri Lanka Transverse Mercator
                area_cache[sid] = float(gdf_proj.geometry.area.iloc[0] / 1e6)
            except Exception:
                area_cache[sid] = np.nan

    rows: list[dict] = []
    for t_idx, tif_path in enumerate(tif_files):
        date_str = tif_path.stem.replace("ASCAT_SWI_V4_", "")
        if verbose:
            print(f"  [{t_idx+1}/{len(tif_files)}] {date_str}", end="\r")
        for sid in stations:
            geom = geom_cache[sid]
            row: dict = {
                "station_id": sid,
                "date": date_str,
                "catch_area_km2": area_cache.get(sid, np.nan),
            }
            if geom is None:
                row.update({
                    "catch_swi005_mean": np.nan, "catch_swi005_std": np.nan,
                    "catch_swi005_max": np.nan, "catch_swi005_wet_frac": np.nan,
                    "catch_swi010_mean": np.nan, "catch_swi010_std": np.nan,
                    "catch_swi010_max": np.nan, "catch_swi010_wet_frac": np.nan,
                    "catch_pixel_count": 0,
                })
            else:
                stats = _extract_basin_stats(tif_path, geom)
                row.update(stats)
            rows.append(row)

    if verbose:
        print()  # newline after progress

    df = pd.DataFrame(rows)
    col_order = [
        "station_id", "date",
        "catch_swi005_mean", "catch_swi005_std",
        "catch_swi005_max", "catch_swi005_wet_frac", "catch_swi005_oversat_frac",
        "catch_swi010_mean", "catch_swi010_std",
        "catch_swi010_max", "catch_swi010_wet_frac", "catch_swi010_oversat_frac",
        "catch_area_km2", "catch_pixel_count",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Feature-merge helper
# ---------------------------------------------------------------------------

CATCH_NOW_COLS = [
    "catch_swi005_mean", "catch_swi005_std",
    "catch_swi005_max", "catch_swi005_wet_frac", "catch_swi005_oversat_frac",
    "catch_swi010_mean", "catch_swi010_std",
    "catch_swi010_max", "catch_swi010_wet_frac", "catch_swi010_oversat_frac",
    "catch_area_km2",
]


def merge_catchment_features(df: pd.DataFrame,
                             catch_table: pd.DataFrame,
                             swi_lag_days: int = 2) -> pd.DataFrame:
    """Left-join *catch_table* into project *df* using (station_id, date - lag).

    Parameters
    ----------
    df : pd.DataFrame
        Project feature table with columns [station_id, forecast_date].
    catch_table : pd.DataFrame
        Output of build_catchment_swi_table().
    swi_lag_days : int
        ASCAT publication lag (default 2 days, matching Phase 13 convention).

    Returns
    -------
    pd.DataFrame with additional catchment SWI columns.
    """
    ct = catch_table.copy()
    ct["date"] = pd.to_datetime(ct["date"])
    # Shift: ASCAT data for date D becomes available on D+swi_lag_days
    ct["forecast_date"] = (ct["date"] + pd.Timedelta(days=swi_lag_days)
                           ).dt.strftime("%Y-%m-%d")

    avail_cols = [c for c in CATCH_NOW_COLS if c in ct.columns]
    merge_key = ct[["station_id", "forecast_date"] + avail_cols].copy()
    merge_key = merge_key.drop_duplicates(["station_id", "forecast_date"])
    out = df.merge(merge_key, on=["station_id", "forecast_date"], how="left")
    return out




# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build catchment-averaged ASCAT SWI feature table (Upgrade 2)")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: c1_satellite/outputs/c1_catchment_swi_features.csv)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="Station IDs to process (default: all)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else \
        ROOT / "c1_satellite" / "outputs" / "c1_catchment_swi_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = build_catchment_swi_table(stations=args.stations, verbose=True)
    table.to_csv(out_path, index=False)
    print(f"\n[Upgrade 2] Saved {len(table)} rows -> {out_path}")
    print(table.head(10).to_string(index=False))
