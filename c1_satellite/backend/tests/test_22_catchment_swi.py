"""Unit tests for Upgrade 2 – catchment_swi.py

Tests:
  1. Registry builds correctly and contains known stations.
  2. Polygon loading returns a Shapely geometry.
  3. Fuzzy matching handles 'nagalagam_street' → 'Nagalagam Street' GeoJSON.
  4. build_catchment_swi_table() runs end-to-end on a single station/day.
  5. All output columns are present and numeric.
  6. merge_catchment_features() correctly shifts dates by swi_lag_days.
  7. NaN rows are produced (not exceptions) when station has no polygon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c1_satellite.src.improvements.catchment_swi import (
    _build_station_polygon_registry,
    load_catchment_polygon,
    build_catchment_swi_table,
    merge_catchment_features,
    CATCH_NOW_COLS,
    ASCAT_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_ascat_tif() -> Path:
    tifs = sorted(ASCAT_DIR.glob("ASCAT_SWI_V4_*.tif"))
    if not tifs:
        if pytest:
            pytest.skip("No ASCAT TIF files available")
        else:
            return None
    return tifs[0]



# ---------------------------------------------------------------------------
# Test 1: Registry
# ---------------------------------------------------------------------------

def test_registry_builds_and_contains_ellagawa():
    reg = _build_station_polygon_registry()
    assert len(reg) >= 10, "Expected at least 10 catchment polygons in registry"
    assert "ellagawa" in reg, "ellagawa must be in registry"


def test_registry_all_paths_exist():
    reg = _build_station_polygon_registry()
    for sid, path in reg.items():
        assert path.exists(), f"GeoJSON path for '{sid}' does not exist: {path}"


# ---------------------------------------------------------------------------
# Test 2: Polygon loading
# ---------------------------------------------------------------------------

def test_load_polygon_returns_geometry():
    geom = load_catchment_polygon("ellagawa")
    assert geom is not None, "Expected a geometry for 'ellagawa'"
    assert hasattr(geom, "area"), "Geometry should have .area attribute"
    assert geom.area > 0, "Catchment area should be positive"


def test_load_polygon_unknown_returns_none():
    geom = load_catchment_polygon("totally_nonexistent_station_xyz")
    assert geom is None, "Unknown station should return None, not raise"


# ---------------------------------------------------------------------------
# Test 3: Fuzzy matching
# ---------------------------------------------------------------------------

def test_fuzzy_match_nagalagam_street():
    geom = load_catchment_polygon("nagalagam_street")
    assert geom is not None, "Fuzzy match should find 'Nagalagam Street' polygon"


# ---------------------------------------------------------------------------
# Test 4 & 5: build_catchment_swi_table for one station / one day
# ---------------------------------------------------------------------------

def test_build_single_station_single_day():
    tif = _first_ascat_tif()
    date_str = tif.stem.replace("ASCAT_SWI_V4_", "")
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["ellagawa"],
        verbose=False,
    )
    # Filter to first day
    day_df = df[df["date"] == date_str]
    assert len(day_df) == 1, "Should have exactly 1 row for 1 station x 1 day"
    row = day_df.iloc[0]
    assert row["station_id"] == "ellagawa"
    assert row["date"] == date_str


def test_output_columns_present():
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["ellagawa"],
        verbose=False,
    )
    expected_cols = [
        "station_id", "date",
        "catch_swi005_mean", "catch_swi005_std", "catch_swi005_max", "catch_swi005_wet_frac", "catch_swi005_oversat_frac",
        "catch_swi010_mean", "catch_swi010_std", "catch_swi010_max", "catch_swi010_wet_frac", "catch_swi010_oversat_frac",
        "catch_area_km2", "catch_pixel_count",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_output_values_are_valid():
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["ellagawa"],
        verbose=False,
    )
    # At least some rows should have real data (not all NaN)
    has_data = df["catch_swi005_mean"].notna().any()
    assert has_data, "Expected at least some non-NaN SWI005 values for ellagawa"
    # All mean values should be >= 0 (CLMS V4 SWI can exceed 100 up to ~160+ during oversaturation/flooding)
    means = df["catch_swi005_mean"].dropna()
    assert (means >= 0).all() and (means <= 200).all(), "SWI005 mean should be in valid range [0, 200]"


def test_wet_fraction_in_range():
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["ellagawa"],
        verbose=False,
    )
    wf = df["catch_swi005_wet_frac"].dropna()
    assert (wf >= 0).all() and (wf <= 1).all(), "Wet fraction must be in [0,1]"
    of = df["catch_swi005_oversat_frac"].dropna()
    assert (of >= 0).all() and (of <= 1).all(), "Oversat fraction must be in [0,1]"



def test_pixel_count_positive_for_known_station():
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["ellagawa"],
        verbose=False,
    )
    counts = df["catch_pixel_count"]
    assert (counts >= 0).all()
    # Ellagawa is 1393 km2, ASCAT is ~25km grid -> expect at least 1 pixel
    assert counts.max() >= 1, "Expected at least 1 ASCAT pixel inside Ellagawa catchment"


def test_nan_station_no_polygon():
    """Station with no GeoJSON should produce NaN rows without crashing."""
    df = build_catchment_swi_table(
        ascat_dir=ASCAT_DIR,
        stations=["this_station_does_not_exist"],
        verbose=False,
    )
    assert len(df) > 0, "Should still produce rows even for unknown station"
    assert df["catch_swi005_mean"].isna().all(), "All rows should be NaN for unknown station"


# ---------------------------------------------------------------------------
# Test 6: merge_catchment_features date shifting
# ---------------------------------------------------------------------------

def test_merge_lag_shifts_dates():
    catch = pd.DataFrame({
        "station_id": ["ellagawa", "ellagawa"],
        "date": ["2025-11-08", "2025-11-09"],
        "catch_swi005_mean": [70.0, 72.0],
        "catch_swi005_std": [3.0, 3.1],
        "catch_swi005_max": [80.0, 82.0],
        "catch_swi005_wet_frac": [0.8, 0.9],
        "catch_swi010_mean": [65.0, 67.0],
        "catch_swi010_std": [2.5, 2.6],
        "catch_swi010_max": [75.0, 77.0],
        "catch_swi010_wet_frac": [0.7, 0.75],
        "catch_area_km2": [1393.0, 1393.0],
        "catch_pixel_count": [5, 5],
    })
    project_df = pd.DataFrame({
        "station_id": ["ellagawa", "ellagawa"],
        "forecast_date": ["2025-11-10", "2025-11-11"],  # date + 2 lag
        "stage_at_issue_m": [1.0, 1.1],
    })
    merged = merge_catchment_features(project_df, catch, swi_lag_days=2)
    assert "catch_swi005_mean" in merged.columns
    # Row for forecast_date=2025-11-10 should get ASCAT data from 2025-11-08 (lag=2)
    row = merged[merged["forecast_date"] == "2025-11-10"].iloc[0]
    assert abs(row["catch_swi005_mean"] - 70.0) < 1e-6, \
        f"Expected 70.0 (date shifted by -2 days), got {row['catch_swi005_mean']}"


def test_merge_unmatched_rows_are_nan():
    """Rows without corresponding catchment data should remain NaN (left join)."""
    catch = pd.DataFrame({
        "station_id": ["ellagawa"],
        "date": ["2025-11-08"],
        "catch_swi005_mean": [70.0],
        "catch_swi005_std": [3.0],
        "catch_swi005_max": [80.0],
        "catch_swi005_wet_frac": [0.8],
        "catch_swi010_mean": [65.0],
        "catch_swi010_std": [2.5],
        "catch_swi010_max": [75.0],
        "catch_swi010_wet_frac": [0.7],
        "catch_area_km2": [1393.0],
        "catch_pixel_count": [5],
    })
    project_df = pd.DataFrame({
        "station_id": ["ellagawa"],
        "forecast_date": ["2099-01-01"],   # far future – no match
        "stage_at_issue_m": [1.0],
    })
    merged = merge_catchment_features(project_df, catch, swi_lag_days=2)
    assert merged["catch_swi005_mean"].isna().all(), \
        "Unmatched rows should produce NaN after left join"
