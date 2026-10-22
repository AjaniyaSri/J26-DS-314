#!/usr/bin/env python
"""Script 22 – Extract Catchment-Averaged ASCAT SWI Features (Upgrade 2).

Loops over all 40 gauge stations and all available ASCAT SWI TIF files,
clips each raster to the upstream catchment polygon, and saves basin-
averaged statistics to:

    c1_satellite/outputs/c1_catchment_swi_features.csv

This CSV is then used by Script 23 (benchmark_with_catchment.py) to
evaluate whether catchment SWI improves over the point-SWI baseline.

Run from project root:
    .venv\\Scripts\\python c1_satellite/scripts/22_extract_catchment_swi_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c1_satellite.src.improvements.catchment_swi import (
    build_catchment_swi_table,
    ASCAT_DIR,
    CATCH_NOW_COLS,
)

OUTPUT_PATH = ROOT / "c1_satellite" / "outputs" / "c1_catchment_swi_features.csv"


def main() -> None:
    print("=" * 60)
    print("Script 22 – Catchment SWI Feature Extraction (Upgrade 2)")
    print("=" * 60)

    # Build full table for all stations
    table = build_catchment_swi_table(ascat_dir=ASCAT_DIR, verbose=True)

    # Summary
    n_total = len(table)
    n_complete = table[CATCH_NOW_COLS].notna().all(axis=1).sum()
    stations_with_data = table.groupby("station_id")[CATCH_NOW_COLS[0]].apply(
        lambda x: x.notna().any()
    )
    n_stations_ok = int(stations_with_data.sum())

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Total station-day rows   : {n_total}")
    print(f"  Rows with complete data  : {n_complete} ({100*n_complete/n_total:.1f}%)")
    print(f"  Stations with any data   : {n_stations_ok} / {table['station_id'].nunique()}")

    # Per-station overview
    print("\nPer-station basin summary:")
    per = table.groupby("station_id").agg(
        area_km2=("catch_area_km2", "first"),
        pixel_count=("catch_pixel_count", "median"),
        swi005_mean=("catch_swi005_mean", "mean"),
        swi010_mean=("catch_swi010_mean", "mean"),
    ).round(1)
    print(per.to_string())

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[Script 22] Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
