"""C1 Phase 3: inventory actual Sentinel-1 VV/VH IW acquisitions per river gauge.

No imagery is downloaded or models trained in this phase. Earth Engine access is
required only for live inventory; --dry-run verifies local inputs offline.

Run from the J26-DS-314 project root:
 python c1_satellite/scripts/03_sentinel1_inventory.py --dry-run
 python c1_satellite/scripts/03_sentinel1_inventory.py --project YOUR_GEE_PROJECT_ID --limit 3
 python c1_satellite/scripts/03_sentinel1_inventory.py --project YOUR_GEE_PROJECT_ID --all-stations
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "c1_satellite" / "local_data" / "FloodRisk_DMC_StationDay_LABELLED_1429.csv"
OUTPUT = ROOT / "c1_satellite" / "outputs"
UTC = timezone.utc
SLST = timezone(timedelta(hours=5, minutes=30))
ISSUE_LOCAL_TIME = time(9, 30)
LOOKBACK_DAYS = 60
LAG_HOURS = 48  # provisional minimum lag; NOT proof of historical publication time
MAX_MULTI_SCENES = 4

SCENE_HEADERS = [
    "station_id", "station_name", "basin", "scene_id", "acquisition_utc",
    "orbit_pass", "relative_orbit", "sensor", "spatial_filter",
]
ELIGIBILITY_HEADERS = [
    "record_id", "station_id", "forecast_date", "chronological_split",
    "nominal_issue_utc", "assumed_product_cutoff_utc", "sar_lookback_days",
    "sar_assumed_lag_hours", "sar_eligible_count", "sar_single_available",
    "sar_latest_scene_id", "sar_latest_acquisition_utc", "sar_latest_age_hours",
    "sar_same_orbit_scene_count", "sar_multi_available", "sar_multi_scene_ids",
    "sar_eligibility_note",
]


def read_labels() -> list[dict[str, str]]:
    if not LABELS.is_file():
        raise FileNotFoundError(f"Missing labelled CSV: {LABELS} (run Phase 1 first)")
    with LABELS.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"record_id", "station_id", "station_name", "basin", "forecast_date",
                "chronological_split", "gauge_lon_wgs84_arcgis", "gauge_lat_wgs84_arcgis"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"Missing required columns: {required - set(rows[0] if rows else [])}")
    if len(rows) != 1429:
        raise ValueError(f"Expected 1,429 labelled rows, got {len(rows)}")
    seen = {(r["station_id"], r["forecast_date"]) for r in rows}
    if len(seen) != len(rows):
        raise ValueError("Duplicate station/date in labelled CSV")
    return rows


def station_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    stations: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = r["station_id"]
        lon = float(r["gauge_lon_wgs84_arcgis"])
        lat = float(r["gauge_lat_wgs84_arcgis"])
        if not (79 <= lon <= 82 and 5 <= lat <= 10):
            raise ValueError(f"Station {sid}: coordinates outside approximate Sri Lanka range")
        info = {"station_id": sid, "station_name": r["station_name"],
                "basin": r["basin"], "lon": lon, "lat": lat}
        if sid in stations and (stations[sid]["lon"], stations[sid]["lat"]) != (lon, lat):
            raise ValueError(f"Station {sid}: conflicting coordinates in shared CSV")
        stations[sid] = info
    if len(stations) != 39:
        raise ValueError(f"Expected 39 labelled gauges, got {len(stations)}")
    return [stations[sid] for sid in sorted(stations)]


def nominal_issue_utc(forecast_date: str) -> datetime:
    local = datetime.combine(date.fromisoformat(forecast_date), ISSUE_LOCAL_TIME, tzinfo=SLST)
    return local.astimezone(UTC)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def serialize_station_scenes(
    station: dict[str, Any], payload: list[list[Any]]
) -> list[dict[str, Any]]:
    """Convert scene columns [index, epoch_ms, pass, relative_orbit] to inventory rows."""
    rows = []
    for info in payload:
        if len(info) != 4:
            raise ValueError(f"Unexpected Earth Engine reducer output: {info}")
        idx, timestamp_ms, orbit_pass, rel_orbit = info
        if idx is None or timestamp_ms is None:
            continue
        rows.append({
            "station_id": station["station_id"],
            "station_name": station["station_name"],
            "basin": station["basin"],
            "scene_id": f"COPERNICUS/S1_GRD/{idx}",
            "acquisition_utc": iso_utc(datetime.fromtimestamp(float(timestamp_ms) / 1000, tz=UTC)),
            "orbit_pass": str(orbit_pass or "UNKNOWN"),
            "relative_orbit": str(rel_orbit) if rel_orbit is not None else "UNKNOWN",
            "sensor": "Sentinel-1 IW VV+VH (GRD dB)",
            "spatial_filter": "intersects_gauge_point_only_not_full_patch",
        })
    return sorted(rows, key=lambda s: (s["acquisition_utc"], s["scene_id"]))


def eligibility_for_row(r: dict[str, str], scenes: list[dict[str, Any]], lag_hours: int) -> dict[str, Any]:
    issue = nominal_issue_utc(r["forecast_date"])
    cutoff = issue - timedelta(hours=lag_hours)
    earliest = issue - timedelta(days=LOOKBACK_DAYS)
    eligible = [s for s in scenes
                if earliest <= datetime.fromisoformat(s["acquisition_utc"].replace("Z", "+00:00")) <= cutoff]
    eligible.sort(key=lambda s: (s["acquisition_utc"], s["scene_id"]))
    latest = eligible[-1] if eligible else None
    same = []
    if latest is not None:
        same = [s for s in eligible if s["orbit_pass"] == latest["orbit_pass"] and
                s["relative_orbit"] == latest["relative_orbit"] and
                s["orbit_pass"] != "UNKNOWN" and s["relative_orbit"] != "UNKNOWN"]
    selected = same[-MAX_MULTI_SCENES:]
    age_h = ((issue - datetime.fromisoformat(latest["acquisition_utc"].replace("Z", "+00:00"))).total_seconds() / 3600) if latest else None
    return {
        "record_id": r["record_id"], "station_id": r["station_id"],
        "forecast_date": r["forecast_date"], "chronological_split": r["chronological_split"],
        "nominal_issue_utc": iso_utc(issue), "assumed_product_cutoff_utc": iso_utc(cutoff),
        "sar_lookback_days": LOOKBACK_DAYS, "sar_assumed_lag_hours": lag_hours,
        "sar_eligible_count": len(eligible), "sar_single_available": int(bool(latest)),
        "sar_latest_scene_id": latest["scene_id"] if latest else "",
        "sar_latest_acquisition_utc": latest["acquisition_utc"] if latest else "",
        "sar_latest_age_hours": round(age_h, 2) if age_h is not None else "",
        "sar_same_orbit_scene_count": len(same),
        "sar_multi_available": int(len(same) >= 2),
        "sar_multi_scene_ids": ";".join(s["scene_id"] for s in selected) if len(same) >= 2 else "",
        "sar_eligibility_note": "PROVISIONAL_48H_LAG_AND_POINT_INTERSECTION_ONLY; VERIFY_FULL_PATCH_AND_PUBLICATION",
    }


def write_csv(path: Path, headers: list[str], data: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def inventory_from_earth_engine(ee: Any, station: dict[str, Any], start: str, end: str) -> list[dict[str, Any]]:
    pt = ee.Geometry.Point([station["lon"], station["lat"]])
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(pt)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start"))
    columns = ["system:index", "system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"]
    result = col.reduceColumns(ee.Reducer.toList(len(columns)), columns).getInfo()
    return serialize_station_scenes(station, result.get("list", []) if result else [])


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inventory real Sentinel-1 scenes at shared DMC gauges")
    parser.add_argument("--project", help="Earth Engine-enabled Google Cloud project ID")
    parser.add_argument("--limit", type=int, default=3, help="Number of test stations (default: 3)")
    parser.add_argument("--all-stations", action="store_true", help="Query all 39 labelled gauges")
    parser.add_argument("--authenticate", action="store_true", help="Launch Earth Engine browser sign-in")
    parser.add_argument("--lag-hours", type=int, default=LAG_HOURS, help="Assumed ingestion lag; default 48h")
    parser.add_argument("--dry-run", action="store_true", help="Validate local data without Earth Engine")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.lag_hours < 0:
        parser.error("--limit must be >=1 and --lag-hours >=0")
    labels = read_labels()
    stations = station_table(labels)
    selected = stations if args.all_stations else stations[:args.limit]
    print("=== C1 SENTINEL-1 SCENE INVENTORY ===")
    print(f"Loaded {len(labels)} labelled pairs, {len(stations)} unique gauges; processing {len(selected)}")
    print("Selected stations:", ", ".join(s["station_id"] for s in selected))
    print("Lookback: 60 days; nominal forecast: 09:30 Asia/Colombo")
    print(f"Provisional availability lag: {args.lag_hours} hours")
    if args.dry_run:
        print("DRY RUN PASS: local labels and coordinates are valid; NO Earth Engine requests made.")
        return
    if not args.project:
        parser.error("Missing --project YOUR_GEE_PROJECT_ID (find it in Google Cloud/Earth Engine)")
    try:
        import ee
    except ImportError as exc:
        raise SystemExit("Install: python -m pip install earthengine-api") from exc
    try:
        if args.authenticate:
            ee.Authenticate(auth_mode="localhost")
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise SystemExit("Earth Engine initialization failed. Confirm Earth Engine registration, "
                         "Google Cloud project ID and browser sign-in. Retry with --authenticate. "
                         f"\nError: {exc}") from exc
    min_issue = min(nominal_issue_utc(r["forecast_date"]) for r in labels)
    max_issue = max(nominal_issue_utc(r["forecast_date"]) for r in labels)
    start = (min_issue - timedelta(days=LOOKBACK_DAYS + 1)).date().isoformat()
    end = (max_issue + timedelta(days=1)).date().isoformat()  # exclusive; row-wise cutoff applied locally
    all_scenes = []
    for idx, st in enumerate(selected, start=1):
        try:
            found = inventory_from_earth_engine(ee, st, start, end)
        except Exception as exc:
            raise SystemExit(f"Earth Engine request failed for station {st['station_id']}: {exc}") from exc
        all_scenes.extend(found)
        print(f"[{idx:02d}/{len(selected)}] {st['station_name']}: {len(found)} actual IW VV/VH acquisitions")
    by_station = defaultdict(list)
    for scene in all_scenes:
        by_station[scene["station_id"]].append(scene)
    selected_ids = {s["station_id"] for s in selected}
    daily = [eligibility_for_row(r, by_station.get(r["station_id"], []), args.lag_hours)
             for r in labels if r["station_id"] in selected_ids]
    coverage = []
    by_split = defaultdict(list)
    for row in daily:
        by_split[row["chronological_split"]].append(row)
    for st in selected:
        subset = [r for r in daily if r["station_id"] == st["station_id"]]
        coverage.append({
            "station_id": st["station_id"], "station_name": st["station_name"], "basin": st["basin"],
            "n_labelled_days": len(subset), "n_distinct_scenes": len(by_station.get(st["station_id"], [])),
            "n_single_eligible_days": sum(int(r["sar_single_available"]) for r in subset),
            "n_multi_eligible_days": sum(int(r["sar_multi_available"]) for r in subset),
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "c1_s1_scene_inventory.csv", SCENE_HEADERS, all_scenes)
    write_csv(OUTPUT / "c1_s1_day_eligibility.csv", ELIGIBILITY_HEADERS, daily)
    write_csv(OUTPUT / "c1_s1_station_coverage.csv", list(coverage[0]), coverage)
    summary = {
        "data_source": "COPERNICUS/S1_GRD", "stations_scanned": len(selected),
        "labelled_rows_scanned": len(daily), "scene_station_records": len(all_scenes),
        "s1_single_eligible_rows": sum(int(r["sar_single_available"]) for r in daily),
        "s1_multi_same_orbit_eligible_rows": sum(int(r["sar_multi_available"]) for r in daily),
        "split_coverage": {s: {"rows": len(rr), "single": sum(int(x["sar_single_available"]) for x in rr),
                              "multi": sum(int(x["sar_multi_available"]) for x in rr)}
                          for s, rr in sorted(by_split.items())},
        "issue_local": "09:30 Asia/Colombo", "assumed_lag_hours": args.lag_hours,
        "lookback_days": LOOKBACK_DAYS,
        "warnings": [
            "Scene acquisition timestamps are real Earth Engine catalog metadata; actual historical product publication times are not verified.",
            "The station point may intersect a scene even if the required full 12.8km patch is not covered.",
            "These outputs are image AVAILABILITY metadata, not downloaded VV/VH arrays or trained satellite models.",
            "40th gauge has no labelled next-day pairs and is intentionally absent from this supervised inventory."
        ],
    }
    (OUTPUT / "c1_s1_inventory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(f"Station-days: {len(daily)}; single-date eligible: {summary['s1_single_eligible_rows']}; "
          f"same-orbit multi-date eligible: {summary['s1_multi_same_orbit_eligible_rows']}")
    print(f"4 metadata files saved under: {OUTPUT}")
    print("NEXT: Verify full-patch coverage, then export actual VV/VH pixels for eligible dates.")


if __name__ == "__main__":
    run()
