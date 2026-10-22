"""C1 Phase 6: safely download distinct real SAR scenes for TRAIN/VALIDATION/TEST.

Reuses Phase 3 eligibility/inventory and Phase 4 *verified* pixel extraction logic.
Writes its own append-only manifest and does not overwrite any earlier script or CSV.

From J26-DS-314 root:
 python c1_satellite/scripts/06_batch_sentinel1_patches.py --split TRAIN --station-limit 3 --max-new 12 --dry-run
 python c1_satellite/scripts/06_batch_sentinel1_patches.py --project YOUR_GEE_PROJECT --split TRAIN --station-limit 3 --max-new 12
 python c1_satellite/scripts/05_sar_feature_table.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'c1_satellite' / 'outputs'
SCRIPTS = ROOT / 'c1_satellite' / 'scripts'
INVENTORY = OUT / 'c1_s1_scene_inventory.csv'
ELIGIBILITY = OUT / 'c1_s1_day_eligibility.csv'
LABELS = ROOT / 'c1_satellite' / 'local_data' / 'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
PATCHES = OUT / 's1_patches'
LOG = OUT / 'c1_s1_batch_qa.csv'
REPORT = OUT / 'c1_s1_batch_summary.json'
HEADERS = [
    'requested_split', 'station_id', 'scene_id', 'acquisition_utc', 'orbit_pass',
    'relative_orbit', 'ee_coverage', 'raster_valid_fraction', 'status',
    'geotiff_path', 'preview_path', 'note',
]


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}; complete Phases 1 and 3 first')
    with path.open(encoding='utf-8-sig', newline='') as f:
        result = list(csv.DictReader(f))
    if not result:
        raise ValueError(f'CSV has no records: {path}')
    return result


def scene_ids(raw: str) -> list[str]:
    return [s.strip() for s in (raw or '').split(';') if s.strip()]


def eligible_scene_candidates(days: list[dict[str, str]], scenes: list[dict[str, str]], split: str):
    """Latest SAR plus previous same-orbit scene only, no fabricated acquisitions.

    Return stations mapped to chronologically newest-first scene rows, distinct by
    station ID + scene ID. Date eligibility is supplied by Phase 3; never query
    scenes not listed as pre-cutoff for a selected labelled forecast date.
    """
    lookups = {(s['station_id'], s['scene_id']): s for s in scenes}
    picks: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    excluded = 0
    for day in days:
        if split != 'ALL' and day.get('chronological_split') != split:
            continue
        st = day['station_id']
        latest = day.get('sar_latest_scene_id', '').strip()
        if not latest:
            continue
        current = lookups.get((st, latest))
        if current is None:
            excluded += 1
            continue
        # Current single SAR, as of eligible forecast date.
        picks[st][latest] = current
        multi = scene_ids(day.get('sar_multi_scene_ids', ''))
        if len(multi) >= 2 and multi[-1] == latest:
            prev = lookups.get((st, multi[-2]))
            if (prev is not None
                and prev.get('orbit_pass') not in ('', 'UNKNOWN')
                and prev.get('relative_orbit') not in ('', 'UNKNOWN')
                and prev['orbit_pass'] == current['orbit_pass']
                and prev['relative_orbit'] == current['relative_orbit']
                and prev['acquisition_utc'][:10] < current['acquisition_utc'][:10]):
                picks[st][prev['scene_id']] = prev
    return {st: sorted(scene_map.values(), key=lambda r: (r['acquisition_utc'],r['scene_id']), reverse=True)
            for st,scene_map in sorted(picks.items())}, excluded


def round_robin_plan(candidates: dict[str, list[dict[str, str]]], station_limit: int,
                     existing_predicate=None) -> list[dict[str, str]]:
    """Alternate stations to avoid exhausting the entire budget on one gauge."""
    stations = list(candidates)[:station_limit]
    plan = []
    for idx in range(max((len(candidates[s]) for s in stations), default=0)):
        for st in stations:
            if idx >= len(candidates[st]):
                continue
            scene = candidates[st][idx]
            if existing_predicate is None or not existing_predicate(scene):
                plan.append(scene)
    return plan


def load_phase4():
    path = SCRIPTS / '04_sentinel1_patch_export.py'
    if not path.exists():
        raise FileNotFoundError(f'Phase 4 source missing: {path}')
    spec = importlib.util.spec_from_file_location('c1_phase4_export', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def append_log(row: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    has_rows = LOG.is_file() and LOG.stat().st_size > 0
    with LOG.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction='ignore')
        if not has_rows:
            writer.writeheader()
        writer.writerow(row)


def cli(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description='Download real SAR train scenes incrementally; no model training')
    ap.add_argument('--split', choices=['TRAIN','VALIDATION','TEST','ALL'], default='TRAIN')
    ap.add_argument('--station-id', help='Optional one exact station ID')
    ap.add_argument('--station-limit', type=int, default=3, help='First N stations in deterministic order, default 3')
    ap.add_argument('--max-new', type=int, default=12, help='Maximum new scene attempts in this run, default 12')
    ap.add_argument('--project', help='Earth Engine-enabled Google Cloud project ID')
    ap.add_argument('--authenticate', action='store_true')
    ap.add_argument('--min-coverage', type=float, default=0.98)
    ap.add_argument('--dry-run', action='store_true', help='Show plan without contacting Earth Engine')
    ap.add_argument('--qa-only', action='store_true', help='Check full-patch coverage, no download')
    ap.add_argument('--previews', action='store_true', help='Generate preview PNGs for new or existing TIFFs')
    args = ap.parse_args(argv)
    if args.station_limit < 1 or args.max_new < 1 or not (0 < args.min_coverage <= 1):
        ap.error('station-limit and max-new must be positive; min-coverage must be in (0,1]')
    days = load_rows(ELIGIBILITY)
    scenes = load_rows(INVENTORY)
    labels = load_rows(LABELS)
    p4 = load_phase4()
    by_station, excluded = eligible_scene_candidates(days, scenes, args.split)
    if args.station_id:
        if args.station_id not in by_station:
            ap.error(f'Station {args.station_id} has no eligible scenes for {args.split}')
        by_station = {args.station_id: by_station[args.station_id]}
    # Do not count previously saved rasters towards budget. Existing TIFFs are
    # validated later by Phase 5 and are never overwritten by this batch.
    plan = round_robin_plan(by_station, args.station_limit)
    existing, missing = [], []
    for scene in plan:
        tiff = PATCHES / (p4.filename_for_scene(scene['station_id'], scene) + '.tif')
        (existing if tiff.is_file() else missing).append(scene)
    print('=== C1 PHASE 6: SAFE BATCH SENTINEL-1 PIXEL EXPORT ===')
    print(f'Split: {args.split}; stations selected: {min(len(by_station),args.station_limit)}')
    print(f'Distinct eligible station/scenes: {len(plan)}; local TIFFs already present: {len(existing)}')
    print(f'Remaining candidate scenes: {len(missing)}; max new attempts this run: {args.max_new}')
    print(f'Rows missing phase-3 latest-inventory reference: {excluded}')
    for s in missing[:min(args.max_new, 12)]:
        print(f"  QUEUED {s['station_id']:18s} {s['acquisition_utc'][:10]}  orbit {s['orbit_pass']}/{s['relative_orbit']}")
    if args.dry_run:
        print('DRY RUN PASS: no Earth Engine access or file changes.')
        return
    if not args.project:
        ap.error('--project is required unless --dry-run')
    try:
        import ee
        from pyproj import Transformer
    except ImportError as exc:
        raise SystemExit('Install: python -m pip install earthengine-api pyproj rasterio numpy requests matplotlib') from exc
    if args.authenticate:
        ee.Authenticate(auth_mode='localhost')
    ee.Initialize(project=args.project)
    PATCHES.mkdir(parents=True, exist_ok=True)
    selected = missing[:args.max_new]
    qa_ok = downloaded = failed = 0
    coordinates = {}
    for station_id in {s['station_id'] for s in selected}:
        coordinates[station_id] = p4.station_coordinates(labels, station_id)
    for i, scene in enumerate(selected, 1):
        st = scene['station_id']
        filename = p4.filename_for_scene(st, scene)
        tif = PATCHES / (filename + '.tif')
        png = PATCHES / (filename + '_preview.png')
        rec = {k:'' for k in HEADERS}
        rec.update({'requested_split': args.split, 'station_id': st,
                    'scene_id': scene['scene_id'], 'acquisition_utc': scene['acquisition_utc'],
                    'orbit_pass': scene['orbit_pass'], 'relative_orbit': scene['relative_orbit']})
        print(f'[{i}/{len(selected)}] {st}: {scene["acquisition_utc"][:10]}')
        try:
            lon,lat = coordinates[st]
            x,y = Transformer.from_crs('EPSG:4326', p4.CRS, always_xy=True).transform(lon,lat)
            grid = p4.grid_for_xy(x,y)
            roi = ee.Geometry.Rectangle([grid['left'],grid['bottom'],grid['right'],grid['top']],
                                        proj=p4.CRS,geodesic=False)
            image = ee.Image(scene['scene_id'])
            fraction = p4.ee_valid_fraction(ee,image,roi,grid['transform'])
            rec['ee_coverage'] = round(fraction,6)
            print(f'    Full-patch coverage: {fraction:.1%}')
            if fraction < args.min_coverage:
                rec['status'] = 'SKIPPED_PARTIAL_PATCH'
            elif args.qa_only:
                qa_ok += 1
                rec['status'] = 'QA_OK_NOT_DOWNLOADED'
            else:
                if tif.exists():
                    # Do not overwrite even if a previous/parallel run wrote it.
                    rec['status'] = 'EXISTING_UNTOUCHED'
                else:
                    p4.download_geotiff(image,grid['transform'],tif)
                    rec['status'] = 'DOWNLOADED'
                raster_fraction = p4.inspect_and_preview(tif,png,make_preview=args.previews)
                rec['raster_valid_fraction'] = round(raster_fraction,6)
                if raster_fraction < args.min_coverage:
                    rec['status']='FAILED_RASTER_COVERAGE'
                else:
                    qa_ok += 1
                    downloaded += rec['status']=='DOWNLOADED'
                rec['geotiff_path']=str(tif.relative_to(ROOT)).replace('\\','/')
                if png.exists():
                    rec['preview_path']=str(png.relative_to(ROOT)).replace('\\','/')
        except Exception as exc:
            rec['status'] = 'ERROR'
            rec['note'] = str(exc)[:350]
            failed += 1
            print(f'    ERROR: {exc}')
        else:
            print(f'    {rec["status"]}')
        append_log(rec)
    report={'mode': 'QA_ONLY' if args.qa_only else 'DOWNLOAD', 'split': args.split,
            'stations_considered': min(len(by_station),args.station_limit),
            'eligible_unique_scenes':len(plan),'already_downloaded_before_run':len(existing),
            'attempted':len(selected),'qa_pass':qa_ok,'downloaded_new':int(downloaded),
            'errors':failed,'manifest':str(LOG.relative_to(ROOT)),
            'cautions':['No flood model trained here.',
                        'Scene candidates obey Phase 3 time eligibility with provisional 48-hour lag; historical product availability not verified.',
                        'Actual 12.8-km patch coverage checked per scene; no fake pixel values.',
                        'Candidate catchments are not official QA-approved boundaries.',
                        'Run Phase 5 again to recompute TRAIN/VALIDATION/TEST real-feature coverage.']}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'SUMMARY: attempted={len(selected)}, QA passed={qa_ok}, new downloads={downloaded}, errors={failed}')
    print(f'Batch QA log: {LOG.relative_to(ROOT)}')
    print('NEXT: python c1_satellite/scripts/05_sar_feature_table.py')


if __name__ == '__main__':
    cli()
