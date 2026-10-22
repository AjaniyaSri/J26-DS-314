"""Copy only missing catchment files from team candidate archives into repo.

Uses the shared 1600-row master manifest to identify paths. Never overwrites
existing repository files. Candidate polygons are provisional / require QA.
Run: python c1_satellite/scripts/02_install_candidate_catchments.py
"""
from __future__ import annotations
import csv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'c1_satellite' / 'local_data'
MASTER = LOCAL / 'FloodRisk_DMC_StationDay_MASTER_1600.csv'
ARCHIVES = [
    LOCAL / 'source_zips' / 'kelani_7_catchments_CANDIDATE_team_share.zip',
    LOCAL / 'source_zips' / 'additional_33_catchments_CANDIDATE_team_share.zip',
]

def main():
    if not MASTER.is_file():
        raise SystemExit(f'Missing DMC master: {MASTER}')
    with MASTER.open(encoding='utf-8-sig', newline='') as f:
        refs = sorted({r['catchment_polygon_path'].replace('\\', '/') for r in csv.DictReader(f) if r.get('catchment_polygon_path')})
    for p in ARCHIVES:
        if not p.is_file():
            raise SystemExit(f'Missing candidate ZIP: {p}. Extract Phase 2 starter into repo root first.')
    copied = 0
    existing = 0
    missing_in_zips = []
    with zipfile.ZipFile(ARCHIVES[0]) as z1, zipfile.ZipFile(ARCHIVES[1]) as z2:
        lookup = {name: z for z in (z1,z2) for name in z.namelist() if name.endswith('.geojson')}
        for ref in refs:
            if not ref.startswith('shared/spatial/catchments/') or '..' in Path(ref).parts:
                raise ValueError(f'Unsafe catchment path in data: {ref}')
            dest = ROOT / ref
            if dest.is_file():
                existing += 1
                continue
            archive = lookup.get(ref)
            if archive is None:
                missing_in_zips.append(ref)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(ref) as src, dest.open('xb') as out:
                while True:
                    chunk = src.read(1024*1024)
                    if not chunk: break
                    out.write(chunk)
            copied += 1
    missing = [ref for ref in refs if not (ROOT/ref).is_file()]
    print('=== C1 CATCHMENT IMPORT ===')
    print('Referenced catchment polygons:',len(refs))
    print('Copied into missing paths: ',copied)
    print('Already present (untouched):',existing)
    print('Still missing:            ',len(missing))
    if missing:
        print('First missing paths:', '\n'.join(missing[:10]))
        raise SystemExit(1)
    print('PASS: 40/40 candidate polygon references resolve locally.')
    print('CAUTION: These are CANDIDATE boundaries, not officially verified catchments.')

if __name__ == '__main__':
    main()
