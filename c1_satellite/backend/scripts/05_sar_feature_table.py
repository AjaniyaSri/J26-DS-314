"""C1 Phase 5: extract measured Sentinel-1 SAR patch features and align station-days.

Use only downloaded and quality-checked real VV/VH GeoTIFFs from Phase 4.
Does NOT train a model, fill missing images, or read future river-level labels.
Run from the repository root:
  python c1_satellite/scripts/05_sar_feature_table.py --dry-run
  python c1_satellite/scripts/05_sar_feature_table.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'c1_satellite'
OUT = BASE / 'outputs'
PATCHES = OUT / 's1_patches'
INVENTORY = OUT / 'c1_s1_scene_inventory.csv'
ELIGIBILITY = OUT / 'c1_s1_day_eligibility.csv'
SCENE_OUT = OUT / 'c1_sar_scene_features.csv'
DAY_OUT = OUT / 'c1_sar_station_day_features.csv'
SUMMARY_OUT = OUT / 'c1_sar_feature_summary.json'
NODATA = -9999.0


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 3 input: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No data in {path}')
    return rows


def scene_stem(station: str, scene: dict):
    date_str = scene['acquisition_utc'].replace('-', '').replace(':', '').replace('T', '_').replace('Z', '')
    uid = hashlib.sha1(scene['scene_id'].encode('utf-8')).hexdigest()[:9]
    return f'{station}_{date_str}_{uid}'


def get_features(tif: Path, threshold: float = .98):
    import numpy as np
    import rasterio
    with rasterio.open(tif) as ds:
        if ds.count != 2 or ds.width != 256 or ds.height != 256:
            raise ValueError(f'Not the expected 2-band 256x256 patch: {tif.name}')
        if ds.crs is None or ds.crs.to_string() != 'EPSG:32644':
            raise ValueError(f'Unexpected CRS in {tif.name}: {ds.crs}')
        if abs(ds.transform.a - 50) > 1e-3 or abs(ds.transform.e + 50) > 1e-3:
            raise ValueError(f'Unexpected pixel size in {tif.name}')
        a = ds.read()
    valid = np.isfinite(a).all(axis=0) & (a[0] != NODATA) & (a[1] != NODATA)
    fraction = float(valid.mean())
    if fraction < threshold:
        raise ValueError(f'Patch valid fraction {fraction:.3f} below {threshold}')
    features = {'sar_patch_valid_fraction': round(fraction, 6)}
    for i, band in enumerate(('vv','vh')):
        pixels = a[i][valid].astype('float64')
        features[f'sar_{band}_mean_db'] = round(float(pixels.mean()), 5)
        features[f'sar_{band}_std_db'] = round(float(pixels.std()), 5)
        features[f'sar_{band}_p10_db'] = round(float(np.percentile(pixels, 10)), 5)
        features[f'sar_{band}_p90_db'] = round(float(np.percentile(pixels, 90)), 5)
    return features


SCENE_FIELDS=['station_id','scene_id','acquisition_utc','orbit_pass','relative_orbit','geotiff_path','qa_status','error',
              'sar_patch_valid_fraction','sar_vv_mean_db','sar_vv_std_db','sar_vv_p10_db','sar_vv_p90_db',
              'sar_vh_mean_db','sar_vh_std_db','sar_vh_p10_db','sar_vh_p90_db']
DAY_FIELDS=['record_id','station_id','forecast_date','chronological_split',
            'sar_latest_local_available','sar_latest_scene_id','sar_latest_patch_path',
            'sar_latest_acquisition_utc','sar_latest_age_hours',
            'sar_vv_mean_db','sar_vh_mean_db','sar_vv_std_db','sar_vh_std_db','sar_vv_p10_db','sar_vv_p90_db',
            'sar_vh_p10_db','sar_vh_p90_db','sar_patch_valid_fraction',
            'sar_multi_local_available','sar_previous_scene_id','sar_previous_patch_path',
            'sar_previous_acquisition_utc','sar_vv_mean_change_db','sar_vh_mean_change_db',
            'sar_feature_status']


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def split_ids(raw):
    return [s.strip() for s in (raw or '').split(';') if s.strip()]


def build_day_feature(eligible, scene_map, local_features):
    """Joins by station_id + scene_id; missing real patches remain missing."""
    station=eligible['station_id']
    last_id=eligible.get('sar_latest_scene_id','')
    out={k:'' for k in DAY_FIELDS}
    for key in ('record_id','station_id','forecast_date','chronological_split'):
        out[key]=eligible.get(key,'')
    out.update(sar_latest_local_available=0,sar_multi_local_available=0,
               sar_latest_scene_id=last_id,sar_feature_status='NO_ELIGIBLE_SAR')
    if not last_id:
        return out
    last=local_features.get((station,last_id))
    if last is None or last.get('qa_status')!='PASS':
        out['sar_feature_status']='ELIGIBLE_SCENE_NOT_DOWNLOADED_OR_INVALID'
        return out
    out.update(sar_latest_local_available=1,sar_latest_patch_path=last['geotiff_path'],
               sar_latest_acquisition_utc=last['acquisition_utc'],
               sar_latest_age_hours=eligible.get('sar_latest_age_hours',''),
               sar_feature_status='SINGLE_PATCH_READY')
    for key in ('sar_vv_mean_db','sar_vh_mean_db','sar_vv_std_db','sar_vh_std_db',
                'sar_vv_p10_db','sar_vv_p90_db','sar_vh_p10_db','sar_vh_p90_db','sar_patch_valid_fraction'):
        out[key]=last[key]
    multi_ids=split_ids(eligible.get('sar_multi_scene_ids',''))
    if len(multi_ids)<2 or multi_ids[-1]!=last_id:
        return out
    prev_id=multi_ids[-2]
    previous=local_features.get((station,prev_id))
    if previous is None or previous.get('qa_status')!='PASS':
        return out
    ref=scene_map.get((station,prev_id))
    if ref is None:
        return out
    if (ref.get('orbit_pass'), ref.get('relative_orbit')) != (last.get('orbit_pass'),last.get('relative_orbit')):
        return out
    if ref['acquisition_utc'][:10]>=last['acquisition_utc'][:10]:
        return out
    out.update(sar_multi_local_available=1,sar_previous_scene_id=prev_id,
               sar_previous_patch_path=previous['geotiff_path'],
               sar_previous_acquisition_utc=previous['acquisition_utc'],
               sar_vv_mean_change_db=round(last['sar_vv_mean_db']-previous['sar_vv_mean_db'],5),
               sar_vh_mean_change_db=round(last['sar_vh_mean_db']-previous['sar_vh_mean_db'],5),
               sar_feature_status='MULTI_PATCH_READY')
    return out


def run():
    ap=argparse.ArgumentParser(description='Build real SAR scene and station-day feature tables (no model training).')
    ap.add_argument('--dry-run',action='store_true',help='List available files and planned work')
    ap.add_argument('--min-valid',type=float,default=.98,help='Required raster valid fraction (default: 0.98)')
    args=ap.parse_args()
    if not 0 < args.min_valid<=1:
        ap.error('--min-valid must be in (0,1]')
    inventory=read_csv(INVENTORY); eligible=read_csv(ELIGIBILITY)
    expected={(r['station_id'],r['scene_id']):r for r in inventory}
    existing=[]
    for key,scene in expected.items():
        tif=PATCHES/(scene_stem(scene['station_id'],scene)+'.tif')
        if tif.is_file():
            existing.append((key,scene,tif))
    print('=== C1 PHASE 5: REAL SENTINEL-1 FEATURE TABLE ===')
    print(f'Phase 3: {len(eligible)} station-days; {len(expected)} station/scene references')
    print(f'Real matching GeoTIFFs found: {len(existing)}, across {len(set(s[0][0] for s in existing))} stations')
    if args.dry_run:
        print('DRY RUN PASS. No model trained; no data changed.')
        return
    local={}
    for key,scene,tif in existing:
        row={k:'' for k in SCENE_FIELDS}
        row.update({k:scene.get(k,'') for k in ('station_id','scene_id','acquisition_utc','orbit_pass','relative_orbit')})
        row['geotiff_path']=str(tif.relative_to(ROOT)).replace('\\','/')
        try:
            row.update(get_features(tif,args.min_valid)); row['qa_status']='PASS'
        except Exception as exc:
            row['qa_status']='FAIL'; row['error']=str(exc)[:260]
        local[key]=row
    days=[build_day_feature(r,expected,local) for r in eligible]
    write_csv(SCENE_OUT,SCENE_FIELDS,list(local.values()))
    write_csv(DAY_OUT,DAY_FIELDS,days)
    status=Counter(r['sar_feature_status'] for r in days)
    by_split={s:{'total':sum(r['chronological_split']==s for r in days),
                 'single':sum(r['chronological_split']==s and r['sar_latest_local_available']==1 for r in days),
                 'multi':sum(r['chronological_split']==s and r['sar_multi_local_available']==1 for r in days)}
              for s in ('TRAIN','VALIDATION','TEST')}
    summary={'source':'real Earth Engine COPERNICUS/S1_GRD 2-band local GeoTIFF patches',
             'total_station_days':len(days),'scene_tifs_found':len(existing),
             'scene_tifs_passed':sum(r['qa_status']=='PASS' for r in local.values()),
             'station_day_status':dict(status),'coverage_by_split':by_split,
             'cautions':['No fitted prediction model yet.',
                         'Only existing real pixels are used, missing scenes left missing.',
                         'Do not use next-day target as input.',
                         'Scene availability remains based on Phase 3 provisional lag; verify for operational use.',
                         'Need multiple genuine historical acquisitions and adequate training split coverage.']}
    SUMMARY_OUT.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Coverage (locally downloaded and QA-valid):')
    for name, data in by_split.items():
        print(f"  {name:10s} total={data['total']:4d} single={data['single']:4d} multi={data['multi']:4d}")
    print('Feature statuses:',dict(status))
    print('Saved:',SCENE_OUT.relative_to(ROOT), DAY_OUT.relative_to(ROOT), SUMMARY_OUT.relative_to(ROOT),sep='\n  ')
    print('NEXT: Collect TRAIN + VALIDATION patches for all eligible stations; join ASCAT SWI and train controlled experiments.')


if __name__=='__main__':
    run()
