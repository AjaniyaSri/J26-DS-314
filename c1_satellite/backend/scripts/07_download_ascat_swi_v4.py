#!/usr/bin/env python3
"""Download authentic Copernicus CLMS ASCAT SWI Global Daily V4 patches.

Requires CDSE Sentinel Hub OAuth client credentials, not Earth Engine credentials.
One 0.1-degree Sri Lanka-wide AOI patch is requested per source date.
Only source observations on that date are requested; empty pixels remain missing.
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

COLLECTION = 'byoc-59c08cd0-762b-4b8a-9eaa-0d55d054a732'
TOKEN_URL = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
PROCESS_URL = 'https://sh.dataspace.copernicus.eu/process/v1'
BANDS = ['SWI001', 'SWI005', 'SWI010', 'QFLAG001', 'QFLAG005', 'QFLAG010', 'SSF', 'dataMask']
HERE = Path(__file__).resolve().parent
C1 = HERE.parent
LABELS = C1 / 'local_data' / 'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
RAW = C1 / 'local_data' / 'ascat_swi_v4' / 'raw'
DEFAULT_CREDS = C1 / 'secrets.env'

EVALSCRIPT = """//VERSION=3
function setup() {
    return {
        input: [{bands: ['SWI001','SWI005','SWI010','QFLAG001','QFLAG005','QFLAG010','SSF','dataMask']}],
        output: { id: 'default', bands: 8, sampleType: 'FLOAT32' }
    };
}
function evaluatePixel(s) {
    if (!s.dataMask) return [-9999,-9999,-9999,-9999,-9999,-9999,-9999,0];
    return [s.SWI001,s.SWI005,s.SWI010,s.QFLAG001,s.QFLAG005,s.QFLAG010,s.SSF,s.dataMask];
}
"""

def station_coordinates(csv_path=LABELS):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing {csv_path}. Run C1 Phase 1 first.')
    stations = {}
    with open(csv_path, encoding='utf-8-sig', newline='') as fin:
        for r in csv.DictReader(fin):
            st = r['station_id']
            lon = float(r['gauge_lon_wgs84_arcgis'])
            lat = float(r['gauge_lat_wgs84_arcgis'])
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f'Invalid gauge coordinates {st}: {lon}, {lat}')
            if st in stations and stations[st] != (lon, lat):
                raise ValueError(f'Conflicting coordinates for {st}')
            stations[st] = (lon, lat)
    return stations

def native_aligned_bounds(stations, pad=0.2):
    # Global SWI 0.1-degree grid cell borders in supplied sample are .05 + n*0.1.
    # Wider than point footprint; choose nearest-neighbour to avoid invented detail.
    xs = [xy[0] for xy in stations.values()]
    ys = [xy[1] for xy in stations.values()]
    lo = lambda x: round(math.floor((x - 0.05) * 10) / 10 + 0.05, 5)
    hi = lambda x: round(math.ceil((x - 0.05) * 10) / 10 + 0.05, 5)
    west, south, east, north = lo(min(xs)-pad), lo(min(ys)-pad), hi(max(xs)+pad), hi(max(ys)+pad)
    width, height = int(round((east - west)*10)), int(round((north-south)*10))
    if width <= 0 or height <= 0 or width > 200 or height > 200:
        raise ValueError(f'Unreasonable bounding box {west,south,east,north}')
    assert all(west < x < east and south < y < north for x,y in stations.values())
    return [west, south, east, north], width, height

def dates_inclusive(start, end):
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if b < a: raise ValueError('End date must be on/after start date')
    return [a + dt.timedelta(days=k) for k in range((b-a).days+1)]

def parse_env(path):
    env = {}
    if path.is_file():
        for ln in path.read_text(encoding='utf-8-sig').splitlines():
            line = ln.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=',1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env

def auth_token(creds):
    import requests
    client_id = os.environ.get('COPERNICUS_SH_CLIENT_ID') or creds.get('COPERNICUS_SH_CLIENT_ID')
    secret = os.environ.get('COPERNICUS_SH_CLIENT_SECRET') or creds.get('COPERNICUS_SH_CLIENT_SECRET')
    if not client_id or not secret or 'YOUR_' in client_id or 'YOUR_' in secret:
        raise ValueError('CDSE OAuth credentials missing. Create c1_satellite/secrets.env following README (DO NOT SHARE).')
    resp = requests.post(TOKEN_URL, data={
        'grant_type':'client_credentials','client_id':client_id,'client_secret':secret
    }, timeout=35)
    if not resp.ok:
        raise RuntimeError(f'CDSE OAuth failure {resp.status_code}: {resp.text[:300]}')
    return resp.json()['access_token']

def request_payload(date, bbox, width, height):
    iso = date.isoformat()
    return {
        'input':{
            'bounds':{'bbox':bbox, 'properties':{'crs':'http://www.opengis.net/def/crs/EPSG/0/4326'}},
            'data':[{'type':COLLECTION, 'dataFilter':{
                'timeRange':{'from':f'{iso}T00:00:00Z','to':f'{iso}T23:59:59Z'}
            }, 'processing':{'upsampling':'NEAREST','downsampling':'NEAREST'}}]
        },
        'output':{'width':width,'height':height,'responses':[{'identifier':'default','format':{'type':'image/tiff'}}]},
        'evalscript':EVALSCRIPT,
    }

def inspect_tif(path, bbox, width, height):
    import rasterio
    import numpy as np
    with rasterio.open(path) as src:
        if (src.count,src.width,src.height) != (8,width,height):
            raise ValueError(f'Unexpected raster size/bands in {path}: {src.count}/{src.width}/{src.height}')
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'Expected EPSG:4326: {src.crs}')
        if any(abs(a-b)>0.02 for a,b in zip(src.bounds,(bbox[0],bbox[1],bbox[2],bbox[3]))):
            raise ValueError(f'Incorrect coverage: {src.bounds}, expected {bbox}')
        mask = src.read(8)
        valid = np.isfinite(mask) & (mask > 0.5)
        count = int(valid.sum())
        return count, float(count/(src.width*src.height))

def run(args):
    import rasterio
    import requests
    stations = station_coordinates()
    bbox, width, height = native_aligned_bounds(stations)
    days = dates_inclusive(args.start, args.end)
    if args.limit:
        days = days[:args.limit]
    print('=== C1 PHASE 7: ASCAT GLOBAL DAILY V4 ===')
    print(f'Gauges: {len(stations)}; dates requested: {len(days)}; bands: {", ".join(BANDS)}')
    print(f'0.1° AOI: {bbox} | pixels: {width} x {height}')
    print('Source: Copernicus CLMS SWI daily V4 (NOT Sentinel-1, NOT fake SWI)')
    if args.dry_run:
        print('DRY RUN PASS. No authentication or download performed.')
        return
    RAW.mkdir(parents=True,exist_ok=True)
    creds=parse_env(Path(args.secrets))
    token=auth_token(creds)
    session=requests.Session()
    headers={'Authorization':f'Bearer {token}', 'Content-Type':'application/json','Accept':'image/tiff'}
    log=[]
    for i, day in enumerate(days,1):
        dest = RAW / f'ASCAT_SWI_V4_{day.isoformat()}.tif'
        status=''; msg=''; valid_pixels=0; fraction=0
        if dest.exists() and not args.force:
            try:
                valid_pixels, fraction = inspect_tif(dest,bbox,width,height)
                status='EXISTING_VALID' if valid_pixels else 'EXISTING_NO_DATA'
            except Exception as exc:
                raise RuntimeError(f'Existing file {dest} failed checks ({exc}). Move it to a backup first.')
        else:
            if args.force and dest.exists(): dest.unlink()
            payload=request_payload(day,bbox,width,height)
            for attempt in range(1,4):
                try:
                    resp=session.post(PROCESS_URL,json=payload,headers=headers,timeout=100)
                    if resp.status_code==401 and attempt==1:
                        token=auth_token(creds)
                        headers['Authorization']=f'Bearer {token}'
                        continue
                    if resp.status_code in (429,500,502,503,504) and attempt<3:
                        time.sleep(5*attempt)
                        continue
                    if not resp.ok:
                        raise RuntimeError(f'HTTP {resp.status_code}: {resp.text[:400]}')
                    temp=dest.with_suffix('.tmp.tif')
                    temp.write_bytes(resp.content)
                    try:
                        valid_pixels,fraction=inspect_tif(temp,bbox,width,height)
                        with rasterio.open(temp,'r+') as src:
                            src.descriptions = tuple(BANDS)
                            src.update_tags(dataset='CLMS SWI Global Daily V4',observed_date=day.isoformat(),collection=COLLECTION)
                        temp.replace(dest)
                    finally:
                        if temp.exists(): temp.unlink()
                    status='DOWNLOADED_VALID' if valid_pixels else 'DOWNLOADED_NO_DATA'
                    break
                except requests.RequestException as exc:
                    if attempt==3: status='ERROR'; msg=str(exc)
                    else: time.sleep(5*attempt)
                except Exception as exc:
                    status='ERROR';msg=str(exc)
                    break
        print(f'[{i}/{len(days)}] {day}: {status} valid pixels={valid_pixels} {msg[:100]}')
        log.append({'date':day.isoformat(),'status':status,'valid_pixels':valid_pixels,
                    'valid_fraction':fraction,'path':str(dest) if dest.exists() else '', 'error':msg})
    manifest=C1/'local_data'/'ascat_swi_v4'/'ascat_download_manifest.csv'
    with open(manifest,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]) if log else ['date','status','valid_pixels','valid_fraction','path','error'])
        w.writeheader();w.writerows(log)
    print(f'Saved manifest: {manifest}')
    print(f'Days with non-empty pixels: {sum(r["valid_pixels"]>0 for r in log)}/{len(log)}')
    print('Next: python c1_satellite/scripts/08_extract_ascat_swi_v4.py')
    if any(r['status']=='ERROR' for r in log):sys.exit(2)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run',action='store_true',help='Show gauge extent and number of requests only')
    ap.add_argument('--start',default='2025-11-08')
    ap.add_argument('--end',default='2025-12-26')
    ap.add_argument('--limit',type=int,default=0,help='First N source dates (test with 2)')
    ap.add_argument('--secrets',default=str(DEFAULT_CREDS))
    ap.add_argument('--force',action='store_true',help='Redownload and replace only Phase 7 raster dates')
    args=ap.parse_args()
    try:run(args)
    except (FileNotFoundError,ValueError,RuntimeError) as ex:
        print('ERROR:',ex,file=sys.stderr);sys.exit(1)

if __name__=='__main__':main()
