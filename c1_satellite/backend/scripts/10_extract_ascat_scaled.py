#!/usr/bin/env python3
"""Extract real ASCAT V4 station-grid SWI observations and leak-free 7-day features.

Produces a station/date observation table and an independent C1 feature table
joined only by station_id + forecast_date. Does not modify DMC label files.
"""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import numpy as np
import rasterio

ROOT=Path(__file__).resolve().parent.parent
LABELS=ROOT/'local_data'/'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
RAW=ROOT/'local_data'/'ascat_swi_v4'/'raw'
OUT=ROOT/'outputs'
SWIS=('SWI001','SWI005','SWI010')
QFLAGS=('QFLAG001','QFLAG005','QFLAG010')
BANDS=(*SWIS,*QFLAGS,'SSF','dataMask')

def read_labels():
    with LABELS.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError(f'Empty labels at {LABELS}')
    stations={}
    for row in rows:
        stations[row['station_id']] = (float(row['gauge_lon_wgs84_arcgis']),float(row['gauge_lat_wgs84_arcgis']))
    return rows,stations

def make_station_daily(stations, min_qflag):
    """Decode native V4 raw UINT8 packing to percentages using scaling=1/2.

    Critical: only SWI/QFLAG bands get scaled. SSF and dataMask do not.
    The Phase-7 evalscript copied raw sample values into FLOAT32 GeoTIFFs,
    so rasterio does not automatically know the original CLMS scale factor.
    """
    obs = {}
    paths = sorted(RAW.glob('ASCAT_SWI_V4_????-??-??.tif'))
    if not paths:
        raise FileNotFoundError(f'No ASCAT V4 rasters at {RAW}')
    for tif in paths:
        day = tif.stem.rsplit('_', 1)[-1]
        with rasterio.open(tif) as src:
            if src.count != 8 or src.crs is None or src.crs.to_epsg() != 4326:
                raise ValueError(f'Expected 8 bands with EPSG:4326 in {tif}')
            descriptions = tuple(x or '' for x in src.descriptions)
            if any(descriptions) and descriptions != BANDS:
                raise ValueError(f'Unexpected band order in {tif.name}: {descriptions}')
            for sid, (lon, lat) in stations.items():
                record = {'station_id': sid, 'observation_date': day,
                          'latitude': lat, 'longitude': lon,
                          'source_file': tif.name, 'quality_threshold': min_qflag,
                          'scaling_applied_to_swi_qflag': 0.5,
                          'quality_status': 'OUTSIDE_RASTER'}
                for b in BANDS:
                    record[b] = ''
                if src.bounds.left <= lon < src.bounds.right and src.bounds.bottom < lat <= src.bounds.top:
                    sample = next(src.sample([(lon, lat)], indexes=list(range(1, 9)), masked=False))
                    raw = dict(zip(BANDS, map(float, sample)))
                    mask = raw['dataMask']
                    if not math.isfinite(mask) or mask < 0.5:
                        record['quality_status'] = 'SOURCE_NODATA'
                    elif any(not math.isfinite(raw[b]) or raw[b] < 0 or raw[b] > 200 for b in (*SWIS, *QFLAGS)):
                        # 255 or -9999 must never become seemingly valid after scaling.
                        record['quality_status'] = 'INVALID_RAW_RANGE'
                    elif not math.isfinite(raw['SSF']) or not 0 <= raw['SSF'] <= 4:
                        record['quality_status'] = 'INVALID_SURFACE_STATE_CODE'
                    else:
                        scaled = {b: raw[b] * 0.5 for b in (*SWIS, *QFLAGS)}
                        if any(not 0 <= scaled[b] <= 100 for b in scaled):
                            record['quality_status'] = 'INVALID_SCALED_RANGE'
                        else:
                            record.update(scaled)
                            record['SSF'] = raw['SSF']
                            record['dataMask'] = mask
                            record['quality_status'] = ('VALID' if all(scaled[b] >= min_qflag
                                                             for b in QFLAGS) else 'LOW_QFLAG')
                obs[(sid, day)] = record
    return obs, len(paths)

def daystr(day): return day.isoformat()

def features(labels,obs,lag):
    result=[]
    for r in labels:
        forecast=dt.date.fromisoformat(r['forecast_date'])
        latest=forecast-dt.timedelta(days=lag)
        dest={'station_id':r['station_id'],'forecast_date':r['forecast_date'],
              'latest_eligible_swi_date':latest.isoformat(), 'swi_publication_lag_days_assumed':lag,
              'swi_product_version':'CLMS_GLOBAL_DAILY_V4', 'swi_feature_available':0,
              'swi_7day_complete':0,'swi_missing_days':7,'swi_quality_policy':'QFLAG threshold applied per observation'}
        last=obs.get((r['station_id'],daystr(latest)))
        valid_days=0
        for k in range(1,8):
            d=latest-dt.timedelta(days=k-1)
            o=obs.get((r['station_id'],daystr(d)))
            valid=bool(o and o['quality_status']=='VALID')
            dest[f'swi_date_d{k}']=str(d)
            dest[f'swi_valid_d{k}']=int(valid)
            if valid:valid_days+=1
            for b in SWIS:
                dest[f'{b.lower()}_d{k}']=o[b] if valid else ''
        dest['swi_feature_available']=dest['swi_valid_d1']
        dest['swi_7day_complete']=int(valid_days==7)
        dest['swi_missing_days']=7-valid_days
        result.append(dest)
    return result

def write_csv(path,records):
    if not records:raise ValueError('No records to export')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader();writer.writerows(records)

def run(args):
    labels,stations=read_labels()
    obs,number_days=make_station_daily(stations,args.min_qflag)
    feature_rows=features(labels,obs,args.lag_days)
    OUT.mkdir(parents=True,exist_ok=True)
    station_path=OUT/'c1_swi_v4_scaled_station_daily.csv'
    features_path=OUT/'c1_swi_v4_scaled_station_day_features.csv'
    summary_path=OUT/'c1_swi_v4_scaled_coverage.json'
    write_csv(station_path,list(obs.values()))
    write_csv(features_path,feature_rows)
    splits={}
    for s in ('TRAIN','VALIDATION','TEST'):
        rows=[r for lab,r in zip(labels,feature_rows) if lab['chronological_split']==s]
        splits[s]={'labelled_rows':len(rows),'single_swi_ready':sum(int(r['swi_feature_available']) for r in rows),
                   'seven_day_swi_ready':sum(int(r['swi_7day_complete']) for r in rows)}
    statuses={}
    for r in obs.values():statuses[r['quality_status']]=statuses.get(r['quality_status'],0)+1
    summary={'source':'CLMS SWI Global Daily V4','bands':list(BANDS),'source_tiff_dates':number_days,
             'stations':len(stations),'station_day_observations':len(obs),'labelled_rows':len(labels),
             'assumed_lag_days':args.lag_days,'min_qflag_threshold':args.min_qflag,'raw_swi_qflag_scale_factor':0.5,
             'note':'Publication-time availability not verified; retrospective hindcast only. Existing Phase-7 FLOAT32 rasters have raw packed SWI/QFLAG values, scaled x0.5 here.',
             'split_coverage':splits,'source_quality_counts':statuses}
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('=== C1 PHASE 10 – CORRECTLY SCALED ASCAT V4 FEATURES ===')
    print(f'GeoTIFF source dates: {number_days} | gauge gridcells: {len(stations)} | station-day rows: {len(obs)}')
    for s,counts in splits.items(): print(s,counts)
    print('Quality statuses:',statuses)
    print('Saved:',station_path,features_path,summary_path,sep='\n  ')
    print('NOTICE: SWI/QFLAG raw values decoded x0.5. SSF/dataMask unchanged.')
    print('NOTICE: lag and quality threshold are provisional; this is retrospective hindcasting.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lag-days',type=int,default=2,help='Conservative assumed publication delay (days)')
    p.add_argument('--min-qflag',type=float,default=50,help='Minimum QFLAG percent for ALL selected SWIs; provisional QA threshold')
    a=p.parse_args()
    if not 0<=a.min_qflag<=100 or not 0<=a.lag_days<=30:p.error('Check lag and quality threshold')
    try:run(a)
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

if __name__=='__main__':main()
