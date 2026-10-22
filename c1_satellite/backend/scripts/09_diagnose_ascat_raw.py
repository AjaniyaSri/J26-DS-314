#!/usr/bin/env python3
"""Non-destructive ASCAT SWI V4 raster/scale diagnostic.

Runs locally within c1_satellite/scripts. Does not modify source TIFFs or outputs.
Shows raw samples, invalid-range source, and candidate 0.5 scale QA counts.
"""
import csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import rasterio

C1 = Path(__file__).resolve().parent.parent
RAW = C1 / 'local_data' / 'ascat_swi_v4' / 'raw'
LABELS = C1 / 'local_data' / 'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
BANDS = ['SWI001', 'SWI005', 'SWI010', 'QFLAG001', 'QFLAG005', 'QFLAG010', 'SSF', 'dataMask']

def main():
    files = sorted(RAW.glob('ASCAT_SWI_V4_????-??-??.tif'))
    if not files:
        raise SystemExit('No raster files found at ' + str(RAW))
    with LABELS.open(encoding='utf-8-sig',newline='') as f:
        d = {r['station_id']:(float(r['gauge_lon_wgs84_arcgis']),float(r['gauge_lat_wgs84_arcgis'])) for r in csv.DictReader(f)}
    if not d:
        raise SystemExit('No stations in labels')
    total = Counter()
    example_count=0
    band_stats=defaultdict(list)
    source_notes=set()
    for path in files:
        with rasterio.open(path) as src:
            if src.count != 8:
                raise SystemExit(f'{path.name}: expected 8 bands; got {src.count}')
            source_notes.add(str(src.descriptions))
            for sid,(lon,lat) in sorted(d.items()):
                if not (src.bounds.left<=lon<src.bounds.right and src.bounds.bottom<lat<=src.bounds.top):
                    total['outside_raster']+=1
                    continue
                v=np.array(next(src.sample([(lon,lat)],indexes=list(range(1,9)),masked=False)),dtype=float)
                if not np.isfinite(v[-1]) or v[-1]<0.5:
                    total['dataMask_zero']+=1
                    continue
                total['dataMask_one']+=1
                if example_count<6:
                    print(f'  {path.stem[-10:]} {sid}: '+', '.join(f'{b}={v[i]:g}' for i,b in enumerate(BANDS)))
                    example_count+=1
                for i,b in enumerate(BANDS):
                    if np.isfinite(v[i]):
                        band_stats[b].append(v[i])
                six=v[:6]
                # As downloaded by current evalscript, six bands must be inspected for raw packing.
                if np.all(np.isfinite(six)) and np.all((six>=0)&(six<=100)):
                    total['raw_six_in_0_100']+=1
                else:
                    total['raw_six_outside_0_100']+=1
                # Candidate only: documented raw packing has scale factor 1/2.
                # 255 and similar unavailable values must not be interpreted as valid after scaling.
                if np.all(np.isfinite(six)) and np.all((six>=0)&(six<=200)):
                    total['candidate_six_0_200']+=1
                    if np.all(six[3:]>=100):
                        total['candidate_scaled_all_qflags_ge_50']+=1
                if np.any(six==255):
                    total['has_255_source_sentinel']+=1
    print('\n=== C1 ASCAT V4 RAW DIAGNOSTIC ===')
    print('Files:',len(files),'Stations:',len(d))
    print('Raster band descriptions:',*source_notes,sep='\n  ')
    print('\nFirst raw station samples:')
    print('(These rows appear above the summary; no values were modified.)')
    for b in BANDS:
        arr=np.asarray(band_stats[b])
        if arr.size:
            print(f'{b:10s} min={arr.min():8.2f}, median={np.median(arr):8.2f}, max={arr.max():8.2f}, raw>100={int((arr>100).sum()):4d}, raw==255={int((arr==255).sum()):4d}')
        else:
            print(f'{b:10s} NO VALUES')
    print('\nCounts:')
    for key in ('outside_raster','dataMask_zero','dataMask_one','raw_six_in_0_100','raw_six_outside_0_100','candidate_six_0_200','candidate_scaled_all_qflags_ge_50','has_255_source_sentinel'):
        print(f'  {key}: {total[key]}')
    print('\nINTERPRETATION: Official V4 SWI/QFLAG source bands use scaling=1/2.')
    print('If the samples contain values 101-200 and candidate_six_0_200 is high, the extractor may be rejecting packed values before applying scaling.')
    print('Do not change TIFF files or assume that all values >100 are valid: check source sentinel values and mask first.')

if __name__=='__main__':
    main()
