#!/usr/bin/env python3
"""Reconstruct calendar-day ASCAT SWI history with *per-band* quality checks.

Does not rewrite original rasters, original DMC labels, or Phase-10 files.
Official CLMS daily V4 QFLAGs are indexed by SWI time parameter; a low
QFLAG001 does not automatically invalidate SWI005 or SWI010.
Publication-time lag and 50%-QFLAG policy remain assumptions, not verified facts.
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
LABELS=ROOT/'local_data'/'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
DAILY=OUT/'c1_swi_v4_scaled_station_daily.csv'
BANDS=('001','005','010')
SPLITS=('TRAIN','VALIDATION','TEST')
KEY=['station_id','forecast_date']


def make_features(labels, daily, *, lag_days=2, min_qflag=50.0):
    required_labels=set(KEY+['chronological_split'])
    required_daily=set(['station_id','observation_date','quality_status','SSF','dataMask']+[f'SWI{x}' for x in BANDS]+[f'QFLAG{x}' for x in BANDS])
    if required_labels-set(labels.columns): raise ValueError(f'Labels missing {sorted(required_labels-set(labels.columns))}')
    if required_daily-set(daily.columns): raise ValueError(f'SWI daily missing {sorted(required_daily-set(daily.columns))}')
    if labels.duplicated(KEY).any(): raise ValueError('Duplicate station/forecast date in labels')
    if daily.duplicated(['station_id','observation_date']).any(): raise ValueError('Duplicate station/observation date in daily table')
    for col in ['SSF','dataMask']+[f'SWI{x}' for x in BANDS]+[f'QFLAG{x}' for x in BANDS]:
        daily[col]=pd.to_numeric(daily[col],errors='coerce')
    obs={(str(r['station_id']),str(r['observation_date'])):r for r in daily.to_dict('records')}
    output=[]
    for label in labels.to_dict('records'):
        forecast=date.fromisoformat(str(label['forecast_date']))
        latest=forecast-timedelta(days=lag_days)
        rec={'station_id':str(label['station_id']),'forecast_date':forecast.isoformat(),
             'latest_eligible_swi_date':latest.isoformat(),'assumed_lag_days':lag_days,
             'quality_threshold':min_qflag}
        for b in BANDS:
            valid_days=0
            for k in range(1,8):
                obs_day=(latest-timedelta(days=k-1)).isoformat()
                r=obs.get((rec['station_id'],obs_day))
                good=bool(r is not None and r['quality_status'] in ('VALID','LOW_QFLAG')
                          and np.isfinite(r['dataMask']) and r['dataMask']>=0.5
                          and np.isfinite(r['SSF']) and 0 <= r['SSF'] <= 4
                          and np.isfinite(r[f'SWI{b}']) and 0 <= r[f'SWI{b}'] <= 100
                          and np.isfinite(r[f'QFLAG{b}']) and min_qflag <= r[f'QFLAG{b}'] <= 100)
                rec[f'swi{b}_d{k}']=float(r[f'SWI{b}']) if good else np.nan
                rec[f'swi{b}_valid_d{k}']=int(good)
                valid_days+=int(good)
            rec[f'swi{b}_valid_days_7']=valid_days
            rec[f'swi{b}_complete_7']=int(valid_days==7)
        rec['swi005010_complete_7']=int(rec['swi005_complete_7']==1 and rec['swi010_complete_7']==1)
        rec['swi_all_three_complete_7']=int(all(rec[f'swi{b}_complete_7']==1 for b in BANDS))
        output.append(rec)
    return pd.DataFrame(output)


def build_summary(labels, feature_rows, min_qflag, lag_days):
    merged=labels[KEY+['chronological_split']].merge(feature_rows,on=KEY,validate='one_to_one')
    grouped={}
    for split in SPLITS:
        x=merged[merged.chronological_split==split]
        grouped[split]={'labelled_n':len(x),
            'complete_7_by_band':{b:int(x[f'swi{b}_complete_7'].sum()) for b in BANDS},
            'complete_7_SWI005_and_SWI010':int(x.swi005010_complete_7.sum()),
            'complete_7_all_three':int(x.swi_all_three_complete_7.sum()),
            'valid_day_counts':{b:{str(k):int((x[f'swi{b}_valid_days_7']==k).sum()) for k in range(8)} for b in BANDS}}
    return {'source':'Already-scaled Phase-10 daily ASCAT V4 CSV','assumed_lag_days':lag_days,
           'provisional_min_per_band_qflag':min_qflag,'policy':'Each SWI is checked against its own QFLAG; no QFLAG001 blanket exclusion of SWI005/010',
           'split_coverage':grouped,'note':'This is a retrospective data audit; independent source publication-time verification still required.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lag-days',type=int,default=2)
    p.add_argument('--min-qflag',type=float,default=50)
    a=p.parse_args()
    if not 0<=a.lag_days<=30 or not 0<=a.min_qflag<=100: p.error('Invalid lag or quality threshold')
    if not LABELS.exists() or not DAILY.exists():p.error('Missing Phase-10 scaled daily data or shared labels. Run Phase 10 first.')
    labels=pd.read_csv(LABELS,dtype={'station_id':str,'forecast_date':str})
    daily=pd.read_csv(DAILY,dtype={'station_id':str,'observation_date':str})
    feats=make_features(labels,daily,lag_days=a.lag_days,min_qflag=a.min_qflag)
    summary=build_summary(labels,feats,a.min_qflag,a.lag_days)
    OUT.mkdir(parents=True,exist_ok=True)
    feat_path=OUT/'c1_swi_bandwise_7day_features.csv'
    summary_path=OUT/'c1_swi_bandwise_7day_coverage.json'
    feats.to_csv(feat_path,index=False)
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('=== C1 PHASE 13: BAND-SPECIFIC SWI VALIDITY ===')
    for split,r in summary['split_coverage'].items():
        print(split, 'n=',r['labelled_n'], 'complete7 SWI001=',r['complete_7_by_band']['001'],
              'SWI005=',r['complete_7_by_band']['005'], 'SWI010=',r['complete_7_by_band']['010'],
              'SWI005+010=',r['complete_7_SWI005_and_SWI010'], 'all3=',r['complete_7_all_three'])
    print('Saved:',feat_path,summary_path,sep='\n  ')
    print('NOTICE: Band-specific quality is an exploratory sensitivity analysis. Original three-band experiment is unchanged.')

if __name__=='__main__':main()
