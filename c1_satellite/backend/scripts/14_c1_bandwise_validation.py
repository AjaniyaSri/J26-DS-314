#!/usr/bin/env python3
"""Research sensitivity test A/B/C/D using SWI005 and SWI010 complete histories.

All four arms use exactly the same genuine, 7-day quality-passing SWI005/010
observations, the same SAR multi-eligible station-days, and fixed chrono split.
This is a TWO-BAND SENSITIVITY ANALYSIS, not the originally planned three-band
experiment and not a CNN/U-Net. Uses held-out TEST only with --evaluate-test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
KEY=['station_id','forecast_date']
SAR_BASE=['sar_latest_age_hours','sar_patch_valid_fraction','sar_vv_mean_db','sar_vh_mean_db',
    'sar_vv_std_db','sar_vh_std_db','sar_vv_p10_db','sar_vv_p90_db','sar_vh_p10_db','sar_vh_p90_db']
SAR_HISTORY=['sar_vv_mean_change_db','sar_vh_mean_change_db']
SWI_NOW=['swi005_d1','swi010_d1']
SWI_HISTORY=[f'swi{band}_d{k}' for k in range(2,8) for band in ('005','010')]
STAGE=['stage_at_issue_m']
ALPHAS=(0.1,1.0,10.0,100.0,1000.0)


def read_unique(path):
    if not path.exists():raise FileNotFoundError(f'Required file missing: {path}')
    df=pd.read_csv(path,dtype={'station_id':str,'forecast_date':str})
    if df.duplicated(KEY).any():raise ValueError(f'Duplicate keys: {path}')
    return df


def prepare():
    labels=read_unique(ROOT/'local_data'/'FloodRisk_DMC_StationDay_LABELLED_1429.csv')
    sar=read_unique(OUT/'c1_sar_station_day_features.csv')
    swi=read_unique(OUT/'c1_swi_bandwise_7day_features.csv')
    # Split, target, current stage and record ID ONLY from the shared label contract.
    sar=sar.drop(columns=['chronological_split','record_id'],errors='ignore')
    df=labels.merge(sar,on=KEY,how='left',validate='one_to_one',suffixes=('','_sar'))
    df=df.merge(swi,on=KEY,how='left',validate='one_to_one',suffixes=('','_swi'))
    required=(['sar_multi_local_available','swi005010_complete_7','chronological_split','record_id','target_delta_stage_m','target_stage_next_day_m']
            +STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+SWI_HISTORY)
    missing=sorted(set(required)-set(df.columns))
    if missing:raise ValueError(f'Missing columns: {missing}')
    cols=[c for c in required if c not in ['chronological_split','record_id']]
    for c in cols:df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Only rows where every real SWI005/010 day and SAR inputs exist. No filling 7-day gaps.
    eligible=(df.sar_multi_local_available.eq(1) & df.swi005010_complete_7.eq(1)
       & df[STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+SWI_HISTORY+['target_delta_stage_m','target_stage_next_day_m']].notna().all(axis=1))
    return df[eligible].sort_values(KEY[::-1]).reset_index(drop=True)


def pipeline(columns,alpha):
    numerical=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    return Pipeline([('column',ColumnTransformer([
        ('num',numerical,columns),('station',OneHotEncoder(handle_unknown='ignore'),['station_id'])])),
        ('ridge',Ridge(alpha=alpha))])


def scores(true, pred):
    return {'MAE_m':float(mean_absolute_error(true,pred)),
            'RMSE_m':float(np.sqrt(mean_squared_error(true,pred)))}


def run(evaluate_test=False,min_train=30,min_validation=8,min_test=8):
    df=prepare()
    splits={s:df[df.chronological_split==s].copy() for s in ('TRAIN','VALIDATION','TEST')}
    counts={k:len(v) for k,v in splits.items()}
    print('=== C1 PHASE 14: TRUE 7-DAY SWI005/010 SENSITIVITY (NOT THREE-BAND) ===')
    print('Common eligible counts:',counts,'Total:',len(df))
    if counts['TRAIN']<min_train or counts['VALIDATION']<min_validation or (evaluate_test and counts['TEST']<min_test):
        print('STOP: insufficient same-cohort complete seven-day samples; no model results produced.')
        return 2
    a,b=splits['TRAIN'],splits['VALIDATION']
    arms={'STAGE_ONLY':STAGE,
      'A_2band':STAGE+SAR_BASE+SWI_NOW,
      'B_2band':STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW,
      'C_2band':STAGE+SAR_BASE+SWI_NOW+SWI_HISTORY,
      'D_2band':STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+SWI_HISTORY}
    val_results=[]
    test_results=[]
    ytr=a['target_delta_stage_m'].to_numpy()
    yval=b['target_delta_stage_m'].to_numpy()
    persist=lambda x:scores(x['target_stage_next_day_m'],x['stage_at_issue_m'])
    for split,x in [('VALIDATION',b)]+([('TEST',splits['TEST'])] if evaluate_test else []):
        row={'model_id':'PERSISTENCE','split':split,'n':len(x),'selected_alpha':''}
        row.update(persist(x))
        (val_results if split=='VALIDATION' else test_results).append(row)
    choices={}
    for name,features in arms.items():
        best=None
        for alpha in ALPHAS:
            candidate=pipeline(features,alpha)
            candidate.fit(a,ytr)
            err=mean_absolute_error(yval,candidate.predict(b))
            if best is None or err<best[0]:best=(err,alpha,candidate)
        _,alpha,fitted=best
        choices[name]=alpha
        print(f'{name}: best VALIDATION alpha={alpha}')
        for split,x in [('VALIDATION',b)]+([('TEST',splits['TEST'])] if evaluate_test else []):
            levels=x.stage_at_issue_m.to_numpy()+fitted.predict(x)
            row={'model_id':name,'split':split,'n':len(x),'selected_alpha':alpha,**scores(x.target_stage_next_day_m,levels)}
            (val_results if split=='VALIDATION' else test_results).append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    val_path=OUT/'c1_bandwise_validation_metrics.csv'
    pd.DataFrame(val_results).to_csv(val_path,index=False)
    print('VALIDATION (same cohort):')
    print(pd.DataFrame(val_results).to_string(index=False,formatters={'MAE_m':'{:.4f}'.format,'RMSE_m':'{:.4f}'.format}))
    print('Saved:',val_path)
    if evaluate_test:
        tpath=OUT/'c1_bandwise_exploratory_test_metrics.csv'
        pd.DataFrame(test_results).to_csv(tpath,index=False)
        print('EXPLORATORY TEST (already examined prior test data; no claim of untouched final holdout):')
        print(pd.DataFrame(test_results).to_string(index=False,formatters={'MAE_m':'{:.4f}'.format,'RMSE_m':'{:.4f}'.format}))
        print('Saved:',tpath)
    note={'method':'Feature-based Ridge sensitivity analysis: only quality-passing complete 7-day SWI005+SWI010; SWI001 EXCLUDED',
          'not_original_three_band_design':True,'counts':counts,'selected_alpha_validation_only':choices,
          'test_evaluated':bool(evaluate_test),'test_status':'EXPLORATORY: old test has been viewed in prior experiments',
          'limitations':['Bandwise QFLAG>=50 is a PROVISIONAL policy, not a product-mandated cutoff.',
                         'Full 3-band 7-day requirement not met under existing policy.',
                         'Current radar pixels summarized statistically; no trained U-Net encoder.',
                         'Multiple samples share satellite acquisitions or SWI cells.',
                         'Publication-time verification incomplete; retrospective analysis only.']}
    (OUT/'c1_bandwise_model_summary.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate-test',action='store_true',help='Explicit exploratory TEST evaluation; withheld by default')
    a=p.parse_args()
    try:raise SystemExit(run(evaluate_test=a.evaluate_test))
    except (ValueError,FileNotFoundError,KeyError) as exc:p.error(str(exc))

if __name__=='__main__':main()
