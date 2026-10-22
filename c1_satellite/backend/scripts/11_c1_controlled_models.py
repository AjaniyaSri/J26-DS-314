#!/usr/bin/env python3
"""C1 preliminary, reproducible A/B/C/D feature-based ridge ablation.

This is NOT a U-Net/CNN or proof of operational flood prediction.
It uses real exported SAR patch STATISTICS and decoded ASCAT SWI observations,
plus today's stage, and predicts next-day STAGE CHANGE.

Runs four comparisons on exactly the same station-days (SAR multi & latest SWI
present); history gaps remain missing, are indicated, and are imputed using
TRAIN-only statistics. Never fit any preprocessors or tune on TEST data.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'local_data' / 'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
SAR = ROOT / 'outputs' / 'c1_sar_station_day_features.csv'
SWI = ROOT / 'outputs' / 'c1_swi_v4_scaled_station_day_features.csv'
OUT = ROOT / 'outputs'

SAR_BASE = ['sar_latest_age_hours','sar_patch_valid_fraction',
            'sar_vv_mean_db','sar_vh_mean_db','sar_vv_std_db','sar_vh_std_db',
            'sar_vv_p10_db','sar_vv_p90_db','sar_vh_p10_db','sar_vh_p90_db']
SAR_HISTORY = ['sar_vv_mean_change_db','sar_vh_mean_change_db']
SWI_NOW = [f'{name}_d1' for name in ('swi001','swi005','swi010')]
SWI_HISTORY = [f'{name}_d{k}' for k in range(2,8) for name in ('swi001','swi005','swi010')]
SWI_MISSING_INDICATORS = [f'swi_valid_d{k}' for k in range(2,8)]
STATION = ['station_id']
STAGE = ['stage_at_issue_m']
KEY = ['station_id','forecast_date']
ALPHAS = [0.1,1.0,10.0,100.0]


def read_unique(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required CSV not found: {path}')
    df = pd.read_csv(path, dtype={'station_id':str,'forecast_date':str})
    if df.duplicated(KEY).any():
        raise ValueError(f'Duplicate station-date keys in {path}')
    return df


def prepare():
    labels=read_unique(LABELS)
    sar=read_unique(SAR)
    swi=read_unique(SWI)
    for label in ('target_delta_stage_m','target_stage_next_day_m','stage_at_issue_m','chronological_split','record_id'):
        if label not in labels.columns:
            raise ValueError(f'Label field missing: {label}')
    for label in ('sar_multi_local_available','sar_latest_local_available'):
        if label not in sar.columns:
            raise ValueError(f'SAR feature field missing: {label}')
    for label in ('swi_feature_available','swi_7day_complete'):
        if label not in swi.columns:
            raise ValueError(f'SWI feature field missing: {label}')
    # Shared LABEL columns are the source of truth. Do not trust split or record ID from feature outputs.
    sar=sar.drop(columns=['record_id','chronological_split'],errors='ignore')
    df=labels.merge(sar,on=KEY,how='left',validate='one_to_one',suffixes=('','_sar'))
    df=df.merge(swi,on=KEY,how='left',validate='one_to_one',suffixes=('','_swi'))
    for col in ('sar_multi_local_available','swi_feature_available','swi_7day_complete'):
        df[col]=pd.to_numeric(df[col],errors='coerce').fillna(0)
    needed=STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+SWI_HISTORY+SWI_MISSING_INDICATORS
    missing=[c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Feature columns missing: {missing}')
    for col in needed+['target_delta_stage_m','target_stage_next_day_m']:
        df[col]=pd.to_numeric(df[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
    # Fair common eligible cohort: multi-orbit actual SAR, valid latest SWI, and verified current/target readings.
    common=df[(df['sar_multi_local_available']==1) & (df['swi_feature_available']==1) &
              df[STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+['target_delta_stage_m','target_stage_next_day_m']].notna().all(axis=1)].copy()
    common=common.sort_values(['forecast_date','station_id']).reset_index(drop=True)
    return df,common


def metric(y,p):
    return {'MAE_m':float(mean_absolute_error(y,p)),
            'RMSE_m':float(np.sqrt(mean_squared_error(y,p)))}


def pipeline(cols, alpha):
    numerical=Pipeline([
        ('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',StandardScaler())
    ])
    xforms=ColumnTransformer([
        ('numeric',numerical,cols),
        ('station',OneHotEncoder(handle_unknown='ignore'),STATION)
    ])
    return Pipeline([('features',xforms),('regression',Ridge(alpha=alpha))])


def run(args):
    allrows, common=prepare()
    if args.require_complete_history:
        common=common[common['swi_7day_complete']==1].copy()
    split_counts={s:int((common['chronological_split']==s).sum()) for s in ('TRAIN','VALIDATION','TEST')}
    print('=== C1: PRELIMINARY FOUR EXPERIMENT ABLATION ===')
    print('DMC labelled pairs:',len(allrows))
    print('Real SAR multi + valid latest ASCAT SWI common cohort:',len(common))
    print('Split counts:',split_counts)
    print('Seven-day SWI fully complete in common cohort:',int((common['swi_7day_complete']==1).sum()),'/',len(common))
    if any(split_counts[s]<k for s,k in [('TRAIN',args.min_train),('VALIDATION',args.min_eval),('TEST',args.min_eval)]):
        print('STOP: Insufficient common eligible rows for valid A/B/C/D comparisons.')
        print('Do not invent results; inspect c1_swi_v4_scaled_coverage.json and SAR coverage.')
        return 2
    if common['station_id'].nunique()<2:
        print('STOP: Only one station in common cohort; results not suitable for group comparisons.')
        return 2
    split={name:common[common['chronological_split']==name].copy() for name in ('TRAIN','VALIDATION','TEST')}
    Xtr,Xva,Xte=(split[s] for s in ('TRAIN','VALIDATION','TEST'))
    feats={
      'STAGE_ONLY':STAGE,
      'A':STAGE+SAR_BASE+SWI_NOW,
      'B':STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW,
      'C':STAGE+SAR_BASE+SWI_NOW+SWI_HISTORY+SWI_MISSING_INDICATORS,
      'D':STAGE+SAR_BASE+SAR_HISTORY+SWI_NOW+SWI_HISTORY+SWI_MISSING_INDICATORS
    }
    target='target_delta_stage_m'
    ytr,yva,yte=[x[target].to_numpy(dtype=float) for x in (Xtr,Xva,Xte)]
    result=[]
    prediction=[]
    # Persistence = predict no change; calculate on SAME eligible common cohort.
    for setname,x,y in [('TRAIN',Xtr,ytr),('VALIDATION',Xva,yva),('TEST',Xte,yte)]:
        for record,pred,actual in zip(x.to_dict('records'),x['stage_at_issue_m'].to_numpy(),x['target_stage_next_day_m'].to_numpy()):
            prediction.append({'record_id':record['record_id'],'station_id':record['station_id'],
               'forecast_date':record['forecast_date'],'chronological_split':setname,
               'model_id':'PERSISTENCE','actual_next_day_stage_m':actual,'predicted_next_day_stage_m':float(pred)})
        err=metric(y,np.zeros(len(y)))
        result.append({'model_id':'PERSISTENCE','split':setname,'n':len(x),
                       'alpha_selected_on_validation':'','feature_count':1,**err})
    for model,columns in feats.items():
        best=None
        for alpha in ALPHAS:
            trial=pipeline(columns,alpha)
            trial.fit(Xtr,ytr)
            pred=trial.predict(Xva)
            score=mean_absolute_error(yva,pred)
            if best is None or score<best[0]:
                best=(score,alpha,trial)
        _,alpha,fitted=best
        print(f'Model {model}: selected Ridge alpha={alpha} using VALIDATION only')
        for setname,x,y in [('TRAIN',Xtr,ytr),('VALIDATION',Xva,yva),('TEST',Xte,yte)]:
            changes=fitted.predict(x)
            levels=x['stage_at_issue_m'].to_numpy()+changes
            true=x['target_stage_next_day_m'].to_numpy()
            score=metric(true,levels)
            result.append({'model_id':model,'split':setname,'n':len(x),
                           'alpha_selected_on_validation':alpha,'feature_count':len(columns),**score})
            for r,p,t in zip(x.to_dict('records'),levels,true):
                prediction.append({'record_id':r['record_id'],'station_id':r['station_id'],
                    'forecast_date':r['forecast_date'],'chronological_split':setname,
                    'model_id':model,'actual_next_day_stage_m':float(t),
                    'predicted_next_day_stage_m':float(p)})
    OUT.mkdir(exist_ok=True,parents=True)
    metric_path=OUT/'c1_ablation_metrics.csv'
    prediction_path=OUT/'c1_ablation_predictions.csv'
    summary_path=OUT/'c1_ablation_summary.json'
    pd.DataFrame(result).to_csv(metric_path,index=False)
    pd.DataFrame(prediction).to_csv(prediction_path,index=False)
    note={
       'method':'Real SAR summary-feature and real quality-filtered ASCAT SWI Ridge regression proof-of-concept; NOT a U-Net'
       ,'baseline_note':'STAGE_ONLY is a separately trained current-stage plus station-ID reference; PERSISTENCE always predicts zero change',
       'target':'delta = next-day stage - stage at issue; reconstruct level using current stage',
       'common_cohort_n':len(common),'split_counts':split_counts,
       'all_7days_valid_n':int((common['swi_7day_complete']==1).sum()),
       'history_gaps_policy':'TRAIN-only imputation and explicit SWI day masks' if not args.require_complete_history else 'complete seven-day histories only',
       'evaluation':'chronological split, same eligible rows for all 4 models and persistence; alpha selected using VALIDATION only',
       'cautions':['Retrospective hindcast; forecast-time availability lag is not independently proven.',
                   'Station-days sharing Sentinel-1 scenes and ASCAT grid cells are correlated, not independent flood events.',
                   'For full architecture claim, U-Net spatial encoder and temporal encoder must be trained separately.',
                   'The current TEST period contains no independently verified major floods.',
                   'Generalization to other flood events not established.']
    }
    summary_path.write_text(json.dumps(note,indent=2),encoding='utf-8')
    print('\nTEST metrics (same cohort, smaller is better):')
    for row in result:
        if row['split']=='TEST':
            print(f"  {row['model_id']:12s} n={row['n']:4d} MAE={row['MAE_m']:.4f}m RMSE={row['RMSE_m']:.4f}m")
    print('Saved metrics:',metric_path)
    print('Saved predictions:',prediction_path)
    print('Saved limitations:',summary_path)
    print('CAUTION: feature-based preliminary prototype, not a finished U-Net or validated flood warning system.')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-complete-history',action='store_true',help='Use only complete 7-day SWI histories; may leave too few samples')
    p.add_argument('--min-train',type=int,default=30)
    p.add_argument('--min-eval',type=int,default=5)
    a=p.parse_args()
    try:
        raise SystemExit(run(a))
    except (ValueError,FileNotFoundError,KeyError) as e:
        p.error(str(e))

if __name__=='__main__':
    main()
