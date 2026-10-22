#!/usr/bin/env python3
"""C1 non-destructive station/temporal features + validation-only model study.

Compares Ridge, ExtraTrees, HistGradientBoosting and optionally XGBoost on the
same quality-controlled two-band SAR+SWI eligible rows as Phase 14. Preserves
fixed chronological TRAIN/VALIDATION/TEST splits. Does not READ TEST labels
for model fitting/scoring, nor produce any TEST metrics. Saves fitted TRAIN-only
pipelines with joblib, reports VALIDATION scores; no guarantee of improvement.

Run after Phase 16:
  python c1_satellite/scripts/17_validation_only_model_improvements.py
Optional: python -m pip install xgboost
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / 'src'))
from improvements.data import (load_project, STAGE_CORE, STAGE_HIST,
                               SAR_BASE, SAR_HIST, SAR_NOW_ENG, SAR_HIST_ENG,
                               SWI_NOW, SWI_HIST, SWI_NOW_ENG, SWI_HIST_ENG)


def variants():
    """Same stage-history features in all enhanced arms; satellite ablations preserved."""
    common = STAGE_CORE + STAGE_HIST
    sar = SAR_BASE + SAR_NOW_ENG
    sw_now = SWI_NOW + SWI_NOW_ENG
    return {
        'STAGE_ONLY_BASE': STAGE_CORE,
        'STAGE_TEMPORAL': common,
        'A_ENH_2B': common + sar + sw_now,
        'B_ENH_2B': common + sar + SAR_HIST + SAR_HIST_ENG + sw_now,
        'C_ENH_2B': common + sar + sw_now + SWI_HIST + SWI_HIST_ENG,
        'D_ENH_2B': common + sar + SAR_HIST + SAR_HIST_ENG + sw_now + SWI_HIST + SWI_HIST_ENG,
    }


def candidate_models():
    defs = [
        ('RIDGE_10', Ridge(alpha=10)),
        ('RIDGE_100', Ridge(alpha=100)),
        ('RIDGE_1000', Ridge(alpha=1000)),
        ('EXTRA_TREES_SHALLOW', ExtraTreesRegressor(n_estimators=120, max_depth=3,
            min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=2)),
        ('EXTRA_TREES_MEDIUM', ExtraTreesRegressor(n_estimators=120, max_depth=5,
            min_samples_leaf=10, max_features=0.8, random_state=42, n_jobs=2)),
        ('HIST_GB_SHALLOW', HistGradientBoostingRegressor(max_iter=90, learning_rate=0.035,
            max_leaf_nodes=5, l2_regularization=10, min_samples_leaf=18, random_state=42)),
    ]
    try:
        from xgboost import XGBRegressor
        defs.extend([
            ('XGBOOST_DEPTH2', XGBRegressor(n_estimators=140, max_depth=2, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.85, reg_lambda=15, min_child_weight=8,
                objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=2)),
            ('XGBOOST_DEPTH3', XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.85, reg_lambda=20, min_child_weight=10,
                objective='reg:squarederror', tree_method='hist', random_state=42, n_jobs=2)),
        ])
    except ImportError:
        print('Optional XGBoost not installed; Ridge + ExtraTrees + HistGradientBoosting will run.')
    return defs


def make_pipeline(cols, estimator):
    # Fit imputation, standardization, one-hot station encoder on TRAIN only.
    prep = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scaler', StandardScaler())]), cols),
        ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['station_id'])
    ], sparse_threshold=0)
    return Pipeline([('preprocessing', prep), ('estimator', estimator)])


def metric(actual, predicted):
    return dict(MAE_m=float(mean_absolute_error(actual, predicted)),
                RMSE_m=float(np.sqrt(mean_squared_error(actual, predicted))))


def run(root=C1, min_train=60, min_validation=20, save_models=True, model_defs=None):
    data, audit = load_project(root)
    train = data[data.chronological_split == 'TRAIN'].copy()
    val = data[data.chronological_split == 'VALIDATION'].copy()
    # TEST is in data only for join/integrity; explicitly never selected or scored.
    if len(train) < min_train or len(val) < min_validation:
        raise ValueError(f'Insufficient common cohort: TRAIN {len(train)}, VALIDATION {len(val)}')
    result_dir = root / 'outputs' / 'improvement_validation'
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir = root / 'saved_models' / 'improvement_experiments'
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
    y_train = train.target_delta_stage_m.to_numpy()
    y_val_stage = val.target_stage_next_day_m.to_numpy()
    persistence = metric(y_val_stage, val.stage_at_issue_m.to_numpy())
    records = [dict(experiment='PERSISTENCE', algorithm='NO_TRAINING',
                    n_train=len(train), n_validation=len(val), n_features=1, **persistence)]
    chosen = {}
    candidates = candidate_models() if model_defs is None else model_defs
    for experiment, columns in variants().items():
        if any(col in ('target_delta_stage_m', 'target_stage_next_day_m', 'chronological_split', 'record_id') for col in columns):
            raise ValueError(f'Future/target/split field in features for {experiment}')
        missing = sorted(set(columns) - set(data))
        if missing:
            raise ValueError(f'{experiment}: missing features {missing}')
        best = None
        for algorithm, estimator in candidates:
            pipe = make_pipeline(columns, estimator)
            pipe.fit(train, y_train)
            predicted_delta = pipe.predict(val)
            predicted_stage = val.stage_at_issue_m.to_numpy() + predicted_delta
            row = dict(experiment=experiment, algorithm=algorithm, n_train=len(train),
                       n_validation=len(val), n_features=len(columns),
                       **metric(y_val_stage, predicted_stage))
            records.append(row)
            if best is None or (row['MAE_m'], row['RMSE_m']) < (best[0]['MAE_m'], best[0]['RMSE_m']):
                best = (row, pipe)
        chosen[experiment] = {**best[0], 'feature_columns': list(columns),
                              'saved_pipeline': f'{experiment}.joblib' if save_models else None}
        if save_models:
            dump(best[1], model_dir / f'{experiment}.joblib')
    results = pd.DataFrame(records).sort_values(['experiment', 'MAE_m'])
    results.to_csv(result_dir / 'candidate_validation_metrics.csv', index=False)
    selected = [dict(experiment='PERSISTENCE', algorithm='NO_TRAINING', n_train=len(train),
                     n_validation=len(val), n_features=1, **persistence)]
    selected.extend({k: v for k, v in r.items() if k in ('experiment', 'algorithm',
        'n_train', 'n_validation', 'n_features', 'MAE_m', 'RMSE_m')} for r in chosen.values())
    pd.DataFrame(selected).to_csv(result_dir / 'selected_validation_metrics.csv', index=False)
    metadata = {
        'status': 'EXPLORATORY_VALIDATION_ONLY',
        'n_train': len(train), 'n_validation': len(val),
        'old_test_used_for_model_selection': False,
        'prior_test_already_seen_in_previous_project_iterations': True,
        'prediction_target': 'delta stage, reconstructed with stage_at_issue_m',
        'cohort': audit, 'selected': chosen,
        'models_fit_on': 'TRAIN_ONLY', 'selection_metric': 'VALIDATION_MAE_m',
        'no_claim_of_improvement_until_run': True,
        'model_artifact_warning': 'joblib files are executable pickle; load only your own trusted locally produced files',
        'independent_new_event_needed': True}
    (result_dir / 'selection_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('=== C1 PHASE 17: VALIDATION-ONLY ACCURACY IMPROVEMENTS ===')
    print(f'Common samples TRAIN={len(train)}, VALIDATION={len(val)}; TEST untouched for decisions')
    print(pd.DataFrame(selected).to_string(index=False, formatters={
        'MAE_m': '{:.4f}'.format, 'RMSE_m': '{:.4f}'.format}))
    print('Saved:', result_dir)
    return pd.DataFrame(selected)


if __name__ == '__main__':
    try:
        run()
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f'IMPROVEMENT STOP: {e}')
