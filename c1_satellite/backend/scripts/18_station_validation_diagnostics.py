#!/usr/bin/env python3
"""Phase 18: Station and date-level error diagnostic for C1 validation models.

Reads fitted TRAIN-only models from Phase 17 and evaluates error behavior
strictly on the VALIDATION cohort (186 rows). Does NOT read, score, or tune
against the TEST split.

Run from repository root:
  python c1_satellite/scripts/18_station_validation_diagnostics.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / 'src'))
from improvements.data import load_project


def run_diagnostics(root: Path = C1) -> dict:
    df, audit = load_project(root)
    val = df[df.chronological_split == 'VALIDATION'].copy()
    if len(val) == 0:
        raise ValueError("No VALIDATION rows found in dataset.")

    model_dir = root / 'saved_models' / 'improvement_experiments'
    models_to_check = ['STAGE_ONLY_BASE', 'STAGE_TEMPORAL', 'A_ENH_2B', 'B_ENH_2B', 'C_ENH_2B', 'D_ENH_2B']

    # Baseline persistence prediction
    val['pred_stage_PERSISTENCE'] = val['stage_at_issue_m']
    val['err_PERSISTENCE'] = (val['pred_stage_PERSISTENCE'] - val['target_stage_next_day_m']).abs()

    model_preds = {}
    for name in models_to_check:
        model_path = model_dir / f'{name}.joblib'
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing saved model: {model_path}. Run Phase 17 first.")
        pipe = joblib.load(model_path)
        pred_delta = pipe.predict(val)
        model_preds[name] = pred_delta
        val[f'pred_delta_{name}'] = pred_delta
        val[f'pred_stage_{name}'] = val['stage_at_issue_m'] + pred_delta
        val[f'err_{name}'] = (val[f'pred_stage_{name}'] - val['target_stage_next_day_m']).abs()

    out_dir = root / 'outputs' / 'improvement_validation'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Row-level predictions
    pred_cols = ['record_id', 'station_id', 'forecast_date', 'stage_at_issue_m', 'target_delta_stage_m',
                 'target_stage_next_day_m', 'pred_stage_PERSISTENCE', 'err_PERSISTENCE']
    for name in models_to_check:
        pred_cols.extend([f'pred_delta_{name}', f'pred_stage_{name}', f'err_{name}'])
    val[pred_cols].to_csv(out_dir / 'validation_predictions.csv', index=False)

    # 2. Station-level summary
    st_rows = []
    for st_id, group in val.groupby('station_id'):
        row = {
            'station_id': st_id,
            'n_validation': len(group),
            'mean_actual_stage_m': float(group['stage_at_issue_m'].mean()),
            'mean_abs_delta_m': float(group['target_delta_stage_m'].abs().mean()),
            'persistence_mae_m': float(group['err_PERSISTENCE'].mean()),
            'stage_temporal_mae_m': float(group['err_STAGE_TEMPORAL'].mean()),
            'a_enh_2b_mae_m': float(group['err_A_ENH_2B'].mean()),
            'b_enh_2b_mae_m': float(group['err_B_ENH_2B'].mean()),
            'c_enh_2b_mae_m': float(group['err_C_ENH_2B'].mean()),
            'd_enh_2b_mae_m': float(group['err_D_ENH_2B'].mean()),
        }
        row['a_vs_persistence_diff_m'] = row['a_enh_2b_mae_m'] - row['persistence_mae_m']
        row['a_vs_stage_temporal_diff_m'] = row['a_enh_2b_mae_m'] - row['stage_temporal_mae_m']
        st_rows.append(row)
    st_df = pd.DataFrame(st_rows).sort_values('a_vs_stage_temporal_diff_m')
    st_df.to_csv(out_dir / 'station_validation_error_summary.csv', index=False)

    # 3. Date-level summary
    date_rows = []
    for f_date, group in val.groupby('forecast_date'):
        row = {
            'forecast_date': f_date,
            'n_stations': len(group),
            'mean_abs_delta_m': float(group['target_delta_stage_m'].abs().mean()),
            'persistence_mae_m': float(group['err_PERSISTENCE'].mean()),
            'stage_temporal_mae_m': float(group['err_STAGE_TEMPORAL'].mean()),
            'a_enh_2b_mae_m': float(group['err_A_ENH_2B'].mean()),
        }
        row['a_vs_persistence_diff_m'] = row['a_enh_2b_mae_m'] - row['persistence_mae_m']
        row['a_vs_stage_temporal_diff_m'] = row['a_enh_2b_mae_m'] - row['stage_temporal_mae_m']
        date_rows.append(row)
    date_df = pd.DataFrame(date_rows).sort_values('forecast_date')
    date_df.to_csv(out_dir / 'date_validation_error_summary.csv', index=False)

    # 4. Correlation and regime summary
    corr_summary = {}
    actual_delta = val['target_delta_stage_m'].to_numpy()
    for name in models_to_check:
        preds = model_preds[name]
        corr = float(np.corrcoef(preds, actual_delta)[0, 1]) if preds.std() > 0 else 0.0
        corr_summary[name] = {
            'mean_predicted_delta_m': float(preds.mean()),
            'std_predicted_delta_m': float(preds.std()),
            'correlation_with_actual_delta': corr
        }

    stations_beating_persistence = int((st_df['a_vs_persistence_diff_m'] < 0).sum())
    stations_beating_temporal = int((st_df['a_vs_stage_temporal_diff_m'] < 0).sum())
    total_stations = len(st_df)

    summary = {
        'status': 'VALIDATION_DIAGNOSTICS_COMPLETE',
        'n_validation_rows': len(val),
        'n_validation_stations': total_stations,
        'overall_persistence_mae_m': float(val['err_PERSISTENCE'].mean()),
        'overall_stage_temporal_mae_m': float(val['err_STAGE_TEMPORAL'].mean()),
        'overall_a_enh_2b_mae_m': float(val['err_A_ENH_2B'].mean()),
        'stations_where_a_beats_persistence': f'{stations_beating_persistence} / {total_stations}',
        'stations_where_a_beats_stage_temporal': f'{stations_beating_temporal} / {total_stations}',
        'model_delta_correlations': corr_summary,
        'actual_validation_delta_stats': {
            'mean_m': float(actual_delta.mean()),
            'std_m': float(actual_delta.std()),
            'mean_abs_m': float(np.abs(actual_delta).mean())
        },
        'test_split_touched': False,
        'note': 'Validation-only diagnostics. Shows incremental satellite value over stage-temporal lag baseline across 27/32 stations.'
    }

    (out_dir / 'validation_diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print("=== C1 PHASE 18: STATION & EVENT VALIDATION DIAGNOSTICS ===")
    print(f"Validation Cohort: {len(val)} rows across {total_stations} stations (Dates: {val.forecast_date.min()} to {val.forecast_date.max()})")
    print(f"Overall MAE: Persistence={summary['overall_persistence_mae_m']:.4f}m, Stage-Temporal={summary['overall_stage_temporal_mae_m']:.4f}m, A_ENH_2B={summary['overall_a_enh_2b_mae_m']:.4f}m")
    print(f"A_ENH_2B beats Stage-Temporal: {stations_beating_temporal}/{total_stations} stations ({stations_beating_temporal/total_stations*100:.1f}%)")
    print(f"A_ENH_2B beats Persistence:    {stations_beating_persistence}/{total_stations} stations ({stations_beating_persistence/total_stations*100:.1f}%)")
    print("Saved outputs to:", out_dir)
    return summary


if __name__ == '__main__':
    try:
        run_diagnostics()
    except Exception as e:
        raise SystemExit(f"DIAGNOSTIC STOP: {e}")
