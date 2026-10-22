#!/usr/bin/env python3
"""Non-destructive leakage/timing/cohort audit of existing C1 project.

No predictions are fitted, no model settings changed, no TEST score used.
Run from J26-DS-314 repository root:
  python c1_satellite/scripts/16_audit_ridge_and_data.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / 'src'))
from improvements.data import load_project, STAGE_HIST, SWI_HIST


def main():
    df, audit = load_project(C1)
    expected = {'TRAIN': 319, 'VALIDATION': 186, 'TEST': 261}
    audit['previous_reported_cohort_counts'] = expected
    audit['matches_previous_reported_cohort'] = audit['eligible_by_split'] == expected
    audit['stage_history_missing_by_split'] = {
        k: {c: int(x[c].isna().sum()) for c in STAGE_HIST}
        for k, x in ((k, df[df.chronological_split == k]) for k in expected)
    }
    audit['two_band_history_missing_in_eligible_rows'] = int(df[SWI_HIST].isna().sum().sum())
    audit['station_count_eligible'] = int(df.station_id.nunique())
    audit['stations_with_train_rows'] = int(df[df.chronological_split == 'TRAIN'].station_id.nunique())
    audit['station_train_n_min_median_max'] = {
        'min': int(df[df.chronological_split == 'TRAIN'].groupby('station_id').size().min()),
        'median': float(df[df.chronological_split == 'TRAIN'].groupby('station_id').size().median()),
        'max': int(df[df.chronological_split == 'TRAIN'].groupby('station_id').size().max())}
    audit['warning'] = ('Published product arrival times are unverified; 48h SAR / 2-day SWI '
                        'lags remain provisional. Existing TEST already inspected; not pristine.')
    output = C1 / 'outputs' / 'improvement_audit'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'audit_summary.json'
    path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('=== C1 PHASE 16: DATA / LEAKAGE / STATION AUDIT ===')
    print('Source labels:', audit['source_labels_n'], 'eligible:', audit['eligible_n'])
    print('Same-cohort split counts:', audit['eligible_by_split'])
    print('Matches Phase-14:', audit['matches_previous_reported_cohort'])
    print('7-day SWI005+010 missing in eligible:', audit['two_band_history_missing_in_eligible_rows'])
    print('SAR forecast-time violations:', audit['cutoff_audit']['observed_future_cutoff_violations'])
    print('Saved:', path)
    if not audit['matches_previous_reported_cohort']:
        print('WARNING: Different sample coverage from Phase 14; investigate before comparing scores.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise SystemExit(f'AUDIT STOP: {e}')
