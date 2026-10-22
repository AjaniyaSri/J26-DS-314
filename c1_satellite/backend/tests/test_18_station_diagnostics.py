"""Unit test for Phase 18 station-level error diagnostic on synthetic data."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / 'src'))

SPEC17 = importlib.util.spec_from_file_location('c1_validation17', C1 / 'scripts' / '17_validation_only_model_improvements.py')
MOD17 = importlib.util.module_from_spec(SPEC17)
SPEC17.loader.exec_module(MOD17)

SPEC18 = importlib.util.spec_from_file_location('c1_diagnostics18', C1 / 'scripts' / '18_station_validation_diagnostics.py')
MOD18 = importlib.util.module_from_spec(SPEC18)
SPEC18.loader.exec_module(MOD18)


def mock_files(root):
    local = root / 'local_data'
    out = root / 'outputs'
    local.mkdir(parents=True)
    out.mkdir(parents=True)
    labels = []
    sar = []
    swi = []
    dates = (list(pd.date_range('2025-11-17', periods=12)) +
             list(pd.date_range('2025-12-10', periods=4)) +
             list(pd.date_range('2025-12-17', periods=3)))
    for st in ('s1', 's2', 's3'):
        st_num = int(st[-1])
        prev_level = 1.0 + st_num / 3.0
        for pos, d in enumerate(dates):
            fd = d.strftime('%Y-%m-%d')
            split = 'TRAIN' if d <= pd.Timestamp('2025-12-09') else ('VALIDATION' if d <= pd.Timestamp('2025-12-16') else 'TEST')
            stage = prev_level + 0.01 * pos
            delta = float(0.02 * st_num + 0.01 * np.sin(pos + st_num))
            labels.append({
                'record_id': f'{st}_{fd}', 'station_id': st, 'forecast_date': fd,
                'chronological_split': split, 'stage_at_issue_m': stage,
                'target_delta_stage_m': delta, 'target_stage_next_day_m': stage + delta
            })
            sar.append({
                'station_id': st, 'forecast_date': fd, 'sar_multi_local_available': 1,
                'sar_latest_acquisition_utc': (d - pd.Timedelta(days=4)).strftime('%Y-%m-%dT03:00:00Z'),
                'sar_previous_acquisition_utc': (d - pd.Timedelta(days=16)).strftime('%Y-%m-%dT03:00:00Z'),
                'sar_latest_age_hours': 100, 'sar_patch_valid_fraction': 1,
                'sar_vv_mean_db': -13 + pos * .01, 'sar_vh_mean_db': -20 + pos * .01,
                'sar_vv_std_db': 3.0, 'sar_vh_std_db': 3.4,
                'sar_vv_p10_db': -17, 'sar_vv_p90_db': -9,
                'sar_vh_p10_db': -24, 'sar_vh_p90_db': -16,
                'sar_vv_mean_change_db': .15, 'sar_vh_mean_change_db': .21
            })
            rec = {
                'station_id': st, 'forecast_date': fd,
                'latest_eligible_swi_date': (d - pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                'swi005010_complete_7': 1
            }
            for k in range(1, 8):
                for b in ('005', '010'):
                    rec[f'swi{b}_d{k}'] = 40 + pos * .1 - k * .15 + (5 if b == '010' else 0)
            swi.append(rec)
    pd.DataFrame(labels).to_csv(local / 'FloodRisk_DMC_StationDay_LABELLED_1429.csv', index=False)
    pd.DataFrame(sar).to_csv(out / 'c1_sar_station_day_features.csv', index=False)
    pd.DataFrame(swi).to_csv(out / 'c1_swi_bandwise_7day_features.csv', index=False)
    return root


class TestDiagnostics18(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = mock_files(Path(self.tmp.name))
        # Fit models on mock data and save them
        MOD17.run(root=self.root, min_train=10, min_validation=5,
                  save_models=True, model_defs=[('RIDGE_MOCK', Ridge(alpha=10))])

    def test_diagnostics_produces_tables_without_touching_test(self):
        summary = MOD18.run_diagnostics(root=self.root)
        self.assertFalse(summary['test_split_touched'])
        self.assertEqual(summary['n_validation_rows'], 12)
        self.assertEqual(summary['n_validation_stations'], 3)
        out = self.root / 'outputs' / 'improvement_validation'
        self.assertTrue((out / 'validation_predictions.csv').is_file())
        self.assertTrue((out / 'station_validation_error_summary.csv').is_file())
        self.assertTrue((out / 'date_validation_error_summary.csv').is_file())
        self.assertTrue((out / 'validation_diagnostic_summary.json').is_file())


if __name__ == '__main__':
    unittest.main()
