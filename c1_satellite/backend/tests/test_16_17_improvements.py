"""Synthetic QA tests; never asserts user data model accuracy."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

C1=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(C1/'src'))
from improvements.data import (load_project, add_stage_history, validate_labels)

SCRIPT=C1/'scripts'/'17_validation_only_model_improvements.py'
spec=importlib.util.spec_from_file_location('c1_validation17',SCRIPT)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def mock_files(root):
    local=root/'local_data'; out=root/'outputs'
    local.mkdir(parents=True); out.mkdir(parents=True)
    labels=[]; sar=[]; swi=[]
    dates=(list(pd.date_range('2025-11-17',periods=12))+
           list(pd.date_range('2025-12-10',periods=4))+
           list(pd.date_range('2025-12-17',periods=3)))
    for st in ('s1','s2','s3'):
        st_num=int(st[-1]); prev_level=1.0+st_num/3.0
        for pos,d in enumerate(dates):
            fd=d.strftime('%Y-%m-%d')
            split='TRAIN' if d <= pd.Timestamp('2025-12-09') else ('VALIDATION' if d <= pd.Timestamp('2025-12-16') else 'TEST')
            stage=prev_level+0.01*pos
            delta=float(0.02*st_num+0.01*np.sin(pos+st_num))
            labels.append({'record_id':f'{st}_{fd}','station_id':st,'forecast_date':fd,
                'chronological_split':split,'stage_at_issue_m':stage,
                'target_delta_stage_m':delta,'target_stage_next_day_m':stage+delta})
            sar.append({'station_id':st,'forecast_date':fd,'sar_multi_local_available':1,
                'sar_latest_acquisition_utc':(d-pd.Timedelta(days=4)).strftime('%Y-%m-%dT03:00:00Z'),
                'sar_previous_acquisition_utc':(d-pd.Timedelta(days=16)).strftime('%Y-%m-%dT03:00:00Z'),
                'sar_latest_age_hours':100,'sar_patch_valid_fraction':1,
                'sar_vv_mean_db':-13+pos*.01,'sar_vh_mean_db':-20+pos*.01,
                'sar_vv_std_db':3.0,'sar_vh_std_db':3.4,
                'sar_vv_p10_db':-17,'sar_vv_p90_db':-9,
                'sar_vh_p10_db':-24,'sar_vh_p90_db':-16,
                'sar_vv_mean_change_db':.15,'sar_vh_mean_change_db':.21})
            rec={'station_id':st,'forecast_date':fd,
                 'latest_eligible_swi_date':(d-pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                 'swi005010_complete_7':1}
            for k in range(1,8):
                for b in ('005','010'):
                    rec[f'swi{b}_d{k}']=40+pos*.1-k*.15+(5 if b=='010' else 0)
            swi.append(rec)
    pd.DataFrame(labels).to_csv(local/'FloodRisk_DMC_StationDay_LABELLED_1429.csv',index=False)
    pd.DataFrame(sar).to_csv(out/'c1_sar_station_day_features.csv',index=False)
    pd.DataFrame(swi).to_csv(out/'c1_swi_bandwise_7day_features.csv',index=False)
    return root


class TestImprovement(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=mock_files(Path(self.tmp.name))

    def test_temporal_history_never_uses_future_target(self):
        labels=pd.read_csv(self.root/'local_data'/'FloodRisk_DMC_StationDay_LABELLED_1429.csv')
        result=add_stage_history(labels)
        row=result[(result.station_id=='s1') & (result.forecast_date=='2025-11-18')].iloc[0]
        earlier=labels[(labels.station_id=='s1') & (labels.forecast_date=='2025-11-17')].iloc[0]
        self.assertAlmostEqual(float(row.stage_lag1_m),float(earlier.stage_at_issue_m))
        self.assertTrue(np.isnan(float(row.stage_lag2_m)))
        # Changing future target cannot change stage history.
        labels.loc[labels.forecast_date=='2025-11-19','target_stage_next_day_m']=999.0
        rebuilt=add_stage_history(labels)
        self.assertAlmostEqual(float(rebuilt.iloc[1].stage_lag1_m),float(result.iloc[1].stage_lag1_m))

    def test_invalid_target_is_rejected(self):
        path=self.root/'local_data'/'FloodRisk_DMC_StationDay_LABELLED_1429.csv'
        df=pd.read_csv(path)
        df.loc[0,'target_delta_stage_m']=999
        with self.assertRaisesRegex(ValueError, 'Label leakage/inconsistency'):
            validate_labels(df)

    def test_realistic_mock_cohort_and_time_cutoff(self):
        df,a=load_project(self.root)
        self.assertEqual(a['eligible_n'],57)
        self.assertEqual(a['eligible_by_split'], {'TRAIN':36,'VALIDATION':12,'TEST':9})
        self.assertEqual(a['cutoff_audit']['observed_future_cutoff_violations'],0)
        self.assertTrue(np.isfinite(df['swi005_mean7']).all())

    def test_future_scene_stops_pipeline(self):
        path=self.root/'outputs'/'c1_sar_station_day_features.csv'
        df=pd.read_csv(path)
        df.loc[0,'sar_latest_acquisition_utc']='2025-11-17T09:00:00Z'
        df.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'newer than assumed forecast cutoff'):
            load_project(self.root)

    def test_validation_training_without_test_scores(self):
        scores=mod.run(root=self.root, min_train=10, min_validation=5,
                       save_models=False, model_defs=[('RIDGE_SMOKE',Ridge(alpha=10))])
        self.assertEqual(set(scores.experiment),set(mod.variants())|{'PERSISTENCE'})
        self.assertEqual(set(scores.n_validation),{12})
        output=self.root/'outputs'/'improvement_validation'/'selection_metadata.json'
        self.assertTrue(output.is_file())
        self.assertNotIn('test_metrics',output.read_text())


if __name__=='__main__':unittest.main()
