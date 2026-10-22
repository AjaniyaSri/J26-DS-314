"""Unit tests for read-only Phase 12 diagnosis (no access to user's data needed)."""
import importlib.util
from pathlib import Path
import pandas as pd
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'12_c1_diagnostics.py'
spec=importlib.util.spec_from_file_location('audit12',SCRIPT)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class Phase12Tests(unittest.TestCase):
    def setUp(self):
        self.labels=pd.DataFrame([
            dict(station_id='alpha',forecast_date='2025-11-17',chronological_split='TRAIN',record_id='a',target_delta_stage_m=2.0),
            dict(station_id='alpha',forecast_date='2025-12-10',chronological_split='VALIDATION',record_id='b',target_delta_stage_m=0.2),
            dict(station_id='beta',forecast_date='2025-12-17',chronological_split='TEST',record_id='c',target_delta_stage_m=-0.3),
        ])
        self.swif=self.labels[['station_id','forecast_date']].copy()
        for k in range(1,8):
            self.swif[f'swi_valid_d{k}']=[int(k<=3),int(k<=5),int(k==1)]
        self.swif['swi_7day_complete']=0
        self.swif['swi_feature_available']=1
        self.sar=self.labels[['station_id','forecast_date']].copy()
        self.sar['sar_multi_local_available']=[1,1,0]
        self.sar['sar_latest_local_available']=[1,1,1]
        self.obs=pd.DataFrame([
            {'station_id':'alpha','observation_date':'2025-11-15','quality_status':'VALID','QFLAG001':75,'QFLAG005':90,'QFLAG010':95},
            {'station_id':'beta','observation_date':'2025-12-15','quality_status':'LOW_QFLAG','QFLAG001':25,'QFLAG005':90,'QFLAG010':95},
        ])
    def test_swi_completeness_and_masks(self):
        coverage,qflag,quality,station,f=audit.swi_audit(self.labels,self.swif,self.obs,self.sar)
        self.assertEqual(coverage[0]['swi_exactly_3_valid_days_n'],1)
        self.assertEqual(coverage[1]['swi_exactly_5_valid_days_n'],1)
        self.assertEqual(coverage[2]['swi_exactly_1_valid_days_n'],1)
        self.assertEqual(coverage[0]['complete_7_day_n'],0)
        self.assertEqual(qflag['QFLAG001_below_50_in_low_qflag_rows'],1)
    def test_regime_summary(self):
        r=audit.summarize_movement(self.labels)
        self.assertAlmostEqual(r[0]['mean_abs_stage_change_m'],2.0)
        self.assertAlmostEqual(r[2]['mean_abs_stage_change_m'],0.3)
    def test_inconsistent_flag_fails(self):
        bad=self.swif.copy()
        bad.loc[0,'swi_7day_complete']=1
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            audit.swi_audit(self.labels,bad,self.obs,self.sar)
    def test_prediction_mismatched_cohort_fails(self):
        self.labels['target_stage_next_day_m']=[3.0,2.0,1.0]
        p=[]
        for j,r in self.labels.iterrows():
            p.append(dict(record_id=r.record_id,station_id=r.station_id,forecast_date=r.forecast_date,
               chronological_split=r.chronological_split,model_id='PERSISTENCE',actual_next_day_stage_m=r.target_stage_next_day_m,
               predicted_next_day_stage_m=r.target_stage_next_day_m))
            if j<2:p.append(dict(record_id=r.record_id,station_id=r.station_id,forecast_date=r.forecast_date,
               chronological_split=r.chronological_split,model_id='A',actual_next_day_stage_m=r.target_stage_next_day_m,
               predicted_next_day_stage_m=r.target_stage_next_day_m))
        with self.assertRaisesRegex(ValueError,'Unfair comparison'):
            audit.prediction_audit(self.labels,pd.DataFrame(p),pd.DataFrame())

if __name__=='__main__': unittest.main()
