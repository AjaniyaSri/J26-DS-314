import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

MODULE=Path(__file__).resolve().parents[1]/'scripts'/'14_c1_bandwise_validation.py'
spec=importlib.util.spec_from_file_location('model_14',MODULE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class ModelTests(unittest.TestCase):
    def test_feature_sets_distinct(self):
        self.assertEqual(len(mod.SWI_NOW),2)
        self.assertEqual(len(mod.SWI_HISTORY),12)
        self.assertTrue(all('001' not in x for x in mod.SWI_NOW+mod.SWI_HISTORY))
    def test_no_target_in_features(self):
        feats=mod.STAGE+mod.SAR_BASE+mod.SAR_HISTORY+mod.SWI_NOW+mod.SWI_HISTORY
        self.assertNotIn('target_delta_stage_m',feats)
        self.assertNotIn('target_stage_next_day_m',feats)
    def test_metrics(self):
        scores=mod.scores(np.array([3,4]),np.array([3,5]))
        self.assertAlmostEqual(scores['MAE_m'],0.5)
        self.assertAlmostEqual(scores['RMSE_m'],np.sqrt(0.5))
    def test_pipeline_can_fit(self):
        x=pd.DataFrame({'stage_at_issue_m':[1,2,3,4],'station_id':['a','b','a','b']})
        y=np.array([0.1,0.2,0.1,0.2])
        p=mod.pipeline(['stage_at_issue_m'],10.0)
        p.fit(x,y)
        self.assertEqual(p.predict(x).shape,(4,))

if __name__=='__main__':unittest.main()
