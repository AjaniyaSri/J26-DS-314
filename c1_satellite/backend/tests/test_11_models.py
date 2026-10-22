import unittest
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
MODULE=Path(__file__).resolve().parents[1]/'scripts'/'11_c1_controlled_models.py'
spec=importlib.util.spec_from_file_location('c1_models',MODULE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TestAblations(unittest.TestCase):
    def test_all_models_fit_on_same_chronological_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            orig=(module.LABELS,module.SAR,module.SWI,module.OUT)
            module.LABELS=tmp/'lab.csv';module.SAR=tmp/'sar.csv';module.SWI=tmp/'swi.csv';module.OUT=tmp
            try:
                labels=[];sar=[];swi=[]
                for j in range(4):
                    sid=f'station{j}'
                    for i in range(16):
                        date=pd.Timestamp('2025-11-17')+pd.Timedelta(days=i)
                        day=date.strftime('%Y-%m-%d')
                        split='TRAIN' if i<10 else 'VALIDATION' if i<13 else 'TEST'
                        stage=1.0+0.1*i+j*0.2
                        delta=0.02+0.004*i+0.001*j
                        labels.append(dict(record_id=f'{sid}_{day}',station_id=sid,forecast_date=day,chronological_split=split,
                                           stage_at_issue_m=stage,target_delta_stage_m=delta,target_stage_next_day_m=stage+delta))
                        s=dict(station_id=sid,forecast_date=day,sar_latest_local_available=1,sar_multi_local_available=1)
                        for k,v in enumerate(module.SAR_BASE+module.SAR_HISTORY):s[v]=k+i*0.03+j*0.05
                        sar.append(s)
                        w=dict(station_id=sid,forecast_date=day,swi_feature_available=1,swi_7day_complete=1)
                        for k,v in enumerate(module.SWI_NOW+module.SWI_HISTORY):w[v]=30+j+i*0.1+k*0.08
                        for k in module.SWI_MISSING_INDICATORS:w[k]=1
                        if i==8:
                            w['swi001_d6']=np.nan;w['swi_valid_d6']=0;w['swi_7day_complete']=0
                        swi.append(w)
                pd.DataFrame(labels).to_csv(module.LABELS,index=False)
                pd.DataFrame(sar).to_csv(module.SAR,index=False)
                pd.DataFrame(swi).to_csv(module.SWI,index=False)
                status=module.run(SimpleNamespace(require_complete_history=False,min_train=30,min_eval=5))
                self.assertEqual(status,0)
                result=pd.read_csv(tmp/'c1_ablation_metrics.csv')
                self.assertEqual(set(result.model_id),set(['STAGE_ONLY','A','B','C','D','PERSISTENCE']))
                # fair cohort: each model gets exact same n in each split
                for split in ['TRAIN','VALIDATION','TEST']:
                    self.assertEqual(result[result.split==split].n.nunique(),1)
                self.assertEqual(result[result.split=='TEST'].n.iloc[0],12)
            finally:
                module.LABELS,module.SAR,module.SWI,module.OUT=orig
if __name__=='__main__':unittest.main()
