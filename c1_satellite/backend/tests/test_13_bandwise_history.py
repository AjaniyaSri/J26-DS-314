import importlib.util
import sys
import unittest
from pathlib import Path
import pandas as pd

MODULE=Path(__file__).resolve().parents[1]/'scripts'/'13_swi_bandwise_history.py'
spec=importlib.util.spec_from_file_location('swi_13',MODULE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class BandwiseTests(unittest.TestCase):
    def records(self):
        labels=pd.DataFrame([{'station_id':'a','forecast_date':'2025-11-17','chronological_split':'TRAIN'}])
        dates=pd.date_range('2025-11-09','2025-11-15')
        records=[]
        for d in dates:
            records.append(dict(station_id='a',observation_date=d.strftime('%Y-%m-%d'),quality_status='LOW_QFLAG',
                SSF=1,dataMask=1,SWI001=54,SWI005=58,SWI010=63,QFLAG001=45,QFLAG005=91,QFLAG010=95))
        return labels,pd.DataFrame(records)

    def test_bandwise_7day_available_even_if_qflag001_low(self):
        labs,daily=self.records()
        row=mod.make_features(labs,daily).iloc[0]
        self.assertEqual(row['latest_eligible_swi_date'],'2025-11-15')
        self.assertEqual(row['swi001_complete_7'],0)
        self.assertEqual(row['swi005_complete_7'],1)
        self.assertEqual(row['swi010_complete_7'],1)
        self.assertEqual(row['swi005010_complete_7'],1)
        self.assertEqual(row['swi_all_three_complete_7'],0)
        self.assertEqual(row['swi005_d7'],58)

    def test_one_bad_day_breaks_only_one_band(self):
        labs,daily=self.records()
        daily.loc[daily.observation_date=='2025-11-12','QFLAG010']=49
        row=mod.make_features(labs,daily).iloc[0]
        self.assertEqual(row['swi005_complete_7'],1)
        self.assertEqual(row['swi010_valid_days_7'],6)
        self.assertEqual(row['swi005010_complete_7'],0)
        self.assertTrue(pd.isna(row['swi010_d4']))
        self.assertEqual(row['swi005_d4'],58)

    def test_mask_zero_invalidates_all_bands_one_day(self):
        labs,daily=self.records()
        daily.loc[daily.observation_date=='2025-11-10','dataMask']=0
        row=mod.make_features(labs,daily).iloc[0]
        self.assertEqual(row['swi005_valid_days_7'],6)
        self.assertEqual(row['swi010_valid_days_7'],6)

    def test_reject_duplicate_dates(self):
        labs,daily=self.records()
        daily=pd.concat([daily,daily.iloc[[0]]])
        with self.assertRaises(ValueError):mod.make_features(labs,daily)

    def test_no_future_data_used(self):
        labs,daily=self.records()
        future=daily.iloc[[0]].copy()
        future.loc[:,'observation_date']='2025-11-18'
        future.loc[:,'SWI005']=99
        row=mod.make_features(labs,pd.concat([daily,future])).iloc[0]
        self.assertEqual(row['swi005_d1'],58)

if __name__=='__main__':unittest.main()
