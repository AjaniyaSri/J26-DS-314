"""Offline unit tests. Run: python -m unittest discover -s c1_satellite/tests -p 'test_06_*.py' -v"""
import importlib.util
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]/'scripts'/'06_batch_sentinel1_patches.py'
SPEC=importlib.util.spec_from_file_location('batch',HERE)
batch=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def scene(st, key, dt, orbit='19'):
    return {'station_id':st, 'scene_id':key, 'acquisition_utc':dt,
            'orbit_pass':'DESCENDING','relative_orbit':orbit}


class BatchPlanTests(unittest.TestCase):
    def setUp(self):
        self.scenes=[scene('x','s0','2025-11-03T00:00:00Z'),
                     scene('x','s1','2025-11-15T00:00:00Z'),
                     scene('x','s2','2025-12-15T00:00:00Z'),
                     scene('y','t0','2025-11-05T00:00:00Z'),
                     scene('y','t1','2025-11-17T00:00:00Z')]
        self.days=[{'station_id':'x','chronological_split':'TRAIN','forecast_date':'2025-11-20',
                    'sar_latest_scene_id':'s1','sar_multi_scene_ids':'s0;s1'},
                   {'station_id':'x','chronological_split':'TEST','forecast_date':'2025-12-20',
                    'sar_latest_scene_id':'s2','sar_multi_scene_ids':'s1;s2'},
                   {'station_id':'y','chronological_split':'TRAIN','forecast_date':'2025-11-21',
                    'sar_latest_scene_id':'t1','sar_multi_scene_ids':'t0;t1'}]

    def test_train_excludes_future_test_scene(self):
        x,missing=batch.eligible_scene_candidates(self.days,self.scenes,'TRAIN')
        self.assertEqual(missing,0)
        self.assertEqual({s['scene_id'] for s in x['x']},{'s0','s1'})
        self.assertNotIn('s2',{s['scene_id'] for v in x.values() for s in v})

    def test_test_includes_only_eligible_history(self):
        x,_=batch.eligible_scene_candidates(self.days,self.scenes,'TEST')
        self.assertEqual({s['scene_id'] for s in x['x']},{'s1','s2'})

    def test_missing_inventory_scene_not_included(self):
        data=[dict(self.days[0],sar_multi_scene_ids='z9;s1')]
        x,_=batch.eligible_scene_candidates(data,self.scenes,'TRAIN')
        self.assertEqual([s['scene_id'] for s in x['x']],['s1'])

    def test_wrong_orbit_predecessor_not_included(self):
        data=[dict(self.days[0],sar_multi_scene_ids='s0;s1')]
        modified=[dict(s,relative_orbit='12') if s['scene_id']=='s0' else s for s in self.scenes]
        x,_=batch.eligible_scene_candidates(data,modified,'TRAIN')
        self.assertEqual([s['scene_id'] for s in x['x']],['s1'])

    def test_round_robin_alternates_stations(self):
        cand,_=batch.eligible_scene_candidates(self.days,self.scenes,'TRAIN')
        plan=batch.round_robin_plan(cand,2)
        self.assertEqual([x['station_id'] for x in plan],['x','y','x','y'])

    def test_deduplicate_station_scene(self):
        rows=self.days[:1]+[dict(self.days[0],forecast_date='2025-11-21')]
        cand,_=batch.eligible_scene_candidates(rows,self.scenes,'TRAIN')
        self.assertEqual(len(cand['x']),2)

if __name__=='__main__':
    unittest.main()
