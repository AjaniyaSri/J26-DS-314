"""Offline tests for C1's Sentinel-1 date cutoffs and orbit-matching logic.
Run from project root: python -m unittest discover -s c1_satellite/tests -p test_03_sentinel1_inventory.py
"""
from __future__ import annotations
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / '03_sentinel1_inventory.py'
spec = importlib.util.spec_from_file_location('c1_s1_inventory', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def scene(sid, iso, orbit='5', direction='ASCENDING'):
    return {'scene_id': sid, 'acquisition_utc': iso,
            'relative_orbit': orbit, 'orbit_pass': direction}


class TestEligibility(unittest.TestCase):
    def setUp(self):
        self.row = {'record_id': 'test1', 'station_id':'test', 'forecast_date':'2025-11-20', 'chronological_split':'TRAIN'}

    def test_issue_time_sri_lanka_to_utc(self):
        self.assertEqual(module.iso_utc(module.nominal_issue_utc('2025-11-20')), '2025-11-20T04:00:00Z')

    def test_48hour_cutoff_and_future_scene_excluded(self):
        xs = [scene('old','2025-11-10T10:00:00Z'), scene('lagged','2025-11-18T04:00:00Z'),
              scene('too_new','2025-11-19T00:00:00Z'),scene('future','2025-11-21T00:00:00Z')]
        out = module.eligibility_for_row(self.row, xs, 48)
        self.assertEqual(out['sar_latest_scene_id'],'lagged')
        self.assertEqual(out['sar_eligible_count'],2)
        self.assertEqual(out['sar_multi_available'],1)
        self.assertNotIn('too_new',out['sar_multi_scene_ids'])

    def test_scenes_with_different_orbit_cannot_be_temporal_pair(self):
        xs = [scene('old','2025-11-10T10:00:00Z','4'),scene('latest','2025-11-15T04:00:00Z','5')]
        out = module.eligibility_for_row(self.row, xs, 48)
        self.assertEqual(out['sar_single_available'],1)
        self.assertEqual(out['sar_multi_available'],0)

    def test_missing_images_are_not_invented(self):
        out = module.eligibility_for_row(self.row, [], 48)
        self.assertEqual(out['sar_single_available'],0)
        self.assertEqual(out['sar_multi_scene_ids'],'')


if __name__ == '__main__':
    unittest.main()
