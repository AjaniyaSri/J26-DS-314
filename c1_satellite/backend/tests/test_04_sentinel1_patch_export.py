"""Offline checks. Run from root: python -m unittest discover -s c1_satellite/tests -p 'test_04*.py' -v"""
import importlib.util
import unittest
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "scripts" / "04_sentinel1_patch_export.py"
spec = importlib.util.spec_from_file_location("c1phase4", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PatchPhase4Tests(unittest.TestCase):
    def test_patch_grid_size_and_alignment(self):
        g = mod.grid_for_xy(300001.6, 900001.1)
        self.assertEqual(g["right"] - g["left"], 12800)
        self.assertEqual(g["top"] - g["bottom"], 12800)
        self.assertEqual(g["transform"][0], 50)
        self.assertEqual(g["transform"][4], -50)
        self.assertEqual(g["transform"][2] % 50, 0)

    def test_latest_multi_eligible_forecast(self):
        rows = [
            {"station_id":"badalgama","forecast_date":"2025-12-20", "sar_multi_scene_ids":"A;B", "sar_latest_scene_id":"B"},
            {"station_id":"badalgama","forecast_date":"2025-12-25", "sar_multi_scene_ids":"A;B;C", "sar_latest_scene_id":"C"},
        ]
        self.assertEqual(mod.choose_forecast_row(rows,"badalgama")["forecast_date"],"2025-12-25")

    def test_choose_scene_rows_constrained_to_inventory(self):
        f = {"station_id":"badalgama","sar_multi_scene_ids":"A;B;C", "sar_latest_scene_id":"C"}
        inv = [{"station_id":"badalgama","scene_id":id,"acquisition_utc":"2025-11-20T00:00:00Z", "orbit_pass":"ASCENDING","relative_orbit":"8"} for id in ["A","B","C"]]
        r = mod.choose_scene_rows(f,inv,2)
        self.assertEqual([s["scene_id"] for s in r],["B","C"])

    def test_mismatched_orbit_fails(self):
        f = {"station_id":"badalgama","sar_multi_scene_ids":"A;B", "sar_latest_scene_id":"B"}
        inv = [
            {"station_id":"badalgama","scene_id":"A","orbit_pass":"ASCENDING","relative_orbit":"1"},
            {"station_id":"badalgama","scene_id":"B","orbit_pass":"DESCENDING","relative_orbit":"1"},
        ]
        with self.assertRaises(ValueError):
            mod.choose_scene_rows(f,inv,2)

    def test_scene_name_stable(self):
        n = mod.filename_for_scene("badalgama",{"scene_id":"COPERNICUS/S1_GRD/hello","acquisition_utc":"2025-12-10T00:00:00Z"})
        self.assertTrue(n.startswith("badalgama_20251210_"))
        self.assertNotIn("/",n)

if __name__ == "__main__":
    unittest.main()
