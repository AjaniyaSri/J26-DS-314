import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / '05_sar_feature_table.py'
spec=importlib.util.spec_from_file_location('phase5',MODULE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class Phase5Tests(unittest.TestCase):
    def setUp(self):
        self.scene1={'station_id':'badalgama','scene_id':'X/S1','acquisition_utc':'2025-12-08T00:25:14Z',
                     'orbit_pass':'DESCENDING','relative_orbit':'19'}
        self.scene2={'station_id':'badalgama','scene_id':'X/S2','acquisition_utc':'2025-12-20T00:25:12Z',
                     'orbit_pass':'DESCENDING','relative_orbit':'19'}
        self.inv={('badalgama','X/S1'):self.scene1,('badalgama','X/S2'):self.scene2}
        self.local={}
        for scene,v in ((self.scene1,-10),(self.scene2,-15)):
            self.local[(scene['station_id'],scene['scene_id'])]={**scene,'qa_status':'PASS',
                'geotiff_path':f"fake-name-{scene['scene_id']}.tif",'sar_vv_mean_db':v,'sar_vh_mean_db':v-4,
                'sar_vv_std_db':2,'sar_vh_std_db':3,'sar_vv_p10_db':-18,'sar_vv_p90_db':-8,
                'sar_vh_p10_db':-25,'sar_vh_p90_db':-12,'sar_patch_valid_fraction':1.0}
        self.e={'record_id':'rid','station_id':'badalgama','forecast_date':'2025-12-25',
            'chronological_split':'TEST','sar_latest_scene_id':'X/S2',
            'sar_multi_scene_ids':'X/S1;X/S2','sar_latest_age_hours':'124'}
    def test_two_actual_scenes(self):
        d=mod.build_day_feature(self.e,self.inv,self.local)
        self.assertEqual(d['sar_latest_local_available'],1)
        self.assertEqual(d['sar_multi_local_available'],1)
        self.assertEqual(d['sar_vv_mean_change_db'],-5)
        self.assertEqual(d['sar_feature_status'],'MULTI_PATCH_READY')
    def test_missing_patch_not_faked(self):
        d=mod.build_day_feature(self.e,self.inv,{})
        self.assertEqual(d['sar_latest_local_available'],0)
        self.assertEqual(d['sar_multi_local_available'],0)
        self.assertEqual(d['sar_vv_mean_db'],'')
    def test_same_day_not_multitemporal(self):
        inv=dict(self.inv)
        inv[('badalgama','X/S1')]=dict(self.scene1,acquisition_utc='2025-12-20T00:20:00Z')
        d=mod.build_day_feature(self.e,inv,self.local)
        self.assertEqual(d['sar_multi_local_available'],0)
    def test_orbit_mismatch_not_multitemporal(self):
        inv=dict(self.inv)
        inv[('badalgama','X/S1')]=dict(self.scene1,relative_orbit='80')
        d=mod.build_day_feature(self.e,inv,self.local)
        self.assertEqual(d['sar_multi_local_available'],0)
    def test_raster_values(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.tif'
            arr=np.stack([np.full((256,256),-12,dtype=np.float32),np.full((256,256),-19,dtype=np.float32)])
            with rasterio.open(p,'w',driver='GTiff',height=256,width=256,count=2,dtype='float32',
                crs='EPSG:32644',transform=from_origin(300000,900000,50,50),nodata=-9999) as ds:
                ds.write(arr)
            stats=mod.get_features(p)
            self.assertEqual(stats['sar_vv_mean_db'],-12)
            self.assertEqual(stats['sar_vh_mean_db'],-19)
            self.assertEqual(stats['sar_patch_valid_fraction'],1)

if __name__=='__main__': unittest.main()
