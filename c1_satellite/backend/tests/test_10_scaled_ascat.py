import unittest
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

MODULE=Path(__file__).resolve().parents[1]/'scripts'/'10_extract_ascat_scaled.py'
spec=importlib.util.spec_from_file_location('ascat_scaled',MODULE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TestScaledSWI(unittest.TestCase):
    def test_raw_uint8_swi_and_qflag_are_half_scaled_but_mask_and_ssf_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            old=module.RAW
            try:
                module.RAW=Path(tmp)
                dest=module.RAW/'ASCAT_SWI_V4_2025-11-15.tif'
                # band 8 mask invalid in left cell, valid in right cell
                vals=[150,120,170,160,140,180,1,1]
                with rasterio.open(dest,'w',driver='GTiff',width=2,height=2,count=8,dtype='float32',crs='EPSG:4326',transform=from_origin(79,8,0.1,0.1)) as ds:
                    for i,v in enumerate(vals,1):
                        arr=np.full((2,2),v,dtype='float32')
                        if i==8:arr[0,0]=0
                        ds.write(arr,i)
                    ds.descriptions=module.BANDS
                records,n=module.make_station_daily({'station':(79.15,7.95),'masked':(79.05,7.95)},50)
                self.assertEqual(n,1)
                r=records[('station','2025-11-15')]
                self.assertEqual(r['quality_status'],'VALID')
                self.assertEqual(r['SWI001'],75.0)
                self.assertEqual(r['SWI005'],60.0)
                self.assertEqual(r['QFLAG001'],80.0)
                self.assertEqual(r['SSF'],1.0)
                self.assertEqual(r['dataMask'],1.0)
                self.assertEqual(records[('masked','2025-11-15')]['quality_status'],'SOURCE_NODATA')
                obs,_=module.make_station_daily({'station':(79.15,7.95)},50)
                lab=[{'station_id':'station','forecast_date':'2025-11-17'}]
                feat=module.features(lab,obs,lag=2)[0]
                self.assertEqual(feat['swi001_d1'],75.0)
                self.assertEqual(feat['swi_7day_complete'],0)
            finally:
                module.RAW=old
if __name__=='__main__':unittest.main()
