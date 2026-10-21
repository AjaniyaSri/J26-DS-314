"""On-disk C4 aligned arrays + lazy verified SAR patch references."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from hashlib import sha256

N=1429
INPUT_KEYS=('X_weather','X_environment','X_ascat_swi','X_ascat_qflag_percent','stage_current_m','sar_available','sar_age_logscaled','sar_valid_pixel_fraction','swi_age_logscaled','swi_area_valid_fraction')

def checksum(path):
    h=sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

class C4Store:
    def __init__(self,repo_root,inputs_npz=None,labels_npz=None,index_csv=None):
        self.root=Path(repo_root).resolve()
        base=self.root/'data/processed/c4/multimodal'
        self.paths={
            'inputs':Path(inputs_npz) if inputs_npz else base/'c4_aligned_inputs_1429_PRIVATE.npz',
            'labels':Path(labels_npz) if labels_npz else base/'c4_aligned_labels_1429_PRIVATE.npz',
            'index':Path(index_csv) if index_csv else base/'c4_aligned_index_1429_PRIVATE.csv',}
        for p in self.paths.values():
            if not p.is_file():raise FileNotFoundError(f'Missing C4 aligned source: {p}')
        self.index=pd.read_csv(self.paths['index'],encoding='utf-8-sig',low_memory=False).sort_values('array_row').reset_index(drop=True)
        with np.load(self.paths['inputs'],allow_pickle=False) as npz:self.inputs={k:npz[k].copy() for k in npz.files}
        with np.load(self.paths['labels'],allow_pickle=False) as npz:self.labels={k:npz[k].copy() for k in npz.files}
        self.verify()
        self.sar_stats=None
    def verify(self):
        a=self.index
        assert len(a)==N and np.array_equal(a['array_row'].to_numpy(),np.arange(N))
        assert a['record_id'].is_unique and a['station_id'].nunique()==39
        assert a.split.value_counts().to_dict()=={'TRAIN':809,'VALIDATION':269,'TEST':351}
        assert np.array_equal(a.record_id.astype(str).to_numpy(),self.inputs['record_id'])
        assert np.array_equal(a.record_id.astype(str).to_numpy(),self.labels['record_id'])
        assert np.array_equal(a.station_id.astype(str).to_numpy(),self.inputs['station_id'])
        assert self.inputs['X_weather'].shape==(N,14,4)
        assert self.inputs['X_environment'].shape==(N,9)
        assert self.inputs['X_ascat_swi'].shape==(N,3)
        assert self.inputs['X_ascat_qflag_percent'].shape==(N,3)
        assert self.labels['y_delta_stage_m'].shape==(N,1)
        assert self.labels['y_next_stage_m'].shape==(N,1)
        assert all(self.inputs[k].shape[0]==N for k in INPUT_KEYS)
        assert all(np.isfinite(self.inputs[k]).all() for k in INPUT_KEYS)
        assert np.isfinite(self.labels['y_delta_stage_m']).all()
        assert np.allclose(self.inputs['stage_current_m']+self.labels['y_delta_stage_m'],self.labels['y_next_stage_m'],atol=1e-5)
        assert a.supervised_complete.astype(bool).all() and a.feature_complete.astype(bool).all()
        assert a.sar_available.astype(bool).all() and a.swi_available.astype(bool).all()
        assert a.sar_age_hours.ge(0).all()
        assert a.sar_age_hours.le(18*24+0.01).all()
        times=pd.to_datetime(a['sar_acquisition_utc'],utc=True)
        origin=pd.to_datetime(a['forecast_date'])+pd.Timedelta(hours=3,minutes=30) # 09:00 SLST
        assert (times.to_numpy()<=origin.dt.tz_localize('UTC').to_numpy()).all()
        assert a.groupby('split').forecast_date.agg(['min','max']).to_dict('index')=={
          'TRAIN':{'min':'2025-11-17','max':'2025-12-09'},'VALIDATION':{'min':'2025-12-10','max':'2025-12-16'},
          'TEST':{'min':'2025-12-17','max':'2025-12-25'}}
        # Require local NPZ references to be inside repo data cache; verify sha before opening.
        for key in ['weather_feature_names','environment_feature_names','ascat_swi_feature_names']:
            assert key in self.inputs
        assert list(self.inputs['weather_feature_names'])==['precipitation_mm','temperature_C','dewpoint_C','surface_pressure_hPa']
        assert len(self.inputs['environment_feature_names'])==9
        assert list(self.inputs['ascat_swi_feature_names'])==['swi001_percent','swi005_percent','swi010_percent']
    def sar_path(self,i):
        rel=str(self.index.iloc[i]['patch_relpath']).replace('\\','/')
        p=(self.root/rel).resolve()
        expected=(self.root/'data/processed/c4/satellite/sentinel1/cache/patches').resolve()
        if not p.is_relative_to(expected):raise ValueError(f'Unexpected SAR cache path: {p}')
        return p
    def check_sar_cache(self):
        seen=set()
        for i,row in self.index.iterrows():
            path=self.sar_path(i)
            if path in seen:continue
            seen.add(path)
            if not path.is_file():raise FileNotFoundError(f'Actual Sentinel-1 cache missing: {path}')
            if checksum(path)!=str(row['patch_sha256']):raise ValueError(f'SAR hash mismatch: {path}')
            with np.load(path,allow_pickle=False) as p:
                if p['sar_db'].shape!=(256,256,2) or p['valid_mask'].shape!=(256,256):raise ValueError(f'Unexpected SAR tensor: {path}')
        return len(seen)
    def fit(self,indices,compute_sar_stats=True):
        ids=np.asarray(indices,dtype=int)
        if len(ids)==0 or not self.index.iloc[ids].split.eq('TRAIN').all():raise ValueError('Fit ONLY on TRAIN indices')
        self.stats={}
        for key in ('X_weather','X_environment','X_ascat_swi','X_ascat_qflag_percent','stage_current_m'):
            a=self.inputs[key][ids].astype('float64')
            mu=a.mean(axis=tuple(range(a.ndim-1)),keepdims=False)
            std=a.std(axis=tuple(range(a.ndim-1)),keepdims=False)
            self.stats[key]=(mu.astype('float32'),np.maximum(std,1e-4).astype('float32'))
        self.sar_stats=None
        if compute_sar_stats:
            # Scene-level weighting (one acquisition per unique patch) avoids repeated-date overweighting.
            seen=set();s=np.zeros(2);sq=np.zeros(2);count=0
            for i in ids:
                path=self.sar_path(i)
                if path in seen:continue
                seen.add(path)
                if not path.is_file():raise FileNotFoundError(f'SAR cache missing: {path}')
                with np.load(path,allow_pickle=False) as p:
                    image=p['sar_db']; mask=p['valid_mask'].astype(bool)
                if image.shape!=(256,256,2) or mask.shape!=(256,256):raise ValueError(f'Invalid SAR shape {path}')
                vals=image[mask].astype('float64')
                if not np.isfinite(vals).all():raise ValueError(f'Invalid SAR pixels {path}')
                s+=vals.sum(0);sq+=(vals*vals).sum(0);count+=len(vals)
            mean=s/count;std=np.sqrt(np.maximum(sq/count-mean**2,1e-6))
            self.sar_stats=(mean.astype('float32'),std.astype('float32'))
        return self
    def normalize(self,key,arr):
        mu,std=self.stats[key]
        return ((arr.astype('float32')-mu)/std).astype('float32')
    def save_stats(self,p):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        content={f'{key}__{part}':value for key,(mu,std) in self.stats.items() for part,value in [('mean',mu),('std',std)]}
        if self.sar_stats:
            content.update(sar_mean=self.sar_stats[0],sar_std=self.sar_stats[1])
        np.savez_compressed(p,**content)

class C4Dataset(Dataset):
    def __init__(self,store,split,require_sar=True):
        self.store=store
        self.rows=np.flatnonzero(store.index.split.eq(split).to_numpy())
        self.require_sar=require_sar
    def __len__(self):return len(self.rows)
    def __getitem__(self,item):
        s=self.store;i=int(self.rows[item]);a=s.inputs
        x={
          'weather':s.normalize('X_weather',a['X_weather'][i]),
          'environment':s.normalize('X_environment',a['X_environment'][i]),
          'swi':s.normalize('X_ascat_swi',a['X_ascat_swi'][i]),
          'quality':s.normalize('X_ascat_qflag_percent',a['X_ascat_qflag_percent'][i]),
          'stage':s.normalize('stage_current_m',a['stage_current_m'][i]),
          'swi_aux':np.array([a['swi_age_logscaled'][i,0],a['swi_area_valid_fraction'][i,0],a['swi_available'][i,0]],dtype='float32'),
          'sar_meta':np.array([a['sar_available'][i,0],a['sar_age_logscaled'][i,0],a['sar_valid_pixel_fraction'][i,0],1.],dtype='float32'),
          'current_raw':a['stage_current_m'][i],
          'target_delta':s.labels['y_delta_stage_m'][i],
          'target_stage':s.labels['y_next_stage_m'][i],
          'array_row':np.array(i,dtype='int64'),
        }
        if self.require_sar:
            path=s.sar_path(i)
            with np.load(path,allow_pickle=False) as p:
                img=p['sar_db'].copy(); mask=p['valid_mask'].astype(bool).copy()
            if img.shape!=(256,256,2) or mask.shape!=(256,256):raise ValueError(path)
            img[~mask]=s.sar_stats[0]  # becomes 0 after training-only scaling
            img=((img-s.sar_stats[0])/s.sar_stats[1]).astype('float32')
            img=np.clip(img,-8,8)
            x['sar']=np.transpose(img,(2,0,1)).copy()
        return {k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else torch.tensor(v) for k,v in x.items()}
