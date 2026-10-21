"""Ablation training and evaluation, one exact common split for all configs."""
import copy,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .models.encoders import C4AblationModel
from .data.dataset import C4Dataset

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def run_epoch(model,loader,device,optimizer=None):
    training=optimizer is not None
    model.train(training)
    preds=[];truth=[];cur=[];ids=[];losses=[]
    for batch in loader:
        batch={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}
        with torch.set_grad_enabled(training):
            predicted_delta=model(batch)
            loss=nn.functional.smooth_l1_loss(predicted_delta,batch['target_delta'],beta=0.3)
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()
        losses.append(loss.item()*len(batch['array_row']))
        preds.extend((predicted_delta.detach()+batch['current_raw']).cpu().view(-1).tolist())
        truth.extend(batch['target_stage'].cpu().view(-1).tolist())
        cur.extend(batch['current_raw'].cpu().view(-1).tolist())
        ids.extend(batch['array_row'].cpu().view(-1).tolist())
    pred=np.array(preds);y=np.array(truth);persist=np.array(cur)
    return {'loss':sum(losses)/max(1,len(pred)), 'mae':float(np.mean(np.abs(y-pred))),
      'rmse':float(np.sqrt(np.mean((y-pred)**2))),
      'persistence_mae':float(np.mean(np.abs(y-persist))),
      'persistence_rmse':float(np.sqrt(np.mean((y-persist)**2))),
      'rows':ids,'pred_stage':preds,'true_stage':truth,'current_stage':cur}

def run_model(store,name,output_dir,*,epochs=18,patience=4,batch_size=8,lr=3e-4,seed=42,pretrained=False,freeze_backbone=True,device=None):
    if not pretrained and freeze_backbone and 'satellite' in C4AblationModel.ACTIVE[name]:
        raise ValueError('Cannot freeze a randomly initialized EfficientNet backbone for research; load pretrained weights or unfreeze.')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    set_seed(seed)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    needs_sar='satellite' in C4AblationModel.ACTIVE[name]
    if needs_sar:
        if store.sar_stats is None:raise ValueError('Compute train-only SAR statistics before satellite training')
        store.check_sar_cache()
    def loader(split,shuffle=False):
        return DataLoader(C4Dataset(store,split,require_sar=needs_sar),batch_size=batch_size,shuffle=shuffle,num_workers=0,pin_memory=False)
    train_loader=loader('TRAIN',True);valid_loader=loader('VALIDATION')
    model=C4AblationModel(name,pretrained,freeze_backbone).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-3)
    best=float('inf');best_state=None;bad=0;history=[]
    for epoch in range(1,epochs+1):
        tr=run_epoch(model,train_loader,device,optimizer)
        va=run_epoch(model,valid_loader,device)
        history.append({'epoch':epoch,'train_loss':tr['loss'],'train_mae':tr['mae'],'val_loss':va['loss'],'val_mae':va['mae']})
        if va['mae']<best-1e-5:
            best=va['mae'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:
            bad+=1
        print(f'{name} epoch {epoch:02d} train MAE={tr["mae"]:.4f} validation MAE={va["mae"]:.4f}',flush=True)
        if bad>=patience:break
    model.load_state_dict(best_state)
    torch.save({'model_state':best_state,'config':name,'seed':seed,'validation_mae_m':best},out/f'{name}_best.pt')
    pd.DataFrame(history).to_csv(out/f'{name}_learning_curve.csv',index=False)
    # UNSEEN test evaluated only after validation model choice; no hyperparam tuning on test.
    test=run_epoch(model,loader('TEST'),device)
    mapping=store.index.iloc[test['rows']][['record_id','station_id','forecast_date','split']].reset_index(drop=True)
    mapping['current_stage_m']=test['current_stage'];mapping['target_stage_m']=test['true_stage']
    mapping['pred_stage_m']=test['pred_stage'];mapping['error_m']=mapping.pred_stage_m-mapping.target_stage_m
    mapping.to_csv(out/f'{name}_test_predictions.csv',index=False)
    result={'model':name,'validation_mae_m':best,'test_mae_m':test['mae'],'test_rmse_m':test['rmse'],
            'persistence_mae_m':test['persistence_mae'],'persistence_rmse_m':test['persistence_rmse'],
            'epochs_completed':len(history),'test_n':len(mapping),'train_n':809,'validation_n':269}
    (out/f'{name}_test_metrics.json').write_text(json.dumps(result,indent=2))
    return result
