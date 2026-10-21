"""C4 independently trained modality encoders. No C1/C2/C3 weights used."""
import torch
from torch import nn
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception:
    efficientnet_b0 = None
    EfficientNet_B0_Weights = None

class SatelliteEncoder(nn.Module):
    def __init__(self, pretrained=False, freeze_backbone=True):
        super().__init__()
        if efficientnet_b0 is None:
            raise RuntimeError('torchvision EfficientNet-B0 is required')
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            net=efficientnet_b0(weights=weights)
        except Exception as exc:
            raise RuntimeError('Pretrained EfficientNet weights unavailable. Cache weights first or explicitly use pretrained=False; no silent fallback.') from exc
        # SAR bands are decibel physical quantities, not RGB channels.
        self.adapter=nn.Conv2d(2,3,kernel_size=1,bias=True)
        self.backbone=net.features
        self.pool=nn.AdaptiveAvgPool2d(1)
        out_dim=net.classifier[1].in_features
        self.sar_head=nn.Sequential(nn.Flatten(),nn.Linear(out_dim,64),nn.ReLU())
        self.swi_head=nn.Sequential(nn.Linear(9,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.meta_head=nn.Sequential(nn.Linear(4,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.merge=nn.Sequential(nn.Linear(88,128),nn.ReLU())
        if freeze_backbone:
            for p in self.backbone.parameters():p.requires_grad_(False)
            # Frozen BN statistics must not drift during training.
            self.backbone.eval()
        self.freeze_backbone=freeze_backbone
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:self.backbone.eval()
        return self
    def forward(self,sar,swi,quality,swi_aux,sar_meta):
        # sar [B,2,256,256] training-only normalized
        z=self.adapter(sar)
        # Keep the frozen backbone differentiable w.r.t. the trainable 2→3 adapter.
        z=self.backbone(z)
        sar_vector=self.sar_head(self.pool(z))
        swi_vector=self.swi_head(torch.cat([swi,quality,swi_aux],dim=1))
        meta_vector=self.meta_head(sar_meta)
        sar_vector=sar_vector*sar_meta[:,0:1] # missing SAR never becomes an observation
        return self.merge(torch.cat([sar_vector,swi_vector,meta_vector],dim=1))

class WeatherEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=4,hidden_size=16,batch_first=True,bidirectional=True)
        self.proj=nn.Sequential(nn.Linear(32,16),nn.ReLU())
    def forward(self,x):
        out,_=self.lstm(x)
        return self.proj(torch.cat((out[:, -1, :16],out[:, 0, 16:]),dim=1))

class EnvironmentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
    def forward(self,x):return self.net(x)

class C4AblationModel(nn.Module):
    ACTIVE={
        'M1':('satellite',),'M2':('weather',),'M3':('environment',),
        'M4':('satellite','weather'),'M5':('satellite','environment'),
        'M6':('weather','environment'),'M7':('satellite','weather','environment')}
    def __init__(self,config='M7',pretrained=False,freeze_backbone=True,dropout=0.2):
        super().__init__()
        if config not in self.ACTIVE:raise ValueError(config)
        self.config=config
        active=self.ACTIVE[config]
        if 'satellite' in active:self.satellite=SatelliteEncoder(pretrained,freeze_backbone)
        if 'weather' in active:self.weather=WeatherEncoder()
        if 'environment' in active:self.environment=EnvironmentEncoder()
        width=sum(dict(satellite=128,weather=16,environment=16)[a] for a in active)
        # Identical current-stage scalar for every model; predicted delta stage is residual.
        self.fusion=nn.Sequential(nn.Linear(width+1,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,batch):
        outputs=[]
        if hasattr(self,'satellite'):
            outputs.append(self.satellite(batch['sar'],batch['swi'],batch['quality'],batch['swi_aux'],batch['sar_meta']))
        if hasattr(self,'weather'):outputs.append(self.weather(batch['weather']))
        if hasattr(self,'environment'):outputs.append(self.environment(batch['environment']))
        return self.fusion(torch.cat(outputs+[batch['stage']],dim=1))
