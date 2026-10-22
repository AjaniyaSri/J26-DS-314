"""Spatial SAR Convolutional Neural Network (CNN) Feature Encoder.

Custom deep learning architecture for dual-polarization Sentinel-1 SAR patches
(VV and VH backscatter bands at 50m resolution, 256x256 pixels).
Extracts compact, regularized spatial embeddings (water extent, floodplain contrast,
and texture gradients) to enhance tabular river stage forecasting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NODATA_VALUE = -9999.0
DEFAULT_EMBEDDING_DIM = 16


class ResidualBlock2D(nn.Module):
    """Residual convolutional block with skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SARResidualEncoder(nn.Module):
    """4-stage Residual CNN custom-built for 2-channel Sentinel-1 SAR patches.
    
    Inputs:
        x: Tensor of shape (Batch, 2, 256, 256) -> [VV, VH] backscatter (dB)
    Outputs:
        embeddings: Tensor of shape (Batch, embedding_dim)
    """
    def __init__(self, in_channels: int = 2, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stage 1: Initial downsampling from 256x256 to 128x128
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 128x128 -> 64x64
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock2D(32),
        )

        # Stage 3: 64x64 -> 32x32
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock2D(64),
        )

        # Stage 4: 32x32 -> 16x16 -> Global Pooling
        self.stage4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Embedding projection head
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return self.projection(out)


class SARGeoTIFFDataset(Dataset):
    """PyTorch Dataset that safely loads and normalizes 2-band SAR GeoTIFF patches."""
    def __init__(self, geotiff_paths: List[Path]):
        self.geotiff_paths = [Path(p) for p in geotiff_paths]

    def __len__(self) -> int:
        return len(self.geotiff_paths)

    @staticmethod
    def load_and_preprocess(tif_path: Path) -> np.ndarray:
        """Reads 2-band GeoTIFF, masks nodata, and normalizes radar dB scale."""
        with rasterio.open(tif_path) as ds:
            arr = ds.read().astype(np.float32)  # Shape: (2, H, W)
            
        # Clean invalid and nodata values
        valid = np.isfinite(arr) & (arr != NODATA_VALUE)
        # Median imputation per band for invalid pixels
        for b in range(arr.shape[0]):
            band_vals = arr[b][valid[b]]
            median_val = float(np.median(band_vals)) if len(band_vals) > 0 else -18.0
            arr[b][~valid[b]] = median_val

        # Clip extreme radar values to reasonable physical range (-35 dB to +5 dB)
        arr = np.clip(arr, -35.0, 5.0)

        # Standardize: typical Sentinel-1 VV/VH mean is ~ -16 dB, std ~ 7 dB
        arr = (arr - (-16.0)) / 7.0
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.geotiff_paths[idx]
        arr = self.load_and_preprocess(path)
        tensor = torch.from_numpy(arr).float()
        return tensor, path.name


def extract_sar_cnn_embeddings(
    tif_paths: List[Path],
    encoder: Optional[SARResidualEncoder] = None,
    batch_size: int = 16,
    device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Passes all GeoTIFF patches through the CNN encoder and returns embedding vectors."""
    if encoder is None:
        torch.manual_seed(42)  # Deterministic initialization
        encoder = SARResidualEncoder(in_channels=2, embedding_dim=DEFAULT_EMBEDDING_DIM)
    
    encoder.to(device)
    encoder.eval()

    dataset = SARGeoTIFFDataset(tif_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings_map = {}
    with torch.no_grad():
        for batch_tensors, batch_names in dataloader:
            batch_tensors = batch_tensors.to(device)
            feats = encoder(batch_tensors).cpu().numpy()
            for name, feat in zip(batch_names, feats):
                embeddings_map[name] = feat

    return embeddings_map
