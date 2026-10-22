"""Unit tests for Phase 20: Spatial SAR CNN Residual Encoder."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
import numpy as np

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / "src"))

from improvements.sar_cnn import (
    ResidualBlock2D,
    SARResidualEncoder,
    SARGeoTIFFDataset,
    DEFAULT_EMBEDDING_DIM
)


class TestSARCNN(unittest.TestCase):
    def test_residual_block_shapes(self):
        block = ResidualBlock2D(channels=32)
        x = torch.randn(2, 32, 64, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 64, 64))

    def test_encoder_forward_pass(self):
        encoder = SARResidualEncoder(in_channels=2, embedding_dim=DEFAULT_EMBEDDING_DIM)
        encoder.eval()
        # Input shape: (Batch=2, Channels=2 [VV, VH], Height=256, Width=256)
        x = torch.randn(2, 2, 256, 256)
        with torch.no_grad():
            emb = encoder(x)
        self.assertEqual(emb.shape, (2, DEFAULT_EMBEDDING_DIM))
        self.assertTrue(torch.isfinite(emb).all())

    def test_real_patch_encoding(self):
        patches_dir = C1 / "outputs" / "s1_patches"
        tifs = list(patches_dir.glob("*.tif"))
        if not tifs:
            self.skipTest("No patches found in s1_patches")

        # Test preprocessor on first real patch
        arr = SARGeoTIFFDataset.load_and_preprocess(tifs[0])
        self.assertEqual(arr.shape, (2, 256, 256))
        self.assertTrue(np.isfinite(arr).all())

        # Test forward pass on real patch
        encoder = SARResidualEncoder(in_channels=2, embedding_dim=DEFAULT_EMBEDDING_DIM)
        encoder.eval()
        t = torch.from_numpy(arr).unsqueeze(0).float()
        with torch.no_grad():
            out = encoder(t)
        self.assertEqual(out.shape, (1, DEFAULT_EMBEDDING_DIM))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
