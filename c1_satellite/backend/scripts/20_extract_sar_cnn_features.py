"""C1 Phase 20: Extract Spatial SAR CNN Embeddings for all Sentinel-1 Patches.

Processes all 204 real 2-band (VV/VH) GeoTIFF patches through the SARResidualEncoder
and aligns the 16 learned spatial features with station-day records.
No future data used. Fully deterministic.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / "src"))

from improvements.sar_cnn import extract_sar_cnn_embeddings, DEFAULT_EMBEDDING_DIM


def main():
    patches_dir = C1 / "outputs" / "s1_patches"
    if not patches_dir.is_dir():
        raise FileNotFoundError(f"Missing patches directory: {patches_dir}")

    tif_files = sorted(list(patches_dir.glob("*.tif")))
    print(f"=== C1 PHASE 20: SPATIAL SAR CNN EMBEDDING EXTRACTION ===")
    print(f"Found {len(tif_files)} real Sentinel-1 GeoTIFF patches in {patches_dir.name}")

    if not tif_files:
        raise ValueError("No GeoTIFF patches found to process.")

    # 1. Extract 16-dimensional spatial embeddings from CNN
    print("Running PyTorch SARResidualEncoder across patches...")
    embeddings_map = extract_sar_cnn_embeddings(tif_files, batch_size=16, device="cpu")
    print(f"Successfully encoded {len(embeddings_map)} SAR patches into {DEFAULT_EMBEDDING_DIM}-dim spatial features.")

    # 2. Convert to DataFrame
    cnn_cols = [f"sar_cnn_f{i+1:02d}" for i in range(DEFAULT_EMBEDDING_DIM)]
    rows = []
    for tif_path in tif_files:
        filename = tif_path.name
        emb = embeddings_map[filename]
        row_dict = {"patch_filename": filename}
        for col, val in zip(cnn_cols, emb):
            row_dict[col] = round(float(val), 6)
        rows.append(row_dict)

    df_cnn = pd.DataFrame(rows)
    patch_csv_path = C1 / "outputs" / "c1_sar_patch_cnn_features.csv"
    df_cnn.to_csv(patch_csv_path, index=False)
    print(f"Saved patch embeddings to: {patch_csv_path.name} ({len(df_cnn)} rows)")

    # 3. Align with station-day feature table
    station_day_path = C1 / "outputs" / "c1_sar_station_day_features.csv"
    if station_day_path.is_file():
        df_days = pd.read_csv(station_day_path, dtype={"station_id": str, "forecast_date": str})
        
        # Extract filename from sar_latest_patch_path
        def get_patch_fname(p):
            if pd.isna(p) or not str(p).strip():
                return ""
            return Path(str(p)).name

        df_days["patch_filename"] = df_days["sar_latest_patch_path"].apply(get_patch_fname)
        
        # Merge on patch_filename
        merged = pd.merge(df_days, df_cnn, on="patch_filename", how="left")
        
        out_day_path = C1 / "outputs" / "c1_sar_cnn_station_day_features.csv"
        merged.to_csv(out_day_path, index=False)
        print(f"Aligned with station-day table: {out_day_path.name} ({len(merged)} rows)")
        matched_count = merged[cnn_cols[0]].notna().sum()
        print(f"Matched real SAR CNN patches to {matched_count} station-day forecast records.")

    print("=== Spatial SAR CNN Feature Extraction Complete ===")


if __name__ == "__main__":
    main()
