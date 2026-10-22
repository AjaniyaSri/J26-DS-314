"""C1 Phase 21: Benchmark Spatial SAR CNN Embeddings against Baseline on Validation Split.

Evaluates whether adding 16-dimensional spatial CNN features (extracted from 204 real
Sentinel-1 GeoTIFF patches) improves 24-hour river stage predictions on the identical
186-sample pristine validation period (Dec 10 - Dec 16, 2025).
Uses the identical station-aware preprocessing pipeline as Phase 17.
No test labels touched. Fully deterministic.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / "src"))

from improvements.data import load_project, STAGE_CORE, STAGE_HIST, SAR_BASE, SAR_NOW_ENG, SWI_NOW, SWI_NOW_ENG


def make_pipeline(cols, estimator):
    prep = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scaler', StandardScaler())]), cols),
        ('station', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['station_id'])
    ], sparse_threshold=0)
    return Pipeline([('preprocessing', prep), ('estimator', estimator)])


def main():
    print("=== C1 PHASE 21: BENCHMARKING SPATIAL SAR CNN EMBEDDINGS ===")
    
    # 1. Load controlled two-band cohort
    df, meta = load_project(C1)
    print(f"Loaded cohort: {len(df)} rows (TRAIN={meta['eligible_by_split']['TRAIN']}, "
          f"VAL={meta['eligible_by_split']['VALIDATION']}, TEST={meta['eligible_by_split']['TEST']})")

    # 2. Merge CNN features from Phase 20
    cnn_file = C1 / "outputs" / "c1_sar_patch_cnn_features.csv"
    if not cnn_file.is_file():
        raise FileNotFoundError(f"Missing CNN features: {cnn_file}. Run Phase 20 first.")

    df_cnn = pd.read_csv(cnn_file)
    cnn_feature_cols = [f"sar_cnn_f{i+1:02d}" for i in range(16)]

    # Extract patch filename from sar_latest_patch_path
    def get_patch_fname(p):
        if pd.isna(p) or not str(p).strip():
            return ""
        return Path(str(p)).name

    df["patch_filename"] = df["sar_latest_patch_path"].apply(get_patch_fname)
    df = pd.merge(df, df_cnn, on="patch_filename", how="left")

    train = df[df["chronological_split"] == "TRAIN"].copy()
    val = df[df["chronological_split"] == "VALIDATION"].copy()

    y_train = train["target_delta_stage_m"].values
    y_val_target = val["target_stage_next_day_m"].values
    stage_val_issue = val["stage_at_issue_m"].values

    # Naive Persistence baseline
    pers_pred = stage_val_issue
    pers_mae = mean_absolute_error(y_val_target, pers_pred)
    pers_rmse = root_mean_squared_error(y_val_target, pers_pred)

    # Feature sets
    feats_a = STAGE_CORE + STAGE_HIST + SAR_BASE + SAR_NOW_ENG + SWI_NOW + SWI_NOW_ENG
    feats_a_cnn = feats_a + cnn_feature_cols

    models = {
        "PERSISTENCE": None,
        "A_ENH_2B (Tabular Only)": feats_a,
        "A_ENH_2B_CNN (Tabular + Spatial CNN)": feats_a_cnn,
    }

    results = []
    estimator = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=42,
        n_jobs=2
    )

    for name, cols in models.items():
        if name == "PERSISTENCE":
            results.append({
                "model": "PERSISTENCE",
                "n_features": 1,
                "val_mae_m": round(pers_mae, 4),
                "val_rmse_m": round(pers_rmse, 4),
                "diff_vs_pers_cm": 0.0,
            })
            continue

        pipe = make_pipeline(cols, estimator)
        pipe.fit(train, y_train)
        pred_delta = pipe.predict(val)
        pred_stage = stage_val_issue + pred_delta

        mae = mean_absolute_error(y_val_target, pred_stage)
        rmse = root_mean_squared_error(y_val_target, pred_stage)
        diff_cm = (mae - pers_mae) * 100

        results.append({
            "model": name,
            "n_features": len(cols),
            "val_mae_m": round(mae, 4),
            "val_rmse_m": round(rmse, 4),
            "diff_vs_pers_cm": round(diff_cm, 2),
        })

    res_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("VALIDATION METRICS (186 Common Samples, Dec 10-16, 2025):")
    print("="*70)
    print(res_df.to_string(index=False))
    print("="*70)

    # Station-by-station comparison
    est_tab = ExtraTreesRegressor(n_estimators=120, max_depth=3, min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=2)
    est_cnn = ExtraTreesRegressor(n_estimators=120, max_depth=3, min_samples_leaf=8, max_features=0.8, random_state=42, n_jobs=2)
    pipe_tab = make_pipeline(feats_a, est_tab).fit(train, y_train)
    pipe_cnn = make_pipeline(feats_a_cnn, est_cnn).fit(train, y_train)

    val_eval = val.copy()
    val_eval["pred_stage_tab"] = stage_val_issue + pipe_tab.predict(val)
    val_eval["pred_stage_cnn"] = stage_val_issue + pipe_cnn.predict(val)

    val_eval["err_tab"] = (val_eval["target_stage_next_day_m"] - val_eval["pred_stage_tab"]).abs()
    val_eval["err_cnn"] = (val_eval["target_stage_next_day_m"] - val_eval["pred_stage_cnn"]).abs()

    st_summary = val_eval.groupby("station_id")[["err_tab", "err_cnn"]].mean()
    cnn_wins = (st_summary["err_cnn"] < st_summary["err_tab"]).sum()
    ties = (st_summary["err_cnn"] == st_summary["err_tab"]).sum()
    total_st = len(st_summary)

    print(f"\nStation-Level Comparison (Total: {total_st} stations):")
    print(f"  CNN improves error over Tabular: {cnn_wins}/{total_st} stations ({cnn_wins/total_st*100:.1f}%)")
    print(f"  Ties / identical:                 {ties}/{total_st} stations")

    # Save summary
    out_csv = C1 / "outputs" / "improvement_validation" / "cnn_vs_tabular_benchmark.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to: {out_csv.name}")


if __name__ == "__main__":
    main()
