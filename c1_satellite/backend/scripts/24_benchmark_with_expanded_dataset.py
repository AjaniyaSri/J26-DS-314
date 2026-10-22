#!/usr/bin/env python
"""Script 24 – Benchmark Expanded Multi-Season & Multi-Temporal Dataset (Upgrade 3).

Evaluates the accuracy impact of expanding training data from 319 records
to the full 809 training observations (+153.6% more training data) combined
with 14-day multi-temporal antecedent catchment memory.

Variants:
  A. Restricted Baseline : 319 TRAIN rows (Phase 17/23 feature set)
  B. Expanded Train      : 809 TRAIN rows (same features + availability flags)
  C. Expanded + Catchment: 809 TRAIN rows + Catchment SWI features
  D. Full Upgrade 3      : 809 TRAIN rows + Catchment SWI + 14-day antecedent memory

Evaluation:
  1. Apples-to-apples comparison on the 186 common validation rows.
  2. Complete validation across all 269 validation station-days.

Run from project root:
    .venv\\Scripts\\python c1_satellite/scripts/24_benchmark_with_expanded_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c1_satellite.src.improvements.data import (
    load_project,
    STAGE_CORE,
    STAGE_HIST,
    SAR_BASE,
    SAR_HIST,
    SWI_NOW,
    SWI_HIST,
    SAR_NOW_ENG,
    SAR_HIST_ENG,
    SWI_NOW_ENG,
    SWI_HIST_ENG,
)
from c1_satellite.src.improvements.catchment_swi import (
    merge_catchment_features,
    CATCH_NOW_COLS,
)
from c1_satellite.src.improvements.dataset_expansion import (
    load_expanded_project,
    ANTECEDENT_COLS,
    AVAILABILITY_COLS,
)

CATCH_CSV = ROOT / "c1_satellite" / "outputs" / "c1_catchment_swi_features.csv"
OUTPUT_JSON = ROOT / "c1_satellite" / "outputs" / "expanded_dataset_benchmark_results.json"
TARGET = "target_delta_stage_m"

BASELINE_FEATURES = (
    STAGE_CORE + STAGE_HIST +
    SAR_BASE + SAR_HIST + SAR_NOW_ENG + SAR_HIST_ENG +
    SWI_NOW + SWI_HIST + SWI_NOW_ENG + SWI_HIST_ENG
)

CATCHMENT_FEATURES = [
    "catch_swi005_mean", "catch_swi005_std", "catch_swi005_max",
    "catch_swi005_wet_frac", "catch_swi005_oversat_frac",
    "catch_swi010_mean", "catch_swi010_std", "catch_swi010_max",
    "catch_swi010_wet_frac", "catch_swi010_oversat_frac",
    "catch_area_km2",
    "catch_swi005_minus_swi010",
    "catch_wet_frac_mean",
    "catch_oversat_frac_mean",
    "catch_swi005_vs_point",
    "catch_swi010_vs_point",
]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: list[str],
    label: str = TARGET,
) -> dict:
    feats = [f for f in features if f in df_train.columns and f in df_val.columns]
    X_tr = df_train[feats].to_numpy(dtype=float)
    y_tr = df_train[label].to_numpy(dtype=float)
    X_val = df_val[feats].to_numpy(dtype=float)
    y_val = df_val[label].to_numpy(dtype=float)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)

    stage_val = df_val["stage_at_issue_m"].to_numpy(dtype=float)
    stage_true = df_val["target_stage_next_day_m"].to_numpy(dtype=float)
    stage_pred = stage_val + preds

    mae_delta = float(np.mean(np.abs(preds - y_val)))
    mae_stage = float(np.mean(np.abs(stage_pred - stage_true)))
    rmse_stage = float(np.sqrt(np.mean((stage_pred - stage_true) ** 2)))

    # Persistence baseline
    pers_mae = float(np.mean(np.abs(stage_val - stage_true)))

    return {
        "n_features": len(feats),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "val_mae_delta_m": round(mae_delta, 4),
        "val_mae_stage_m": round(mae_stage, 4),
        "val_rmse_stage_m": round(rmse_stage, 4),
        "persistence_mae_m": round(pers_mae, 4),
    }


def main() -> None:
    print("=" * 65)
    print("Script 24 – Expanded Dataset & Temporal Ingestion Benchmark (Upgrade 3)")
    print("=" * 65)

    # 1. Load restricted dataset (319 TRAIN, 186 VAL)
    print("\n[1] Loading restricted dataset (Phase 17/Upgrade 2 baseline) ...")
    df_restr, restr_info = load_project()
    if CATCH_CSV.exists():
        catch_tbl = pd.read_csv(CATCH_CSV, dtype={"station_id": str})
        df_restr = merge_catchment_features(df_restr, catch_tbl, swi_lag_days=2)
        df_restr["catch_swi005_minus_swi010"] = df_restr["catch_swi005_mean"] - df_restr["catch_swi010_mean"]
        df_restr["catch_wet_frac_mean"] = (df_restr["catch_swi005_wet_frac"] + df_restr["catch_swi010_wet_frac"]) / 2.0
        df_restr["catch_oversat_frac_mean"] = (df_restr["catch_swi005_oversat_frac"] + df_restr["catch_swi010_oversat_frac"]) / 2.0
        df_restr["catch_swi005_vs_point"] = df_restr["catch_swi005_mean"] - df_restr["swi005_d1"]
        df_restr["catch_swi010_vs_point"] = df_restr["catch_swi010_mean"] - df_restr["swi010_d1"]

    train_restr = df_restr[df_restr["chronological_split"] == "TRAIN"].copy()
    val_restr = df_restr[df_restr["chronological_split"] == "VALIDATION"].copy()
    print(f"    Restricted: TRAIN = {len(train_restr)}, VAL = {len(val_restr)}")

    # 2. Load expanded dataset (809 TRAIN, 269 VAL)
    print("\n[2] Loading expanded dataset (Upgrade 3) ...")
    df_exp, exp_meta = load_expanded_project(include_antecedent=True)
    train_exp = df_exp[df_exp["chronological_split"] == "TRAIN"].copy()
    val_exp = df_exp[df_exp["chronological_split"] == "VALIDATION"].copy()
    print(f"    Expanded  : TRAIN = {len(train_exp)} (+{len(train_exp)-len(train_restr)} rows / +{100*(len(train_exp)-len(train_restr))/len(train_restr):.1f}%), VAL = {len(val_exp)}")

    # Subsets of training data
    train_sar516 = train_exp[train_exp["sar_available_flag"] == 1].copy()
    val_sar217 = val_exp[val_exp["sar_available_flag"] == 1].copy()
    print(f"    Real SAR available : TRAIN = {len(train_sar516)} (+{len(train_sar516)-len(train_restr)} rows / +{100*(len(train_sar516)-len(train_restr))/len(train_restr):.1f}% over restricted)")

    # Define feature sets
    feat_baseline = BASELINE_FEATURES
    feat_expanded_sar = BASELINE_FEATURES + CATCHMENT_FEATURES
    feat_full_upgrade3 = BASELINE_FEATURES + CATCHMENT_FEATURES + ANTECEDENT_COLS
    feat_all_modalities = BASELINE_FEATURES + AVAILABILITY_COLS + CATCHMENT_FEATURES + ANTECEDENT_COLS

    results = {}

    # -----------------------------------------------------------------------
    # Part 1: Apples-to-apples comparison on Common Validation Set (186 rows)
    # -----------------------------------------------------------------------
    print("\n[3] Benchmarking on Common Validation Set (186 rows) ...")
    results["A_Restricted_Train319"] = evaluate(train_restr, val_restr, feat_baseline)
    results["B_Expanded_SAR_Train516"] = evaluate(train_sar516, val_restr, feat_expanded_sar)
    results["C_Full_Upgrade3_Train516"] = evaluate(train_sar516, val_restr, feat_full_upgrade3)
    results["D_AllModalities_Train809"] = evaluate(train_exp, val_restr, feat_all_modalities)

    # -----------------------------------------------------------------------
    # Part 2: Complete validation across available SAR scenes (217 rows) & full 269
    # -----------------------------------------------------------------------
    print("[4] Benchmarking on Expanded Validation Sets ...")
    results["C_Full_Upgrade3_ValSAR217"] = evaluate(train_sar516, val_sar217, feat_full_upgrade3)
    results["D_AllModalities_FullVal269"] = evaluate(train_exp, val_exp, feat_all_modalities)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("Benchmark Results (Next-Day River Stage Forecast)")
    print("=" * 75)
    print(f"{'Variant':<28} {'Feats':>5} {'Train':>5} {'Val':>5} "
          f"{'MAE (m)':>10} {'RMSE (m)':>10} {'Persist (m)':>12}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<28} {r['n_features']:>5} {r['n_train']:>5} {r['n_val']:>5} "
              f"{r['val_mae_stage_m']:>10.4f} {r['val_rmse_stage_m']:>10.4f} {r['persistence_mae_m']:>12.4f}")

    # Calculate gain from expanding dataset
    base_mae = results["A_Restricted_Train319"]["val_mae_stage_m"]
    sar_mae = results["B_Expanded_SAR_Train516"]["val_mae_stage_m"]
    up3_mae = results["C_Full_Upgrade3_Train516"]["val_mae_stage_m"]
    gain_b = (base_mae - sar_mae) / base_mae * 100
    gain_c = (base_mae - up3_mae) / base_mae * 100

    print("\n" + "=" * 75)
    print(f"[Upgrade 3] Expanded Real SAR (319 -> 516 rows) gain   : {gain_b:+.2f}%")
    print(f"[Upgrade 3] Full Upgrade 3 (516 rows + Catchment + 14d) : {gain_c:+.2f}%")
    print("=" * 75)


    # Save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Script 24] Benchmark saved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
