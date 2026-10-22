#!/usr/bin/env python
"""Script 23 – Benchmark Catchment SWI Features (Upgrade 2).

Evaluates whether adding basin-averaged upstream catchment SWI features
improves 24-hour river stage prediction over the Phase 17 tabular baseline.

Pipeline variants compared:
  A. Baseline        – tabular features only (Stage history + point SWI + SAR)
  B. Catchment SWI   – baseline + 9 new catch_swi* columns
  C. Full Upgrade2   – Baseline + Catchment SWI + derived catch features

Metric: Mean Absolute Error (MAE) on the pristine VALIDATION split.

Run from project root:
    .venv\\Scripts\\python c1_satellite/scripts/23_benchmark_with_catchment.py
"""
from __future__ import annotations

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
    STAGE_CORE, STAGE_HIST, SAR_BASE, SAR_HIST,
    SWI_NOW, SWI_HIST, SAR_NOW_ENG, SAR_HIST_ENG, SWI_NOW_ENG, SWI_HIST_ENG,
)
from c1_satellite.src.improvements.catchment_swi import (
    merge_catchment_features,
    CATCH_NOW_COLS,
)

CATCH_CSV = ROOT / "c1_satellite" / "outputs" / "c1_catchment_swi_features.csv"

TARGET = "target_delta_stage_m"

BASELINE_FEATURES = (
    STAGE_CORE + STAGE_HIST +
    SAR_BASE + SAR_HIST + SAR_NOW_ENG + SAR_HIST_ENG +
    SWI_NOW + SWI_HIST + SWI_NOW_ENG + SWI_HIST_ENG
)

CATCHMENT_DERIVED = [
    "catch_swi005_minus_swi010",       # spatial moisture contrast
    "catch_wet_frac_mean",             # average of both wet fractions
    "catch_oversat_frac_mean",         # avg fraction fully waterlogged (SWI>100)
    "catch_swi005_vs_point",           # basin average vs point reading
    "catch_swi010_vs_point",
]


def build_sklearn_pipe(n_features: int) -> Pipeline:
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


def evaluate(df_train: pd.DataFrame, df_val: pd.DataFrame,
             features: list[str], label: str = TARGET) -> dict:
    feats = [f for f in features if f in df_train.columns]
    X_tr = df_train[feats].to_numpy(dtype=float)
    y_tr = df_train[label].to_numpy(dtype=float)
    X_val = df_val[feats].to_numpy(dtype=float)
    y_val = df_val[label].to_numpy(dtype=float)

    pipe = build_sklearn_pipe(X_tr.shape[1])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)

    stage_val = df_val["stage_at_issue_m"].to_numpy(dtype=float)
    stage_true = df_val["target_stage_next_day_m"].to_numpy(dtype=float)
    stage_pred = stage_val + preds

    mae_delta = float(np.mean(np.abs(preds - y_val)))
    mae_stage = float(np.mean(np.abs(stage_pred - stage_true)))
    return {
        "n_features": len(feats),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "val_mae_delta_m": round(mae_delta, 4),
        "val_mae_stage_m": round(mae_stage, 4),
    }


def add_derived_catch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["catch_swi005_minus_swi010"] = df["catch_swi005_mean"] - df["catch_swi010_mean"]
    df["catch_wet_frac_mean"] = (df["catch_swi005_wet_frac"] + df["catch_swi010_wet_frac"]) / 2.0
    df["catch_oversat_frac_mean"] = (df["catch_swi005_oversat_frac"] + df["catch_swi010_oversat_frac"]) / 2.0
    df["catch_swi005_vs_point"] = df["catch_swi005_mean"] - df["swi005_d1"]
    df["catch_swi010_vs_point"] = df["catch_swi010_mean"] - df["swi010_d1"]
    return df



def main() -> None:
    print("=" * 60)
    print("Script 23 – Catchment SWI Benchmark (Upgrade 2)")
    print("=" * 60)

    # Load base project data
    print("\n[1] Loading project data ...")
    eligible, info = load_project()
    print(f"    Total eligible rows: {info['eligible_n']}")
    for k, v in info["eligible_by_split"].items():
        print(f"      {k}: {v}")

    # Load catchment table
    if not CATCH_CSV.exists():
        print(f"\n[!] Catchment CSV not found at {CATCH_CSV}")
        print("    Run Script 22 first: .venv\\Scripts\\python c1_satellite/scripts/22_extract_catchment_swi_features.py")
        sys.exit(1)

    catch_table = pd.read_csv(CATCH_CSV, dtype={"station_id": str})
    print(f"\n[2] Catchment SWI table loaded: {len(catch_table)} rows")

    # Merge catchment features into feature matrix
    enriched = merge_catchment_features(eligible, catch_table, swi_lag_days=2)
    enriched = add_derived_catch(enriched)

    # Split
    train = enriched[enriched["chronological_split"] == "TRAIN"].copy()
    val = enriched[enriched["chronological_split"] == "VALIDATION"].copy()
    print(f"\n[3] Split sizes — TRAIN: {len(train)}, VALIDATION: {len(val)}")

    # Count rows with complete catchment features
    n_train_catch = train[CATCH_NOW_COLS].notna().all(axis=1).sum()
    n_val_catch = val[CATCH_NOW_COLS].notna().all(axis=1).sum()
    print(f"    Rows with complete catchment SWI — TRAIN: {n_train_catch}, VAL: {n_val_catch}")

    # -----------------------------------------------------------------------
    # Evaluate 3 variants
    # -----------------------------------------------------------------------
    results = {}

    print("\n[4] Variant A: Baseline (tabular only) ...")
    results["A_Baseline"] = evaluate(train, val, BASELINE_FEATURES)

    print("[5] Variant B: Baseline + Catchment SWI columns ...")
    b_features = BASELINE_FEATURES + CATCH_NOW_COLS
    results["B_CatchSWI"] = evaluate(train, val, b_features)

    print("[6] Variant C: Baseline + Catchment SWI + derived features ...")
    c_features = BASELINE_FEATURES + CATCH_NOW_COLS + CATCHMENT_DERIVED
    results["C_Full_Upgrade2"] = evaluate(train, val, c_features)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results (VALIDATION split, 24h forecast)")
    print("=" * 60)
    print(f"{'Variant':<22} {'Features':>8} {'Train':>6} {'Val':>5} "
          f"{'MAE Delta (m)':>14} {'MAE Stage (m)':>14}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<22} {r['n_features']:>8} {r['n_train']:>6} {r['n_val']:>5} "
              f"{r['val_mae_delta_m']:>14.4f} {r['val_mae_stage_m']:>14.4f}")

    best = min(results, key=lambda k: results[k]["val_mae_stage_m"])
    baseline_mae = results["A_Baseline"]["val_mae_stage_m"]
    best_mae = results[best]["val_mae_stage_m"]
    improvement = (baseline_mae - best_mae) / baseline_mae * 100

    print("\n" + "=" * 60)
    if best != "A_Baseline":
        print(f"[Upgrade 2] Best variant : {best}")
        print(f"[Upgrade 2] Improvement  : {improvement:+.2f}% over baseline")
    else:
        print("[Upgrade 2] Baseline is still best (catchment data may be sparse).")
        print("  Tip: run Script 22 first, or expand the ASCAT date range.")
    print("=" * 60)

    # Save results
    out = ROOT / "c1_satellite" / "outputs" / "catchment_benchmark_results.json"
    import json
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Script 23] Results saved -> {out}")


if __name__ == "__main__":
    main()
