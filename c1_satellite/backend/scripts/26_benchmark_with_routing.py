#!/usr/bin/env python
"""Script 26 – Benchmark Physics-Guided River Routing Integration (Upgrade 5).

Evaluates whether augmenting satellite and stage features with upstream-to-downstream
river network topological routing and kinematic wave propagation improves next-day
forecast accuracy, especially for high-risk downstream gauges (Nagalagam Street,
Hanwella, Putupaula, Manampitiya, Baddegama).

Variants:
  A. Isolated Baseline    : Satellite + Catchment features (no river routing)
  B. Kinematic Routing    : A + Upstream surge, acceleration, and kinematic wave potential
  C. Physics-Constrained : B + Physics-guided mass conservation clamping

Run from project root:
    .venv\\Scripts\\python c1_satellite/scripts/26_benchmark_with_routing.py
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
from c1_satellite.src.improvements.dataset_expansion import (
    load_expanded_project,
    ANTECEDENT_COLS,
)
from c1_satellite.src.improvements.river_routing import (
    add_river_routing_features,
    apply_physics_guided_constraints,
    ROUTING_FEATURE_COLS,
    RIVER_ROUTING_NETWORK,
)

OUTPUT_JSON = ROOT / "c1_satellite" / "outputs" / "routing_benchmark_results.json"
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


def evaluate_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features: list[str],
    apply_physics: bool = False,
    downstream_only: bool = False,
) -> dict:
    if downstream_only:
        downstream_sids = set(RIVER_ROUTING_NETWORK.keys())
        eval_val = df_val[df_val["station_id"].str.lower().isin(downstream_sids)].copy()
    else:
        eval_val = df_val.copy()

    feats = [f for f in features if f in df_train.columns and f in eval_val.columns]
    X_tr = df_train[feats].to_numpy(dtype=float)
    y_tr = df_train[TARGET].to_numpy(dtype=float)
    X_val = eval_val[feats].to_numpy(dtype=float)
    y_val = eval_val[TARGET].to_numpy(dtype=float)

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)

    stage_val = eval_val["stage_at_issue_m"].to_numpy(dtype=float)
    stage_true = eval_val["target_stage_next_day_m"].to_numpy(dtype=float)
    stage_pred = stage_val + preds

    if apply_physics:
        up_chg = eval_val["upstream_stage_change_1d_m"].to_numpy(dtype=float)
        hw = eval_val["is_headwater_station"].to_numpy(dtype=float)
        stage_pred = apply_physics_guided_constraints(stage_val, stage_pred, up_chg, hw)

    mae_delta = float(np.mean(np.abs(preds - y_val)))
    mae_stage = float(np.mean(np.abs(stage_pred - stage_true)))
    rmse_stage = float(np.sqrt(np.mean((stage_pred - stage_true) ** 2)))
    pers_mae = float(np.mean(np.abs(stage_val - stage_true)))

    return {
        "n_features": len(feats),
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "val_mae_stage_m": round(mae_stage, 4),
        "val_rmse_stage_m": round(rmse_stage, 4),
        "persistence_mae_m": round(pers_mae, 4),
    }


def main() -> None:
    print("=" * 75)
    print("Script 26 – Physics-Guided River Routing Integration (Upgrade 5)")
    print("=" * 75)

    # 1. Load expanded dataset with satellite and catchment features
    print("\n[1] Loading dataset with satellite, catchment, and antecedent features ...")
    df, meta = load_expanded_project(include_antecedent=True)

    # 2. Add river routing topological features
    print("\n[2] Computing topological river routing & kinematic wave features ...")
    df = add_river_routing_features(df)

    n_downstream = int((df["is_headwater_station"] == 0.0).sum())
    print(f"    Downstream station-days connected to upstream parent: {n_downstream} ({100*n_downstream/len(df):.1f}%)")

    # Filter to real SAR available rows for cleanest training
    train = df[(df["chronological_split"] == "TRAIN") & (df["sar_available_flag"] == 1)].copy()
    val = df[(df["chronological_split"] == "VALIDATION") & (df["sar_available_flag"] == 1)].copy()
    print(f"    Split sizes (real SAR): TRAIN = {len(train)}, VALIDATION = {len(val)}")

    feat_baseline = BASELINE_FEATURES + CATCHMENT_FEATURES + ANTECEDENT_COLS
    feat_routing = BASELINE_FEATURES + CATCHMENT_FEATURES + ANTECEDENT_COLS + ROUTING_FEATURE_COLS

    results = {}

    # -----------------------------------------------------------------------
    # Part 1: Overall validation across all 39 stations
    # -----------------------------------------------------------------------
    print("\n[3] Benchmarking across ALL gauges (Validation split) ...")
    results["All_A_Isolated"] = evaluate_model(train, val, feat_baseline)
    results["All_B_KinematicRouting"] = evaluate_model(train, val, feat_routing)
    results["All_C_PhysicsConstrained"] = evaluate_model(train, val, feat_routing, apply_physics=True)

    # -----------------------------------------------------------------------
    # Part 2: High-risk downstream flood outlets only (Nagalagam Street, Hanwella, Putupaula, etc.)
    # -----------------------------------------------------------------------
    print("\n[4] Benchmarking specifically on DOWNSTREAM flood-risk gauges ...")
    results["Downstream_A_Isolated"] = evaluate_model(train, val, feat_baseline, downstream_only=True)
    results["Downstream_B_KinematicRouting"] = evaluate_model(train, val, feat_routing, downstream_only=True)
    results["Downstream_C_PhysicsConstrained"] = evaluate_model(train, val, feat_routing, apply_physics=True, downstream_only=True)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("Benchmark Results (VALIDATION Split, 24h River Stage Forecast)")
    print("=" * 75)
    print(f"{'Variant':<32} {'Feats':>5} {'Val':>5} {'MAE (m)':>10} {'RMSE (m)':>10} {'Persist (m)':>12}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<32} {r['n_features']:>5} {r['n_val']:>5} "
              f"{r['val_mae_stage_m']:>10.4f} {r['val_rmse_stage_m']:>10.4f} {r['persistence_mae_m']:>12.4f}")

    # Gain on downstream gauges
    down_iso_mae = results["Downstream_A_Isolated"]["val_mae_stage_m"]
    down_rout_mae = results["Downstream_B_KinematicRouting"]["val_mae_stage_m"]
    down_gain = (down_iso_mae - down_rout_mae) / down_iso_mae * 100

    print("\n" + "=" * 75)
    print(f"[Upgrade 5] Downstream river gauge accuracy gain from routing: {down_gain:+.2f}%")
    print(f"  Downstream isolated MAE : {down_iso_mae:.4f} m")
    print(f"  Downstream routed MAE   : {down_rout_mae:.4f} m")
    print("=" * 75)

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Script 26] Results saved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
