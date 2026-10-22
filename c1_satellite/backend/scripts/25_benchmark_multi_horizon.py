#!/usr/bin/env python
"""Script 25 – Benchmark Multi-Horizon Quantile Forecasting (Upgrade 4).

Evaluates 12h, 24h, 48h, and 72h river stage forecast accuracy and calibrated
quantile prediction intervals (P10, P50, P90) on the validation split.

Metrics per horizon:
  - MAE & RMSE of median prediction (P50)
  - Persistence benchmark comparison at each lead time
  - Empirical coverage of the 80% prediction interval [P10, P90]
  - Mean interval width (uncertainty envelope growth over lead time)

Run from project root:
    .venv\\Scripts\\python c1_satellite/scripts/25_benchmark_multi_horizon.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c1_satellite.src.improvements.dataset_expansion import (
    load_expanded_project,
    ANTECEDENT_COLS,
    AVAILABILITY_COLS,
)
from c1_satellite.src.improvements.multi_horizon import (
    add_multi_horizon_targets,
    MultiHorizonQuantileModel,
    HORIZONS,
)
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


OUTPUT_JSON = ROOT / "c1_satellite" / "outputs" / "multi_horizon_benchmark_results.json"


def main() -> None:
    print("=" * 75)
    print("Script 25 – Multi-Horizon River Stage Forecasting & Quantile Uncertainty (Upgrade 4)")
    print("=" * 75)

    # 1. Load expanded dataset with satellite and catchment features
    print("\n[1] Loading dataset with satellite, catchment, and antecedent features ...")
    df, meta = load_expanded_project(include_antecedent=True)
    print(f"    Loaded {len(df)} records across {meta['n_stations']} stations.")

    # 2. Add multi-horizon future targets (12h, 24h, 48h, 72h)
    print("\n[2] Constructing multi-horizon forecast targets (12h, 24h, 48h, 72h) ...")
    df_horizons = add_multi_horizon_targets(df)

    for h in HORIZONS:
        valid_count = df_horizons[f"target_stage_{h}_m"].notna().sum()
        print(f"    Horizon +{h:<4}: {valid_count} valid targets ({100*valid_count/len(df_horizons):.1f}%)")

    # Filter to real SAR-available rows for cleanest satellite learning
    train_df = df_horizons[(df_horizons["chronological_split"] == "TRAIN") & (df_horizons["sar_available_flag"] == 1)].copy()
    val_df = df_horizons[(df_horizons["chronological_split"] == "VALIDATION") & (df_horizons["sar_available_flag"] == 1)].copy()

    print(f"\n[3] Model splits (real SAR available): TRAIN = {len(train_df)}, VALIDATION = {len(val_df)}")

    # 3. Features
    feature_cols = BASELINE_FEATURES + CATCHMENT_FEATURES + ANTECEDENT_COLS
    print(f"    Feature vector length: {len(feature_cols)} features")

    # 4. Train multi-horizon quantile models (P10, P50, P90)
    print("\n[4] Training Multi-Horizon Quantile Forecaster (P10, P50, P90 across 4 horizons) ...")
    model = MultiHorizonQuantileModel(
        horizons=HORIZONS,
        quantiles=[0.10, 0.50, 0.90],
        n_estimators=180,
        max_depth=4,
        random_state=42,
    )
    model.fit(train_df, feature_cols)
    print("    Fitting complete for all lead times and quantiles.")

    # 5. Evaluate on pristine VALIDATION split
    print("\n[5] Evaluating probabilistic forecasts on VALIDATION split ...")
    val_metrics = model.evaluate(val_df)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Multi-Horizon Performance & Uncertainty Verification (VALIDATION Split)")
    print("=" * 80)
    print(f"{'Horizon':<10} {'Samples':>8} {'P50 MAE (m)':>12} {'RMSE (m)':>10} "
          f"{'Persist (m)':>12} {'80% Coverage':>14} {'Interval Width':>16}")
    print("-" * 80)

    for h in HORIZONS:
        m = val_metrics[h]
        cov_str = f"{100*m['coverage_p10_p90']:.1f}%"
        width_str = f"{m['mean_interval_width_m']:.4f} m"
        print(f"+{h:<9} {m['n_samples']:>8} {m['mae_stage_p50_m']:>12.4f} {m['rmse_stage_p50_m']:>10.4f} "
              f"{m['persistence_mae_m']:>12.4f} {cov_str:>14} {width_str:>16}")

    # Check uncertainty widening
    widths = [val_metrics[h]["mean_interval_width_m"] for h in HORIZONS]
    is_widening = all(widths[i] <= widths[i+1] for i in range(len(widths)-1))

    print("\n" + "=" * 80)
    print(f"[Hydrological Verification] Uncertainty envelope widens with lead time: {is_widening}")
    print(f"  +12h interval width : {widths[0]:.4f} m")
    print(f"  +24h interval width : {widths[1]:.4f} m")
    print(f"  +48h interval width : {widths[2]:.4f} m")
    print(f"  +72h interval width : {widths[3]:.4f} m")
    print("=" * 80)

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\n[Script 25] Results saved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
