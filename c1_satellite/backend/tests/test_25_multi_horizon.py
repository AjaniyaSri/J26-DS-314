"""Unit tests for Upgrade 4 – multi_horizon.py

Tests:
  1. Target construction creates columns for 12h, 24h, 48h, 72h.
  2. Multi-horizon targets maintain arithmetic consistency (stage = now + delta).
  3. Quantile predictions strictly preserve monotonic ordering (P10 <= P50 <= P90).
  4. Prediction interval width (P90 - P10) is non-negative everywhere.
  5. End-to-end fit and evaluation run on mock data without errors.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c1_satellite.src.improvements.multi_horizon import (
    add_multi_horizon_targets,
    MultiHorizonQuantileModel,
    HORIZONS,
)


def _make_mock_station_days() -> pd.DataFrame:
    records = []
    dates = pd.date_range("2025-11-17", periods=10, freq="D").strftime("%Y-%m-%d")
    for sid in ["station_A", "station_B"]:
        base_stage = 2.0 if sid == "station_A" else 5.0
        for i, d in enumerate(dates):
            stage = base_stage + 0.1 * i + 0.05 * np.sin(i)
            records.append({
                "station_id": sid,
                "forecast_date": d,
                "stage_at_issue_m": stage,
                "stage_lag1_m": stage - 0.08,
                "stage_lag2_m": stage - 0.15,
                "stage_change_1d_m": 0.08,
                "stage_change_prev_1d_m": 0.07,
                "stage_acceleration_m": 0.01,
                "feature_x1": np.random.randn(),
                "feature_x2": np.random.randn(),
            })
    return pd.DataFrame(records)


def test_add_multi_horizon_targets_creates_columns():
    df = _make_mock_station_days()
    out = add_multi_horizon_targets(df)
    for h in HORIZONS:
        assert f"target_stage_{h}_m" in out.columns, f"Missing target_stage_{h}_m"
        assert f"target_delta_{h}_m" in out.columns, f"Missing target_delta_{h}_m"


def test_target_arithmetic_consistency():
    df = _make_mock_station_days()
    out = add_multi_horizon_targets(df)
    for h in HORIZONS:
        s_now = out["stage_at_issue_m"]
        delta = out[f"target_delta_{h}_m"]
        s_future = out[f"target_stage_{h}_m"]
        mask = s_future.notna()
        diff = (s_now[mask] + delta[mask]) - s_future[mask]
        assert np.allclose(diff, 0.0, atol=1e-5), f"Arithmetic mismatch for horizon {h}"


def test_quantile_ordering_guarantee():
    df = _make_mock_station_days()
    df_targets = add_multi_horizon_targets(df)

    feats = ["stage_at_issue_m", "stage_lag1_m", "stage_change_1d_m", "feature_x1", "feature_x2"]
    model = MultiHorizonQuantileModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(df_targets, feats)

    preds = model.predict_quantiles(df_targets)

    for h in HORIZONS:
        p10 = preds[h]["stage_p10"]
        p50 = preds[h]["stage_p50"]
        p90 = preds[h]["stage_p90"]

        assert (p10 <= p50 + 1e-6).all(), f"Quantile crossing P10 > P50 in horizon {h}"
        assert (p50 <= p90 + 1e-6).all(), f"Quantile crossing P50 > P90 in horizon {h}"


def test_interval_width_non_negative():
    df = _make_mock_station_days()
    df_targets = add_multi_horizon_targets(df)

    feats = ["stage_at_issue_m", "stage_lag1_m", "stage_change_1d_m", "feature_x1", "feature_x2"]
    model = MultiHorizonQuantileModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(df_targets, feats)

    preds = model.predict_quantiles(df_targets)
    for h in HORIZONS:
        w = preds[h]["interval_width"]
        assert (w >= -1e-6).all(), f"Negative interval width detected in horizon {h}"


def test_evaluation_metrics_produced():
    df = _make_mock_station_days()
    df_targets = add_multi_horizon_targets(df)

    feats = ["stage_at_issue_m", "stage_lag1_m", "stage_change_1d_m", "feature_x1", "feature_x2"]
    model = MultiHorizonQuantileModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(df_targets, feats)

    eval_results = model.evaluate(df_targets)
    for h in HORIZONS:
        assert h in eval_results, f"Missing metrics for horizon {h}"
        assert "mae_stage_p50_m" in eval_results[h]
        assert "coverage_p10_p90" in eval_results[h]
        assert "mean_interval_width_m" in eval_results[h]
        assert 0.0 <= eval_results[h]["coverage_p10_p90"] <= 1.0
