"""Unit tests for Upgrade 5 – river_routing.py

Tests:
  1. Routing network contains all major Sri Lankan river basins.
  2. Headwater stations correctly identified (is_headwater_station == 1).
  3. Downstream stations (Nagalagam Street, Hanwella, Putupaula) correctly route upstream surges.
  4. Kinematic wave potential is non-negative and properly scaled.
  5. Physics-guided constraints prevent non-physical drops during upstream flood surges.
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

from c1_satellite.src.improvements.river_routing import (
    RIVER_ROUTING_NETWORK,
    ROUTING_FEATURE_COLS,
    add_river_routing_features,
    apply_physics_guided_constraints,
)


def _make_mock_river_network() -> pd.DataFrame:
    # 2-station network: Hanwella (upstream) -> Nagalagam Street (downstream)
    dates = ["2025-11-17", "2025-11-18"]
    records = [
        # Hanwella (upstream)
        {
            "station_id": "hanwella",
            "forecast_date": "2025-11-17",
            "stage_at_issue_m": 4.0,
            "stage_change_1d_m": 0.2,
            "stage_acceleration_m": 0.05,
        },
        {
            "station_id": "hanwella",
            "forecast_date": "2025-11-18",
            "stage_at_issue_m": 5.5,
            "stage_change_1d_m": 1.5,  # Surge of +1.5m!
            "stage_acceleration_m": 1.3,
        },
        # Nagalagam Street (downstream)
        {
            "station_id": "nagalagam_street",
            "forecast_date": "2025-11-17",
            "stage_at_issue_m": 1.0,
            "stage_change_1d_m": 0.1,
            "stage_acceleration_m": 0.02,
        },
        {
            "station_id": "nagalagam_street",
            "forecast_date": "2025-11-18",
            "stage_at_issue_m": 1.4,
            "stage_change_1d_m": 0.4,
            "stage_acceleration_m": 0.3,
        },
    ]
    return pd.DataFrame(records)


def test_routing_network_topology():
    assert "nagalagam_street" in RIVER_ROUTING_NETWORK
    assert "hanwella" in RIVER_ROUTING_NETWORK["nagalagam_street"]
    assert "putupaula" in RIVER_ROUTING_NETWORK
    assert "manampitiya" in RIVER_ROUTING_NETWORK


def test_headwater_stations_flagged():
    df = _make_mock_river_network()
    routed = add_river_routing_features(df)

    hw_row = routed[routed["station_id"] == "hanwella"].iloc[0]
    assert hw_row["is_headwater_station"] == 0.0  # Hanwella has Glencourse upstream in full graph

    # Test pure headwater
    df_pure = pd.DataFrame([{
        "station_id": "norwood",
        "forecast_date": "2025-11-17",
        "stage_at_issue_m": 3.0,
        "stage_change_1d_m": 0.1,
        "stage_acceleration_m": 0.0,
    }])
    routed_pure = add_river_routing_features(df_pure)
    assert routed_pure["is_headwater_station"].iloc[0] == 1.0


def test_downstream_routes_upstream_surge():
    df = _make_mock_river_network()
    routed = add_river_routing_features(df)

    # On 2025-11-18, Hanwella had a +1.5m surge
    naga_18 = routed[(routed["station_id"] == "nagalagam_street") & (routed["forecast_date"] == "2025-11-18")].iloc[0]

    assert abs(naga_18["upstream_stage_change_1d_m"] - 1.5) < 1e-5
    assert naga_18["upstream_flood_surge_flag"] == 1.0
    assert naga_18["kinematic_wave_potential_m"] > 1.0


def test_physics_constraints_clamp_unphysical_drops():
    stage_now = np.array([5.0, 5.0])
    # Predicted stage tries to drop drastically to 2.0m during upstream flood surge
    pred_stage = np.array([2.0, 2.0])
    upstream_change = np.array([1.2, 0.0])  # Sample 0 has +1.2m surge upstream
    is_headwater = np.array([0.0, 1.0])

    constrained = apply_physics_guided_constraints(
        stage_now, pred_stage, upstream_change, is_headwater
    )

    # Sample 0: clamped because upstream has active surge (+1.2m)
    assert constrained[0] >= stage_now[0] - 0.10, "Should clamp drop during upstream surge"
    # Sample 1: clamped by natural recession limit (not drop more than 1.8m)
    assert constrained[1] >= stage_now[1] - 1.80
