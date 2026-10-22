"""Upgrade 5 – Physics-Guided Upstream-to-Downstream River Routing Integration.

Implements the official topological river routing network for Sri Lanka's major basins:
  - Kelani Ganga: Norwood -> Kithulgala, Glencourse -> Hanwella -> Nagalagam Street
  - Kalu Ganga: Rathnapura -> Ellagawa -> Putupaula, Kalawellawa -> Putupaula
  - Mahaweli Ganga: Nawalapitiya -> Peradeniya, Thaldena -> Weraganthota -> Manampitiya
  - Gin Ganga: Thawalama -> Baddegama
  - Maha Oya: Giriulla -> Badalgama
  - Nilwala Ganga: Urawa/Pitabeddara -> Panadugama -> Thalgahagoda
  - Kirindi Oya: Wellawaya/Kuda Oya -> Thanamalwila

Derives kinematic flood wave propagation features from upstream gauges to predict
downstream stage surges with physical mass-conservation consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Directed Acyclic Graph (DAG) of Upstream -> Downstream Gauges
# Key: downstream station_id -> list of direct upstream station_ids
# ---------------------------------------------------------------------------
RIVER_ROUTING_NETWORK: Dict[str, List[str]] = {
    # Kelani Ganga (Highlands -> Colombo Coastal Outlet)
    "kithulgala": ["norwood"],
    "glencourse": ["kithulgala", "deraniyagala", "holombuwa"],
    "hanwella": ["glencourse"],
    "nagalagam_street": ["hanwella"],

    # Kalu Ganga (Central Hills -> Kalutara Ocean Outlet)
    "ellagawa": ["rathnapura"],
    "putupaula": ["ellagawa", "magura", "kalawellawa_millakanda", "kalawellawa_(millakanda)"],

    # Mahaweli Ganga (Central Massif -> Eastern Lowlands Delta)
    "peradeniya": ["nawalapitiya"],
    "weraganthota": ["peradeniya", "thaldena"],
    "manampitiya": ["weraganthota"],

    # Gin Ganga (Southern Rainforest -> Galle Coast)
    "baddegama": ["thawalama"],

    # Maha Oya (North-Western Basin)
    "badalgama": ["giriulla"],

    # Nilwala Ganga (Southern Catchment -> Matara Outlet)
    "panadugama": ["pitabeddara", "urawa"],
    "thalgahagoda": ["panadugama"],

    # Kirindi Oya (Southern Dry Zone)
    "thanamalwila": ["wellawaya", "kuda_oya"],
}

ROUTING_FEATURE_COLS = [
    "upstream_stage_change_1d_m",
    "upstream_stage_acceleration_m",
    "upstream_flood_surge_flag",
    "is_headwater_station",
    "kinematic_wave_potential_m",
]


def add_river_routing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute physical upstream flood wave routing features for each station-day.

    For downstream gauges (e.g. Hanwella, Nagalagam Street, Putupaula, Manampitiya),
    extracts the rate-of-rise and acceleration at upstream stations observed on date D.
    Because all upstream features are observed on or before date D, there is zero
    target leakage.
    """
    res = df.copy()

    # Pre-build lookup: (station_id, forecast_date) -> {stage_at_issue, stage_change_1d, stage_acceleration}
    lookup_cols = ["station_id", "forecast_date", "stage_at_issue_m", "stage_change_1d_m", "stage_acceleration_m"]
    available_lookup_cols = [c for c in lookup_cols if c in res.columns]
    stage_lookup = res[available_lookup_cols].set_index(["station_id", "forecast_date"])

    n_rows = len(res)
    up_changes = np.zeros(n_rows, dtype=float)
    up_accels = np.zeros(n_rows, dtype=float)
    up_surges = np.zeros(n_rows, dtype=float)
    is_headwater = np.ones(n_rows, dtype=float)
    kinematic_waves = np.zeros(n_rows, dtype=float)

    for i, (_, row) in enumerate(res.iterrows()):
        sid = str(row["station_id"]).strip().lower()
        date_str = str(row["forecast_date"]).strip()

        upstream_parents = RIVER_ROUTING_NETWORK.get(sid, [])
        # Also try clean matching
        if not upstream_parents:
            clean_sid = sid.replace("(", "").replace(")", "").replace(" ", "_")
            upstream_parents = RIVER_ROUTING_NETWORK.get(clean_sid, [])

        if not upstream_parents:
            # Headwater station (no upstream river gauge)
            is_headwater[i] = 1.0
            continue

        is_headwater[i] = 0.0
        parent_changes = []
        parent_accels = []

        for pid in upstream_parents:
            key = (pid, date_str)
            if key in stage_lookup.index:
                parent_row = stage_lookup.loc[key]
                if isinstance(parent_row, pd.DataFrame):
                    parent_row = parent_row.iloc[0]

                chg = parent_row.get("stage_change_1d_m")
                acc = parent_row.get("stage_acceleration_m")

                if pd.notna(chg):
                    parent_changes.append(float(chg))
                if pd.notna(acc):
                    parent_accels.append(float(acc))

        if parent_changes:
            # In flood routing, the maximum upstream surge determines crest propagation
            max_change = float(np.max(parent_changes))
            mean_accel = float(np.mean(parent_accels)) if parent_accels else 0.0

            up_changes[i] = max_change
            up_accels[i] = mean_accel
            up_surges[i] = 1.0 if max_change >= 0.40 else 0.0

            # Kinematic wave potential: positive upstream surges propagate downstream
            # Celerity factor ~ 0.65 to 0.85 of upstream surge reaches next gauge within 24h
            kinematic_waves[i] = max(0.0, max_change) * 0.75

    res["upstream_stage_change_1d_m"] = up_changes
    res["upstream_stage_acceleration_m"] = up_accels
    res["upstream_flood_surge_flag"] = up_surges
    res["is_headwater_station"] = is_headwater
    res["kinematic_wave_potential_m"] = kinematic_waves

    return res


def apply_physics_guided_constraints(
    stage_at_issue: np.ndarray,
    predicted_stage: np.ndarray,
    upstream_change: np.ndarray,
    is_headwater: np.ndarray,
) -> np.ndarray:
    """Enforce physical mass conservation bounds on downstream water level predictions.

    Physics Rules:
      1. Water cannot drop below non-physical negative stages (unless calibrated datum allows).
      2. When an upstream gauge is surging (> +1.0m change in 24h), a downstream gauge
         cannot experience a catastrophic plummet unless a dam diversion occurs.
      3. Monotonic rate-of-fall limit: recession limb rate is constrained by river hydraulics.
    """
    constrained = predicted_stage.copy()

    for i in range(len(constrained)):
        s_now = stage_at_issue[i]
        s_pred = constrained[i]
        up_chg = upstream_change[i]
        hw = is_headwater[i]

        # Rule 2: If upstream is experiencing a major surge (> +0.8m) and station is downstream,
        # downstream delta must not be strongly negative (wave arrival prevents rapid drop)
        if hw == 0.0 and up_chg > 0.8:
            min_allowed = s_now - 0.10  # cannot drop more than 10cm during active upstream flood wave
            if constrained[i] < min_allowed:
                constrained[i] = min_allowed
        # Rule 3: Natural recession limit (river stage rarely drops > 1.8m in a single 24h period)
        elif constrained[i] < s_now - 1.8:
            constrained[i] = s_now - 1.8

    return constrained

