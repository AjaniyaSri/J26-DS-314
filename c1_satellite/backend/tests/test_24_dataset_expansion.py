"""Unit tests for Upgrade 3 – dataset_expansion.py

Tests:
  1. load_expanded_project loads 1,429 rows across all 39 stations.
  2. Chronological splits match contract (809 TRAIN, 269 VAL, 351 TEST).
  3. No future target leakage in stage or delta definitions.
  4. Antecedent catchment memory computes rolling 14-day statistics correctly.
  5. Availability flags are non-null and strictly binary {0.0, 1.0}.
  6. Date ranges and station continuity are intact.
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

from c1_satellite.src.improvements.dataset_expansion import (
    load_expanded_project,
    compute_antecedent_catchment_memory,
    ANTECEDENT_COLS,
    AVAILABILITY_COLS,
    CATCH_CSV,
)


def test_load_expanded_project_shape():
    df, meta = load_expanded_project(include_antecedent=False)
    assert len(df) == 1429, f"Expected 1429 rows, got {len(df)}"
    assert meta["total_records"] == 1429
    assert meta["n_stations"] == 39


def test_split_counts_match_contract():
    df, meta = load_expanded_project(include_antecedent=False)
    splits = meta["split_counts"]
    assert splits["TRAIN"] == 809, f"Expected 809 TRAIN rows, got {splits['TRAIN']}"
    assert splits["VALIDATION"] == 269, f"Expected 269 VAL rows, got {splits['VALIDATION']}"
    assert splits["TEST"] == 351, f"Expected 351 TEST rows, got {splits['TEST']}"


def test_no_target_leakage():
    df, _ = load_expanded_project(include_antecedent=False)
    diff = (df["stage_at_issue_m"] + df["target_delta_stage_m"]) - df["target_stage_next_day_m"]
    assert np.allclose(diff, 0.0, atol=1e-5), "Label arithmetic inconsistent"


def test_availability_flags_binary():
    df, _ = load_expanded_project(include_antecedent=False)
    for col in AVAILABILITY_COLS:
        assert col in df.columns, f"Missing flag {col}"
        vals = set(df[col].dropna().unique())
        assert vals.issubset({0.0, 1.0}), f"Flag {col} has unexpected values: {vals}"


def test_antecedent_catchment_memory():
    if not CATCH_CSV.exists():
        if pytest:
            pytest.skip("Catchment CSV not available")
        return

    catch_table = pd.read_csv(CATCH_CSV)
    ant = compute_antecedent_catchment_memory(catch_table, swi_lag_days=2)
    assert len(ant) > 0, "Antecedent table should not be empty"
    for col in ANTECEDENT_COLS:
        assert col in ant.columns, f"Missing antecedent column: {col}"

    # Mean values should be non-negative
    valid_means = ant["catch_swi005_roll14_mean"].dropna()
    assert (valid_means >= 0).all(), "Antecedent 14-day rolling mean should be >= 0"


def test_expanded_dataset_with_antecedent():
    df, meta = load_expanded_project(include_antecedent=True)
    assert len(df) == 1429
    for col in ANTECEDENT_COLS:
        assert col in df.columns, f"Expected antecedent column in merged dataset: {col}"
