import numpy as np
import pandas as pd
import pytest

from src.baseline import persistence
from src.data_loader import load_config, load_dataset
from src.data_validator import require_valid, validate_dataset
from src.evaluator import metrics
from src.features import FEATURES, feature_matrix


@pytest.fixture
def raw():
    return load_dataset(load_config())


def test_exact_order_and_numerical_features(raw):
    frame, _ = require_valid(raw)
    X = feature_matrix(frame)
    assert tuple(X.columns) == FEATURES
    assert X.shape == (1314, 10)
    assert all(dtype.kind == "f" for dtype in X.dtypes)


def test_calendar_lags_report_coverage_without_overclaiming(raw):
    _, audit = require_valid(raw)
    report = audit.markdown()
    assert "stage_lag_1d_m: 1259 calendar references available; 55 unavailable" in report
    assert "stage_lag_2d_m: 1204 calendar references available; 110 unavailable" in report
    assert "cannot be independently verified" in report


@pytest.mark.parametrize("column", ["stage_lag_1d_m", "stage_lag_2d_m", "target_stage_next_day_m"])
def test_detects_wrong_calendar_reference(raw, column):
    # This date has all required reference dates in the original CSV.
    index = raw.index[(raw.station_id == "badalgama") & (raw.forecast_date == "2025-11-22")][0]
    raw.loc[index, column] = str(float(raw.loc[index, column]) + 0.123)
    assert any(f"Calendar/station mismatch in {column}" in x for x in validate_dataset(raw)[1].errors)


def test_detects_stage_change_mismatch(raw):
    raw.loc[0, "stage_change_1d_m"] = "999"
    assert any("Stage-change mismatch" in x for x in validate_dataset(raw)[1].errors)


def test_static_environment_must_be_constant_within_station(raw):
    raw.loc[0, "mean_elevation_m"] = "12345"
    assert any("Static feature varies" in x for x in validate_dataset(raw)[1].errors)


def test_persistence_uses_only_current_stage(raw):
    frame, _ = require_valid(raw)
    np.testing.assert_array_equal(persistence(frame), frame.stage_at_issue_m)
    frame["target_stage_next_day_m"] = 999
    np.testing.assert_array_equal(persistence(frame), frame.stage_at_issue_m)


def test_metric_definitions_and_signed_error():
    actual = metrics([0, 2, 4], [1, 2, 2])
    assert actual["n_samples"] == 3
    assert actual["mae_m"] == pytest.approx(1)
    assert actual["rmse_m"] == pytest.approx(np.sqrt(5 / 3))
    assert actual["mean_signed_error_m"] == pytest.approx(-1 / 3)


@pytest.mark.parametrize("observed,predicted", [([], []), ([1], [1, 2]), ([np.nan], [1])])
def test_metrics_reject_misalignment_and_nonfinite_values(observed, predicted):
    with pytest.raises(ValueError):
        metrics(observed, predicted)
