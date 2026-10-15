import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.model2_data import load_model2_config, validate_model2, weather_windows


@pytest.fixture(scope="module")
def sources():
    config = load_model2_config()
    return load_dataset(config), load_dataset({"dataset": config["model1_dataset"]})


def test_every_source_is_previous_day_and_exact_example_windows(sources):
    raw, _ = sources
    windows = weather_windows(raw)
    assert windows.source_age_days.eq(1).all()
    row = windows[windows.forecast_date.eq("2025-11-20")].iloc[0]
    assert row.c2_weather_source_date == "2025-11-19"
    assert row.rain3_start_inclusive == "2025-11-16"
    assert row.rain3_end_inclusive == "2025-11-18"
    assert row.rain7_start_inclusive == "2025-11-12"
    assert row.rain7_end_inclusive == "2025-11-18"
    assert row.rain14_start_inclusive == "2025-11-05"
    assert row.rain14_end_inclusive == "2025-11-18"


@pytest.mark.parametrize("value", ["2025-11-19", "2025-11-20", "2025-11-17", "not-a-date", ""])
def test_same_day_future_extra_shift_or_malformed_source_dates_are_rejected(sources, value):
    raw, reference = sources
    raw = raw.copy()
    raw.loc[0, "c2_weather_source_date"] = value
    assert validate_model2(raw, reference)[1].errors


def test_wrong_alignment_metadata_is_rejected(sources):
    raw, reference = sources
    raw = raw.copy()
    raw.loc[0, "c2_weather_alignment"] = "same_day"
    assert validate_model2(raw, reference)[1].errors


def test_non_nested_one_day_values_are_warned_not_clipped(sources):
    raw, reference = sources
    frame, audit = validate_model2(raw, reference)
    assert not audit.errors
    assert "157 rows have 1-day" in audit.markdown()
    assert (frame.c2_rainfall_mm > frame.c2_rainfall_3day_mm + 1e-6).sum() == 157
    pd.testing.assert_series_equal(frame.c2_rainfall_mm, pd.to_numeric(raw.c2_rainfall_mm))
    assert "not proven real-time" in audit.markdown()
    assert "not independently certified" in audit.markdown()


def test_nested_rolling_windows_must_be_consistent_under_declared_definition(sources):
    raw, reference = sources
    changed = raw.copy()
    changed.loc[0, "c2_rainfall_3day_mm"] = str(float(changed.loc[0, "c2_rainfall_7day_mm"]) + 10)
    assert any("Non-nested rolling totals" in error for error in validate_model2(changed, reference)[1].errors)
