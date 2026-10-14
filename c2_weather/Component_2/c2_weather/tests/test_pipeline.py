import pandas as pd
import pytest

from src.feature_engineering import (
    assert_identical_samples,
    build_feature_frames,
    past_rainfall_windows,
)
from src.evaluate import prediction_metrics, validate_identical_evaluation_samples
from src.train_xgboost import validate_prediction_shape


def _config():
    return {
        "data": {
            "station_column": "station",
            "date_column": "date",
            "level_column": "water_level_9am_m",
        },
        "features": {
            "raw_daily_weather": ["rainfall_mm", "temperature_C"],
            "rainfall_windows": ["rainfall_3day_mm"],
            "water_level_lags_days": [1],
            "daily_weather_lag_days": 1,
        },
    }


def _source(station, dates, levels, rain):
    return pd.DataFrame(
        {
            "station": station,
            "date": pd.to_datetime(dates),
            "water_level_9am_m": levels,
            "rainfall_mm": rain,
            "temperature_C": [20.0] * len(dates),
            "rainfall_3day_mm": [1.0] * len(dates),
        }
    )


def test_target_alignment_is_exact_next_day_and_station_local():
    dates = ["2025-01-01", "2025-01-02", "2025-01-04"]
    first = _source(["A"] * 3, dates, [1.0, 2.0, 4.0], [5.0, 6.0, 8.0])
    second = _source(["B"] * 3, dates, [10.0, 20.0, 40.0], [50.0, 60.0, 80.0])
    data = {name: pd.concat([first, second], ignore_index=True) for name in ("raw", "engineered", "hybrid")}

    frames = build_feature_frames(data, _config())

    assert set(frames["raw"]["station"]) == {"A", "B"}
    by_station = frames["raw"].set_index(["station", "date"])
    assert by_station.loc[("A", pd.Timestamp("2025-01-01")), "target_water_level_m"] == 2.0
    assert ("A", pd.Timestamp("2025-01-02")) not in by_station.index
    assert by_station.loc[("B", pd.Timestamp("2025-01-01")), "target_water_level_m"] == 20.0


def test_missing_date_does_not_create_a_next_day_target():
    data = _source(["A"] * 3, ["2025-01-01", "2025-01-02", "2025-01-04"], [1.0, 2.0, 4.0], [1.0] * 3)
    datasets = {name: data.copy() for name in ("raw", "engineered", "hybrid")}

    frames = build_feature_frames(datasets, _config())

    assert pd.Timestamp("2025-01-02") not in set(frames["raw"]["date"])


def test_missing_date_does_not_become_a_calendar_weather_lag():
    data = _source(
        ["A"] * 4,
        ["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-05"],
        [1.0, 2.0, 4.0, 5.0],
        [10.0, 20.0, 40.0, 50.0],
    )
    datasets = {name: data.copy() for name in ("raw", "engineered", "hybrid")}

    frames = build_feature_frames(datasets, _config())
    fourth = frames["raw"].set_index("date").loc[pd.Timestamp("2025-01-04")]

    assert pd.isna(fourth["rainfall_mm_lag_1d"])
    assert pd.isna(fourth["water_level_lag_1d_m"])


def test_weather_features_are_lagged_one_day():
    data = _source(["A"] * 3, ["2025-01-01", "2025-01-02", "2025-01-03"], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    datasets = {name: data.copy() for name in ("raw", "engineered", "hybrid")}

    frames = build_feature_frames(datasets, _config())
    second_day = frames["raw"].set_index("date").loc[pd.Timestamp("2025-01-02")]

    assert second_day["rainfall_mm_lag_1d"] == 10.0
    assert second_day["temperature_C_lag_1d"] == 20.0
    assert "target_water_level_m" not in frames["raw"].columns[5:]


def test_engineered_rainfall_windows_are_lagged_for_availability():
    data = _source(["A"] * 3, ["2025-01-01", "2025-01-02", "2025-01-03"], [1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    data["rainfall_3day_mm"] = [100.0, 200.0, 300.0]
    datasets = {name: data.copy() for name in ("raw", "engineered", "hybrid")}

    frames = build_feature_frames(datasets, _config())
    second_day = frames["engineered"].set_index("date").loc[pd.Timestamp("2025-01-02")]

    assert second_day["rainfall_3day_mm_lag_1d"] == 100.0
    assert "rainfall_3day_mm" not in frames["engineered"].columns


def test_past_rainfall_windows_require_complete_prior_dates():
    rainfall = pd.DataFrame(
        {
            "station": ["A"] * 4,
            "date": pd.date_range("2025-01-01", periods=4),
            "rainfall_mm": [1.0, 2.0, 3.0, 4.0],
        }
    )

    windows = past_rainfall_windows(rainfall, "station", "date", "rainfall_mm", [3])

    assert windows["rainfall_3day_mm"].iloc[:3].isna().all()
    assert windows["rainfall_3day_mm"].iloc[3] == pytest.approx(6.0)


def test_rainfall_window_is_missing_when_a_calendar_date_is_absent():
    rainfall = pd.DataFrame(
        {
            "station": ["A", "A", "A"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"]),
            "rainfall_mm": [1.0, 2.0, 4.0],
        }
    )

    windows = past_rainfall_windows(rainfall, "station", "date", "rainfall_mm", [2])

    assert pd.isna(windows.loc[windows["date"].eq(pd.Timestamp("2025-01-04")), "rainfall_2day_mm"]).all()


def test_experiments_must_have_identical_station_date_samples():
    common = pd.DataFrame({"station": ["A"], "date": [pd.Timestamp("2025-01-01")]})
    different = pd.DataFrame({"station": ["B"], "date": [pd.Timestamp("2025-01-01")]})

    with pytest.raises(AssertionError, match="identical"):
        assert_identical_samples({"raw": common, "engineered": different})


def test_evaluation_rejects_different_test_samples():
    common = pd.DataFrame(
        {"station": ["A"], "date": [pd.Timestamp("2025-01-01")], "target_date": [pd.Timestamp("2025-01-02")]}
    )
    different = common.assign(station="B")

    with pytest.raises(AssertionError, match="different test samples"):
        validate_identical_evaluation_samples({"raw": common, "hybrid": different})


def test_metrics_require_matching_prediction_shapes():
    with pytest.raises(ValueError, match="matching one-dimensional"):
        prediction_metrics([1.0, 2.0], [1.0])


def test_model_prediction_shape_matches_sample_count():
    validate_prediction_shape(pd.Series([1.0, 2.0]).to_numpy(), 2)

    with pytest.raises(AssertionError, match="Expected prediction shape"):
        validate_prediction_shape(pd.Series([1.0]).to_numpy(), 2)