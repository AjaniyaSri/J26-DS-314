import numpy as np
import pandas as pd

from src.delta_experiment import build_delta_samples, reconstruct_level


def _config():
    return {
        "data": {
            "station_column": "station",
            "date_column": "date",
            "level_column": "water_level_9am_m",
        },
        "features": {
            "raw_daily_weather": ["rainfall_mm", "temperature_C"],
            "rainfall_windows": ["rainfall_3day_mm", "rainfall_7day_mm", "rainfall_14day_mm"],
            "water_level_lags_days": [1, 3, 7],
        },
    }


def _inputs(dates, stations=None, levels=None):
    stations = stations or ["A"] * len(dates)
    levels = levels or [float(index) for index in range(len(dates))]
    dates = pd.to_datetime(dates)
    water = pd.DataFrame(
        {"station": stations, "date": dates, "water_level_9am_m": levels}
    )
    weather = pd.DataFrame(
        {
            "station": stations,
            "date": dates,
            "rainfall_mm": [10.0 + index for index in range(len(dates))],
            "temperature_C": [20.0 + index for index in range(len(dates))],
            "rainfall_3day_mm": [30.0 + index for index in range(len(dates))],
            "rainfall_7day_mm": [70.0 + index for index in range(len(dates))],
            "rainfall_14day_mm": [140.0 + index for index in range(len(dates))],
        }
    )
    return water, weather


def test_delta_target_is_station_local_and_reconstructs_next_level():
    dates = pd.date_range("2025-01-01", periods=3)
    stations = ["A", "A", "A", "B", "B", "B"]
    all_dates = list(dates) + list(dates)
    levels = [1.0, 2.5, 3.0, 10.0, 9.0, 12.0]
    water, weather = _inputs(all_dates, stations, levels)

    samples = build_delta_samples(water, weather, _config(), maximum_origin_date="2025-01-02")
    row = samples.loc[(samples.station == "A") & (samples.date == pd.Timestamp("2025-01-01"))].iloc[0]
    other = samples.loc[(samples.station == "B") & (samples.date == pd.Timestamp("2025-01-01"))].iloc[0]

    assert row.delta_water_level_m == 1.5
    assert other.delta_water_level_m == -1.0
    assert reconstruct_level([row.water_level_current_m], [row.delta_water_level_m])[0] == 2.5


def test_missing_calendar_date_is_not_used_as_water_level_or_weather_lag():
    dates = ["2025-01-01", "2025-01-02", "2025-01-04", "2025-01-05"]
    water, weather = _inputs(dates, levels=[1.0, 2.0, 4.0, 5.0])

    samples = build_delta_samples(water, weather, _config(), maximum_origin_date="2025-01-04")
    row = samples.loc[samples.date.eq(pd.Timestamp("2025-01-04"))].iloc[0]

    assert pd.isna(row.water_level_lag_1d_m)
    assert pd.isna(row.rainfall_mm_lag_1d)
    assert pd.isna(row.water_level_change_1d_m)
    assert pd.isna(row.water_level_change_std_7d_m)


def test_features_use_only_previous_calendar_day_weather():
    water, weather = _inputs(pd.date_range("2025-01-01", periods=3), levels=[1.0, 2.0, 3.0])

    samples = build_delta_samples(water, weather, _config(), maximum_origin_date="2025-01-02")
    row = samples.loc[samples.date.eq(pd.Timestamp("2025-01-02"))].iloc[0]

    assert row.rainfall_mm_lag_1d == 10.0
    assert row.temperature_C_lag_1d == 20.0
    assert row.rainfall_3day_mm_lag_1d == 30.0


def test_delta_builder_does_not_emit_origins_beyond_validation_cutoff():
    water, weather = _inputs(pd.date_range("2025-01-01", periods=5), levels=[1.0, 2.0, 3.0, 4.0, 5.0])

    samples = build_delta_samples(water, weather, _config(), maximum_origin_date="2025-01-03")

    assert samples.date.max() == pd.Timestamp("2025-01-03")
    assert samples.target_date.max() == pd.Timestamp("2025-01-04")
    assert np.all((samples.target_date - samples.date).dt.days == 1)