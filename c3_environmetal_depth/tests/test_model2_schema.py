import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset
from src.features import FEATURES, TARGET
from src.model2_data import (DATASET, MODEL2_FEATURES, MODEL2_SCHEMA, WEATHER,
                             load_model2_config, validate_model2, align_model1)
from src.model2_training import Model2Regressor


@pytest.fixture(scope="module")
def sources():
    config = load_model2_config()
    return load_dataset(config), load_dataset({"dataset": config["model1_dataset"]})


def test_actual_exact_file_schema_and_gauge_correspondence(sources):
    raw, reference = sources
    assert load_model2_config()["dataset"] == DATASET
    assert raw.shape == (1314, 30)
    assert tuple(raw.columns) == MODEL2_SCHEMA
    frame, audit = validate_model2(raw, reference)
    assert not audit.errors
    assert len(MODEL2_FEATURES) == 16
    assert frame.station_id.nunique() == 39
    assert frame.chronological_split.value_counts().to_dict() == {"TRAIN": 700, "VALIDATION": 265, "TEST": 349}
    assert frame.record_id.tolist() == raw.record_id.tolist()
    assert frame.chronological_split.tolist() == raw.chronological_split.tolist()
    assert np.isfinite(frame[list(MODEL2_FEATURES)].to_numpy()).all()


@pytest.mark.parametrize("column,value", [
    (WEATHER[0], ""), (WEATHER[0], "-1"), (WEATHER[1], "inf"),
    (WEATHER[2], "not numeric"), (WEATHER[4], "1.1"), (WEATHER[5], "-0.01"),
    (TARGET, ""), ("chronological_split", "random"), ("catchment_area_km2", "0"),
])
def test_invalid_required_values_prevent_training(sources, column, value):
    raw, reference = sources
    changed = raw.copy()
    changed.loc[0, column] = value
    _, audit = validate_model2(changed, reference)
    assert audit.errors
    assert "badalgama" in audit.markdown()


def test_schema_rejects_extra_future_column_and_missing_weather(sources):
    raw, reference = sources
    future = raw.assign(future_rainfall=1)
    assert validate_model2(future, reference)[1].errors
    assert validate_model2(raw.drop(columns=WEATHER[0]), reference)[1].errors


def test_station_date_keys_and_targets_cannot_change(sources):
    raw, reference = sources
    duplicate = raw.copy()
    duplicate.loc[1, ["station_id", "forecast_date"]] = duplicate.loc[0, ["station_id", "forecast_date"]].values
    with pytest.raises(ValueError, match="duplicate"):
        align_model1(duplicate, reference)
    with pytest.raises(ValueError, match="keys differ"):
        align_model1(raw.iloc[1:], reference)
    changed = raw.copy()
    changed.loc[0, TARGET] = str(float(changed.loc[0, TARGET]) + .01)
    with pytest.raises(ValueError, match="correspondence failed"):
        align_model1(changed, reference)


def test_matrix_excludes_temperature_dates_metadata_and_future_response(sources):
    frame, audit = validate_model2(*sources)
    assert not audit.errors
    wrapper = Model2Regressor("delta", None)
    original = wrapper.matrix(frame)
    changed = frame.copy()
    for column in (TARGET, "c2_temperature_C", "c2_weather_source_date", "station_name", "chronological_split"):
        changed[column] = "changed"
    pd.testing.assert_frame_equal(wrapper.matrix(changed), original)
    assert tuple(original.columns) == MODEL2_FEATURES
    assert not set((TARGET, "c2_temperature_C", "c2_weather_source_date")).intersection(original.columns)


def test_primary_cannot_accidentally_use_only_ten_or_add_a_seventeenth_input():
    with pytest.raises(ValueError, match="allowlist"):
        Model2Regressor("absolute", None, FEATURES)
    with pytest.raises(ValueError, match="allowlist"):
        Model2Regressor("absolute", None, (*MODEL2_FEATURES, "c2_temperature_C"))
    assert Model2Regressor("delta", None, FEATURES, is_control=True).is_control
