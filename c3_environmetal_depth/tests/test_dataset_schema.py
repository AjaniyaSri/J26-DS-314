"""Source-data checks and deliberate malformed-data rejection."""
import pytest

from src.data_loader import load_config, load_dataset, local_path
from src.data_validator import require_valid, validate_dataset
from src.features import SCHEMA


@pytest.fixture
def raw():
    return load_dataset(load_config())


def test_actual_schema_and_shape(raw):
    assert raw.shape == (1314, 21)
    assert tuple(raw.columns) == SCHEMA
    frame, audit = require_valid(raw)
    assert not audit.errors
    assert frame.station_id.nunique() == 39


@pytest.mark.parametrize("column,value", [
    ("mean_elevation_m", "not-a-number"), ("stage_at_issue_m", "inf"),
    ("target_stage_next_day_m", "NaN"), ("station_name", ""),
    ("catchment_area_km2", "0"), ("tree_fraction", "1.01"),
    ("cropland_fraction", "-0.1"), ("builtup_fraction", "2"),
    ("forecast_date", "not-a-date"),
])
def test_rejects_bad_cells(raw, column, value):
    raw.loc[0, column] = value
    _, audit = validate_dataset(raw)
    assert audit.errors
    assert "badalgama" in audit.markdown()
    with pytest.raises(ValueError, match="training prohibited"):
        require_valid(raw)


def test_rejects_unapproved_column(raw):
    raw["target_delta_stage_m"] = 1.0
    assert validate_dataset(raw)[1].errors


def test_missing_required_column(raw):
    assert validate_dataset(raw.drop(columns="stage_lag_1d_m"))[1].errors


@pytest.mark.parametrize("path", ["../outside.csv", "C:/outside.csv"])
def test_paths_cannot_escape_project(path):
    with pytest.raises(ValueError):
        local_path(path)
