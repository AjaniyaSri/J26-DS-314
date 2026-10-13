import pytest

from src.data_loader import load_config, load_dataset
from src.data_validator import EXPECTED_SPLITS, require_valid, validate_dataset


@pytest.fixture
def raw():
    return load_dataset(load_config())


def test_preserves_all_original_assignments_and_rows(raw):
    frame, _ = require_valid(raw)
    assert frame.record_id.tolist() == raw.record_id.tolist()
    assert frame.chronological_split.tolist() == raw.chronological_split.tolist()
    assert frame.chronological_split.value_counts().to_dict() == {
        split: spec[0] for split, spec in EXPECTED_SPLITS.items()}


def test_rejects_shuffled_split_assignments_even_when_counts_match(raw):
    a = raw.index[raw.chronological_split.eq("TRAIN")][0]
    b = raw.index[raw.chronological_split.eq("TEST")][0]
    raw.loc[a, "chronological_split"], raw.loc[b, "chronological_split"] = "TEST", "TRAIN"
    assert any("date window" in error for error in validate_dataset(raw)[1].errors)


def test_rejects_unknown_split(raw):
    raw.loc[0, "chronological_split"] = "train"
    assert any("Unexpected split names" in error for error in validate_dataset(raw)[1].errors)


def test_rejects_duplicate_station_dates(raw):
    raw.loc[1, ["station_id", "forecast_date"]] = raw.loc[0, ["station_id", "forecast_date"]].values
    assert any("Duplicate station/date" in error for error in validate_dataset(raw)[1].errors)


def test_rejects_duplicate_record_ids(raw):
    raw.loc[1, "record_id"] = raw.loc[0, "record_id"]
    assert any("Duplicate record_id" in error for error in validate_dataset(raw)[1].errors)
