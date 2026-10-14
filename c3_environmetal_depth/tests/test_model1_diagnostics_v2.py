"""Leakage, representation, selection and preservation checks for v2."""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, load_dataset, write_text_new
from src.data_validator import require_valid
from src.evaluator import metrics
from src.features import FEATURES, TARGET
from src.model1_diagnostics import evaluate_revised, load_v2_config, DEFAULT_CONFIG
from src.model1_variants import (StageVariant, choose_environmental, fit_variant,
                                 recommended_method)


@pytest.fixture(scope="module")
def partitions():
    frame, _ = require_valid(load_dataset(load_config()))
    return (frame[frame.chronological_split.eq("TRAIN")].copy(),
            frame[frame.chronological_split.eq("VALIDATION")].copy())


def test_predeclared_fit_budget_and_seed():
    config = load_v2_config(DEFAULT_CONFIG)
    assert len(config["candidates"]) + 2 == 8
    assert config["seed"] == 42
    assert config["near_tie_mae_m"] == .005


def test_zero_delta_is_exactly_persistence(partitions):
    _, validation = partitions
    variant = StageVariant("delta")
    np.testing.assert_array_equal(variant.reconstruct(validation, np.zeros(len(validation))),
                                  validation.stage_at_issue_m)


@pytest.mark.parametrize("representation", ["absolute", "delta", "relative"])
def test_response_round_trip_and_reconstructed_metrics(partitions, representation):
    train, validation = partitions
    variant = StageVariant(representation).fit_reference(train)
    np.testing.assert_allclose(variant.reconstruct(validation, variant.response(validation)),
                               validation[TARGET], atol=1e-12, rtol=0)
    response_prediction = variant.response(validation) + np.linspace(-.4, .8, len(validation))
    response_scores = metrics(variant.response(validation), response_prediction)
    stage_scores = metrics(validation[TARGET], variant.reconstruct(validation, response_prediction))
    for name in ("mae_m", "rmse_m", "mean_signed_error_m"):
        assert stage_scores[name] == pytest.approx(response_scores[name], abs=1e-12)


def test_station_reference_uses_train_current_observations_only(partitions):
    train, validation = partitions
    variant = StageVariant("relative").fit_reference(train)
    expected = train.groupby("station_id").stage_at_issue_m.median().to_dict()
    assert variant.references == expected
    changed_targets = train.copy()
    changed_targets[TARGET] = 999999
    assert StageVariant("relative").fit_reference(changed_targets).references == expected
    changed_validation = validation.copy()
    changed_validation["stage_at_issue_m"] = 999999
    variant.matrix(changed_validation)
    assert variant.references == expected
    with pytest.raises(ValueError, match="TRAIN-only"):
        StageVariant("relative").fit_reference(pd.concat([train, validation]))


def test_relative_centering_changes_three_levels_not_daily_change_or_environment(partitions):
    train, validation = partitions
    variant = StageVariant("relative").fit_reference(train)
    X = variant.matrix(validation)
    offset = validation.station_id.map(variant.references).to_numpy()
    for feature in FEATURES[:3]:
        np.testing.assert_allclose(X[feature], validation[feature] - offset)
    for feature in FEATURES[3:]:
        np.testing.assert_array_equal(X[feature], validation[feature])


def test_centering_is_invariant_to_consistent_station_datum_shift(partitions):
    train, validation = partitions
    shift = {name: 100.0 + i for i, name in enumerate(train.station_id.unique())}
    shifted_train, shifted_validation = train.copy(), validation.copy()
    for frame in (shifted_train, shifted_validation):
        for column in (*FEATURES[:3], TARGET):
            frame[column] = frame[column] + frame.station_id.map(shift)
    original = StageVariant("relative").fit_reference(train)
    shifted = StageVariant("relative").fit_reference(shifted_train)
    np.testing.assert_allclose(original.matrix(validation), shifted.matrix(shifted_validation), atol=1e-12)
    np.testing.assert_allclose(original.response(validation), shifted.response(shifted_validation), atol=1e-12)


def test_centering_fails_for_unknown_station_or_insufficient_history(partitions):
    train, validation = partitions
    variant = StageVariant("relative").fit_reference(train)
    unknown = validation.iloc[:1].copy()
    unknown["station_id"] = "unseen"
    with pytest.raises(ValueError, match="No frozen TRAIN reference"):
        variant.matrix(unknown)
    with pytest.raises(ValueError, match="Insufficient TRAIN"):
        StageVariant("relative").fit_reference(train, minimum_rows=1000)


@pytest.mark.parametrize("representation", ["absolute", "delta", "relative"])
def test_prediction_does_not_read_future_target(partitions, representation):
    train, validation = partitions
    variant = StageVariant(representation).fit_reference(train)

    class ZeroResponse:
        def predict(self, X):
            assert tuple(X.columns) == FEATURES
            return np.zeros(len(X))

    variant.estimator = ZeroResponse()
    without_target = validation.drop(columns=TARGET)
    first = variant.predict_stage(without_target)
    changed = validation.copy()
    changed[TARGET] = -999999
    np.testing.assert_array_equal(first, variant.predict_stage(changed))


@pytest.mark.parametrize("features", [(*FEATURES, TARGET), (*FEATURES[:9], "station_id"), ("target_delta",)])
def test_variant_feature_allowlist_cannot_include_labels_or_metadata(features):
    with pytest.raises(ValueError, match="approved ten inputs"):
        StageVariant("delta", features)


def test_fit_uses_train_response_and_validation_only_for_early_stopping(partitions, monkeypatch):
    train, validation = partitions
    recorded = []

    class RecordingEstimator:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, **kwargs):
            assert tuple(X.columns) == FEATURES
            assert X.index.tolist() == train.index.tolist()
            np.testing.assert_array_equal(y, train[TARGET] - train.stage_at_issue_m)
            assert kwargs["eval_set"][0][0].index.tolist() == validation.index.tolist()
            np.testing.assert_array_equal(kwargs["eval_set"][0][1], validation[TARGET] - validation.stage_at_issue_m)
            recorded.append(len(X))

    monkeypatch.setattr("src.model1_variants.lgb.LGBMRegressor", RecordingEstimator)
    fit_variant(train, validation, "delta", {"n_estimators": 20, "metric": "rmse"})
    assert recorded == [700]
    with pytest.raises(ValueError, match="TRAIN-only"):
        fit_variant(pd.concat([train, validation]), validation, "delta", {})
    invalid = validation.copy()
    invalid["chronological_split"] = "TEST"
    with pytest.raises(ValueError, match="VALIDATION-only"):
        fit_variant(train, invalid, "delta", {})


def test_matched_ablation_has_no_extra_validation_search(partitions, monkeypatch):
    train, validation = partitions
    calls = []

    class RecordingEstimator:
        def __init__(self, **params):
            assert params["n_estimators"] == 12

        def fit(self, X, y, **kwargs):
            assert tuple(X.columns) == FEATURES[:4]
            assert kwargs == {}
            calls.append(len(X))

    monkeypatch.setattr("src.model1_variants.lgb.LGBMRegressor", RecordingEstimator)
    fit_variant(train, validation, "delta", {"n_estimators": 500}, features=FEATURES[:4], fixed_iterations=12)
    assert calls == [700]


def selection_example():
    return pd.DataFrame([
        {"model": "persistence", "mae_m": .23, "eligible_primary": False, "fitted_leaf_count": 0, "transformation_count": 0, "candidate_order": -1},
        {"model": "complex_delta", "mae_m": .20, "eligible_primary": True, "fitted_leaf_count": 100, "transformation_count": 1, "candidate_order": 0},
        {"model": "simple_delta", "mae_m": .203, "eligible_primary": True, "fitted_leaf_count": 20, "transformation_count": 1, "candidate_order": 1},
        {"model": "history_only_ablation", "mae_m": .01, "eligible_primary": False, "fitted_leaf_count": 1, "transformation_count": 1, "candidate_order": 2},
    ])


def test_selection_respects_simplicity_band_and_excludes_ablation():
    table = selection_example()
    assert choose_environmental(table, .005) == "simple_delta"
    assert choose_environmental(table, 0) == "complex_delta"
    assert recommended_method("simple_delta", table, .005) == "simple_delta"


def test_recommend_persistence_if_candidates_lose_or_gain_is_tiny():
    table = selection_example()
    table.loc[table.model.eq("simple_delta"), "mae_m"] = .24
    assert recommended_method("simple_delta", table, .005) == "persistence"
    table.loc[table.model.eq("simple_delta"), "mae_m"] = .228
    assert recommended_method("simple_delta", table, .005) == "persistence"


def test_no_silent_file_overwrite(tmp_path):
    path = tmp_path / "artifact.txt"
    write_text_new(path, "original")
    with pytest.raises(FileExistsError):
        write_text_new(path, "changed")
    assert path.read_text() == "original"


def test_repeat_test_assessment_is_blocked_before_loading_model(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "test_evaluation_started.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already started"):
        evaluate_revised(tmp_path)
