import numpy as np
import pandas as pd
import pytest

from src.baseline import persistence
from src.data_loader import ROOT, load_dataset
from src.features import FEATURES, TARGET
from src.model2_data import MODEL2_FEATURES, load_model2_config, validate_model2
from src.model2_training import Model2Regressor, fit_model2, select_winner
from src.model_comparison import (aligned_predictions, evaluate_model2, prediction_rows,
                                  summarize, per_station_metrics, original_model1_predictions)


@pytest.fixture(scope="module")
def partitions():
    config = load_model2_config()
    frame, audit = validate_model2(load_dataset(config), load_dataset({"dataset": config["model1_dataset"]}))
    assert not audit.errors
    return frame[frame.chronological_split.eq("TRAIN")], frame[frame.chronological_split.eq("VALIDATION")]


def test_saved_model1_predictions_align_by_keys_not_row_position(partitions):
    _, validation = partitions
    saved = pd.read_csv(ROOT / "results/predictions/model1_validation_predictions.csv")
    first = aligned_predictions(validation, saved)
    shuffled = aligned_predictions(validation, saved.sample(frac=1, random_state=7))
    np.testing.assert_array_equal(first, shuffled)
    with pytest.raises(ValueError, match="evaluation rows differ"):
        aligned_predictions(validation, saved.iloc[1:])
    changed = saved.copy()
    changed.loc[0, TARGET] += .001
    with pytest.raises(ValueError, match="targets differ"):
        aligned_predictions(validation, changed)
    changed = pd.concat([saved, saved.iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate"):
        aligned_predictions(validation, changed)


def test_model1_reproduction_uses_original_source_despite_export_roundoff(partitions):
    _, validation = partitions
    original = original_model1_predictions(validation, "VALIDATION", reproduce=True)
    assert original.shape == (265,)


@pytest.mark.parametrize("representation", ["absolute", "delta"])
def test_stage_prediction_never_reads_target(partitions, representation):
    _, validation = partitions

    class ZeroResponse:
        def predict(self, X):
            assert tuple(X.columns) == MODEL2_FEATURES
            return np.zeros(len(X))

    wrapper = Model2Regressor(representation, ZeroResponse())
    prediction = wrapper.predict_stage(validation.drop(columns=TARGET))
    expected = persistence(validation) if representation == "delta" else np.zeros(len(validation))
    np.testing.assert_array_equal(prediction, expected)
    changed = validation.copy()
    changed[TARGET] = 99999
    np.testing.assert_array_equal(wrapper.predict_stage(changed), prediction)


def test_fit_sees_only_train_and_validation_response_is_proper_delta(partitions, monkeypatch):
    train, validation = partitions
    recorded = []

    class EstimatorSpy:
        def __init__(self, **params):
            pass

        def fit(self, X, y, **options):
            assert X.index.tolist() == train.index.tolist()
            assert tuple(X.columns) == MODEL2_FEATURES
            np.testing.assert_array_equal(y, train[TARGET] - train.stage_at_issue_m)
            X_val, y_val = options["eval_set"][0]
            assert X_val.index.tolist() == validation.index.tolist()
            np.testing.assert_array_equal(y_val, validation[TARGET] - validation.stage_at_issue_m)
            recorded.append(len(X))

    monkeypatch.setattr("src.model2_training.lgb.LGBMRegressor", EstimatorSpy)
    fit_model2(train, validation, "delta", {"metric": "rmse"})
    assert recorded == [700]
    with pytest.raises(ValueError, match="TRAIN-only"):
        fit_model2(pd.concat([train, validation]), validation, "delta", {"metric": "rmse"})
    disguised_test = validation.copy()
    disguised_test["chronological_split"] = "TEST"
    with pytest.raises(ValueError, match="VALIDATION-only"):
        fit_model2(train, disguised_test, "delta", {"metric": "rmse"})


def test_matched_control_has_ten_inputs_fixed_iterations_no_validation_search(partitions, monkeypatch):
    train, validation = partitions
    calls = []

    class ControlSpy:
        def __init__(self, **params):
            assert params["n_estimators"] == 12

        def fit(self, X, y, **options):
            assert tuple(X.columns) == FEATURES
            assert options == {}
            calls.append(len(X))

    monkeypatch.setattr("src.model2_training.lgb.LGBMRegressor", ControlSpy)
    fit_model2(train, validation, "absolute", {}, control=True, iterations=12)
    assert calls == [700]
    with pytest.raises(ValueError, match="Only matched controls"):
        fit_model2(train, validation, "absolute", {}, iterations=12)


def test_winner_selection_is_by_validation_mae_and_excludes_controls():
    table = pd.DataFrame([
        {"model": "absolute", "role": "candidate", "family": "absolute", "mae_m": .2, "fitted_leaves": 10, "family_order": 0, "candidate_order": 0},
        {"model": "delta", "role": "candidate", "family": "delta", "mae_m": .19, "fitted_leaves": 20, "family_order": 1, "candidate_order": 0},
        {"model": "control", "role": "matched_control", "family": "delta", "mae_m": .01, "fitted_leaves": 1, "family_order": -1, "candidate_order": -1},
    ])
    assert select_winner(table) == "delta"
    assert select_winner(table, "absolute") == "absolute"
    table.loc[table.model.eq("delta"), "mae_m"] = .2
    assert select_winner(table) == "absolute"


def test_paired_metrics_and_macro_weights(partitions):
    _, validation = partitions
    # Deliberately different station coverage; macro must weight gauges equally.
    station_ids = validation.station_id.unique()[:2]
    first = validation[validation.station_id.eq(station_ids[0])].iloc[:1]
    second = validation[validation.station_id.eq(station_ids[1])].iloc[:3]
    frame = pd.concat([first, second])
    errors = np.array([1., 2., 2., 2.])
    out = prediction_rows(frame, frame[TARGET].to_numpy() + errors, "example", frame[TARGET].to_numpy() + 3.)
    score = summarize(out)
    assert score["mae_m"] == pytest.approx(1.75)
    assert score["rmse_m"] == pytest.approx(np.sqrt(13/4))
    assert score["station_macro_mae_m"] == pytest.approx(1.5)
    assert score["mean_signed_error_m"] == pytest.approx(1.75)
    assert score["mae_improvement_vs_original_model1_pct"] == pytest.approx(100*(1-1.75/3))
    assert per_station_metrics(out).n_samples.sum() == 4


def test_test_repeat_guard_precedes_any_model_or_dataset_access(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "test_evaluation_started.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already started"):
        evaluate_model2(tmp_path)
