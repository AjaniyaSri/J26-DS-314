import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_config, load_dataset
from src.data_validator import require_valid
from src.evaluator import evaluate_frozen
from src.features import FEATURES, METADATA, TARGET, feature_matrix
from src.trainer import select_model


def test_target_and_metadata_cannot_enter_feature_contract():
    assert TARGET not in FEATURES
    assert set(FEATURES).isdisjoint(METADATA)
    assert len(FEATURES) == 10
    for forbidden in (TARGET, "target_delta_stage_m", "station_id"):
        with pytest.raises(ValueError):
            feature_matrix(pd.DataFrame(), (*FEATURES, forbidden))


def test_changing_future_label_does_not_change_inputs():
    frame, _ = require_valid(load_dataset(load_config()))
    original = feature_matrix(frame)
    frame[TARGET] = np.arange(len(frame)) * 999
    for column in METADATA:
        frame[column] = "changed"
    pd.testing.assert_frame_equal(feature_matrix(frame), original)


def test_selection_fits_train_only_and_evaluates_validation_only(monkeypatch):
    frame, _ = require_valid(load_dataset(load_config()))
    train = frame[frame.chronological_split.eq("TRAIN")]
    validation = frame[frame.chronological_split.eq("VALIDATION")]
    fitted = []

    class RecordingModel:
        best_iteration_ = 3

        def __init__(self, **parameters):
            pass

        def fit(self, X, y, *, eval_set, **kwargs):
            pd.testing.assert_frame_equal(X, feature_matrix(train))
            pd.testing.assert_series_equal(y, train[TARGET])
            pd.testing.assert_frame_equal(eval_set[0][0], feature_matrix(validation))
            pd.testing.assert_series_equal(eval_set[0][1], validation[TARGET])
            fitted.append(len(X))

        def predict(self, X):
            pd.testing.assert_frame_equal(X, feature_matrix(validation))
            return X.stage_at_issue_m.to_numpy()

    monkeypatch.setattr("src.trainer.lgb.LGBMRegressor", RecordingModel)
    _, candidate, trials = select_model(train, validation, load_config())
    assert fitted == [700] * 6
    assert len(trials) == 6
    assert candidate == 1  # Equal MAE ties retain predefined order.


def test_final_evaluation_marker_blocks_repeated_test_scoring(tmp_path):
    directory = tmp_path / "models/model1_static"
    directory.mkdir(parents=True)
    (directory / "test_evaluation_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already started"):
        evaluate_frozen(tmp_path)
