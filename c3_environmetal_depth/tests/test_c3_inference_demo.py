"""Inference regression and target-separation checks; never train models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from scripts import demo_c3_inference as demo


@pytest.fixture(scope="module")
def inputs():
    return demo.load_inputs()


@pytest.fixture(scope="module", params=["original", "revised"])
def models(request):
    return demo.load_models(request.param)


def test_input_reader_never_parses_target(monkeypatch):
    original = pd.read_csv
    calls = []

    def guarded(*args, **kwargs):
        columns = kwargs.get("usecols")
        assert columns is not None and demo.TARGET not in columns
        calls.append(columns)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    first, second = demo.load_inputs()
    assert len(calls) == 2 and demo.TARGET not in first and demo.TARGET not in second


def test_models_reproduce_all_saved_test_predictions(models, inputs, monkeypatch):
    import lightgbm as lgb

    def forbidden(*args, **kwargs):
        pytest.fail("Inference must never fit an estimator")

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", forbidden)
    first, second = inputs
    x1 = first.loc[first.chronological_split.eq("TEST")].sort_values(demo.KEYS)
    x2 = second.loc[second.chronological_split.eq("TEST")].sort_values(demo.KEYS)
    values = demo.predict(models, x1[list(demo.FEATURES)], x2[list(demo.MODEL2_FEATURES)])
    path1 = ("results/predictions/model1_test_predictions.csv" if models.variant == "original"
             else "experiments/model1_diagnostics_v2/predictions/model1_improved_test_predictions.csv")
    for label, path in [("Model 1", path1), ("Model 2", demo.RUN / "predictions/model2_test_predictions.csv")]:
        saved = pd.read_csv(demo.ROOT / path).sort_values(demo.KEYS)
        assert list(zip(x1.station_id, x1.forecast_date)) == list(zip(saved.station_id, saved.forecast_date))
        np.testing.assert_allclose(values[label], saved.predicted_stage_next_day_m, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(values["Persistence"], x1.stage_at_issue_m)
    # Check reconstruction explicitly: a delta is added once; absolute is not offset.
    raw1 = (models.model1.predict(x1[list(demo.FEATURES)]) if models.variant == "original"
            else models.model1.estimator.predict(x1[list(demo.FEATURES)]))
    expected1 = raw1 if models.variant == "original" else raw1 + x1.stage_at_issue_m.to_numpy()
    np.testing.assert_allclose(values["Model 1"], expected1, atol=1e-12, rtol=0)
    raw2 = models.model2.estimator.predict(x2[list(demo.MODEL2_FEATURES)])
    np.testing.assert_allclose(values["Model 2"], raw2 + x2.stage_at_issue_m.to_numpy(), atol=1e-12, rtol=0)


@pytest.mark.parametrize("station,day,message", [
    ("unknown_station", "2025-12-17", "Unknown"),
    ("badalgama", "2025-02-30", "valid calendar"),
    ("badalgama", "17/12/2025", "YYYY-MM-DD"),
    ("badalgama", "2026-10-23", "unavailable"),
])
def test_invalid_selection(inputs, station, day, message):
    with pytest.raises(ValueError, match=message):
        demo.select_rows(*inputs, station, day)


def test_station_name_supported(inputs):
    row, first, second = demo.select_rows(*inputs, "Badalgama", "2025-12-17")
    assert row.station_id == "badalgama"
    assert tuple(first.columns) == demo.FEATURES
    assert tuple(second.columns) == demo.MODEL2_FEATURES


@pytest.mark.parametrize("corruption", ["missing", "infinite", "text", "fraction", "order", "target"])
def test_invalid_features(inputs, corruption):
    x = inputs[0].iloc[[0]][list(demo.FEATURES)].copy()
    if corruption == "order":
        x = x[list(reversed(demo.FEATURES))]
    elif corruption == "target":
        x[demo.TARGET] = 3.0
    elif corruption == "fraction":
        x["tree_fraction"] = 1.5
    else:
        x["stage_at_issue_m"] = {"missing": np.nan, "infinite": np.inf, "text": "bad"}[corruption]
    with pytest.raises(ValueError):
        demo.validate_features(x, demo.FEATURES)


def test_estimator_feature_order_rejected(models):
    with pytest.raises(ValueError, match="feature names/order"):
        demo.check_estimator(models.model2.estimator, tuple(reversed(demo.MODEL2_FEATURES)))


def test_frozen_hash_mismatch_rejected():
    with pytest.raises(ValueError, match="differs from its frozen metadata"):
        demo.verify_hash(demo.STATIC, "0" * 64)


def test_actual_never_read_when_hidden(models, inputs, monkeypatch, capsys):
    def forbidden(*args):
        pytest.fail("Actual must not be read when reveal is declined")

    monkeypatch.setattr(demo, "reveal_actual", forbidden)
    monkeypatch.setattr(demo, "ask", lambda prompt: "n")
    demo.show_prediction(models, *inputs, "badalgama", "2025-12-17")
    assert "Actual next-day stage remains hidden" in capsys.readouterr().out


def test_actual_revealed_after_predictions(models, inputs, monkeypatch, capsys):
    original = demo.reveal_actual

    def checked(*args):
        output = capsys.readouterr().out
        assert "Model 1" in output and "Model 2" in output and "Persistence" in output
        assert "Actual next-day stage:" not in output
        return original(*args)

    monkeypatch.setattr(demo, "reveal_actual", checked)
    demo.show_prediction(models, *inputs, "badalgama", "2025-12-17", auto_reveal=True)
    output = capsys.readouterr().out
    assert "Actual next-day stage: 2.330000 m" in output
    assert "signed error" in output


def test_invalid_cli_has_clear_error(capsys):
    assert demo.main(["--station", "unknown", "--date", "2025-12-17"]) == 2
    assert "Unknown station" in capsys.readouterr().err


def test_interactive_retry_and_exit(monkeypatch, capsys):
    answers = iter(["not_a_station", "q"])
    monkeypatch.setattr(demo, "ask", lambda prompt: next(answers))
    assert demo.main([]) == 0
    assert "Choose a listed" in capsys.readouterr().out
