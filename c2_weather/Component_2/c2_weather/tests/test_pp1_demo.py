import numpy as np

from src.pp1_demo import evaluate_record, load_demo_records, verify_saved_predictions


def test_demo_records_are_common_held_out_test_rows():
    records = load_demo_records()

    assert len(records) == 351
    assert records["station"].nunique() == 39
    assert ((records["target_date"] - records["date"]).dt.days == 1).all()
    assert records["date"].min() == np.datetime64("2025-12-16")
    assert records["date"].max() == np.datetime64("2025-12-24")


def test_demo_prediction_uses_saved_models_and_matches_saved_row():
    row = load_demo_records().iloc[0]
    result = evaluate_record(row)

    assert result["target_date"] == result["forecast_origin"] + np.timedelta64(1, "D")
    assert set(result["predictions"]) == {"persistence", "raw", "engineered", "hybrid"}
    assert all(
        result["predictions"][variant]["matches_saved_prediction"]
        for variant in ("raw", "engineered", "hybrid")
    )
    assert result["predictions"]["persistence"]["predicted_water_level_m"] == result["current_water_level_m"]


def test_three_record_verification_has_no_prediction_mismatches():
    checks = verify_saved_predictions(3)

    assert len(checks) == 9
    assert checks["within_tolerance"].all()
    assert checks["absolute_difference_m"].max() <= 1e-6