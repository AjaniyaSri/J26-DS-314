"""Metrics, frozen-model evaluation and a guarded final TEST assessment."""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import persistence
from src.data_loader import (ROOT, load_dataset, sha256, timestamp, write_csv_new,
                             write_json_new, write_text_new)
from src.data_validator import require_valid
from src.features import FEATURES, TARGET, feature_matrix


def metrics(observed, predicted) -> dict:
    y = np.asarray(observed, dtype=float)
    prediction = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not y.size:
        raise ValueError("Metrics require nonempty, aligned one-dimensional arrays.")
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError("Nonfinite observations or predictions in metrics.")
    return {"n_samples": len(y), "mae_m": float(mean_absolute_error(y, prediction)),
            "rmse_m": float(np.sqrt(mean_squared_error(y, prediction))),
            "mean_signed_error_m": float(np.mean(prediction - y))}


def prediction_frame(frame: pd.DataFrame, predictions) -> pd.DataFrame:
    result = frame[["record_id", "station_id", "station_name", "basin", "forecast_date",
                    "chronological_split", TARGET]].copy()
    result["predicted_stage_next_day_m"] = np.asarray(predictions)
    result["persistence_prediction_m"] = persistence(frame)
    # Positive error means overprediction; target is never used to construct inputs.
    result["prediction_error_m"] = result.predicted_stage_next_day_m - result[TARGET]
    return result


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in predictions.groupby("chronological_split", sort=False):
        for model, col in (("lightgbm_model1", "predicted_stage_next_day_m"),
                           ("persistence", "persistence_prediction_m")):
            rows.append({"split": split, "model": model, **metrics(group[TARGET], group[col])})
    return pd.DataFrame(rows)


def station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["chronological_split", "station_id", "station_name", "basin"]
    for key, group in predictions.groupby(keys, sort=True):
        for model, col in (("lightgbm_model1", "predicted_stage_next_day_m"),
                           ("persistence", "persistence_prediction_m")):
            rows.append({**dict(zip(keys, key)), "model": model,
                         **metrics(group[TARGET], group[col])})
    return pd.DataFrame(rows)


LIMITATIONS = """1. The supplied study is described as about 40 consecutive days; actual forecast-input rows span 37 days, 2025-11-19 to 2025-12-25. Antecedent history and next-day labels extend that span slightly.
2. Catchment polygons are candidate boundaries pending validation; QA discrepancies remain included.
3. ESA WorldCover describes 2021 land cover, not contemporaneous conditions.
4. Station-specific gauge datums may differ; these are gauge stages, not a common elevation datum. Pooled errors can be dominated by a subset of gauges; per-station errors and macro means are reported.
5. Gauges contribute different numbers of records (20–37 in the supplied CSV).
6. Related gauges may have overlapping upstream catchments. This is temporal evaluation at known gauges, not validation on unseen independent catchments.
7. Random row splits are inappropriate here; supplied chronological assignments remain unchanged.
8. Current observations are described as 09:00 readings; assumed issue timestamps are 09:30 Asia/Colombo. Every publication-time verification flag is False. Thus the horizon to next-day 09:00 is nominally one day from observation, about 23.5 hours from assumed issuance. Operational availability is not established.
9. This short window does not establish seasonal or long-term generalization. Validation also lies later in the same short sequence; selection over six configurations can overfit that window.
10. Predictions are river stages, not flood classes, flood depths or validated flood-inundation maps.
11. Group approval of the revised river-stage objective is not supplied: research governance confirmation remains required.
12. Missing source-date references prevent independent verification of every lag and label. Measurement units are specified by the column names; independent source unit certification is unavailable.
13. Boundary labels occur on the next calendar day: the last TRAIN label coincides with the first VALIDATION input date, and similarly for VALIDATION/TEST. Under the assumed 09:30 issue time, that day's 09:00 observation could be available before issuance, but publication timing is unverified. This is a known-gauge, daily rolling-information experiment, not a forecast of the entire TEST block from one fixed origin. Later TEST rows may use earlier observed TEST stages as allowed historical inputs; no TEST labels are used for fitting or selection.
14. Feature importance measures fitted associations, not hydrological causes. Static features may partly encode station/basin identity. A gauge-history-only ablation is needed to establish their incremental predictive value; it is outside this initial Model 1 run.
15. No rainfall or soil moisture is available here. Model 2 is not implemented or trained.
"""


def experiment_summary(base: Path, metadata: dict, table: pd.DataFrame,
                       per_station: pd.DataFrame) -> str:
    lines = ["# Model 1 experiment summary", "", "## Final observed results", "",
             "Errors are in metres; signed error is prediction minus observation.", "",
             "| Split | Model | n | MAE (m) | RMSE (m) | Signed error (m) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in table.itertuples():
        lines.append(f"| {row.split} | {row.model} | {row.n_samples} | {row.mae_m:.6f} | "
                     f"{row.rmse_m:.6f} | {row.mean_signed_error_m:.6f} |")
    for split in ("VALIDATION", "TEST"):
        group = table[table.split.eq(split)].set_index("model")
        model, baseline = group.loc["lightgbm_model1"], group.loc["persistence"]
        lines += ["", f"{split}: LightGBM MAE is "
                  f"{'lower' if model.mae_m < baseline.mae_m else 'higher or equal'} than persistence."]
        for metric in ("mae_m", "rmse_m"):
            gain = 100 * (1 - model[metric] / baseline[metric]) if baseline[metric] else None
            lines.append(f"- {metric} improvement over persistence: "
                         + (f"{gain:.2f}% (negative means worse)." if gain is not None else "undefined (baseline zero)."))
    lines += ["", "## Station-balanced diagnostics", "",
              "These are arithmetic means of individual station metrics, with each gauge weighted equally. "
              "Mean station RMSE is not the pooled RMSE.", ""]
    macro = per_station.groupby(["chronological_split", "model"])[["mae_m", "rmse_m"]].mean()
    for (split, model), row in macro.iterrows():
        lines.append(f"- {split}, {model}: macro MAE {row.mae_m:.6f} m; mean station RMSE {row.rmse_m:.6f} m.")
    lines += ["", "## Frozen procedure and provenance", "",
              "TRAIN alone fits every candidate. VALIDATION chooses boosting iterations by RMSE early stopping "
              "and selects the candidate by MAE (candidate order breaks exact ties). Six configurations were "
              "fixed before training. The selected TRAIN-fitted model is retained without TRAIN+VALIDATION refitting. "
              "TEST is evaluated once after selection and freezing. No transformations or scaling are applied.", "",
              f"Selected candidate: {metadata['selected_candidate']}; retained trees: {metadata['best_iteration']}.", "",
              "```json", json.dumps(metadata["model_parameters"], indent=2), "```", "",
              f"Source SHA-256: `{metadata['dataset_sha256']}`.", "",
              "Metadata records feature order, configuration, source-code hashes, dependency versions and model hash. "
              "The native LightGBM text model and joblib estimator are both saved. "
              "Treat joblib files as trusted local artifacts only.", "",
              "## Research limitations", "", LIMITATIONS]
    return "\n".join(lines) + "\n"


def evaluate_frozen(base: Path) -> pd.DataFrame:
    """Exclusive marker prevents accidental repeated access to final TEST outcomes."""
    model_dir = base / "models/model1_static"
    marker = model_dir / "test_evaluation_started.json"
    if marker.exists():
        raise FileExistsError("Final TEST evaluation was already started for this run. "
                              "Read its saved results; do not tune against TEST or overwrite outputs.")
    metadata = json.loads((model_dir / "model1_metadata.json").read_text(encoding="utf-8"))
    config = metadata["configuration"]
    dataset = ROOT / config["dataset"]
    if sha256(dataset) != metadata["dataset_sha256"]:
        raise ValueError("Source CSV changed after model freezing.")
    model_path = model_dir / "lightgbm_model1.joblib"
    if sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("Frozen model changed after selection.")
    for relative, digest in metadata["source_code_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Source code changed after selection: {relative}")
    if tuple(metadata["features"]) != FEATURES or metadata["target"] != TARGET:
        raise ValueError("Frozen model feature/target contract mismatch.")
    frame, _ = require_valid(load_dataset(config))
    model = joblib.load(model_path)
    if tuple(model.feature_name_) != FEATURES:
        raise ValueError("Saved estimator has unexpected input feature order.")
    # Marker is written before final TEST scoring; a failed run is retained for diagnosis.
    write_json_new(marker, {"started_utc": timestamp(), "model_sha256": metadata["model_sha256"]})
    test = frame.loc[frame.chronological_split.eq("TEST")]
    test_predictions = prediction_frame(test, model.predict(feature_matrix(test)))
    write_csv_new(base / "results/predictions/model1_test_predictions.csv", test_predictions)
    test_metrics = metric_rows(test_predictions)
    write_csv_new(base / "results/metrics/model1_test_metrics.csv", test_metrics)
    validation = pd.read_csv(base / "results/predictions/model1_validation_predictions.csv",
                             dtype={"record_id": str, "station_id": str})
    predictions = pd.concat([validation, test_predictions], ignore_index=True)
    table = metric_rows(predictions)
    write_csv_new(base / "results/metrics/persistence_baseline_metrics.csv",
                  table.loc[table.model.eq("persistence")])
    per_station = station_metrics(predictions)
    write_csv_new(base / "results/metrics/model1_station_metrics.csv", per_station)
    from src.plotting import make_figures
    make_figures(base, test_predictions, table, per_station, model)
    write_text_new(base / "reports/model1_experiment_summary.md",
                   experiment_summary(base, metadata, table, per_station))
    if sha256(dataset) != metadata["dataset_sha256"]:
        raise RuntimeError("Source file changed during evaluation.")
    write_json_new(model_dir / "test_evaluation_complete.json", {"completed_utc": timestamp()})
    logging.info("Final results:\n%s", table.to_string(index=False))
    return table
