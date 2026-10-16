"""Paired stage metrics, preserved Model 1 comparison and final descriptive scoring."""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline import persistence
from src.data_loader import (ROOT, local_path, load_dataset, sha256, timestamp, versions,
                             write_csv_new, write_json_new, write_text_new)
from src.evaluator import metrics
from src.features import FEATURES, TARGET, feature_matrix
from src.model2_data import KEYS, MODEL2_FEATURES, load_and_audit, load_model2_config


def verify_manifest(manifest: dict) -> None:
    failures = [name for name, digest in manifest.items()
                if not local_path(name).is_file() or sha256(local_path(name)) != digest]
    if failures:
        raise ValueError(f"Frozen or preserved files changed: {failures}")


def aligned_predictions(frame: pd.DataFrame, saved: pd.DataFrame) -> np.ndarray:
    """Align by station/date and reject any change of rows, labels or identifiers."""
    for label, part in (("requested", frame), ("saved", saved)):
        if part.duplicated(KEYS).any():
            raise ValueError(f"Duplicate {label} station-date predictions.")
    columns = [*KEYS, "record_id", "station_name", "chronological_split", TARGET]
    left = frame[columns].copy()
    left["row_order"] = np.arange(len(left))
    right = saved[[*columns, "predicted_stage_next_day_m"]]
    merged = left.merge(right, on=KEYS, how="outer", suffixes=("_input", "_saved"),
                        validate="one_to_one", indicator=True, sort=False)
    if len(merged) != len(frame) or not merged._merge.eq("both").all():
        raise ValueError("Saved prediction evaluation rows differ; no silent common subset is allowed.")
    for column in ("record_id", "station_name", "chronological_split"):
        if not merged[column + "_input"].eq(merged[column + "_saved"]).all():
            raise ValueError(f"Saved prediction metadata mismatch: {column}")
    if not np.allclose(merged[TARGET + "_input"], merged[TARGET + "_saved"], atol=1e-10, rtol=0):
        raise ValueError("Saved prediction targets differ from authoritative dataset.")
    prediction = merged.sort_values("row_order").predicted_stage_next_day_m.to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Saved predictions contain nonfinite values.")
    return prediction


def original_model1_predictions(frame: pd.DataFrame, split: str, reproduce=False) -> np.ndarray:
    if split not in ("VALIDATION", "TEST") or not frame.chronological_split.eq(split).all():
        raise ValueError("Model 1 comparison requires the declared evaluation partition.")
    saved = pd.read_csv(ROOT / f"results/predictions/model1_{split.lower()}_predictions.csv")
    prediction = aligned_predictions(frame, saved)
    if reproduce:
        if split != "VALIDATION":
            raise ValueError("Artifact reproduction is restricted to VALIDATION in this workflow.")
        model = joblib.load(ROOT / "models/model1_static/lightgbm_model1.joblib")
        canonical = original_model1_frame(frame)
        np.testing.assert_allclose(model.predict(feature_matrix(canonical)), prediction, atol=1e-10, rtol=0)
    return prediction


def original_model1_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Use Model 1's own source for reproduction, without changing Model 2 inputs.

    Tiny export roundoff can cross old tree thresholds. Station-date and target
    correspondence does not guarantee bit-identical predictor values.
    """
    from src.data_validator import require_valid
    metadata = json.loads((ROOT / "models/model1_static/model1_metadata.json").read_text(encoding="utf-8"))
    canonical, _ = require_valid(load_dataset({"dataset": metadata["source_filename"]}))
    indexed = canonical.set_index(KEYS)
    result = indexed.loc[pd.MultiIndex.from_frame(rows[KEYS])].reset_index()
    if result.record_id.tolist() != rows.record_id.tolist():
        raise ValueError("Original Model 1 source row alignment failed.")
    np.testing.assert_allclose(result[TARGET], rows[TARGET], rtol=0, atol=1e-10)
    return result


def discover_model2_run(parent: Path) -> Path:
    """Find the latest frozen run and ignore failed startup attempts."""
    candidates = [*parent.glob("models/model2_metadata.json"), *parent.glob("runs/*/models/model2_metadata.json")]
    if not candidates:
        raise FileNotFoundError("No frozen Model 2 run exists. Run train_model2.py first.")
    newest = max(candidates, key=lambda p: json.loads(p.read_text(encoding="utf-8"))["frozen_utc"])
    return newest.parent.parent


def prediction_rows(frame: pd.DataFrame, prediction, name: str, model1_prediction) -> pd.DataFrame:
    out = frame[["record_id", *KEYS, "station_name", "basin", "chronological_split",
                 "stage_at_issue_m", "stage_change_1d_m", TARGET]].copy()
    predicted = np.asarray(prediction, dtype=float)
    if predicted.shape != (len(frame),) or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and row aligned.")
    out["model"] = name
    out["predicted_stage_next_day_m"] = predicted
    out["persistence_prediction_m"] = persistence(frame)
    out["original_model1_prediction_m"] = np.asarray(model1_prediction)
    out["prediction_error_m"] = predicted - out[TARGET]
    return out


def summarize(predictions: pd.DataFrame) -> dict:
    score = metrics(predictions[TARGET], predictions.predicted_stage_next_day_m)
    station = [metrics(part[TARGET], part.predicted_stage_next_day_m)
               for _, part in predictions.groupby("station_id", sort=False)]
    baseline = metrics(predictions[TARGET], predictions.persistence_prediction_m)
    original = metrics(predictions[TARGET], predictions.original_model1_prediction_m)
    return {"model": str(predictions.model.iloc[0]), "split": str(predictions.chronological_split.iloc[0]),
            **score, "n_stations": len(station),
            "station_macro_mae_m": float(np.mean([r["mae_m"] for r in station])),
            "station_mean_rmse_m": float(np.mean([r["rmse_m"] for r in station])),
            "mae_improvement_vs_persistence_pct": 100 * (1 - score["mae_m"] / baseline["mae_m"]) if baseline["mae_m"] else None,
            "rmse_improvement_vs_persistence_pct": 100 * (1 - score["rmse_m"] / baseline["rmse_m"]) if baseline["rmse_m"] else None,
            "mae_improvement_vs_original_model1_pct": 100 * (1 - score["mae_m"] / original["mae_m"]) if original["mae_m"] else None}


def per_station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["chronological_split", "model", "station_id", "station_name", "basin"]
    for key, part in predictions.groupby(keys, sort=False):
        base = metrics(part[TARGET], part.persistence_prediction_m)
        original = metrics(part[TARGET], part.original_model1_prediction_m)
        score = metrics(part[TARGET], part.predicted_stage_next_day_m)
        rows.append({**dict(zip(keys, key)), **score, "persistence_mae_m": base["mae_m"],
                     "original_model1_mae_m": original["mae_m"],
                     "excess_mae_vs_persistence_m": score["mae_m"] - base["mae_m"],
                     "excess_mae_vs_original_model1_m": score["mae_m"] - original["mae_m"]})
    return pd.DataFrame(rows)


def score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([summarize(part) for _, part in predictions.groupby(["chronological_split", "model"], sort=False)])


def metric_markdown(table: pd.DataFrame) -> str:
    lines = ["| Model | Split | n | MAE (m) | RMSE (m) | Bias (m) | Macro MAE (m) | MAE gain vs persistence (%) | MAE gain vs original M1 (%) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table.itertuples():
        lines.append(f"| {row.model} | {row.split} | {row.n_samples} | {row.mae_m:.6f} | {row.rmse_m:.6f} | "
                     f"{row.mean_signed_error_m:.6f} | {row.station_macro_mae_m:.6f} | "
                     f"{row.mae_improvement_vs_persistence_pct:.2f} | {row.mae_improvement_vs_original_model1_pct:.2f} |")
    return "\n".join(lines)


def evaluate_model2(base: Path) -> pd.DataFrame:
    marker = base / "models/test_evaluation_started.json"
    if marker.exists():
        raise FileExistsError("Model 2 TEST assessment already started. Read saved outputs; do not rescore.")
    metadata = json.loads((base / "models/model2_metadata.json").read_text(encoding="utf-8"))
    verify_manifest(json.loads((base / "models/original_preservation.json").read_text(encoding="utf-8")))
    verify_manifest(metadata["source_code_sha256"])
    verify_manifest(metadata["frozen_artifact_sha256"])
    if versions() != metadata["dependencies"]:
        raise ValueError("Dependency versions changed after freezing.")
    config = metadata["configuration"]
    frame, audit = load_and_audit(config)
    if audit.errors:
        raise ValueError(audit.markdown())
    test = frame[frame.chronological_split.eq("TEST")].copy()
    models = {family: joblib.load(base / "models" / f"model2_{family}.joblib") for family in ("absolute", "delta")}
    controls = {family: joblib.load(base / "models" / f"control_{family}_no_weather.joblib") for family in models}
    for family, model in models.items():
        if model.features != MODEL2_FEATURES or model.representation != family:
            raise ValueError("Frozen model lost its family or 16-feature contract.")
    write_json_new(marker, {"started_utc": timestamp(), "test_status": "previously inspected project holdout; descriptive"})
    # Model 1 TEST predictions are reused, not refitted or newly re-predicted.
    original = original_model1_predictions(test, "TEST")
    frames = [prediction_rows(test, persistence(test), "persistence", original),
              prediction_rows(test, original, "original_model1", original)]
    for family, model in models.items():
        name = metadata["family_winners"][family]
        result = prediction_rows(test, model.predict_stage(test.drop(columns=TARGET)), name, original)
        result["recommended_method"] = metadata["recommended_method"]
        write_csv_new(base / "predictions" / f"model2_{family}_test_predictions.csv", result)
        frames.append(result)
        control = prediction_rows(test, controls[family].predict_stage(test.drop(columns=TARGET)),
                                  f"control_{family}_no_weather", original)
        frames.append(control)
    all_test = pd.concat(frames, ignore_index=True)
    selected = all_test[all_test.model.eq(metadata["selected_model2"])].copy()
    selected["recommended_prediction_m"] = (selected.persistence_prediction_m
                                             if metadata["recommended_method"] == "persistence"
                                             else selected.predicted_stage_next_day_m)
    write_csv_new(base / "predictions/model2_test_predictions.csv", selected)
    write_csv_new(base / "predictions/model_comparison_test_predictions.csv", all_test)
    table = score_table(all_test)
    write_csv_new(base / "metrics/model2_test_metrics.csv", table)
    station = per_station_metrics(all_test)
    write_csv_new(base / "metrics/model2_test_station_metrics.csv", station)
    validation = pd.read_csv(base / "metrics/model2_validation_metrics.csv")
    write_csv_new(base / "metrics/model1_vs_model2_metrics.csv", pd.concat([validation, table], ignore_index=True))
    make_figures(base, all_test, table, station, metadata["selected_model2"], "test")
    write_final_comparison(base, metadata, validation, table, station)
    verify_manifest(json.loads((base / "models/original_preservation.json").read_text(encoding="utf-8")))
    write_json_new(base / "models/test_evaluation_complete.json", {"completed_utc": timestamp(),
        "original_files_unchanged": True, "no_test_based_selection": True,
        "warning": "Previously inspected project TEST; descriptive assessment only."})
    logging.info("Final descriptive TEST metrics:\n%s", table.to_string(index=False))
    return table


LIMITATIONS_MODEL2 = """- Forecast inputs span 37 days within an approximately 40-day study; only 700 TRAIN records, with uneven gauge coverage.
- Candidate upstream watershed polygons and outstanding catchment QA flags remain unverified; some static inputs use ESA WorldCover 2021.
- Station gauge datums differ or may differ; nearby gauges and overlapping catchments reduce independence. Evaluation is at known stations, not unseen catchments.
- Nominal observations are at 09:00. Availability before assumed 09:30 Asia/Colombo issuance is unverified; the target is next-day 09:00 (one day from observation, about 23.5 hours from assumed issuance).
- CHIRPS and ERA5-Land product publication availability is unverified. Weather source dates alone do not establish actual availability at forecast issuance.
- The previous-day join introduces an extra day of lag for 3/7/14-day totals under the user-supplied C2 script. This alignment is authoritative for this run, not claimed optimal.
- The 1-day rainfall observation window, soil-moisture aggregation windows, weather timezone and spatial aggregation provenance cannot be independently certified from this CSV. Soil moisture is presumed volumetric m3/m3, with a preliminary [0,1] screen; this is not independent unit verification.
- Calendar lags/labels can only be checked where corresponding source-date rows exist. Raw gauge reports and original weather assets were not supplied. No value is corrected, clipped, imputed or dropped.
- Static features and weather may proxy station or short-period regime differences. Importance reflects association, not physical causation. No-weather controls are selection-conditioned, not unbiased estimates of the best possible weather-free model.
- The original Model 1 TEST set has already been inspected. Model 2 assessment is descriptive, not a fresh untouched-test confirmation. Further TEST-driven tuning would invalidate independent generalization claims.
- Six candidate fits reuse seven validation dates; selection optimism and correlated errors limit confidence. Independent future periods and multiple monsoon seasons are needed.
- This is retrospective preliminary river-stage prediction, not a deployment-ready real-time warning system. Flood thresholds, inundation extents and operational warning performance are not evaluated.
- Formal team approval of the revised river-stage objective remains a governance item unless separately confirmed by the group.
"""


def write_final_comparison(base, metadata, validation, test, station):
    selected = metadata["selected_model2"]
    val = validation.set_index("model").loc[selected]
    test_row = test.set_index("model").loc[selected]
    selected_station = station[station.model.eq(selected)]
    val_station = pd.read_csv(base / "metrics/model2_validation_station_metrics.csv")
    val_station = val_station[val_station.model.eq(selected)]
    text = ("# Original Model 1 versus Model 2\n\n"
            f"Validation-selected Model 2: **{selected}**. Validation recommendation: "
            f"**{metadata['recommended_method']}**. Both were frozen before scoring TEST.\n\n"
            "The primary Model 2 experiment predicts absolute stage. The delta experiment is a separately "
            "labelled secondary experiment. Family winners and the overall selected model are reported "
            "separately; no original Model 1 result is relabelled or overwritten.\n\n"
            "## Validation results used for selection\n\n" + metric_markdown(validation) + "\n\n"
            "## Final descriptive TEST assessment\n\n"
            "**This project TEST set was previously examined. These results are descriptive and do not "
            "establish independent confirmation of generalization.** Both frozen family winners and their "
            "predeclared no-weather controls were evaluated once. TEST did not select candidates, "
            "boosting iterations, calibration or feature definitions.\n\n" + metric_markdown(test) + "\n\n")
    text += ("## Interpretation\n\n"
             f"Selected validation MAE is {val.mae_m:.6f} m, with {val.mae_improvement_vs_original_model1_pct:.2f}% "
             f"MAE improvement over original Model 1 and {val.mae_improvement_vs_persistence_pct:.2f}% over persistence. "
             f"Descriptive TEST MAE is {test_row.mae_m:.6f} m. Negative improvement means worse. "
             "Beating the original Model 1 alone is insufficient to recommend Model 2 over persistence.\n\n"
             f"The selected model beats persistence at {int(val_station.excess_mae_vs_persistence_m.lt(0).sum())}/39 "
             f"validation gauges and {int(selected_station.excess_mae_vs_persistence_m.lt(0).sum())}/39 TEST gauges. "
             "Per-gauge wins and losses are retained rather than hidden by pooled metrics. Macro MAE gives "
             "each station equal weight; mean station RMSE is not pooled RMSE.\n\n"
             "## Controlled weather comparison\n\n")
    for family, winner in metadata["family_winners"].items():
        values = validation.set_index("model")
        weather = values.loc[winner]
        control = values.loc[f"control_{family}_no_weather"]
        text += (f"- {family}: weather-model validation MAE {weather.mae_m:.6f} m; matched no-weather "
                 f"control {control.mae_m:.6f} m; control-minus-weather difference "
                 f"{control.mae_m-weather.mae_m:+.6f} m.\n")
    text += ("\nControls use the same representation, objective, hyperparameters and selected weather model's "
             "boosting count, fitted on TRAIN once without further validation search. Positive differences "
             "favour weather for this configuration; negative differences favour no weather. These controlled "
             "comparisons isolate removing the six weather inputs under a fixed procedure, but are conditioned "
             "on weather-model selection and do not establish causal hydrological effects. The original "
             "Model 1 comparison also includes differences in selected training configurations.\n\n"
             "## Preservation and reproducibility\n\n"
             "The exact authoritative CSV and pre-existing Model 1 project artifacts were fingerprinted "
             "and verified unchanged. All new outputs remain local in this experiment directory. Metadata "
             "records feature order, target representation, parameters, seeds, source-code and data checksums, "
             "package versions and the frozen selection. Native delta models output changes and require "
             "adding current stage; use the saved wrapper's predict_stage method for river stages.\n\n"
             "## Scientific limitations\n\n" + LIMITATIONS_MODEL2)
    write_text_new(base / "reports/model1_vs_model2_comparison.md", text)


def make_figures(base, predictions, table, station, selected, split):
    from src.plotting import plt, save

    names = predictions.model.unique().tolist()
    absolute = [n for n in names if n.startswith("model2_absolute_")]
    delta = [n for n in names if n.startswith("model2_delta_")]
    if len(absolute) != 1 or len(delta) != 1:
        raise ValueError("Figures require exactly one frozen winner from each Model 2 family.")
    methods = ["original_model1", absolute[0], delta[0], "persistence"]
    labels = {"original_model1": "Original Model 1", absolute[0]: "Model 2 absolute (primary)",
              delta[0]: "Model 2 delta (secondary)", "persistence": "Persistence"}
    colors = dict(zip(methods, ("#d55e00", "#0072b2", "#cc79a7", "#009e73")))
    subset = table.set_index("model").loc[methods]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("mae_m", "rmse_m"), ("MAE", "RMSE")):
        ax.barh([labels[m] for m in methods], subset[metric], color=[colors[m] for m in methods])
        ax.invert_yaxis()
        ax.set(xlabel=f"{title} (m), lower is better", title=f"{split.upper()} {title}")
    fig.tight_layout()
    save(fig, base / "figures" / f"model1_vs_model2_{split}.png")

    chosen = predictions[predictions.model.eq(selected)]
    baseline = predictions[predictions.model.eq("persistence")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, part, title in ((axes[0], chosen, f"Selected Model 2: {selected}"), (axes[1], baseline, "Persistence")):
        ax.scatter(part[TARGET], part.predicted_stage_next_day_m, s=15, alpha=.6)
        limits = [min(part[TARGET].min(), part.predicted_stage_next_day_m.min()),
                  max(part[TARGET].max(), part.predicted_stage_next_day_m.max())]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlabel="Observed next-day stage (m)", ylabel="Predicted stage (m)", title=title)
    fig.suptitle(f"{split.upper()}: {len(chosen)} paired station-days")
    fig.tight_layout()
    save(fig, base / "figures" / f"model2_observed_vs_predicted_{split}.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    focus = predictions[predictions.model.isin(methods)]
    edges = np.linspace(focus.prediction_error_m.min(), focus.prediction_error_m.max(), 40)
    for method in methods:
        part = focus[focus.model.eq(method)]
        ax.hist(part.prediction_error_m, bins=edges, density=True, histtype="step", linewidth=1.5,
                label=labels[method], color=colors[method])
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Prediction minus observation (m)", ylabel="Density", title=f"{split.upper()}: residual distribution")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    save(fig, base / "figures" / f"model2_residual_distribution_{split}.png")

    paired_model_error = np.abs(chosen.predicted_stage_next_day_m - chosen[TARGET])
    paired_base_error = np.abs(chosen.persistence_prediction_m - chosen[TARGET])
    limit = max(paired_model_error.max(), paired_base_error.max())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(paired_base_error, paired_model_error, s=18, alpha=.6)
    ax.plot([0, limit], [0, limit], "k--", linewidth=1)
    ax.set(xlabel="Persistence absolute error (m)", ylabel="Selected Model 2 absolute error (m)",
           title=f"{split.upper()}: below line favours Model 2")
    save(fig, base / "figures" / f"model2_vs_persistence_errors_{split}.png")

    subset_station = station[station.model.isin(methods)]
    pivot = subset_station.pivot(index="station_name", columns="model", values="mae_m")
    pivot = pivot.loc[pivot[selected].sort_values().index, methods]
    fig, ax = plt.subplots(figsize=(11, 14))
    pivot.rename(columns=labels).plot.barh(ax=ax, color=[colors[m] for m in methods], width=.85)
    ax.set(xlabel="Station MAE (m)", ylabel="Gauge", title=f"{split.upper()}: all 39 gauges", ylim=(-.7, 38.7))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, fontsize=8, title=None)
    save(fig, base / "figures" / f"model2_station_mae_{split}.png")


def write_validation_details(base, train, validation, all_predictions, wrappers, winners, selected):
    from src.plotting import plt, save
    import matplotlib.dates as mdates

    importance = []
    for name, wrapper in wrappers.items():
        for feature, gain, count in zip(wrapper.features, wrapper.estimator.booster_.feature_importance("gain"),
                                        wrapper.estimator.booster_.feature_importance("split")):
            importance.append({"model": name, "feature": feature, "group": "weather" if feature.startswith("c2_") else "gauge_or_static",
                               "gain": float(gain), "split_count": int(count)})
    write_csv_new(base / "metrics/model2_feature_importance.csv", pd.DataFrame(importance))
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (family, name) in zip(axes, winners.items()):
        values = pd.DataFrame(importance)
        values = values[values.model.eq(name)].sort_values("gain")
        gain_sum = values.gain.sum()
        fraction = values.gain / gain_sum * 100 if gain_sum > 0 else values.gain
        ax.barh(values.feature, fraction, color=["#0072b2" if c == "weather" else "#888888" for c in values.group])
        ax.set(xlabel="Share of within-model training split gain (%)", title=f"{family}: {name}")
    fig.suptitle("Weather in blue; fitted associations, not physical causes")
    fig.tight_layout()
    save(fig, base / "figures/model2_feature_importance.png")

    # Station examples selected from TRAIN variability, not favourable validation performance.
    variation = train.groupby("station_id").stage_at_issue_m.std().sort_values(ascending=False)
    choices = list(dict.fromkeys([*variation.head(3).index.tolist(), variation.index[len(variation) // 2]]))
    write_json_new(base / "models/diagnostic_station_choices.json", {"definition": "three highest TRAIN stage standard deviations plus median-ranked gauge",
                                                                  "station_ids": choices})
    methods = ["original_model1", winners["absolute"], winners["delta"], "persistence"]
    labels = ["Original Model 1", "Model 2 absolute", "Model 2 delta", "Persistence"]
    colors = ["#d55e00", "#0072b2", "#cc79a7", "#009e73"]
    fig, axes = plt.subplots(len(choices), 1, figsize=(11, 3 * len(choices)), squeeze=False)
    for ax, station in zip(axes[:, 0], choices):
        observations = validation[validation.station_id.eq(station)].sort_values("forecast_date")
        ax.plot(pd.to_datetime(observations.forecast_date) + pd.Timedelta(days=1), observations[TARGET],
                "ko-", markersize=4, label="Observed next-day stage")
        for method, label, color in zip(methods, labels, colors):
            part = all_predictions[all_predictions.station_id.eq(station) & all_predictions.model.eq(method)].sort_values("forecast_date")
            ax.plot(pd.to_datetime(part.forecast_date) + pd.Timedelta(days=1), part.predicted_stage_next_day_m,
                    "o--", color=color, markersize=3, linewidth=1, label=label)
        ax.set(ylabel="Gauge stage (m)", title=observations.station_name.iloc[0])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Target date: forecast input date plus one day")
    fig.suptitle("VALIDATION station examples chosen by TRAIN variability", y=1.02)
    fig.tight_layout()
    save(fig, base / "figures/model2_validation_station_timeseries.png")


def write_methodology(base, metadata, table):
    config = metadata["configuration"]
    selected = metadata["selected_model2"]
    text = ("# Model 2 experiment summary\n\n"
            "## Objective and architecture\n\n"
            "Predict nominal next-day 09:00 gauge stage in metres using 4 gauge-history + 6 static "
            "catchment + 4 accumulated rainfall + 2 soil-moisture inputs (16 numerical predictors). "
            "The estimator is LightGBM gradient-boosted trees, without scaling or neural layers. "
            "Identifiers, targets, source dates, split labels, temperature and QA fields are not predictors.\n\n"
            f"Authoritative dataset: `{config['dataset']}`. SHA-256: `{metadata['dataset_sha256']}`. "
            "All 1,314 original station-days are retained with their supplied chronological assignments: "
            "TRAIN 700, VALIDATION 265, TEST 349; 39 gauges in each.\n\n"
            "## Exact feature order and units\n\n")
    for feature in MODEL2_FEATURES:
        if feature.startswith("c2_rainfall"):
            unit = "mm accumulated totals; 1-day interval unverified where applicable"
        elif feature.startswith("c2_soil"):
            unit = "presumed m3/m3; units and temporal aggregation not independently verified"
        elif feature in FEATURES[:4]:
            unit = "metres, per gauge dataset specification; independent source certification unavailable"
        elif feature in ("tree_fraction", "cropland_fraction", "builtup_fraction"):
            unit = "fraction [0,1]"
        else:
            unit = {"mean_elevation_m": "metres", "mean_slope_deg": "degrees", "catchment_area_km2": "km2"}[feature]
        text += f"- `{feature}`: {unit}.\n"
    text += ("\n## Weather observation windows\n\n"
             "Every source date is forecast day minus one. Under the user-supplied C2 script, rolling "
             "totals use [source date minus N days, source date), so they end forecast day minus two. "
             "For a forecast dated 2025-11-20, source date is 2025-11-19: 3-day rainfall covers Nov 16–18; "
             "7-day covers Nov 12–18; 14-day covers Nov 5–18. This deliberately retains the supplied "
             "additional one-day lag; no rebuild or mixing with another CSV occurs. Exact per-record "
             "windows are saved in metrics/model2_weather_windows.csv.\n\n"
             "There are 157 rows where the 1-day rainfall exceeds the 3-day total. Because the 1-day "
             "interval is unverified, these are warnings rather than corrections. The 3/7/14-day nesting "
             "checks pass under the supplied shared-window definition. Rainfall is nonnegative; soil "
             "moisture passes the provisional [0,1] screen. Publication timing remains unverified.\n\n"
             "## Frozen training and selection procedure\n\n"
             "Three predefined configurations are fitted for primary absolute-stage prediction, and "
             "the same three for secondary delta prediction. TRAIN alone fits all models; VALIDATION "
             "determines early stopping (40-round patience, maximum 500 or 250 trees). L2 objective "
             "models monitor validation RMSE; the MAE-loss model monitors validation MAE. Family winners "
             "and overall selection use reconstructed-stage validation MAE. Exact ties prefer fewer "
             "retained leaves, then the primary absolute representation, then candidate order. "
             "There is no TRAIN+VALIDATION refit and no target-derived input.\n\n"
             "Delta training response is next-day stage minus current stage. Adding current stage "
             "back produces river-stage predictions. All tables explicitly evaluate reconstructed "
             "stages, not an unlabelled delta response.\n\n"
             "One no-weather control is then fitted for each family winner, retaining the original "
             "ten features and the same parameters and selected boosting count, without further "
             "validation search. Controls are excluded from selection. Eight estimator fits total.\n\n"
             f"Family winners: `{metadata['family_winners']}`. Overall selected Model 2: **{selected}**. "
             f"Validation recommendation: **{metadata['recommended_method']}**. Persistence is preferred "
             "unless selected Model 2 validation MAE is strictly lower. No improvement claim depends "
             "solely on training error.\n\n"
             "## Validation results\n\n" + metric_markdown(table) + "\n\n"
             "MAE and RMSE are in metres, with identical sample sets and counts for comparisons. Signed "
             "bias is prediction minus observation. Macro MAE averages per-station MAE with equal gauge "
             "weight; mean station RMSE differs from pooled RMSE. Percentage gains use each comparator's "
             "error as denominator; negative values mean worse. The original Model 1 remains the "
             "preserved absolute-stage reference, not the separately revised delta model. Its saved "
             "validation predictions were reproduced from the original artifact on its original CSV. "
             "The weather export differs by up to about 1.5e-15 m in stage change. Tiny differences "
             "changed three old-model validation predictions by up to 0.1014 m through tree-threshold "
             "routing. Model 1 therefore uses its canonical original-source predictions aligned by keys "
             "and targets. Model 2 and its controls retain the weather CSV unchanged. No input columns "
             "are transplanted into Model 2 and no data-rebuild is performed.\n\n"
             "## Selected configuration\n\n```json\n" + json.dumps(metadata["selected_parameters"], indent=2) +
             "\n```\n\n"
             f"Retained trees: {metadata['selected_retained_trees']}. Seed: {metadata['seed']}. "
             "Complete fit parameters, feature order, source snapshot and dependency versions are in "
             "models/model2_metadata.json and configs/model2_config.yaml.\n\n"
             "## TEST policy\n\n"
             "At the time of this report the procedure is frozen and Model 2 TEST scoring has not "
             "occurred. The final guarded comparison produces a separate report and scores the two "
             "frozen family winners and two predeclared controls once. The project TEST was previously "
             "inspected during Model 1 work, so that assessment is descriptive only. No further "
             "selection follows TEST.\n\n## Limitations\n\n" + LIMITATIONS_MODEL2)
    write_text_new(base / "reports/model2_experiment_summary.md", text)
    location = base.relative_to(ROOT).as_posix()
    usage = ("# Model 2 local usage\n\n"
             "All commands run from C:\\Users\\LENOVO\\Desktop\\RP. Python 3.11 and installed "
             "dependencies are reused; no activation, package installation or downloads are needed.\n\n"
             "```powershell\n.\\.venv\\Scripts\\python.exe scripts/audit_model2_dataset.py\n"
             ".\\.venv\\Scripts\\python.exe -m pytest -q\n"
             ".\\.venv\\Scripts\\python.exe scripts/train_model2.py\n"
             f".\\.venv\\Scripts\\python.exe scripts/compare_models.py --run-dir {location}\n```\n\n"
             "The training command stops after validation and freezing. A later training invocation "
             "uses a new timestamped run directory and prints its path; pass that path to compare_models.py. "
             "For a completed run, compare_models.py displays saved metrics without rescoring. "
             "A started-but-incomplete TEST marker blocks automatic retry and preserves failure artifacts. "
             "Do not optimize further using known TEST results.\n\n"
             "models/model2_absolute.joblib is the primary absolute-stage wrapper; model2_delta.joblib "
             "is the secondary delta wrapper. model2_selected.joblib is the validation-selected wrapper. "
             "Use wrapper.predict_stage(frame) with the 16 input columns to obtain metres; a delta "
             "wrapper also needs stage_at_issue_m (already in those inputs). The raw selected estimator "
             "and native text models output their training response: add current stage when that "
             "response is delta. Only load trusted local joblib artifacts.\n\n"
             "Temperature is excluded. Read model2_data_audit.md for units and timing uncertainty, "
             "model2_experiment_summary.md for the frozen protocol and model1_vs_model2_comparison.md "
             "for final results. Original data, configurations and outputs are preserved.\n")
    write_text_new(base / "reports/model2_usage.md", usage)
