"""Versioned diagnostics and bounded experiments; original artifacts stay immutable."""
import json
import logging
from pathlib import Path
import random
import zipfile

import joblib
import numpy as np
import pandas as pd
import yaml

from src.baseline import persistence
from src.data_loader import (ROOT, configure_logging, load_config, load_dataset, local_path,
                             sha256, timestamp, versions, write_csv_new, write_json_new,
                             write_text_new)
from src.data_validator import validate_dataset, require_valid
from src.evaluator import LIMITATIONS, metrics
from src.features import FEATURES, TARGET
from src.model1_variants import (StageVariant, choose_environmental, fit_variant,
                                 recommended_method)

DEFAULT_CONFIG = "experiments/model1_diagnostics_v2/configs/model1_diagnostics_v2.yaml"
DEFAULT_RUN = "experiments/model1_diagnostics_v2"


def original_files() -> list[Path]:
    metadata = json.loads((ROOT / "models/model1_static/model1_metadata.json").read_text(encoding="utf-8"))
    paths = [ROOT / p for p in metadata["source_code_sha256"]]
    paths += [ROOT / p for p in ("README.md", "requirements.txt", ".gitignore",
                                "configs/model1_config.yaml", metadata["source_filename"])]
    paths += [ROOT / "tests" / p for p in ("test_dataset_schema.py", "test_feature_integrity.py",
                                           "test_split_integrity.py", "test_no_target_leakage.py")]
    for directory in ("models", "results", "reports", "logs"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    return sorted(set(paths))


def fingerprint(paths: list[Path]) -> dict:
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def verify_fingerprints(manifest: dict) -> None:
    changed = [name for name, digest in manifest.items()
               if not local_path(name).is_file() or sha256(local_path(name)) != digest]
    if changed:
        raise ValueError(f"Preservation/frozen-file checksum mismatch: {changed}")


def load_v2_config(path: str) -> dict:
    config = yaml.safe_load(local_path(path).read_text(encoding="utf-8"))
    if config["selection_metric"] != "validation_stage_mae_m":
        raise ValueError("Select using reconstructed-stage VALIDATION MAE only.")
    if config["test_policy"] != "previously_inspected_holdout_final_descriptive_assessment_only":
        raise ValueError("The original TEST set must be acknowledged as previously inspected.")
    if len(config["candidates"]) != 6 or len({c["id"] for c in config["candidates"]}) != 6:
        raise ValueError("The approved plan permits exactly six revised primary candidates.")
    if config["near_tie_mae_m"] < 0:
        raise ValueError("Invalid simplicity band.")
    if config["base_parameters"]["random_state"] != config["seed"]:
        raise ValueError("Model and experiment seeds must agree.")
    for candidate in config["candidates"]:
        if candidate["representation"] not in ("absolute", "delta", "relative"):
            raise ValueError("Unapproved representation.")
        if not candidate["id"].replace("_", "").isalnum():
            raise ValueError("Candidate IDs must be safe filename components.")
    local_path(config["dataset"])
    return config


def reserve_experiment(config_path: str) -> Path:
    source = local_path(config_path)
    parent = local_path(DEFAULT_RUN)
    if (parent / "models/run_manifest.json").exists():
        parent = parent / "runs" / timestamp()
    for name in ("configs", "metrics", "predictions", "figures", "models", "logs", "reports"):
        (parent / name).mkdir(parents=True, exist_ok=True)
    destination = parent / "configs/model1_diagnostics_v2.yaml"
    if source != destination:
        write_text_new(destination, source.read_text(encoding="utf-8"))
    write_json_new(parent / "models/run_manifest.json", {"reserved_utc": timestamp()})
    return parent


def stage_predictions(frame: pd.DataFrame, predicted, method: str) -> pd.DataFrame:
    columns = ["record_id", "station_id", "station_name", "basin", "forecast_date",
               "chronological_split", *FEATURES, TARGET]
    out = frame[columns].copy()
    out["model"] = method
    out["predicted_stage_next_day_m"] = np.asarray(predicted, dtype=float)
    out["persistence_prediction_m"] = persistence(frame)
    out["prediction_error_m"] = out.predicted_stage_next_day_m - out[TARGET]
    return out


def station_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["chronological_split", "model", "station_id", "station_name", "basin"]
    rows = []
    for key, group in predictions.groupby(columns, sort=False):
        rows.append({**dict(zip(columns, key)),
                     **metrics(group[TARGET], group.predicted_stage_next_day_m)})
    result = pd.DataFrame(rows)
    baseline = result[result.model.eq("persistence")][["chronological_split", "station_id", "mae_m"]]
    baseline = baseline.rename(columns={"mae_m": "persistence_mae_m"})
    result = result.merge(baseline, on=["chronological_split", "station_id"], validate="many_to_one")
    result["excess_mae_over_persistence_m"] = result.mae_m - result.persistence_mae_m
    return result


def summarize_method(predictions: pd.DataFrame, method: str) -> dict:
    station = predictions.groupby("station_id", sort=False)
    per_station = [metrics(group[TARGET], group.predicted_stage_next_day_m) for _, group in station]
    baseline = metrics(predictions[TARGET], predictions.persistence_prediction_m)
    score = metrics(predictions[TARGET], predictions.predicted_stage_next_day_m)
    return {"model": method, **score,
            "station_macro_mae_m": float(np.mean([r["mae_m"] for r in per_station])),
            "station_mean_rmse_m": float(np.mean([r["rmse_m"] for r in per_station])),
            "n_stations": len(per_station),
            "mae_improvement_over_persistence_pct": 100 * (1 - score["mae_m"] / baseline["mae_m"])
            if baseline["mae_m"] else None}


def diagnostic_tables(base: Path, frame: pd.DataFrame, predictions: pd.DataFrame,
                      variants: dict[str, StageVariant]) -> None:
    """All diagnostic summaries use TRAIN/VALIDATION, never TEST outcomes."""
    train = frame[frame.chronological_split.eq("TRAIN")]
    used = frame[frame.chronological_split.isin(["TRAIN", "VALIDATION"])].copy()
    used["next_day_delta_m"] = used[TARGET] - used.stage_at_issue_m
    description = used.groupby("chronological_split")[[*FEATURES[:4], TARGET, "next_day_delta_m"]].describe()
    write_csv_new(base / "metrics/stage_distributions.csv", description.reset_index().pipe(
        lambda d: d.set_axis(["_".join(filter(None, c)) if isinstance(c, tuple) else c for c in d.columns], axis=1)))
    corr = train[list(FEATURES[:4])].corr().rename_axis("feature").reset_index()
    write_csv_new(base / "metrics/train_history_correlations.csv", corr)
    medians = train.groupby(["station_id", "station_name", "basin"]).stage_at_issue_m.agg(["count", "median", "mean", "std", "min", "max"]).reset_index()
    write_csv_new(base / "metrics/train_station_stage_references.csv", medians)
    static = train.groupby(["station_id", "station_name", "basin"])[list(FEATURES[4:])].first().reset_index()
    write_csv_new(base / "metrics/train_static_environment.csv", static)

    # Stage bins are frozen from TRAIN current-stage quantiles, then applied to
    # observed stage purely for retrospective error stratification, not prediction.
    inner = np.unique(train.stage_at_issue_m.quantile([0.25, 0.5, 0.75]).to_numpy())
    bins = [-np.inf, *inner.tolist(), np.inf]
    labelled = predictions.copy()
    labelled["observed_stage_bin"] = pd.cut(labelled[TARGET], bins=bins).astype(str)
    labelled["issue_trend"] = np.select(
        [labelled.stage_change_1d_m > 1e-6, labelled.stage_change_1d_m < -1e-6],
        ["rising", "falling"], default="steady")
    labelled["realized_next_day_trend"] = np.select(
        [labelled[TARGET] - labelled.stage_at_issue_m > 1e-6,
         labelled[TARGET] - labelled.stage_at_issue_m < -1e-6],
        ["rising", "falling"], default="steady")
    grouped = []
    for dimension in ("basin", "observed_stage_bin", "issue_trend", "realized_next_day_trend"):
        for key, part in labelled.groupby(["chronological_split", "model", dimension], sort=False):
            grouped.append({"split": key[0], "model": key[1], "dimension": dimension,
                            "group": key[2], **metrics(part[TARGET], part.predicted_stage_next_day_m)})
    write_csv_new(base / "metrics/grouped_error_analysis.csv", pd.DataFrame(grouped))
    residuals = []
    for (split, method), part in predictions.groupby(["chronological_split", "model"], sort=False):
        values = part.prediction_error_m
        residuals.append({"split": split, "model": method, "n": len(part),
                          **{f"q{q:g}": float(values.quantile(q)) for q in (0, .1, .25, .5, .75, .9, 1)}})
    write_csv_new(base / "metrics/residual_quantiles.csv", pd.DataFrame(residuals))
    importance = []
    for method, variant in variants.items():
        for feature, gain, count in zip(variant.features,
                                        variant.estimator.booster_.feature_importance("gain"),
                                        variant.estimator.booster_.feature_importance("split")):
            importance.append({"model": method, "feature": feature, "gain": float(gain), "split_count": int(count)})
    write_csv_new(base / "metrics/feature_importance.csv", pd.DataFrame(importance))
    write_json_new(base / "models/diagnostic_definitions.json", {
        "stage_bin_internal_edges_m": inner.tolist(), "stage_bin_source": "TRAIN current-stage quartiles",
        "issue_trend": "sign of current stage minus lag 1; available at issuance",
        "realized_next_day_trend": "sign of observed next-day change; retrospective diagnostic only",
        "station_reference": "TRAIN current-stage median; subtract from current/lag1/lag2 and target, not change"})


def run_diagnostics(config_path: str = DEFAULT_CONFIG) -> Path:
    config = load_v2_config(config_path)
    preservation = fingerprint(original_files())
    base = reserve_experiment(config_path)
    configure_logging(base)
    write_json_new(base / "models/original_preservation_manifest.json", preservation)
    logging.info("Versioned experiment: %s", base)
    original_meta = json.loads((ROOT / "models/model1_static/model1_metadata.json").read_text(encoding="utf-8"))
    original_config = load_config(config["original_config"])
    if original_config != original_meta["configuration"] or config["dataset"] != original_meta["source_filename"]:
        raise ValueError("Original configuration or approved dataset changed.")
    verify_fingerprints({**original_meta["source_code_sha256"],
                         config["dataset"]: original_meta["dataset_sha256"],
                         config["original_model"]: original_meta["model_sha256"]})
    frame, audit = validate_dataset(load_dataset(original_config))
    write_text_new(base / "reports/data_audit_v2.md", audit.markdown())
    if audit.errors:
        raise ValueError(audit.markdown())
    train = frame[frame.chronological_split.eq("TRAIN")].copy()
    validation = frame[frame.chronological_split.eq("VALIDATION")].copy()
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    original = StageVariant("absolute", estimator=joblib.load(local_path(config["original_model"])))
    saved_validation = pd.read_csv(ROOT / "results/predictions/model1_validation_predictions.csv")
    if saved_validation.record_id.tolist() != validation.record_id.tolist():
        raise ValueError("Original saved validation row ordering differs.")
    np.testing.assert_allclose(original.predict_stage(validation), saved_validation.predicted_stage_next_day_m,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(persistence(validation), saved_validation.persistence_prediction_m, rtol=0, atol=0)
    reproduced = fit_variant(train, validation, "absolute", original_meta["model_parameters"],
                             config["early_stopping_rounds"], config["minimum_station_training_rows"])
    differences = {}
    for split, part in (("TRAIN", train), ("VALIDATION", validation)):
        before, after = original.predict_stage(part), reproduced.predict_stage(part)
        differences[split] = float(np.max(np.abs(before - after)))
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-10)
    write_json_new(base / "models/original_reproduction.json", {
        "max_prediction_difference_m": differences, "best_iteration": reproduced.estimator.best_iteration_,
        "test_not_repredicted": True})
    logging.info("Original selected fit reproduced: %s", differences)

    variants = {"original_lightgbm": reproduced}
    properties = {"original_lightgbm": {"eligible_primary": True, "candidate_order": 0,
                                        "representation": "absolute", "transformation_count": 0}}
    for order, candidate in enumerate(config["candidates"], start=1):
        params = {**config["base_parameters"], **candidate["parameters"]}
        variant = fit_variant(train, validation, candidate["representation"], params,
                              config["early_stopping_rounds"], config["minimum_station_training_rows"])
        name = candidate["id"]
        variants[name] = variant
        properties[name] = {"eligible_primary": True, "candidate_order": order,
                            "representation": candidate["representation"],
                            "transformation_count": {"absolute": 0, "delta": 1, "relative": 2}[candidate["representation"]]}
        logging.info("%s: validation stage metrics %s", name,
                     metrics(validation[TARGET], variant.predict_stage(validation)))

    def comparison_rows():
        rows = [dict(summarize_method(stage_predictions(validation, persistence(validation), "persistence"), "persistence"),
                     eligible_primary=False, candidate_order=-1, representation="persistence",
                     transformation_count=0, fitted_leaf_count=0, fitted_trees=0)]
        for name, variant in variants.items():
            rows.append({**summarize_method(stage_predictions(validation, variant.predict_stage(validation), name), name),
                         **properties[name], "fitted_leaf_count": variant.fitted_leaf_count(),
                         "fitted_trees": variant.estimator.booster_.num_trees()})
        return pd.DataFrame(rows)

    comparison = comparison_rows()
    selected = choose_environmental(comparison, config["near_tie_mae_m"])
    recommendation = recommended_method(selected, comparison, config["near_tie_mae_m"])
    primary = variants[selected]
    ablation = fit_variant(train, validation, primary.representation, primary.estimator.get_params(),
                           config["early_stopping_rounds"], config["minimum_station_training_rows"],
                           features=FEATURES[:4], fixed_iterations=primary.estimator.booster_.num_trees())
    variants["history_only_ablation"] = ablation
    properties["history_only_ablation"] = {"eligible_primary": False, "candidate_order": 7,
                                           "representation": primary.representation,
                                           "transformation_count": properties[selected]["transformation_count"]}
    comparison = comparison_rows()
    comparison["selected_environmental"] = comparison.model.eq(selected)
    comparison["recommended"] = comparison.model.eq(recommendation)
    write_csv_new(base / "metrics/model1_validation_comparison.csv", comparison)
    logging.info("Selected environmental: %s; recommendation: %s", selected, recommendation)
    logging.info("Validation comparison:\n%s", comparison.to_string(index=False))

    all_predictions = []
    for split, part in (("TRAIN", train), ("VALIDATION", validation)):
        all_predictions.append(stage_predictions(part, persistence(part), "persistence"))
        for name, variant in variants.items():
            all_predictions.append(stage_predictions(part, variant.predict_stage(part), name))
    predictions = pd.concat(all_predictions, ignore_index=True)
    write_csv_new(base / "predictions/model1_train_validation_diagnostics.csv", predictions)
    per_station = station_errors(predictions)
    write_csv_new(base / "metrics/model1_station_error_analysis.csv", per_station)
    training = pd.DataFrame([summarize_method(g, name) for name, g in
                             predictions[predictions.chronological_split.eq("TRAIN")].groupby("model", sort=False)])
    write_csv_new(base / "metrics/model1_training_comparison.csv", training)
    chosen_predictions = predictions[(predictions.chronological_split.eq("VALIDATION")) & predictions.model.eq(selected)].copy()
    chosen_predictions["recommended_method"] = recommendation
    chosen_predictions["recommended_prediction_m"] = (chosen_predictions.persistence_prediction_m
                                                       if recommendation == "persistence" else chosen_predictions.predicted_stage_next_day_m)
    write_csv_new(base / "predictions/model1_improved_validation_predictions.csv", chosen_predictions)
    diagnostic_tables(base, frame, predictions, variants)
    diagnostic_figures(base, predictions, comparison, per_station, selected, train)
    diagnostic_report(base, frame, predictions, comparison, per_station, selected, recommendation, config)

    for name, variant in variants.items():
        with (base / "models" / f"{name}.joblib").open("xb") as handle:
            joblib.dump(variant, handle)
    with (base / "models/selected_environmental_model.joblib").open("xb") as handle:
        joblib.dump(primary, handle)
    write_text_new(base / "models/selected_environmental_model.txt", primary.estimator.booster_.model_to_string())
    write_json_new(base / "models/frozen_station_references.json", primary.references)
    code_files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")])
    code_manifest = fingerprint(code_files)
    with zipfile.ZipFile(base / "models/source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in code_files:
            archive.write(file, file.relative_to(ROOT).as_posix())
    verify_fingerprints(preservation)
    write_json_new(base / "models/frozen_experiment.json", {
        "frozen_utc": timestamp(), "configuration": config, "selected_environmental": selected,
        "recommended_method": recommendation, "features": list(FEATURES), "target": TARGET,
        "representation": primary.representation, "parameters": primary.estimator.get_params(),
        "fitted_trees": primary.estimator.booster_.num_trees(),
        "station_reference_definition": "TRAIN current-stage median; minimum 5 rows; no fallback",
        "station_references": primary.references, "dataset_sha256": sha256(local_path(config["dataset"])),
        "source_code_sha256": code_manifest, "dependencies": versions(),
        "model_sha256": sha256(base / "models/selected_environmental_model.joblib"),
        "config_sha256": sha256(base / "configs/model1_diagnostics_v2.yaml"),
        "validation_comparison_sha256": sha256(base / "metrics/model1_validation_comparison.csv"),
        "actual_fits": 8, "test_status": "not evaluated in v2; original TEST was previously inspected",
    })
    logging.info("Procedure frozen. TEST was not scored. Run evaluate_model1_v2.py for one descriptive assessment.")
    return base


def evaluate_revised(base: Path) -> pd.DataFrame:
    started = base / "models/test_evaluation_started.json"
    if started.exists():
        raise FileExistsError("Revised TEST assessment already started; read saved outputs without rescoring.")
    meta = json.loads((base / "models/frozen_experiment.json").read_text(encoding="utf-8"))
    preservation = json.loads((base / "models/original_preservation_manifest.json").read_text(encoding="utf-8"))
    verify_fingerprints(preservation)
    verify_fingerprints(meta["source_code_sha256"])
    if sha256(base / "configs/model1_diagnostics_v2.yaml") != meta["config_sha256"]:
        raise ValueError("Frozen configuration changed.")
    if sha256(base / "metrics/model1_validation_comparison.csv") != meta["validation_comparison_sha256"]:
        raise ValueError("Frozen validation comparison changed.")
    model_path = base / "models/selected_environmental_model.joblib"
    if sha256(model_path) != meta["model_sha256"]:
        raise ValueError("Frozen selected model changed.")
    if sha256(local_path(meta["configuration"]["dataset"])) != meta["dataset_sha256"]:
        raise ValueError("Dataset changed after freezing.")
    if versions() != meta["dependencies"]:
        raise ValueError("Dependency versions changed after freezing.")
    variant = joblib.load(model_path)
    if variant.features != FEATURES or tuple(variant.estimator.feature_name_) != FEATURES:
        raise ValueError("Primary environmental model lost the ten-feature contract.")
    frame, _ = require_valid(load_dataset(meta["configuration"]))
    test = frame[frame.chronological_split.eq("TEST")]
    # All selection, plotting and diagnostic development has ended before this marker.
    write_json_new(started, {"started_utc": timestamp(), "status": "descriptive; previously inspected holdout"})
    chosen = stage_predictions(test, variant.predict_stage(test), meta["selected_environmental"])
    chosen["recommended_method"] = meta["recommended_method"]
    chosen["recommended_prediction_m"] = (chosen.persistence_prediction_m if meta["recommended_method"] == "persistence"
                                            else chosen.predicted_stage_next_day_m)
    write_csv_new(base / "predictions/model1_improved_test_predictions.csv", chosen)
    rows = [summarize_method(chosen, meta["selected_environmental"])]
    baseline = stage_predictions(test, persistence(test), "persistence")
    rows.append(summarize_method(baseline, "persistence"))
    # Reuse original saved TEST predictions; do not fit/reselect/repredict original.
    saved = pd.read_csv(ROOT / "results/predictions/model1_test_predictions.csv")
    if saved.record_id.tolist() != test.record_id.tolist():
        raise ValueError("Original TEST row IDs no longer align.")
    np.testing.assert_allclose(saved[TARGET], test[TARGET], atol=1e-12, rtol=0)
    original = stage_predictions(test, saved.predicted_stage_next_day_m, "original_lightgbm")
    if meta["selected_environmental"] != "original_lightgbm":
        rows.append(summarize_method(original, "original_lightgbm"))
    comparison = pd.DataFrame(rows)
    write_csv_new(base / "metrics/model1_test_descriptive_comparison.csv", comparison)
    station_frames = [chosen, baseline]
    if meta["selected_environmental"] != "original_lightgbm":
        station_frames.append(original)
    write_csv_new(base / "metrics/model1_test_station_descriptive.csv",
                  station_errors(pd.concat(station_frames, ignore_index=True)))
    final_report(base, meta, comparison)
    verify_fingerprints(preservation)
    write_json_new(base / "models/test_evaluation_complete.json", {"completed_utc": timestamp(),
        "original_artifacts_unchanged": True, "warning": "descriptive TEST assessment, not independent confirmation"})
    logging.info("Final descriptive TEST comparison:\n%s", comparison.to_string(index=False))
    return comparison


def markdown_table(table: pd.DataFrame) -> str:
    columns = ["model", "n_samples", "mae_m", "rmse_m", "mean_signed_error_m",
               "station_macro_mae_m", "mae_improvement_over_persistence_pct"]
    lines = ["| Model | n | MAE (m) | RMSE (m) | Bias (m) | Macro MAE (m) | MAE gain vs persistence (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join([str(row[0]), str(row[1]),
                                         *[f"{value:.6f}" if value is not None else "undefined" for value in row[2:]]]) + " |")
    return "\n".join(lines)


def diagnostic_report(base, frame, predictions, comparison, per_station, selected, recommendation, config):
    train = frame[frame.chronological_split.eq("TRAIN")]
    val = frame[frame.chronological_split.eq("VALIDATION")]
    training = pd.read_csv(base / "metrics/model1_training_comparison.csv")
    reference = per_station[(per_station.chronological_split.eq("VALIDATION")) & per_station.model.eq("original_lightgbm")]
    winner = per_station[(per_station.chronological_split.eq("VALIDATION")) & per_station.model.eq(selected)]
    original_wins = int(reference.excess_mae_over_persistence_m.lt(0).sum())
    selected_wins = int(winner.excess_mae_over_persistence_m.lt(0).sum())
    total_ss = ((train.stage_at_issue_m - train.stage_at_issue_m.mean()) ** 2).sum()
    within_ss = ((train.stage_at_issue_m - train.groupby("station_id").stage_at_issue_m.transform("mean")) ** 2).sum()
    fraction = 1 - within_ss / total_ss
    static = train.groupby("station_id")[list(FEATURES[4:])].first()
    references = train.groupby("station_id").stage_at_issue_m.median()
    importance = pd.read_csv(base / "metrics/feature_importance.csv")
    original_importance = importance[importance.model.eq("original_lightgbm")].set_index("feature")
    stage_gain = original_importance.loc["stage_at_issue_m", "gain"] / original_importance.gain.sum()
    selected_row = comparison.set_index("model").loc[selected]
    ablation_row = comparison.set_index("model").loc["history_only_ablation"]
    title = "# Model 1 diagnostic report — version 2\n\n"
    report = title + "## Original findings and reproduction\n\n"
    report += ("Original files were fingerprinted before the run. The selected original configuration "
               "was refitted on TRAIN and early-stopped using VALIDATION. Predictions agree with the "
               "original model within 1e-10 m on TRAIN and VALIDATION; the precise differences are saved "
               "in models/original_reproduction.json. No baseline, alignment, feature-contract or metric "
               "implementation error was found. No preprocessing was fitted to future partitions.\n\n")
    report += "## Training diagnostics\n\n" + markdown_table(training) + "\n\n"
    report += ("These are in-sample training metrics, not independent forecasting performance. "
               "Training errors must not be compared to validation errors without accounting for the "
               "different stage-change distributions.\n\n## Validation comparison\n\n"
               + markdown_table(comparison) + "\n\n")
    report += (f"Selected environmental model: **{selected}**. Recommended forecasting method: "
               f"**{recommendation}**. Primary candidates all retain the six static environmental inputs. "
               "The history-only ablation is excluded from primary selection.\n\n"
               f"The predefined near-tie band is {config['near_tie_mae_m']:.3f} m. Among eligible models "
               "within that band of the minimum validation MAE, select the fewest fitted tree leaves, "
               "then fewer representation transformations, then predefined candidate order. Persistence "
               "is recommended unless the selected environmental model beats it by more than that band. "
               "This is a pragmatic simplicity preference, not a confidence interval or significance test.\n\n")
    report += "## Diagnostic findings supported by these data\n\n"
    report += (f"- The original model beats persistence at {original_wins}/39 validation gauges; the selected "
               f"environmental model does so at {selected_wins}/39. Gauge-by-gauge differences remain in "
               "metrics/model1_station_error_analysis.csv.\n"
               f"- Mean absolute observed next-day change is {(train[TARGET]-train.stage_at_issue_m).abs().mean():.6f} m "
               f"in TRAIN and {(val[TARGET]-val.stage_at_issue_m).abs().mean():.6f} m in VALIDATION. "
               "The later quieter window makes persistence a stronger comparator.\n"
               f"- Mean next-day change is {(train[TARGET]-train.stage_at_issue_m).mean():+.6f} m in TRAIN and "
               f"{(val[TARGET]-val.stage_at_issue_m).mean():+.6f} m in VALIDATION.\n"
               f"- Using TRAIN only, {fraction:.2%} of current-stage sum of squares lies between station means. "
               f"Station current-stage medians span {references.min():.3f}–{references.max():.3f} m. "
               "This mixes physical river differences with possible datum differences; it cannot identify their causes.\n"
               f"- The six static features have {len(static.drop_duplicates())} distinct joint signatures among 39 stations. "
               "They can act as gauge proxies, but uniqueness alone does not prove how the model uses them.\n"
               f"- Current stage contributes {stage_gain:.2%} of original fitted split gain. "
               "Gain is an association measure and is affected by correlated inputs.\n\n")
    report += "## Controlled environmental ablation\n\n"
    report += (f"The four-history-input ablation uses the selected primary model's **{selected_row.representation}** "
               "representation, same objective, hyperparameters and frozen boosting iteration count. "
               "It is fitted once on TRAIN, with no extra validation search. The primary model's validation "
               f"MAE is {selected_row.mae_m:.6f} m; the matched ablation's is {ablation_row.mae_m:.6f} m. "
               f"The ablation-minus-primary MAE difference is {ablation_row.mae_m-selected_row.mae_m:+.6f} m. "
               "A positive difference favours environmental inputs for this fitted comparison; a negative "
               "difference favours history alone. This single short-window, selection-conditioned comparison "
               "does not establish causal environmental effects or robust incremental value.\n\n")
    report += "## Proposed explanations, distinct from established findings\n\n"
    report += ("The absolute-stage tree model approximates an already informative current-stage reading and "
               "can introduce smoothing errors during quiet periods. TRAIN has stronger excursions than "
               "VALIDATION, and the validation overprediction is consistent with a changed regime and/or "
               "pooled station effects. The experiment tests useful alternative representations; it does not "
               "uniquely identify the mechanism. Low training mean bias does not justify applying a validation-derived "
               "bias correction and scoring on those same validation rows. Bias correction was therefore not fitted.\n\n")
    report += "## Exact representations and comparisons\n\n"
    report += ("- Absolute: predict next-day stage directly.\n"
               "- Delta: fit next-day stage minus current stage; add current stage to predictions. "
               "All reported metrics use reconstructed next-day stages in metres.\n"
               "- Relative: fit station medians from TRAIN current observations only; subtract the same "
               "reference from current stage, lag 1, lag 2 and target. Leave daily change and environmental "
               "inputs unchanged. Add the reference back at prediction. At least five training rows are "
               "required per gauge; missing/unseen gauge references fail without a future-data fallback. "
               "The supplied gauges have 8–21 TRAIN rows each.\n"
               "- Two representations share standard and smaller L2 configurations. Two further absolute "
               "variants test strong regularization and MAE-aligned loss. Early stopping monitors the configured "
               "validation response metric: RMSE for L2 models and MAE for the MAE-loss model. Since the "
               "same offset is added to observed and predicted responses, these errors equal reconstructed-stage "
               "errors algebraically; reporting and comparison still explicitly reconstruct stages.\n"
               "- Stage bins use TRAIN current-stage quartiles. Binning by observed label and realized next-day "
               "trend is retrospective diagnosis only. Separate issue-time rising/falling summaries use stage_change_1d_m.\n\n")
    report += "## Reproducibility, outputs and remaining uncertainties\n\n"
    report += ("There are eight fits: one original reproduction, six revised candidates and one matched ablation. "
               "No TRAIN+VALIDATION refit is performed. Configuration, models, references, source snapshot, "
               "checksums, package versions, predictions, residuals and grouped errors are saved locally. "
               "The primary file names contain 'improved' as requested; those names are not evidence of improvement. "
               "Recommendation and measured improvements are reported separately.\n\n"
               "Repeated model comparison reuses seven validation dates, so selection optimism is possible. "
               "The original TEST set was previously inspected. A later v2 assessment is descriptive only; "
               "new future periods are needed for independent confirmation. No TEST outcomes were used "
               "to choose these fitted candidates.\n\n## Research limitations\n\n" + LIMITATIONS)
    write_text_new(base / "reports/model1_diagnostic_report.md", report)


def diagnostic_figures(base, predictions, comparison, per_station, selected, train):
    from src.plotting import plt, save
    import matplotlib.dates as mdates

    methods = list(dict.fromkeys(["original_lightgbm", selected, "persistence"]))
    validation = predictions[predictions.chronological_split.eq("VALIDATION")]
    labels = {"original_lightgbm": "Original absolute model", "persistence": "Persistence", selected: selected}
    colors = {"original_lightgbm": "#d55e00", selected: "#0072b2", "persistence": "#009e73"}
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        part = validation[validation.model.eq(method)]
        ax.scatter(part[TARGET], part.predicted_stage_next_day_m, s=15, alpha=.6, color=colors[method])
        limits = [min(part[TARGET].min(), part.predicted_stage_next_day_m.min()),
                  max(part[TARGET].max(), part.predicted_stage_next_day_m.max())]
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlabel="Observed next-day stage (m)", ylabel="Predicted stage (m)", title=labels[method])
    fig.suptitle("VALIDATION: identical 265 station-days")
    fig.tight_layout()
    save(fig, base / "figures/validation_observed_vs_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ("TRAIN", "VALIDATION")):
        subset = predictions[predictions.chronological_split.eq(split) & predictions.model.isin(methods)]
        edges = np.linspace(subset.prediction_error_m.min(), subset.prediction_error_m.max(), 45)
        for method in methods:
            values = subset.loc[subset.model.eq(method), "prediction_error_m"]
            ax.hist(values, bins=edges, histtype="step", density=True, linewidth=1.5,
                    color=colors[method], label=labels[method])
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Prediction minus observation (m)", ylabel="Density", title=split)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, base / "figures/prediction_error_histograms.png")

    part = per_station[per_station.chronological_split.eq("VALIDATION") & per_station.model.isin(methods)]
    pivot = part.pivot(index="station_name", columns="model", values="mae_m")
    pivot = pivot.loc[pivot.original_lightgbm.sort_values().index, methods]
    fig, ax = plt.subplots(figsize=(11, 13))
    pivot.rename(columns=labels).plot.barh(ax=ax, color=[colors[m] for m in methods], width=.8)
    ax.set(xlabel="Validation station MAE (m)", ylabel="Gauge", title="All 39 gauges: identical rows per method")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=8, title=None)
    save(fig, base / "figures/validation_station_mae.png")

    original_station = part[part.model.eq("original_lightgbm")].sort_values("excess_mae_over_persistence_m", ascending=False)
    choices = original_station.head(3).station_id.tolist()
    median_station = original_station.iloc[len(original_station) // 2].station_id
    choices = list(dict.fromkeys([*choices, median_station]))
    fig, axes = plt.subplots(len(choices), 1, figsize=(11, 3 * len(choices)), squeeze=False)
    for ax, station in zip(axes[:, 0], choices):
        subset = validation[validation.station_id.eq(station)]
        observations = subset[subset.model.eq("persistence")].sort_values("forecast_date")
        dates = pd.to_datetime(observations.forecast_date) + pd.Timedelta(days=1)
        ax.plot(dates, observations[TARGET], "ko-", label="Observed next-day stage", linewidth=1.5)
        for method in methods:
            values = subset[subset.model.eq(method)].sort_values("forecast_date")
            ax.plot(pd.to_datetime(values.forecast_date) + pd.Timedelta(days=1), values.predicted_stage_next_day_m,
                    "o--", markersize=3, linewidth=1, color=colors[method], label=labels[method])
        ax.set(ylabel="Gauge stage (m)", title=observations.station_name.iloc[0])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.17), ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Target date (input date + one day)")
    fig.suptitle("Validation examples: three largest original excess errors and median-ranked gauge", y=1.02)
    fig.tight_layout()
    save(fig, base / "figures/validation_station_timeseries.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("mae_m", "rmse_m"), ("MAE", "RMSE")):
        ordered = comparison.sort_values(metric, ascending=False)
        bars = ["#009e73" if m == "persistence" else "#0072b2" if m == selected else "#888888" for m in ordered.model]
        ax.barh(ordered.model, ordered[metric], color=bars)
        ax.set(xlabel=f"Validation {title} (m)", title=f"All measured variants: {title}")
    fig.tight_layout()
    save(fig, base / "figures/validation_model_comparison.png")

    importance = pd.read_csv(base / "metrics/feature_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, method in zip(axes, ("original_lightgbm", selected)):
        values = importance[importance.model.eq(method)].sort_values("gain")
        ax.barh(values.feature, values.gain)
        ax.set(xlabel="Training split gain", title=labels[method])
    fig.suptitle("Fitted associations, not hydrological causes")
    fig.tight_layout()
    save(fig, base / "figures/feature_importance_comparison.png")

    static = train.groupby("station_id")[list(FEATURES[4:])].first()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, feature in zip(axes.flat, FEATURES[4:]):
        ax.hist(static[feature], bins=12, color="#0072b2", edgecolor="white")
        ax.set(xlabel=feature, ylabel="Gauges", title="TRAIN: 39 gauge values")
    fig.tight_layout()
    save(fig, base / "figures/static_environment_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, part in predictions[predictions.model.eq("persistence")].groupby("chronological_split"):
        axes[0].hist(part.stage_at_issue_m, bins=30, histtype="step", density=True, label=split)
        axes[1].hist(part[TARGET] - part.stage_at_issue_m, bins=30, histtype="step", density=True, label=split)
    axes[0].set(xlabel="Current stage (m)", ylabel="Density", title="Gauge-stage inputs")
    axes[1].set(xlabel="Realized next-day change (m)", ylabel="Density", title="Retrospective response distributions")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    save(fig, base / "figures/stage_and_change_distributions.png")


def final_report(base, metadata, test_comparison):
    validation = pd.read_csv(base / "metrics/model1_validation_comparison.csv")
    chosen = metadata["selected_environmental"]
    original_val = validation.set_index("model").loc["original_lightgbm"]
    chosen_val = validation.set_index("model").loc[chosen]
    gain_original = 100 * (1 - chosen_val.mae_m / original_val.mae_m)
    ablation = validation.set_index("model").loc["history_only_ablation"]
    text = "# Model 1 final comparison — version 2\n\n"
    text += ("## Interpretation\n\n"
             f"Validation-selected environmental model: **{chosen}**. Recommended method: "
             f"**{metadata['recommended_method']}**. This decision was frozen before the v2 TEST assessment. "
             f"Selected environmental validation MAE is {chosen_val.mae_m:.6f} m, compared with "
             f"{original_val.mae_m:.6f} m for the original model ({gain_original:.2f}% reduction). "
             f"Its validation MAE improvement over persistence is {chosen_val.mae_improvement_over_persistence_pct:.2f}%. "
             "Negative improvement means worse. Training completion alone is not success evidence.\n\n"
             "## Validation evidence used for selection\n\n" + markdown_table(validation) + "\n\n"
             "## Final descriptive TEST assessment\n\n"
             "**This TEST set was already inspected in the original experiment. These revised results "
             "are descriptive, not an independent confirmation or a new untouched-test claim.** "
             "Only the frozen environmental model and persistence were newly scored once. Original "
             "TEST predictions were read from the preserved original file. There was no TEST-based "
             "selection, calibration or refitting.\n\n" + markdown_table(test_comparison) + "\n\n")
    text += ("## Environmental ablation and uncertainty\n\n"
             f"Matched history-only validation MAE: {ablation.mae_m:.6f} m; selected environmental MAE: "
             f"{chosen_val.mae_m:.6f} m. This ablation holds representation, training procedure and boosting "
             "iteration count fixed while removing environmental inputs. It is conditional on selection "
             "of the environmental configuration and is not causal evidence. Any gauge-level wins and "
             "losses are retained in station-error tables. Seven validation dates, repeated comparison "
             "and correlated gauges limit confidence. New future periods and a longer seasonal record "
             "are needed to establish generalization.\n\n"
             "## Preservation and reproducibility\n\n"
             "The original CSV, configuration, source files, models, reports, logs, predictions and metrics "
             "were verified against the pre-run preservation manifest. No original files were overwritten. "
             "All v2 artifacts remain under this experiment directory. Frozen metadata records the configuration, "
             "feature order, representation, station references, dependency versions and checksums. "
             "No Git operations, uploads, package installations or external downloads occurred in this v2 work.\n\n"
             "## Reproduce or inspect locally\n\n"
             "From the RP root:\n\n```powershell\n"
             ".\\.venv\\Scripts\\python.exe scripts/diagnose_model1_v2.py\n"
             f".\\.venv\\Scripts\\python.exe scripts/evaluate_model1_v2.py --run-dir {base.relative_to(ROOT).as_posix()}\n"
             "```\n\n"
             "The first command creates a fresh versioned run if this experiment already exists, and "
             "prints its path. It stops after validation selection and freezing; it does not score TEST. "
             "Pass that printed path to the second command for a deliberate final descriptive assessment. "
             "For the completed run shown above, the second command only displays saved results. "
             "Do not iteratively tune using the known TEST outcomes.\n\n"
             "## Research limitations\n\n" + LIMITATIONS)
    write_text_new(base / "reports/model1_final_comparison.md", text)
