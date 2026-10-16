"""Six predeclared Model 2 fits and two controlled no-weather fits; no TEST scoring."""
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import zipfile

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baseline import persistence
from src.data_loader import (ROOT, local_path, sha256, timestamp, versions, write_csv_new,
                             write_json_new, write_text_new)
from src.features import FEATURES, TARGET
from src.model1_variants import require_partition
from src.model2_data import MODEL2_FEATURES, load_model2_config, load_and_audit, weather_windows
from src.model_comparison import (verify_manifest, original_model1_predictions, original_model1_frame, prediction_rows,
                                  summarize, per_station_metrics, make_figures,
                                  write_methodology, write_validation_details)


@dataclass
class Model2Regressor:
    representation: str
    estimator: object
    features: tuple[str, ...] = MODEL2_FEATURES
    is_control: bool = False

    def __post_init__(self):
        if self.representation not in ("absolute", "delta"):
            raise ValueError("Only absolute stage and separately labelled delta are approved.")
        self.features = tuple(self.features)
        expected = FEATURES if self.is_control else MODEL2_FEATURES
        if self.features != expected:
            raise ValueError("Wrong feature allowlist for primary Model 2 or matched control.")

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[list(self.features)].astype(float)
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Model 2 inputs are nonfinite.")
        return X

    def response(self, frame: pd.DataFrame) -> np.ndarray:
        y = frame[TARGET].to_numpy(dtype=float)
        return y - frame.stage_at_issue_m.to_numpy(dtype=float) if self.representation == "delta" else y

    def predict_stage(self, frame: pd.DataFrame) -> np.ndarray:
        predicted = np.asarray(self.estimator.predict(self.matrix(frame)), dtype=float)
        return predicted + frame.stage_at_issue_m.to_numpy(dtype=float) if self.representation == "delta" else predicted

    def tree_count(self) -> int:
        return self.estimator.booster_.num_trees()

    def leaf_count(self) -> int:
        return sum(tree["num_leaves"] for tree in self.estimator.booster_.dump_model()["tree_info"])


def fit_model2(train, validation, representation, parameters, patience=40, *, control=False, iterations=None):
    require_partition(train, "TRAIN")
    require_partition(validation, "VALIDATION")
    if control != (iterations is not None):
        raise ValueError("Only matched controls use fixed iterations, without extra validation search.")
    params = dict(parameters)
    if control:
        if iterations < 1:
            raise ValueError("A matched control needs a positive boosting count.")
        params["n_estimators"] = iterations
    wrapper = Model2Regressor(representation, lgb.LGBMRegressor(**params),
                              FEATURES if control else MODEL2_FEATURES, control)
    fit_options = {} if control else {
        "eval_set": [(wrapper.matrix(validation), wrapper.response(validation))],
        "eval_metric": params["metric"],
        "callbacks": [lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
    }
    wrapper.estimator.fit(wrapper.matrix(train), wrapper.response(train), **fit_options)
    return wrapper


def select_winner(table: pd.DataFrame, family: str | None = None) -> str:
    eligible = table[table.role.eq("candidate")]
    if family is not None:
        eligible = eligible[eligible.family.eq(family)]
    if eligible.empty:
        raise ValueError("No eligible Model 2 candidates.")
    # MAE is primary; complexity and family order only break exact ties.
    return str(eligible.sort_values(["mae_m", "fitted_leaves", "family_order", "candidate_order"],
                                    kind="stable").iloc[0].model)


def configure_model2_logging(base: Path):
    (base / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(base / "logs" / f"model2_{timestamp()}.log",
                                                     encoding="utf-8", mode="x"), logging.StreamHandler()], force=True)


def reserve_model2(config: dict, config_path: str) -> Path:
    parent = local_path(config["output_root"])
    preservation = parent / "models/original_preservation.json"
    if preservation.exists():
        manifest = json.loads(preservation.read_text(encoding="utf-8"))
        verify_manifest(manifest)
    else:
        # A fresh copy of the project can create its own preservation manifest.
        paths = [p for p in ROOT.iterdir() if p.is_file()]
        for directory in ("configs", "src", "scripts", "tests", "models", "results", "reports", "logs", "experiments/model1_diagnostics_v2"):
            paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(paths))}
    base = parent / "runs" / timestamp() if (parent / "models/run_manifest.json").exists() else parent
    for directory in ("configs", "models", "metrics", "predictions", "figures", "reports", "logs"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    if not (base / "models/original_preservation.json").exists():
        write_json_new(base / "models/original_preservation.json", manifest)
    write_json_new(base / "models/run_manifest.json", {"reserved_utc": timestamp(), "fit_budget": 8})
    write_text_new(base / "configs/model2_config.yaml", local_path(config_path).read_text(encoding="utf-8"))
    return base


def train_model2(config_path="configs/model2_config.yaml") -> Path:
    config = load_model2_config(config_path)
    base = reserve_model2(config, config_path)
    configure_model2_logging(base)
    logging.info("Model 2 run: %s", base)
    frame, audit = load_and_audit(config)
    report = base / "reports/model2_data_audit.md"
    if report.exists():
        report = base / "reports" / f"model2_data_audit_{timestamp()}.md"
    write_text_new(report, audit.markdown())
    if audit.errors:
        raise ValueError("Data audit failed; no fits performed.\n" + audit.markdown())
    write_csv_new(base / "metrics/model2_schema.csv", pd.DataFrame(audit.schema_table))
    write_csv_new(base / "metrics/model2_weather_windows.csv", weather_windows(frame))
    from importlib.metadata import distributions
    installed = sorted({d.metadata["Name"] + "==" + d.version for d in distributions()}, key=str.lower)
    write_text_new(base / "reports/environment_lock.txt", "# Python " + versions()["python"] + "\n" + "\n".join(installed) + "\n")
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    train = frame[frame.chronological_split.eq("TRAIN")].copy()
    validation = frame[frame.chronological_split.eq("VALIDATION")].copy()
    original_validation = original_model1_predictions(validation, "VALIDATION", reproduce=True)
    original_train_model = joblib.load(ROOT / "models/model1_static/lightgbm_model1.joblib")
    original_train = original_train_model.predict(original_model1_frame(train)[list(FEATURES)].astype(float))
    exported_validation_prediction = original_train_model.predict(validation[list(FEATURES)].astype(float))
    affected = np.abs(exported_validation_prediction - original_validation) > 1e-10
    drift = validation.loc[affected, ["record_id", "station_id", "forecast_date", "stage_change_1d_m"]].copy()
    canonical_validation = original_model1_frame(validation)
    drift["original_source_stage_change_1d_m"] = canonical_validation.loc[affected, "stage_change_1d_m"].to_numpy()
    drift["old_model_prediction_from_weather_export_m"] = exported_validation_prediction[affected]
    drift["original_model1_prediction_m"] = original_validation[affected]
    write_csv_new(base / "metrics/model1_source_export_roundoff.csv", drift)
    metadata_original = json.loads((ROOT / "models/model1_static/model1_metadata.json").read_text(encoding="utf-8"))
    verify_manifest({**metadata_original["source_code_sha256"],
                     "models/model1_static/lightgbm_model1.joblib": metadata_original["model_sha256"],
                     config["model1_dataset"]: metadata_original["dataset_sha256"]})
    logging.info("Model 1 validation artifact reproduced; persistence: %s",
                 summarize(prediction_rows(validation, persistence(validation), "persistence", original_validation)))

    wrappers, properties, validation_predictions, training_predictions = {}, {}, [], []
    for family_order, family in enumerate(config["family_order"]):
        for candidate_order, candidate in enumerate(config["candidates"]):
            parameters = {**config["base_parameters"], **candidate["parameters"]}
            name = f"model2_{family}_{candidate['id']}"
            model = fit_model2(train, validation, family, parameters, config["early_stopping_rounds"])
            wrappers[name] = model
            properties[name] = {"family": family, "role": "candidate", "family_order": family_order,
                                "candidate_order": candidate_order, "fitted_trees": model.tree_count(),
                                "fitted_leaves": model.leaf_count(), "objective": parameters["objective"],
                                "early_stopping_metric": parameters["metric"]}
            val_rows = prediction_rows(validation, model.predict_stage(validation), name, original_validation)
            validation_predictions.append(val_rows)
            training_predictions.append(prediction_rows(train, model.predict_stage(train), name, original_train))
            logging.info("%s: %s", name, summarize(val_rows))
    table = pd.DataFrame([{**summarize(rows), **properties[rows.model.iloc[0]]} for rows in validation_predictions])
    winners = {family: select_winner(table, family) for family in config["family_order"]}
    selected = select_winner(table)
    persistence_mae = summarize(prediction_rows(validation, persistence(validation), "persistence", original_validation))["mae_m"]
    selected_mae = float(table.set_index("model").loc[selected, "mae_m"])
    recommendation = selected if selected_mae < persistence_mae else "persistence"
    # Two post-selection controls have no influence on the winners/recommendation.
    for family, winner in winners.items():
        primary = wrappers[winner]
        control = fit_model2(train, validation, family, primary.estimator.get_params(),
                             control=True, iterations=primary.tree_count())
        name = f"control_{family}_no_weather"
        wrappers[name] = control
        properties[name] = {"family": family, "role": "matched_control", "family_order": -1,
                            "candidate_order": -1, "fitted_trees": control.tree_count(),
                            "fitted_leaves": control.leaf_count(), "objective": control.estimator.get_params()["objective"],
                            "early_stopping_metric": "none; fixed selected weather-model iterations"}
        validation_predictions.append(prediction_rows(validation, control.predict_stage(validation), name, original_validation))
        training_predictions.append(prediction_rows(train, control.predict_stage(train), name, original_train))
    for method, val_pred, train_pred in (("persistence", persistence(validation), persistence(train)),
                                         ("original_model1", original_validation, original_train)):
        validation_predictions.append(prediction_rows(validation, val_pred, method, original_validation))
        training_predictions.append(prediction_rows(train, train_pred, method, original_train))
        properties[method] = {"family": "reference", "role": "baseline", "family_order": -1,
                              "candidate_order": -1, "fitted_trees": 0, "fitted_leaves": 0,
                              "objective": "reference", "early_stopping_metric": "not searched in Model 2"}
    all_val = pd.concat(validation_predictions, ignore_index=True)
    validation_table = pd.DataFrame([{**summarize(rows), **properties[rows.model.iloc[0]]} for rows in validation_predictions])
    validation_table["family_winner"] = validation_table.model.isin(winners.values())
    validation_table["selected_model2"] = validation_table.model.eq(selected)
    validation_table["recommended"] = validation_table.model.eq(recommendation)
    write_csv_new(base / "metrics/model2_validation_metrics.csv", validation_table)
    write_csv_new(base / "predictions/model2_validation_candidates.csv", all_val)
    write_csv_new(base / "metrics/model2_training_metrics.csv", pd.DataFrame([summarize(rows) for rows in training_predictions]))
    station = per_station_metrics(all_val)
    write_csv_new(base / "metrics/model2_validation_station_metrics.csv", station)
    for family, winner in winners.items():
        rows = all_val[all_val.model.eq(winner)].copy()
        write_csv_new(base / "predictions" / f"model2_{family}_validation_predictions.csv", rows)
    chosen_rows = all_val[all_val.model.eq(selected)].copy()
    chosen_rows["recommended_method"] = recommendation
    chosen_rows["recommended_prediction_m"] = (chosen_rows.persistence_prediction_m
                                                if recommendation == "persistence" else chosen_rows.predicted_stage_next_day_m)
    write_csv_new(base / "predictions/model2_validation_predictions.csv", chosen_rows)
    focus = ["persistence", "original_model1", *winners.values(), "control_absolute_no_weather", "control_delta_no_weather"]
    make_figures(base, all_val[all_val.model.isin(focus)], validation_table,
                  station, selected, "validation")
    write_validation_details(base, train, validation, all_val, wrappers, winners, selected)
    for name, wrapper in wrappers.items():
        with (base / "models" / f"{name}.joblib").open("xb") as handle:
            joblib.dump(wrapper, handle)
    for family, winner in winners.items():
        model = wrappers[winner]
        with (base / "models" / f"model2_{family}.joblib").open("xb") as handle:
            joblib.dump(model, handle)
        write_text_new(base / "models" / f"model2_{family}.txt", model.estimator.booster_.model_to_string())
    chosen = wrappers[selected]
    with (base / "models/model2_selected.joblib").open("xb") as handle:
        joblib.dump(chosen, handle)
    with (base / "models/model2_selected_estimator.joblib").open("xb") as handle:
        joblib.dump(chosen.estimator, handle)
    write_text_new(base / "models/model2_selected.txt", chosen.estimator.booster_.model_to_string())
    source_files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")])
    source_manifest = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files}
    with zipfile.ZipFile(base / "models/source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in source_files:
            archive.write(file, file.relative_to(ROOT).as_posix())
    artifacts = [*base.glob("models/*.joblib"), *base.glob("models/*.txt"), base / "models/source_snapshot.zip",
                 base / "models/original_preservation.json", base / "configs/model2_config.yaml",
                 base / "metrics/model2_validation_metrics.csv", base / "predictions/model2_validation_predictions.csv",
                 base / "reports/environment_lock.txt",
                 local_path(config["dataset"]), local_path(config["model1_dataset"]), local_path(config_path)]
    metadata = {
        "frozen_utc": timestamp(), "configuration": config, "source_filename": config["dataset"],
        "dataset_sha256": sha256(local_path(config["dataset"])), "features": list(MODEL2_FEATURES),
        "target": TARGET, "seed": config["seed"], "split_counts": frame.chronological_split.value_counts().to_dict(),
        "family_winners": winners, "selected_model2": selected, "recommended_method": recommendation,
        "selected_representation": chosen.representation, "selected_parameters": chosen.estimator.get_params(),
        "selected_retained_trees": chosen.tree_count(), "actual_fits": 8,
        "all_fit_parameters": {name: wrapper.estimator.get_params() for name, wrapper in wrappers.items()},
        "dependencies": versions(), "source_code_sha256": source_manifest,
        "frozen_artifact_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in artifacts},
        "test_status": "not yet scored for Model 2; project holdout previously inspected",
        "native_model_output": "delta requires adding current stage; absolute directly outputs metres",
        "no_refit_on_train_plus_validation": True,
    }
    write_methodology(base, metadata, validation_table)
    verify_manifest(json.loads((base / "models/original_preservation.json").read_text(encoding="utf-8")))
    write_json_new(base / "models/model2_metadata.json", metadata)
    logging.info("Frozen family winners %s; selected %s; recommended %s. TEST not scored.", winners, selected, recommendation)
    return base
