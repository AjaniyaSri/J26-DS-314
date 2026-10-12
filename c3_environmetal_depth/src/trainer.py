"""Small predefined validation search; TEST does not participate in selection."""
import logging
import random

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baseline import persistence
from src.data_loader import (ROOT, configure_logging, load_config, load_dataset, local_path,
                             reserve_run, sha256, timestamp, versions, write_csv_new,
                             write_json_new, write_text_new)
from src.data_validator import validate_dataset
from src.evaluator import evaluate_frozen, metric_rows, metrics, prediction_frame
from src.features import FEATURES, TARGET, feature_matrix


def select_model(train: pd.DataFrame, validation: pd.DataFrame, config: dict):
    """Only these two partitions enter selection; no TEST argument exists."""
    X_train, X_valid = feature_matrix(train), feature_matrix(validation)
    y_train, y_valid = train[TARGET], validation[TARGET]
    best_model, best_score, best_id = None, float("inf"), None
    trials = []
    for candidate_id, overrides in enumerate(config["candidates"], start=1):
        parameters = {**config["model"], **overrides}
        model = lgb.LGBMRegressor(**parameters)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(config["early_stopping_rounds"],
                                               first_metric_only=True, verbose=False)])
        scores = metrics(y_valid, model.predict(X_valid))
        trials.append({"candidate": candidate_id, "best_iteration": model.best_iteration_,
                       **parameters, **scores})
        logging.info("Candidate %s: validation MAE %.6f m; RMSE %.6f m; trees %s",
                     candidate_id, scores["mae_m"], scores["rmse_m"], model.best_iteration_)
        if scores["mae_m"] < best_score:
            best_model, best_score, best_id = model, scores["mae_m"], candidate_id
    if best_model is None:
        raise ValueError("No candidate models were configured.")
    return best_model, best_id, pd.DataFrame(trials)


def train_experiment(config_path="configs/model1_config.yaml"):
    config = load_config(config_path)
    base = reserve_run()
    configure_logging(base)
    logging.info("Run directory: %s", base)
    source = local_path(config["dataset"])
    initial_hash = sha256(source)
    frame, audit = validate_dataset(load_dataset(config))
    audit_path = base / "reports/data_quality_audit.md"
    if audit_path.exists():
        audit_path = base / "reports" / f"data_quality_audit_{timestamp()}.md"
    write_text_new(audit_path, audit.markdown())
    if audit.errors:
        raise ValueError(f"Data validation failed; training stopped. See {audit_path}\n" + audit.markdown())
    logging.info("Dataset shape %s; split counts %s", frame.shape,
                 frame.chronological_split.value_counts().to_dict())
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    train = frame.loc[frame.chronological_split.eq("TRAIN")].copy()
    validation = frame.loc[frame.chronological_split.eq("VALIDATION")].copy()
    # No TEST scores or predictions are calculated until model freezing is complete.
    baseline = metrics(validation[TARGET], persistence(validation))
    logging.info("Validation persistence: %s", baseline)
    write_csv_new(base / "results/metrics/persistence_validation_before_selection.csv",
                  pd.DataFrame([{"split": "VALIDATION", "model": "persistence", **baseline}]))
    model, candidate, trials = select_model(train, validation, config)
    write_csv_new(base / "results/metrics/model1_validation_search.csv", trials)
    validation_predictions = prediction_frame(validation, model.predict(feature_matrix(validation)))
    write_csv_new(base / "results/predictions/model1_validation_predictions.csv", validation_predictions)
    write_csv_new(base / "results/metrics/model1_validation_metrics.csv", metric_rows(validation_predictions))
    model_dir = base / "models/model1_static"
    path = model_dir / "lightgbm_model1.joblib"
    with path.open("xb") as handle:
        joblib.dump(model, handle)
    write_text_new(model_dir / "lightgbm_model1.txt", model.booster_.model_to_string())
    write_text_new(model_dir / "model1_config.yaml", local_path(config_path).read_text(encoding="utf-8"))
    if sha256(source) != initial_hash:
        raise RuntimeError("Source CSV changed during training; final evaluation prohibited.")
    source_files = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")])
    metadata = {
        "frozen_utc": timestamp(), "features": list(FEATURES), "target": TARGET,
        "random_seed": config["seed"], "source_filename": config["dataset"],
        "dataset_sha256": initial_hash, "dataset_shape": list(frame.shape),
        "split_counts": {k: int(v) for k, v in frame.chronological_split.value_counts().items()},
        "configuration": config, "model_parameters": model.get_params(),
        "selected_candidate": candidate, "best_iteration": int(model.best_iteration_),
        "selection_metric": "validation MAE; earlier candidate breaks exact ties",
        "early_stopping_metric": "validation RMSE", "fit_partition": "TRAIN only; no refit",
        "dependencies": versions(), "model_sha256": sha256(path),
        "source_code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
        "test_policy": "Evaluate once after this metadata is written; never tune on TEST.",
    }
    write_json_new(model_dir / "model1_metadata.json", metadata)
    logging.info("Selected candidate %s frozen before final TEST evaluation.", candidate)
    evaluate_frozen(base)
    return base
