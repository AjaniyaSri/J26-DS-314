"""Explicit, invertible stage representations for the versioned C3 experiment.

IDs are lookup keys for TRAIN-only centering, never estimator inputs. A delta
label is constructed only as a training/validation response, never as a feature.
"""
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURES, TARGET


def require_partition(frame: pd.DataFrame, expected: str) -> None:
    if frame.empty or not frame.chronological_split.eq(expected).all():
        raise ValueError(f"Expected a nonempty {expected}-only partition.")


@dataclass
class StageVariant:
    representation: str
    features: tuple[str, ...] = FEATURES
    references: dict[str, float] = field(default_factory=dict)
    estimator: object | None = None

    def __post_init__(self):
        if self.representation not in ("absolute", "delta", "relative"):
            raise ValueError("Unknown stage representation.")
        if tuple(self.features) not in (FEATURES, FEATURES[:4]):
            raise ValueError("Only the approved ten inputs or four-input ablation are allowed.")
        self.features = tuple(self.features)

    def fit_reference(self, train: pd.DataFrame, minimum_rows: int = 5):
        require_partition(train, "TRAIN")
        if self.representation == "relative":
            counts = train.groupby("station_id").stage_at_issue_m.count()
            insufficient = counts[counts < minimum_rows]
            if not insufficient.empty:
                raise ValueError(f"Insufficient TRAIN observations for centering: {insufficient.to_dict()}")
            self.references = train.groupby("station_id").stage_at_issue_m.median().astype(float).to_dict()
        return self

    def offset(self, frame: pd.DataFrame) -> np.ndarray:
        if self.representation == "delta":
            return frame.stage_at_issue_m.to_numpy(dtype=float)
        if self.representation == "relative":
            reference = frame.station_id.map(self.references)
            if reference.isna().any():
                raise ValueError(f"No frozen TRAIN reference for stations: "
                                 f"{frame.loc[reference.isna(), 'station_id'].unique().tolist()}")
            return reference.to_numpy(dtype=float)
        return np.zeros(len(frame), dtype=float)

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.loc[:, list(self.features)].astype(float).copy()
        if self.representation == "relative":
            reference = self.offset(frame)
            # Daily change is already a difference; never subtract the datum from it.
            for col in FEATURES[:3]:
                X[col] = X[col] - reference
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Nonfinite variant input.")
        return X

    def response(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[TARGET].to_numpy(dtype=float) - self.offset(frame)

    def reconstruct(self, frame: pd.DataFrame, response_prediction) -> np.ndarray:
        predicted = np.asarray(response_prediction, dtype=float)
        if predicted.shape != (len(frame),):
            raise ValueError("Response predictions are not aligned with rows.")
        return predicted + self.offset(frame)

    def predict_stage(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Variant has not been fitted.")
        # Prediction works on frames with no future-label column.
        return self.reconstruct(frame, self.estimator.predict(self.matrix(frame)))

    def fitted_leaf_count(self) -> int:
        return sum(tree["num_leaves"] for tree in self.estimator.booster_.dump_model()["tree_info"])


def fit_variant(train: pd.DataFrame, validation: pd.DataFrame, representation: str,
                parameters: dict, patience: int = 40, minimum_rows: int = 5,
                features=FEATURES, fixed_iterations: int | None = None) -> StageVariant:
    require_partition(train, "TRAIN")
    require_partition(validation, "VALIDATION")
    variant = StageVariant(representation, tuple(features)).fit_reference(train, minimum_rows)
    params = dict(parameters)
    if fixed_iterations is not None:
        if fixed_iterations < 1:
            raise ValueError("Fixed boosting iterations must be positive.")
        params["n_estimators"] = fixed_iterations
    variant.estimator = lgb.LGBMRegressor(**params)
    if fixed_iterations is None:
        variant.estimator.fit(
            variant.matrix(train), variant.response(train),
            eval_set=[(variant.matrix(validation), variant.response(validation))],
            eval_metric=params.get("metric", "rmse"),
            callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
        )
    else:
        # Matched ablation: no extra validation tuning or iteration search.
        variant.estimator.fit(variant.matrix(train), variant.response(train))
    return variant


def choose_environmental(comparison: pd.DataFrame, band: float) -> str:
    """Select only ten-input environmental candidates, never the ablation."""
    candidates = comparison.loc[comparison.eligible_primary.astype(bool)].copy()
    if candidates.empty or band < 0:
        raise ValueError("No eligible environmental candidates or invalid tie band.")
    near_best = candidates[candidates.mae_m <= candidates.mae_m.min() + band]
    return str(near_best.sort_values(["fitted_leaf_count", "transformation_count", "candidate_order"],
                                    kind="stable").iloc[0].model)


def recommended_method(selected: str, comparison: pd.DataFrame, band: float) -> str:
    table = comparison.set_index("model")
    gain = float(table.loc["persistence", "mae_m"] - table.loc[selected, "mae_m"])
    return selected if gain > band else "persistence"
