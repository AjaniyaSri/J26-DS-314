"""An explicit feature allowlist prevents accidental metadata/label leakage."""
import pandas as pd

FEATURES = (
    "stage_at_issue_m", "stage_lag_1d_m", "stage_lag_2d_m", "stage_change_1d_m",
    "mean_elevation_m", "mean_slope_deg", "tree_fraction", "cropland_fraction",
    "builtup_fraction", "catchment_area_km2",
)
TARGET = "target_stage_next_day_m"
METADATA = (
    "record_id", "station_id", "station_name", "basin", "forecast_date",
    "chronological_split", "catchment_qa_status", "environment_feature_status",
    "issue_time_assumed_local", "source_publication_time_verified",
)
SCHEMA = (*METADATA[:6], *FEATURES, TARGET, *METADATA[6:])


def feature_matrix(frame: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """Require the exact, ordered ten-feature contract; never infer inputs."""
    if tuple(features) != FEATURES:
        raise ValueError("Model 1 requires exactly the ten approved features in order.")
    return frame.loc[:, list(FEATURES)].astype(float)
