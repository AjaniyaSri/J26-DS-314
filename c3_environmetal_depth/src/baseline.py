"""Persistence predicts tomorrow's stage equals today's observed stage."""
import numpy as np
import pandas as pd


def persistence(frame: pd.DataFrame) -> np.ndarray:
    return frame["stage_at_issue_m"].to_numpy(dtype=float, copy=True)
