"""C3 Model 1: static-environment-conditioned river-stage forecasting."""
import os
from pathlib import Path

# Set this before LightGBM can import matplotlib through its optional integrations.
_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_root / ".cache/matplotlib"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_root / ".cache/joblib"))
