"""Project-relative input, immutable outputs and reproducibility helpers."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
from importlib.metadata import version

import pandas as pd
import yaml

from src.features import FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("pandas", "numpy", "lightgbm", "scikit-learn", "matplotlib", "joblib", "PyYAML", "pytest")


def local_path(relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Use a project-relative path: {relative}")
    resolved = (ROOT / candidate).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path escapes RP: {relative}")
    return resolved


def load_config(path="configs/model1_config.yaml") -> dict:
    with local_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if tuple(config["features"]) != FEATURES or config["target"] != TARGET:
        raise ValueError("Configuration changed the approved feature/target contract.")
    if config["selection_metric"] != "mae_m":
        raise ValueError("The predefined model selection metric is validation MAE.")
    if config["model"]["random_state"] != config["seed"]:
        raise ValueError("Configuration seeds must agree.")
    local_path(config["dataset"])
    return config


def load_dataset(config: dict) -> pd.DataFrame:
    path = local_path(config["dataset"])
    if not path.is_file():
        raise FileNotFoundError(f"Source CSV is missing: {path}")
    # Keep raw strings so validation can report malformed cells, not coerce/drop them.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict:
    return {"python": platform.python_version(), **{p: version(p) for p in PACKAGES}}


def write_text_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)


def write_json_new(path: Path, value: dict) -> None:
    write_text_new(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    write_text_new(path, frame.to_csv(index=False))


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def reserve_run() -> Path:
    """First experiment uses requested paths; subsequent ones use new run folders."""
    occupied = (ROOT / "models/model1_static/run_manifest.json").exists()
    occupied |= any((ROOT / d).exists() and any((ROOT / d).iterdir()) for d in (
        "models/model1_static", "results/metrics", "results/predictions", "results/figures"))
    base = ROOT / "runs" / timestamp() if occupied else ROOT
    for d in ("models/model1_static", "results/metrics", "results/predictions",
              "results/figures", "reports", "logs"):
        (base / d).mkdir(parents=True, exist_ok=True)
    write_json_new(base / "models/model1_static/run_manifest.json", {
        "created_utc": timestamp(), "status": "reserved; see metadata and final report"})
    return base


def configure_logging(base: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(base / "logs" / f"model1_{timestamp()}.log",
                                                     encoding="utf-8", mode="x"),
                                  logging.StreamHandler()], force=True)
