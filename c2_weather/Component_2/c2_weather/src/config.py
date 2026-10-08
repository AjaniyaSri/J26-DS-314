from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def load_config(config_path=None):
    path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = path
    config["_project_root"] = path.parent.parent
    return config


def project_path(config, value):
    path = Path(value)
    if not path.is_absolute():
        path = config["_project_root"] / path
    return path.resolve()