"""Read source CSV and save a new audit report without training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import ROOT, load_config, load_dataset, timestamp, write_text_new
from src.data_validator import validate_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model1_config.yaml")
    args = parser.parse_args()
    _, audit = validate_dataset(load_dataset(load_config(args.config)))
    path = ROOT / "reports/data_quality_audit.md"
    if path.exists():
        path = ROOT / "reports" / f"data_quality_audit_{timestamp()}.md"
    write_text_new(path, audit.markdown())
    print(audit.markdown())
    print(f"Saved: {path}")
    return 1 if audit.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
