"""C1 local dataset audit and no-change persistence benchmark.

Run from J26-DS-314 repository root:
    python c1_satellite/scripts/01_c1_local_audit.py

Uses only local DMC labels; it does not extract satellite features or train ML.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "c1_satellite" / "local_data"
OUT = ROOT / "c1_satellite" / "outputs"
LABELS = DATA / "FloodRisk_DMC_StationDay_LABELLED_1429.csv"
MASTER = DATA / "FloodRisk_DMC_StationDay_MASTER_1600.csv"
EXPECTED = {"TRAIN": 809, "VALIDATION": 269, "TEST": 351}


def load(path: Path):
    if not path.is_file():
        raise SystemExit(
            f"Missing: {path}\n"
            "Extract the original FloodRisk_DMC_StationDay_TEAM_SHARE ZIP into "
            "c1_satellite/local_data (not into shared/)."
        )
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def evaluate(rows):
    errors = [float(r["target_stage_next_day_m"]) - float(r["stage_at_issue_m"]) for r in rows]
    if not errors:
        raise ValueError("No rows to evaluate")
    return {"n": len(errors), "mae_m": round(sum(map(abs, errors)) / len(errors), 6),
            "rmse_m": round(math.sqrt(sum(x*x for x in errors) / len(errors)), 6)}


def main():
    labels, master = load(LABELS), load(MASTER)
    if len(labels) != 1429 or len(master) != 1600:
        raise ValueError(f"Unexpected dataset row counts: {len(labels)} labels, {len(master)} master")
    for name, rows in [("labels", labels), ("master", master)]:
        keys = [(r["station_id"], r["forecast_date"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate station-date keys in {name}")
    count = Counter(r["chronological_split"].strip().upper() for r in labels)
    if dict(count) != EXPECTED:
        raise ValueError(f"Split mismatch: {count}")
    nmaster = len({r["station_id"] for r in master})
    nlabelled = len({r["station_id"] for r in labels})
    if (nmaster, nlabelled) != (40, 39):
        raise ValueError(f"Unexpected station counts: {nmaster} master, {nlabelled} labelled")

    for row in labels:
        current = float(row["stage_at_issue_m"])
        target = float(row["target_stage_next_day_m"])
        delta = float(row["target_delta_stage_m"])
        if abs((target-current)-delta) > 1e-5:
            raise ValueError(f"Target delta mismatch on {row['record_id']}")
        if row["next_day_date"] != str(date.fromisoformat(row["forecast_date"]) + timedelta(days=1)):
            raise ValueError(f"Nonconsecutive target on {row['record_id']}")

    scores = {split: evaluate([r for r in labels if r["chronological_split"].strip().upper() == split])
              for split in EXPECTED}
    test = scores["TEST"]
    assert abs(test["mae_m"] - 0.2172) < .0001
    assert abs(test["rmse_m"] - 0.4066) < .0001
    paths = sorted({r.get("catchment_polygon_path", "") for r in master if r.get("catchment_polygon_path")})
    missing = [p for p in paths if not (ROOT / p).is_file()]
    OUT.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(Counter)
    for r in master:
        counts[r["station_id"]]["master_dates"] += 1
    for r in labels:
        counts[r["station_id"]]["valid_pairs"] += 1
        counts[r["station_id"]][r["chronological_split"].strip().lower()] += 1

    with (OUT / "c1_station_coverage.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["station_id", "master_dates", "valid_pairs", "train", "validation", "test"]
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for sid, d in sorted(counts.items()):
            wr.writerow({"station_id": sid, **{k: d[k] for k in fields if k != "station_id"}})

    with (OUT / "c1_persistence_test_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        fields = ["record_id", "station_id", "forecast_date", "model_id", "split",
                  "predicted_next_day_stage_m", "true_next_day_stage_m", "absolute_error_m"]
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for row in labels:
            if row["chronological_split"].strip().upper() != "TEST":
                continue
            today, truth = float(row["stage_at_issue_m"]), float(row["target_stage_next_day_m"])
            wr.writerow({"record_id": row["record_id"], "station_id": row["station_id"],
                         "forecast_date": row["forecast_date"], "model_id": "PERSISTENCE",
                         "split": "TEST", "predicted_next_day_stage_m": today,
                         "true_next_day_stage_m": truth, "absolute_error_m": round(abs(truth-today), 8)})

    report = {"experiment": "DATA_AUDIT_AND_STAGE_ONLY_PERSISTENCE",
              "master_rows": len(master), "labelled_pairs": len(labels),
              "master_stations": nmaster, "labelled_stations": nlabelled,
              "split_results": scores, "candidate_catchments_referenced": len(paths),
              "catchment_files_not_found_locally": missing,
              "satellite_features_extracted": False, "ml_model_trained": False}
    (OUT / "c1_baseline_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== C1 DATA AUDIT / PERSISTENCE BASELINE ===")
    print(f"Station-day master : {len(master)} rows / {nmaster} stations")
    print(f"Valid target pairs  : {len(labels)} rows / {nlabelled} stations")
    for split, result in scores.items():
        print(f"{split:11s} n={result['n']:3d}  MAE={result['mae_m']:.4f} m  RMSE={result['rmse_m']:.4f} m")
    print(f"Polygon paths referenced: {len(paths)}; not found in local repo: {len(missing)}")
    print(f"Saved 3 result files in {OUT}")
    print("NEXT: Acquire actual Sentinel-1 and ASCAT SWI data. No satellite model was trained.\n")


if __name__ == "__main__":
    main()
