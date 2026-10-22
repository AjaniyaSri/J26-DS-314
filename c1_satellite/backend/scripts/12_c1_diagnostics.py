#!/usr/bin/env python3
"""Read-only QA of C1 satellite prototype; no model fitting or TEST tuning.

Expected existing outputs from phases 5, 10 and 11. Run from repo root:
python c1_satellite/scripts/12_c1_diagnostics.py

Reports source-data quality, complete-vs-masked SWI history, station/split
coverage, stage-change regime shifts and station-wise performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "local_data"
OUT = ROOT / "outputs"
KEY = ["station_id", "forecast_date"]
SPLITS = ["TRAIN", "VALIDATION", "TEST"]


def csv_req(p: Path, required, unique=None):
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run earlier C1 phase first")
    df = pd.read_csv(p, dtype={"station_id": "string", "forecast_date": "string"})
    missed = sorted(set(required) - set(df.columns))
    if missed:
        raise ValueError(f"{p.name} missing columns: {missed}")
    if unique and df.duplicated(unique).any():
        raise ValueError(f"{p.name}: duplicate keys {unique}")
    return df


def n(x):
    return pd.to_numeric(x, errors="coerce")


def safe(v):
    if isinstance(v, dict):
        return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if not np.isfinite(v) else float(v)
    if pd.isna(v):
        return None
    return v


def summarize_movement(labels):
    data=labels.copy()
    data["absolute_daily_change_m"] = n(data["target_delta_stage_m"]).abs()
    summary=[]
    for split in SPLITS:
        z=data[data.chronological_split==split]
        summary.append({"split":split,"n":len(z),
          "mean_abs_stage_change_m":float(z.absolute_daily_change_m.mean()),
          "median_abs_stage_change_m":float(z.absolute_daily_change_m.median()),
          "p90_abs_stage_change_m":float(z.absolute_daily_change_m.quantile(0.9))})
    return summary


def swi_audit(labels, features, observations, sar):
    flags=[f"swi_valid_d{k}" for k in range(1,8)]
    if any(c not in features.columns for c in flags):
        raise ValueError("SWI feature file lacks swi_valid_d1..d7; run phase 10")
    f=labels[KEY+["chronological_split"]].merge(features, on=KEY, how="left", validate="one_to_one")
    sar_small=sar[KEY+["sar_multi_local_available", "sar_latest_local_available"]]
    f=f.merge(sar_small,on=KEY,how="left",validate="one_to_one")
    for c in flags+["sar_multi_local_available", "sar_latest_local_available", "swi_7day_complete", "swi_feature_available"]:
        f[c]=n(f[c]).fillna(0).astype(int)
    f["valid_swi_days_of_7"]=f[flags].sum(axis=1)
    f["has_all_7"]=(f.valid_swi_days_of_7==7)
    if ((f["swi_7day_complete"]==1)!=(f.has_all_7)).any():
        raise ValueError("SWI complete-history flag inconsistent with seven day-mask columns")
    f["has_real_sar_multi_and_latest_swi"]=(f.sar_multi_local_available==1)&(f.swi_feature_available==1)
    buckets=[]
    for split in SPLITS:
        z=f[f.chronological_split==split]
        rows={"split":split,"labelled_n":int(len(z)),"sar_single_n":int(z.sar_latest_local_available.sum()),
            "sar_multi_n":int(z.sar_multi_local_available.sum()),
            "latest_swi_n":int(z.swi_feature_available.sum()),
            "complete_7_day_n":int(z.has_all_7.sum()),
            "sar_multi_and_latest_swi_n":int(z.has_real_sar_multi_and_latest_swi.sum())}
        for k in range(8):rows[f"swi_exactly_{k}_valid_days_n"]=int((z.valid_swi_days_of_7==k).sum())
        for k in range(1,8):rows[f"swi_day{k}_valid_n"]=int(z[flags[k-1]].sum())
        buckets.append(rows)
    # Count number of failed band-quality tests. LOW_QFLAG is a documented quality classification.
    low=observations[observations.quality_status=="LOW_QFLAG"].copy()
    qflag_summary={"LOW_QFLAG_station_dates":int(len(low))}
    for c in ("QFLAG001","QFLAG005","QFLAG010"):
        if c in low.columns:
            qflag_summary[c+"_below_50_in_low_qflag_rows"]=int((n(low[c])<50).sum())
    quality_counts={str(k):int(v) for k,v in observations.quality_status.value_counts(dropna=False).items()}
    station_daily=(f.groupby("station_id",dropna=False).agg(labelled_records=("forecast_date","size"),
        sar_multi_records=("sar_multi_local_available","sum"),
        swi_latest_records=("swi_feature_available","sum"),
        swi_complete_7day_records=("has_all_7","sum"),
        mean_valid_swi_days=("valid_swi_days_of_7","mean")
       ).reset_index().sort_values("station_id"))
    return buckets, qflag_summary, quality_counts, station_daily, f


def prediction_audit(labels, preds, metrics):
    needed=["record_id","station_id","forecast_date","chronological_split","model_id",
            "actual_next_day_stage_m","predicted_next_day_stage_m"]
    if preds.duplicated(["record_id","model_id"]).any():
        raise ValueError("Prediction file has duplicate record_id + model_id keys")
    actual_truth=labels[["record_id","target_stage_next_day_m"]].copy()
    pred=preds.merge(actual_truth,on="record_id",how="left",validate="many_to_one")
    if pred.target_stage_next_day_m.isna().any():
        raise ValueError("Predictions contain unknown record IDs")
    pred["actual_next_day_stage_m"]=n(pred.actual_next_day_stage_m)
    pred["predicted_next_day_stage_m"]=n(pred.predicted_next_day_stage_m)
    pred["target_stage_next_day_m"]=n(pred.target_stage_next_day_m)
    if not np.allclose(pred.actual_next_day_stage_m, pred.target_stage_next_day_m, atol=1e-7):
        raise ValueError("Predictions have actual labels inconsistent with canonical DMC labels")
    pred["abs_error_m"]=(pred.actual_next_day_stage_m-pred.predicted_next_day_stage_m).abs()
    models=set(pred.model_id.unique())
    for split in SPLITS:
        cohort=pred[pred.chronological_split==split]
        expected=set(cohort[cohort.model_id=="PERSISTENCE"].record_id)
        if set(cohort.model_id.unique()) != models:
            raise ValueError(f"Unfair comparison in {split}: model list differs from other splits")
        for model in models:
            if set(cohort[cohort.model_id==model].record_id)!=expected:
                raise ValueError(f"Unfair comparison in {split}: {model} uses a different record cohort")
    station_metrics=pred.groupby(["chronological_split","station_id","model_id"],dropna=False).agg(
        records=("record_id","size"),MAE_m=("abs_error_m","mean")).reset_index()
    # Avoid presenting station-level results with only one or two dates as robust findings.
    stage_results=[]
    for model in ["PERSISTENCE","STAGE_ONLY","A","B","C","D"]:
        rows=pred[(pred.chronological_split=="TEST") & (pred.model_id==model)]
        if len(rows)==0:continue
        stage_results.append({"model_id":model,"test_n":len(rows),"MAE_m":round(float(rows.abs_error_m.mean()),4)})
    # TEST errors here are descriptive only; do NOT use them to select/tune methods.
    station_metrics=station_metrics.sort_values(["chronological_split","station_id","model_id"])
    return station_metrics, stage_results


def run():
    labels=csv_req(DATA/"FloodRisk_DMC_StationDay_LABELLED_1429.csv",
        KEY+["record_id","chronological_split","target_stage_next_day_m","target_delta_stage_m","stage_at_issue_m"],unique=KEY)
    sar=csv_req(OUT/"c1_sar_station_day_features.csv",KEY+["sar_multi_local_available","sar_latest_local_available"],unique=KEY)
    swif=csv_req(OUT/"c1_swi_v4_scaled_station_day_features.csv",KEY+["swi_7day_complete","swi_feature_available"],unique=KEY)
    obs=csv_req(OUT/"c1_swi_v4_scaled_station_daily.csv",["station_id","observation_date","quality_status"],unique=["station_id","observation_date"])
    pred=csv_req(OUT/"c1_ablation_predictions.csv",["record_id","station_id","forecast_date","chronological_split","model_id","actual_next_day_stage_m","predicted_next_day_stage_m"])
    metrics=csv_req(OUT/"c1_ablation_metrics.csv",["model_id","split","n","MAE_m","RMSE_m"])
    movements=summarize_movement(labels)
    coverage,qflags,quality,station_coverage,features=swi_audit(labels,swif,obs,sar)
    station_scores, test_scores=prediction_audit(labels,pred,metrics)
    OUTPUT=OUT/"c1_phase12_diagnostics"
    OUTPUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUTPUT/"coverage_by_split.csv",index=False)
    station_coverage.to_csv(OUTPUT/"station_swi_sar_coverage.csv",index=False)
    station_scores.to_csv(OUTPUT/"station_model_errors.csv",index=False)
    pd.DataFrame(movements).to_csv(OUTPUT/"stage_change_regime_by_split.csv",index=False)
    report={"note":"Read-only retrospective research audit; do not use TEST to tune or claim operational validation.",
            "source_quality_statuses":quality,"low_qflag_breakdown":qflags,
            "coverage_by_split":coverage,"daily_stage_change_by_split":movements,
            "descriptive_test_MAE":test_scores,
            "methodological_alerts": [
                "A forecast history window is NOT a fully observed 7-day sequence unless every validity mask is 1.",
                "Masked SWI days must not be described as measured SWI values.",
                "Same SAR scene / SWI grid cell can serve multiple station-days: rows are correlated.",
                "Different train/test hydrological regimes may explain worse test generalization.",
                "Keep TEST locked; use TRAIN and VALIDATION for diagnostic choices only."]}
    (OUTPUT/"phase12_summary.json").write_text(json.dumps(safe(report),indent=2),encoding="utf-8")
    print("=== C1 PHASE 12 – READ-ONLY RESEARCH DIAGNOSTIC ===")
    for s in coverage:
        print(f"{s['split']:10s} labels={s['labelled_n']:3d} SARmulti={s['sar_multi_n']:3d} SWIlatest={s['latest_swi_n']:3d} "+
              f"complete7={s['complete_7_day_n']:3d} SARmulti+SWIlatest={s['sar_multi_and_latest_swi_n']:3d}")
        print("  SWI valid-days distribution (0..7):", [s[f'swi_exactly_{k}_valid_days_n'] for k in range(8)])
    print("Quality:",quality)
    print("LOW_QFLAG broken down:",qflags)
    print("Daily stage change by split:")
    for r in movements:print(f"  {r['split']:10s}: mean={r['mean_abs_stage_change_m']:.4f} m; median={r['median_abs_stage_change_m']:.4f} m; P90={r['p90_abs_stage_change_m']:.4f} m")
    print("Descriptive TEST MAE (not for model tuning):")
    for r in test_scores:print(f"  {r['model_id']:12s}: {r['MAE_m']:.4f} m (n={r['test_n']})")
    print("Saved QA tables and JSON in:", OUTPUT)
    print("RESEARCH CAUTION: C/D are masked partial-history experiments until valid 7-day histories exist.")

if __name__=="__main__":
    try:run()
    except (FileNotFoundError,ValueError,KeyError) as exc:
        raise SystemExit(f"STOP – audit cannot be trusted: {exc}")
