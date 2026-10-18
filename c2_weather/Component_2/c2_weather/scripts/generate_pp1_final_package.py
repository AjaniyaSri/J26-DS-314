from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "pp1_final_package"
MODEL_ORDER = ["persistence", "raw", "engineered", "hybrid"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "raw": "Raw XGBoost",
    "engineered": "Engineered XGBoost",
    "hybrid": "Hybrid XGBoost",
}
COLORS = {
    "persistence": "#3f4a56",
    "raw": "#247a84",
    "engineered": "#ce674b",
    "hybrid": "#637bb5",
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _table_rows(frame, metric_columns):
    labels = {
        "mae_m": "MAE (m)",
        "rmse_m": "RMSE (m)",
        "r2": "R²",
        "rising_mae_m": "Rising MAE (m)",
    }
    lines = [
        "| Evaluation | Model | n | "
        + " | ".join(labels.get(column, column) for column in metric_columns)
        + " | Status |",
        "|---|---|---:|" + "---:|" * len(metric_columns) + "---|",
    ]
    for row in frame.itertuples(index=False):
        values = [f"{getattr(row, column):.4f}" if pd.notna(getattr(row, column)) else "N/A" for column in metric_columns]
        lines.append(
            f"| {row.evaluation} | {row.model_label} | {int(row.n)} | "
            + " | ".join(values)
            + f" | {row.status} |"
        )
    return "\n".join(lines)


def _build_tables(metrics_dir, diagnostics_dir, delta_dir, package_dir):
    test_metrics = pd.read_csv(metrics_dir / "model_comparison.csv")
    test_table = test_metrics.copy()
    test_table["evaluation"] = "Held-out test"
    test_table["model_label"] = test_table["model"].map(MODEL_LABELS)
    test_table["status"] = "Held-out; common 351 records"
    test_table = test_table[["evaluation", "model_label", "n", "mae_m", "rmse_m", "r2", "status"]]
    test_table = test_table.rename(columns={"model_label": "model"})
    model_rank = {MODEL_LABELS[name]: index for index, name in enumerate(MODEL_ORDER)}
    test_table["_rank"] = test_table["model"].map(model_rank)
    test_table = test_table.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    delta_metrics = pd.read_csv(delta_dir / "validation_model_comparison.csv")
    allowed = delta_metrics[delta_metrics["model"].isin([
        "persistence",
        "delta_hybrid_base_station",
        "delta_hybrid_base_pooled",
        "delta_hybrid_trends_station",
        "delta_hybrid_trends_pooled",
    ])].copy()
    allowed["evaluation"] = "Delta validation"
    allowed["model_label"] = allowed["model"].map(
        {
            "persistence": "Persistence",
            "delta_hybrid_base_station": "Delta Hybrid, no trends, station-aware",
            "delta_hybrid_base_pooled": "Delta Hybrid, no trends, pooled",
            "delta_hybrid_trends_station": "Delta Hybrid + trends, station-aware",
            "delta_hybrid_trends_pooled": "Delta Hybrid + trends, pooled",
        }
    )
    allowed["status"] = np.where(
        allowed["model"].eq("persistence"),
        "Validation baseline",
        "Train-only fit; selected on validation",
    )
    delta_table = allowed[[
        "evaluation",
        "model_label",
        "n",
        "mae_m",
        "rmse_m",
        "r2",
        "rising_mae_m",
        "rising_n",
        "rising_underprediction_share",
        "rapid_rise_n",
        "rapid_rise_mae_m",
        "rapid_rise_underprediction_share",
        "status",
    ]]
    delta_table = delta_table.rename(columns={"model_label": "model"})
    test_rising = pd.read_csv(diagnostics_dir / "regime_metrics.csv")
    test_rising = test_rising.loc[test_rising["regime"].eq("rising"), ["model", "mae_m"]].rename(
        columns={"mae_m": "rising_mae_m"}
    )
    test_rising["model"] = test_rising["model"].map(MODEL_LABELS)
    test_table = test_table.merge(test_rising, on="model", how="left", validate="one_to_one")
    test_table.to_csv(package_dir / "original_test_comparison.csv", index=False)
    delta_table.to_csv(package_dir / "delta_validation_comparison.csv", index=False)
    return test_table, delta_table


def _copy_existing_figures(root, figure_dir):
    sources = {
        "test_actual_vs_predicted_persistence.png": "outputs/figures/persistence_actual_vs_predicted.png",
        "test_actual_vs_predicted_raw.png": "outputs/figures/raw_actual_vs_predicted.png",
        "test_actual_vs_predicted_engineered.png": "outputs/figures/engineered_actual_vs_predicted.png",
        "test_actual_vs_predicted_hybrid.png": "outputs/figures/hybrid_actual_vs_predicted.png",
        "test_residuals_persistence.png": "outputs/figures/persistence_residuals.png",
        "test_residuals_raw.png": "outputs/figures/raw_residuals.png",
        "test_residuals_engineered.png": "outputs/figures/engineered_residuals.png",
        "test_residuals_hybrid.png": "outputs/figures/hybrid_residuals.png",
        "shap_summary_raw.png": "outputs/figures/raw_shap_summary.png",
        "shap_summary_engineered.png": "outputs/figures/engineered_shap_summary.png",
        "shap_summary_hybrid.png": "outputs/figures/hybrid_shap_summary.png",
        "shap_importance_raw.png": "outputs/figures/raw_shap_importance.png",
        "shap_importance_engineered.png": "outputs/figures/engineered_shap_importance.png",
        "shap_importance_hybrid.png": "outputs/figures/hybrid_shap_importance.png",
        "test_regime_mae.png": "outputs/diagnostics/regime_mae.png",
        "test_largest_rise_events.png": "outputs/diagnostics/largest_rise_events.png",
        "test_station_mae_delta.png": "outputs/diagnostics/station_mae_delta.png",
        "test_change_calibration.png": "outputs/diagnostics/predicted_vs_observed_change.png",
        "delta_validation_overall_rising.png": "outputs/experiments/delta_xgboost/validation_overall_and_rising_mae.png",
        "delta_validation_change_calibration.png": "outputs/experiments/delta_xgboost/delta_change_calibration.png",
        "model_architecture.png": "outputs/pp1_evidence/model_architecture.png",
    }
    for name, source in sources.items():
        source_path = root / source
        if not source_path.is_file():
            raise FileNotFoundError(f"Required saved evidence figure is missing: {source_path}")
        shutil.copy2(source_path, figure_dir / name)
    return sources


def _make_score_figure(test_table, delta_table, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    left = test_table.set_index("model").loc[[MODEL_LABELS[name] for name in MODEL_ORDER]]
    axes[0].bar(
        left.index,
        left["mae_m"],
        color=[COLORS[name] for name in MODEL_ORDER],
    )
    axes[0].set_title("Original models: untouched test")
    axes[0].set_ylabel("MAE (m)")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].grid(axis="y", alpha=0.2)
    for index, value in enumerate(left["mae_m"]):
        axes[0].text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)

    delta_labels = [
        "Persistence",
        "Delta base\nstation",
        "Delta base\npooled",
        "Delta + trend\nstation",
        "Delta + trend\npooled",
    ]
    values = delta_table["mae_m"].to_numpy(dtype=float)
    colors = [COLORS["persistence"], "#247a84", "#62a6a3", "#ce674b", "#e79a68"]
    axes[1].bar(delta_labels, values, color=colors)
    axes[1].set_title("Delta experiment: validation only")
    axes[1].set_ylabel("Reconstructed-level MAE (m)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.006, f"{value:.3f}", ha="center", fontsize=9)
    figure.suptitle("Evaluation periods are separate; validation scores are not test results", fontsize=13)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _make_split_figure(summary, output_path):
    figure, axis = plt.subplots(figsize=(10.5, 3.5))
    axis.set_xlim(0, 39)
    axis.set_ylim(0, 1)
    axis.axis("off")
    segments = [
        (0, 21, "Train\nNov 17-Dec 07\n737 samples", "#4c9f70"),
        (21, 29, "Validation\nDec 08-Dec 15\n303 samples", "#d39537"),
        (29, 38, "Held-out test\nDec 16-Dec 24\n351 samples", "#537ca5"),
    ]
    for start, end, label, color in segments:
        axis.barh(0.5, end - start, left=start, height=0.36, color=color)
        axis.text((start + end) / 2, 0.5, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
    axis.text(38.5, 0.5, "Dec 25\nlast target", ha="left", va="center", fontsize=9)
    axis.annotate("Forecast origin dates are chronological; each label is next-day 9 AM", xy=(19, 0.15), ha="center", fontsize=9)
    axis.set_title("Chronological split and eligible station-date counts", fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_comparison_markdown(test_table, delta_table, path):
    combined_test = test_table[
        ["evaluation", "model", "n", "mae_m", "rmse_m", "r2", "rising_mae_m", "status"]
    ].copy()
    combined_delta = delta_table[
        ["evaluation", "model", "n", "mae_m", "rmse_m", "r2", "rising_mae_m", "status"]
    ].copy()
    combined = pd.concat([combined_test, combined_delta], ignore_index=True)
    combined["model_label"] = combined["model"]
    combined["status"] = combined["status"].astype(str)
    combined["mae_m"] = combined["mae_m"].astype(float)
    combined["rmse_m"] = combined["rmse_m"].astype(float)
    combined["r2"] = combined["r2"].astype(float)
    path.write_text(_table_rows(combined, ["mae_m", "rmse_m", "r2", "rising_mae_m"]), encoding="utf-8")


def _write_report(package_dir, summary, quality, metrics, test_table, delta_table):
    missing_levels = quality["raw"]["water_level_missing"]
    date_min = quality["raw"]["date_min"]
    date_max = quality["raw"]["date_max"]
    selected = _read_json(ROOT / "outputs/experiments/delta_xgboost/experiment_summary.json")
    delta_best = delta_table.sort_values(["mae_m", "rmse_m"]).iloc[1]
    test_persistence = test_table.loc[test_table.model.eq("Persistence")].iloc[0]
    best_test = test_table.loc[test_table.model.isin(["Raw XGBoost", "Engineered XGBoost", "Hybrid XGBoost"])].sort_values("mae_m").iloc[0]
    architecture_rows = []
    for variant in ("raw", "engineered", "hybrid"):
        metadata = _read_json(ROOT / "models" / "xgboost" / variant / "metadata.json")
        params = metadata["selected_parameters"]
        architecture_rows.append(
            {
                "evaluation": "Saved final model",
                "model": MODEL_LABELS[variant],
                "n": metadata["training_rows"] + metadata["validation_rows"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "min_child_weight": params["min_child_weight"],
                "reg_lambda": params["reg_lambda"],
                "n_estimators": metadata["selected_n_estimators"],
                "status": "Fit on train + validation",
            }
        )
    architecture_table = "\n".join(
        [
            "| Model | Fit rows | Depth | Learning rate | Min child weight | L2 regularization | Trees |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {row['model']} | {row['n']} | {row['max_depth']} | {row['learning_rate']:.3f} | "
                f"{row['min_child_weight']} | {row['reg_lambda']:.1f} | {row['n_estimators']} |"
                for row in architecture_rows
            ],
        ]
    )
    report = f"""# C2 Weather-Specific Depth Study: PP1 Progress Report

## Research objective

Predict the river water level at 9:00 AM on calendar day t+1, in metres as named by the DMC `water_level_9am_m` field, from information available at the 9:00 AM cutoff on day t. This is a 24-hour-ahead, station-specific forecast. The physical gauge unit and vertical datum still require authoritative confirmation.

## Data and sample construction

The study uses DMC 9 AM water-level readings and weather features supplied as CHIRPS/ERA5-Land-derived rainfall, soil-moisture, and temperature values. The CSVs do not carry full source provenance, observation timestamps, publication times, or timezone metadata, so product lineage and actual weather availability at the cutoff remain unverified.

There are 39 stations with daily rows from {date_min} through {date_max} (39 days; 1,521 station-date rows per source file). There are {missing_levels} missing water-level cells. No targets are imputed. Exact next-calendar-day alignment and nonmissing current/target readings produce {summary['common_eligible_samples']:,} common forecast examples: {summary['split_counts']['train']} train, {summary['split_counts']['validation']} validation, and {summary['split_counts']['test']} test. The original test predictions contain 351 identical station-date-target records across all models.

## Feature experiments and workflow

- **Raw:** current 9 AM water level, exact 1/3/7-day water-level lags, and prior-day rainfall, soil moisture, and temperature.
- **Engineered:** the same water-level inputs, prior-day soil moisture/temperature, and prior-day 3/7/14-day rainfall windows.
- **Hybrid:** all Raw features plus the engineered rainfall windows.

Targets are formed station-locally from the next observation only when it is exactly one calendar day later. Water-level lags and trends require exact calendar dates; gaps are not bridged. Daily weather and supplied rainfall windows are shifted one day to avoid same-day full-day leakage. Missing targets are dropped; remaining missing predictors are passed to XGBoost's native missing-value handling. All experiments share the same eligible records and chronological forecast-origin splits.

## Model and split design

Each level model is a shallow `XGBRegressor` with squared-error objective, histogram tree method, conservative depth/regularization, fixed random seed, and validation early stopping. Station is one-hot encoded with unknown categories ignored. Candidate selection uses validation only. The persisted final models were then fit on train plus validation; the test block remained untouched for the original test evaluation.

All models use `objective=reg:squarederror`, `tree_method=hist`, and `random_state=42`. The saved final-model parameters are:

{architecture_table}

| Period | Forecast origins | Samples | Label availability |
|---|---|---:|---|
| Train | Nov 17-Dec 07 | 737 | Through Dec 08 9 AM |
| Validation | Dec 08-Dec 15 | 303 | Through Dec 16 9 AM |
| Test | Dec 16-Dec 24 | 351 | Through Dec 25 9 AM |

## Original held-out test results

{_table_rows(test_table.assign(model_label=test_table['model'], status=test_table['status']), ['mae_m', 'rmse_m', 'r2', 'rising_mae_m'])}

Persistence achieves MAE {test_persistence.mae_m:.4f} m and RMSE {test_persistence.rmse_m:.4f} m. The best XGBoost test MAE is {best_test.mae_m:.4f} m ({best_test['model']}); no XGBoost model beats persistence. Raw and Hybrid RMSEs are close but not identical: {metrics['raw']['rmse_m']:.12f} m versus {metrics['hybrid']['rmse_m']:.12f} m (difference {abs(metrics['raw']['rmse_m']-metrics['hybrid']['rmse_m']):.3g} m). The reported four-decimal equality is rounding.

## Rapid rises and SHAP

On the held-out test set, 62 records rose by at least 0.10 m/day. XGBoost underpredicted the target in 90.3% of Raw cases and 83.9% of Engineered and Hybrid cases. For the 12 top-decile rises (cutoff 0.852 m/day), all three XGBoost models underpredicted; their MAEs were 1.339 m Raw, 1.349 m Engineered, and 1.312 m Hybrid, versus 1.396 m persistence. The modest event-MAE reduction does not remove the systematic underprediction.

Actual-versus-predicted plots, residual plots, and SHAP summary/importance plots are included in `figures/`. SHAP and gain explanations are model-specific descriptive attributions, not causal evidence. Current level and some station indicators are influential in the original models; station heterogeneity is plausible but the brief record cannot identify its cause.

## Separate Delta regression exploration

Delta XGBoost predicts next-day change, `next_level - current_level`, then reconstructs level as `current_level + predicted_delta`. Four Hybrid ablations compared trend features on/off and one-hot station encoding on/off. These are **validation-only** results, not test results. The best Delta validation MAE is {selected['best_delta_validation_mae_m']:.4f} m (RMSE {delta_best.rmse_m:.4f} m) versus persistence validation MAE {selected['persistence_validation_mae_m']:.4f} m: Delta is worse by {abs(selected['delta_improvement_vs_persistence_mae_m']):.4f} m. It underpredicted all 11 validation top-decile rises. Legacy Raw/Engineered/Hybrid validation scores are in-sample because those saved models were refit on train plus validation; they are not fair validation comparators. No held-out test examples or test metrics were used for Delta selection.

## Limitations and next work

The record spans only 39 days, with one short validation and test period; 39 stations are heterogeneous and individual station samples are sparse. Weather publication availability/timezone and rainfall aggregation semantics are not verified. Gauge units, datums, and station metadata need confirmation. Current-level persistence is a strong short-horizon baseline, while both level and Delta XGBoost underpredict rapid rises.

Priorities are to expand data across seasons and flood events, verify gauge/source metadata and forecast-time weather availability, use multiple rolling-origin evaluations, and evaluate event-aware objectives without sacrificing overall metrics. LSTM is future work only. Multimodal integration may combine gauge history with verified near-real-time meteorology, rainfall products, upstream gauges, and catchment/satellite context after temporal availability checks.

## Evidence and reproducibility

All tables and figures in this package are assembled from existing predictions, saved metrics, model metadata, diagnostics, and Delta validation artifacts. No model was retrained or altered. The existing detailed diagnostics and Delta report remain available alongside this package; original source CSVs and prior research evidence were not modified.
"""
    (package_dir / "PP1_progress_report.md").write_text(report, encoding="utf-8")


def _write_viva_script(package_dir, test_table, selected, summary):
    test = test_table.set_index("model")
    delta_mae = selected["best_delta_validation_mae_m"]
    text = f"""# PP1 Viva Explanation Script (about 5 minutes)

## Opening and objective

My C2 study investigates whether weather and recent river-level history can predict a station's 9 AM water level 24 hours ahead. For a forecast made at 9 AM on date t, the target is the observed 9 AM level on t+1. The persistence benchmark simply carries today's level forward.

## Data

The current evidence covers 39 stations from Nov 17 to Dec 25, 2025. Each source table contains 1,521 station-date rows. There are 71 missing water-level values, which are not interpolated. Exact next-day targets and usable current readings yield {summary['common_eligible_samples']:,} common examples, split chronologically into 737 training, 303 validation, and 351 held-out test samples.

The files contain DMC 9 AM levels and weather fields intended from CHIRPS and ERA5-Land. The data do not include publication time, timezone, gauge datum, or complete provenance, so those need verification before operational interpretation.

## Method

I compare Raw weather, Engineered rainfall windows, and Hybrid features. Every model receives the current water level and exact calendar-day water-level lags. Same-day full-day weather is not used; daily weather and rainfall windows are shifted by one day. Targets are station-local and kept only for the exact next calendar date. Chronological splits are shared across experiments.

The regressors are shallow, regularized XGBoost models. Validation selects the model settings, and the original final models use train plus validation. The test set is reserved for the final evaluation. SHAP figures describe which fitted features influenced model outputs, but they do not show causation.

## Main result

On the untouched test set, persistence achieved MAE {test.loc['Persistence','mae_m']:.4f} metres. Raw, Engineered, and Hybrid XGBoost achieved {test.loc['Raw XGBoost','mae_m']:.4f}, {test.loc['Engineered XGBoost','mae_m']:.4f}, and {test.loc['Hybrid XGBoost','mae_m']:.4f} metres. Therefore, none of the XGBoost models outperformed persistence. This is an important negative result: the weather feature additions did not demonstrate an overall benefit on this short test window.

On rapid rises, the XGBoost variants underpredicted every one of the 12 top-decile test events. Their mean absolute event errors were slightly lower than persistence, but their predictions still fell short of observed rise magnitudes. This is a safety-relevant weakness, not evidence that the models are reliable for flood peaks.

## Delta exploration and caveats

I also tested a separate Delta model, which predicts tomorrow's change and adds it back to today's reading. On validation its best MAE was {delta_mae:.4f} metres, worse than validation persistence at {selected['persistence_validation_mae_m']:.4f} metres. It underpredicted all 11 top-decile validation rises. This is validation evidence only; I did not use the test block for Delta selection.

The saved Raw/Engineered/Hybrid models were refit on train plus validation, so their validation scores are in-sample and are not fair validation comparisons. The dataset is only 39 days long, weather delivery times are unverified, and station gauge datums may differ. I cannot claim strong generalization.

## Close and future work

Next I will expand the record across seasons and flood events, validate gauge units/datums and weather release times, and evaluate with multiple rolling-origin holdouts. LSTM remains future work. I will also investigate multimodal inputs, such as upstream gauge context and verified near-real-time weather, while preserving strict forecast-time availability. The next model must beat persistence on untouched future periods and show improved rise behavior before I would recommend it.
"""
    (package_dir / "viva_script.md").write_text(text, encoding="utf-8")


def _write_captions(package_dir):
    captions = """# Figure Captions

- `chronological_split.png`: common forecast-origin split dates and eligible sample counts; the Dec 16 reading closes the final validation label and begins the test-origin period.
- `test_metrics_and_delta_validation.png`: original held-out test MAE and separate Delta validation MAE; panels are intentionally separated because these are different evaluation periods.
- `test_actual_vs_predicted_*.png`: saved held-out test predictions against observed levels; diagonal agreement indicates accurate level prediction.
- `test_residuals_*.png`: held-out residuals (prediction minus observation) versus predictions.
- `shap_summary_*.png`: SHAP summary for each original XGBoost model; explanations are associational/model-specific.
- `shap_importance_*.png`: mean absolute SHAP feature importance for each original model.
- `test_regime_mae.png`: original test MAE for rising, stable, and falling level-change regimes.
- `test_largest_rise_events.png`: actual and predicted level changes for the largest observed test rises.
- `test_station_mae_delta.png`: per-station model MAE differences from persistence; positive values mean the XGBoost model is worse.
- `test_change_calibration.png`: predicted versus observed daily changes on the held-out test data.
- `delta_validation_overall_rising.png`: overall and rising validation MAE for persistence and Delta ablations.
- `delta_validation_change_calibration.png`: predicted versus observed change for the separate Delta validation models.
- `model_architecture.png`: saved PP1 model flow diagram for the original level models.
"""
    (package_dir / "figure_captions.md").write_text(captions, encoding="utf-8")


def _write_slide_outline(package_dir):
    outline = """# Suggested Slide Order

1. **Question and forecast definition:** 9 AM to next-day 9 AM; station-level target.
2. **Data and verified coverage:** 39 stations, 39 dates, missing levels, usable exact-next-day samples.
3. **Features and leakage controls:** Raw, Engineered, Hybrid; exact calendar lags; one-day weather shift.
4. **Experiment design:** XGBoost architecture and chronological train/validation/test windows.
5. **Held-out test comparison:** persistence versus three XGBoost variants; emphasize persistence wins.
6. **Error behavior:** actual-vs-predicted, residuals, regimes, station variation, rapid-rise misses.
7. **Interpretability:** SHAP summaries with non-causal caveat.
8. **Exploratory Delta model:** validation-only negative result, separate from test findings.
9. **Limitations and future work:** more flood/seasonal data, metadata validation, rolling origins, LSTM and multimodal integration as future work.
"""
    (package_dir / "slide_outline.md").write_text(outline, encoding="utf-8")


def generate_package():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing PP1 package: {OUTPUT}")
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    metrics_dir = ROOT / "outputs" / "metrics"
    diagnostics_dir = ROOT / "outputs" / "diagnostics"
    delta_dir = ROOT / "outputs" / "experiments" / "delta_xgboost"
    summary = _read_json(metrics_dir / "training_dataset_summary.json")
    quality = _read_json(metrics_dir / "data_quality_report.json")
    diagnostics = _read_json(diagnostics_dir / "diagnostics.json")
    delta_summary = _read_json(delta_dir / "experiment_summary.json")
    test_metrics = pd.read_csv(metrics_dir / "model_comparison.csv")
    recomputed = diagnostics["recomputed_metrics"]
    for row in test_metrics.itertuples(index=False):
        score = recomputed[row.model]
        if not np.isclose(row.mae_m, score["mae_m"]) or not np.isclose(row.rmse_m, score["rmse_m"]):
            raise AssertionError(f"Stored test metrics no longer agree with verified diagnostic for {row.model}")

    test_table, delta_table = _build_tables(metrics_dir, diagnostics_dir, delta_dir, OUTPUT)
    figure_sources = _copy_existing_figures(ROOT, figures)
    _make_score_figure(test_table, delta_table, figures / "test_metrics_and_delta_validation.png")
    _make_split_figure(summary, figures / "chronological_split.png")
    _write_comparison_markdown(test_table, delta_table, OUTPUT / "model_comparison_tables.md")
    _write_report(OUTPUT, summary, quality, recomputed, test_table, delta_table)
    _write_viva_script(OUTPUT, test_table, delta_summary, summary)
    _write_captions(OUTPUT)
    _write_slide_outline(OUTPUT)

    manifest = {
        "package": "C2 PP1 final evidence package",
        "source_of_results": [
            "outputs/metrics/model_comparison.csv",
            "outputs/metrics/training_dataset_summary.json",
            "outputs/metrics/data_quality_report.json",
            "outputs/diagnostics/diagnostics.json",
            "outputs/experiments/delta_xgboost/experiment_summary.json",
            "outputs/experiments/delta_xgboost/validation_model_comparison.csv",
            "saved existing figures and SHAP outputs",
        ],
        "models_retrained_or_modified": False,
        "test_predictions_recomputed": False,
        "original_evidence_overwritten": False,
        "copied_figures": list(figure_sources),
        "test_metrics_match_diagnostic_recomputation": True,
        "held_out_test_rows": 351,
        "delta_validation_rows": int(delta_summary["validation_rows"]),
    }
    (OUTPUT / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return OUTPUT


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description="Assemble final C2 PP1 evidence from saved results without retraining.")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    OUTPUT = Path(args.output).resolve()
    print(generate_package())


if __name__ == "__main__":
    main()