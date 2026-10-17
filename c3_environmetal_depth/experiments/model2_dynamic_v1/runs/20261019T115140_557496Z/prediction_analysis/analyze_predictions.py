"""Describe existing predictions only; never load estimators, datasets or fit models.

Run from RP with .venv/Scripts/python.exe <this script>. Existing outputs are
immutable. A second invocation refuses to overwrite the completed analysis.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
ROOT = RUN.parents[3]
NUMERIC_TOLERANCE_M = 1e-12
OBSERVED_CHANGE_TOLERANCE_M = 1e-6
KEYS = ["station_id", "forecast_date", "record_id"]
METHODS = {
    "model1": "original_model1",
    "model2_absolute": "model2_absolute_mae_loss",
    "model2_delta": "model2_delta_mae_loss",
    "no_weather_delta": "control_delta_no_weather",
    "persistence": "persistence",
}
LABELS = {"model1": "Original Model 1", "model2_absolute": "Model 2 absolute",
          "model2_delta": "Model 2 stage-change", "no_weather_delta": "No-weather stage-change",
          "persistence": "Persistence"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_csv(path, frame):
    write_text(path, frame.to_csv(index=False))


def scores(actual, predicted):
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("Metric arrays are not aligned.")
    if not len(y):
        return {"mae_m": np.nan, "rmse_m": np.nan, "signed_bias_m": np.nan}
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Nonfinite comparison values.")
    error = p - y
    return {"mae_m": float(np.mean(np.abs(error))), "rmse_m": float(np.sqrt(np.mean(error**2))),
            "signed_bias_m": float(np.mean(error))}


def percentage_gain(reference, candidate):
    return 100 * (reference - candidate) / reference if reference > 0 else np.nan


def load_comparisons():
    comparisons, evidence = [], []
    common = [*KEYS, "station_name", "basin", "chronological_split", "stage_at_issue_m", "target_stage_next_day_m"]
    definitions = [("VALIDATION", "model2_validation_candidates.csv", 265),
                   ("TEST", "model_comparison_test_predictions.csv", 349)]
    for split, filename, count in definitions:
        source = pd.read_csv(RUN / "predictions" / filename)
        metrics = pd.read_csv(RUN / "metrics" / f"model2_{split.lower()}_metrics.csv").set_index("model")
        reference, result = None, None
        for slug, name in METHODS.items():
            part = source[source.model.eq(name)].sort_values(KEYS).reset_index(drop=True)
            required = [*common, "predicted_stage_next_day_m", "prediction_error_m", "persistence_prediction_m"]
            if len(part) != count or part.station_id.nunique() != 39 or not part.chronological_split.eq(split).all():
                raise ValueError(f"Unexpected split coverage for {name}.")
            if part[required].isna().any().any() or part[KEYS].astype(str).apply(lambda s: s.str.strip().eq("")).any().any():
                raise ValueError(f"Missing comparison values for {name}.")
            if part.duplicated(KEYS).any() or part.duplicated(["station_id", "forecast_date"]).any() or not part.record_id.is_unique:
                raise ValueError(f"Duplicate comparison keys for {name}.")
            if reference is None:
                reference = part[common].copy()
                result = reference.rename(columns={"target_stage_next_day_m": "actual_next_day_stage_m"}).copy()
            else:
                pd.testing.assert_frame_equal(reference, part[common])
            prediction = part.predicted_stage_next_day_m.to_numpy()
            measured = scores(part.target_stage_next_day_m, prediction)
            for metric in ("mae_m", "rmse_m"):
                if not np.isclose(measured[metric], metrics.loc[name, metric], atol=NUMERIC_TOLERANCE_M, rtol=0):
                    raise ValueError(f"Saved metrics disagree for {split}, {name}, {metric}.")
            np.testing.assert_allclose(part.prediction_error_m, prediction-part.target_stage_next_day_m,
                                       atol=NUMERIC_TOLERANCE_M, rtol=0)
            if slug == "persistence":
                np.testing.assert_allclose(prediction, part.stage_at_issue_m, atol=NUMERIC_TOLERANCE_M, rtol=0)
            result[slug + "_prediction_m"] = prediction
            result[slug + "_abs_error_m"] = np.abs(prediction-part.target_stage_next_day_m)
            evidence.append({"split": split, "model": name, "n": count, **measured,
                             "source_csv": filename, "source_columns": source.dtypes.astype(str).to_dict()})
            if slug in ("model2_absolute", "model2_delta", "model1"):
                if slug == "model1":
                    path = ROOT / "results/predictions" / f"model1_{split.lower()}_predictions.csv"
                else:
                    family = "absolute" if slug == "model2_absolute" else "delta"
                    path = RUN / "predictions" / f"model2_{family}_{split.lower()}_predictions.csv"
                separate = pd.read_csv(path).sort_values(KEYS).reset_index(drop=True)
                pd.testing.assert_frame_equal(part[KEYS], separate[KEYS])
                np.testing.assert_allclose(part.target_stage_next_day_m, separate.target_stage_next_day_m,
                                           atol=NUMERIC_TOLERANCE_M, rtol=0)
                np.testing.assert_allclose(prediction, separate.predicted_stage_next_day_m,
                                           atol=NUMERIC_TOLERANCE_M, rtol=0)
        result["target_date"] = (pd.to_datetime(result.forecast_date, format="%Y-%m-%d") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        result["actual_delta_m"] = result.actual_next_day_stage_m - result.stage_at_issue_m
        result["observed_stage_direction"] = np.select(
            [result.actual_delta_m > OBSERVED_CHANGE_TOLERANCE_M,
             result.actual_delta_m < -OBSERVED_CHANGE_TOLERANCE_M], ["rising", "falling"], default="unchanged")
        result["weather_absolute_error_reduction_m"] = result.no_weather_delta_abs_error_m-result.model2_delta_abs_error_m
        comparisons.append(result)
    return pd.concat(comparisons, ignore_index=True), evidence


def analyze_stations(comparison):
    rows = []
    keys = ["chronological_split", "station_id", "station_name", "basin"]
    for key, part in comparison.groupby(keys, sort=True):
        row = {**dict(zip(keys, key)), "n_observations": len(part)}
        for slug in METHODS:
            values = scores(part.actual_next_day_stage_m, part[slug + "_prediction_m"])
            row.update({slug + "_" + name: value for name, value in values.items()})
        row["weather_mae_reduction_m"] = row["no_weather_delta_mae_m"]-row["model2_delta_mae_m"]
        row["weather_rmse_reduction_m"] = row["no_weather_delta_rmse_m"]-row["model2_delta_rmse_m"]
        row["weather_mae_improvement_pct"] = percentage_gain(row["no_weather_delta_mae_m"], row["model2_delta_mae_m"])
        change = row["weather_mae_reduction_m"]
        row["weather_effect_mae"] = "helps" if change > NUMERIC_TOLERANCE_M else "worsens" if change < -NUMERIC_TOLERANCE_M else "ties"
        rows.append(row)
    table = pd.DataFrame(rows)
    for slug in ("model2_delta", "model2_absolute"):
        table[slug + "_mae_rank_best_first"] = table.groupby("chronological_split")[slug + "_mae_m"].rank(method="min", ascending=True).astype(int)
    return table.sort_values(["chronological_split", "model2_delta_mae_rank_best_first", "station_id"])


def summarize(comparison, stations):
    rows = []
    for split, part in comparison.groupby("chronological_split", sort=False):
        station = stations[stations.chronological_split.eq(split)]
        for slug in METHODS:
            rows.append({"split": split, "model": slug, "n_observations": len(part), "n_stations": len(station),
                         **scores(part.actual_next_day_stage_m, part[slug + "_prediction_m"]),
                         "station_macro_mae_m": float(station[slug + "_mae_m"].mean()),
                         "mean_station_rmse_m": float(station[slug + "_rmse_m"].mean())})
    return pd.DataFrame(rows)


def rising_falling(comparison):
    rows = []
    for split, part in comparison.groupby("chronological_split", sort=False):
        for direction in ("rising", "falling", "unchanged"):
            group = part[part.observed_stage_direction.eq(direction)]
            for slug in ("model2_delta", "no_weather_delta", "persistence"):
                station_metrics = [scores(s.actual_next_day_stage_m, s[slug + "_prediction_m"])
                                   for _, s in group.groupby("station_id")]
                rows.append({"split": split, "observed_direction": direction, "model": slug,
                             "n_observations": len(group), "n_stations": group.station_id.nunique(),
                             **scores(group.actual_next_day_stage_m, group[slug + "_prediction_m"]),
                             "station_macro_mae_m": float(np.mean([s["mae_m"] for s in station_metrics])) if station_metrics else np.nan,
                             "near_zero_change_tolerance_m": OBSERVED_CHANGE_TOLERANCE_M})
    return pd.DataFrame(rows)


def select_examples(stations):
    rows = []
    criteria = [("lowest_model2_delta_mae", "model2_delta_mae_m", True),
                ("highest_model2_delta_mae", "model2_delta_mae_m", False),
                ("greatest_weather_benefit", "weather_mae_reduction_m", False),
                ("greatest_weather_deterioration", "weather_mae_reduction_m", True)]
    for split, part in stations.groupby("chronological_split", sort=False):
        for name, column, ascending in criteria:
            chosen = part.sort_values([column, "station_id"], ascending=[ascending, True], kind="stable").iloc[0]
            rows.append({"split": split, "selection_rule": name, "station_id": chosen.station_id,
                         "station_name": chosen.station_name, "selection_value_m": chosen[column],
                         "n_observations": int(chosen.n_observations),
                         "figure": f"{split.lower()}_{chosen.station_id}.png"})
    return pd.DataFrame(rows)


def make_figures(comparison, examples):
    os.environ["MPLCONFIGDIR"] = str(OUT / ".matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    folder = OUT / "station_actual_vs_predicted"
    folder.mkdir(exist_ok=False)
    for (split, station_id), selections in examples.groupby(["split", "station_id"], sort=False):
        part = comparison[comparison.chronological_split.eq(split) & comparison.station_id.eq(station_id)].sort_values("target_date")
        dates = pd.to_datetime(part.target_date)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(dates, part.actual_next_day_stage_m, "ko-", linewidth=1.8, markersize=4, label="Actual next-day stage")
        for slug, color in (("model2_delta", "#0072b2"), ("no_weather_delta", "#d55e00"), ("persistence", "#009e73")):
            axes[0].plot(dates, part[slug+"_prediction_m"], "o--", linewidth=1.2, markersize=3,
                         color=color, label=LABELS[slug])
            axes[1].plot(dates, part[slug+"_abs_error_m"], "o-", linewidth=1, markersize=3, color=color, label=LABELS[slug])
        axes[0].set(ylabel="River stage (m)", title=f"{part.station_name.iloc[0]} — {split} ({len(part)} observations)")
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.17), ncol=2, fontsize=8)
        axes[1].set(ylabel="Absolute error (m)", xlabel="Forecast target date (input date + one calendar day)")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        rules = ", ".join(selections.selection_rule.str.replace("_", " "))
        fig.suptitle("Selection: " + rules, y=1.04, fontsize=10, wrap=True)
        fig.tight_layout()
        fig.savefig(folder / selections.figure.iloc[0], dpi=180, bbox_inches="tight")
        plt.close(fig)


def metric_table(frame):
    columns = ["model", "n_observations", "mae_m", "rmse_m", "signed_bias_m", "station_macro_mae_m"]
    lines = ["| Method | n | MAE (m) | RMSE (m) | Signed bias (m) | Station-macro MAE (m) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join([LABELS[row[0]], str(row[1]), *[f"{v:.6f}" for v in row[2:]]]) + " |")
    return "\n".join(lines)


def make_report(comparison, stations, pooled, events, examples, evidence):
    text = ("# Actual prediction analysis\n\n"
            "This analysis uses existing saved predictions only. No models were loaded or fitted, no "
            "new model predictions were calculated, no configurations were changed and no source datasets "
            "were regenerated or used. VALIDATION and TEST are reported separately. The project TEST "
            "has already been examined; these are descriptive analyses, not independent confirmation.\n\n"
            "## Prediction integrity and interpretation\n\n"
            "All five methods align on the same 265 validation and 349 TEST station-days, with 39 gauges "
            "in both splits. Station/date/record IDs are unique, required comparison cells are present, "
            "and observed targets and current stages agree. Recomputed MAE and RMSE match saved metrics "
            "within 1e-12 m. Separate original Model 1 and Model 2 family files agree with combined "
            "artifacts within that tolerance. Harmless CSV-roundtrip differences reach about 4.44e-16 m.\n\n"
            "The saved `predicted_stage_next_day_m` fields are reconstructed absolute stages, as documented "
            "in frozen metadata and methodology. Errors equal prediction minus observed next-day stage. "
            "Raw predicted deltas were not saved separately, so their reconstruction cannot be independently "
            "rechecked here without evaluating an estimator, which is intentionally outside this task.\n\n"
            "Validation source: `../predictions/model2_validation_candidates.csv`; TEST source: "
            "`../predictions/model_comparison_test_predictions.csv`. Identifiers, dates, labels and model "
            "names are text; stage, prediction and error columns are float64. TEST's optional "
            "recommended-method metadata is not needed for comparisons. All individual values and absolute "
            "errors are in prediction_comparison.csv.\n\n")
    for split in ("VALIDATION", "TEST"):
        summary = pooled[pooled.split.eq(split)]
        station = stations[stations.chronological_split.eq(split)]
        ranked = station.sort_values(["model2_delta_mae_m", "station_id"])
        easy = ranked.iloc[0]
        hard = station.sort_values(["model2_delta_mae_m", "station_id"], ascending=[False, True]).iloc[0]
        by_weather = station.sort_values(["weather_mae_reduction_m", "station_id"])
        worst_weather = by_weather.iloc[0]
        best_weather = station.sort_values(["weather_mae_reduction_m", "station_id"], ascending=[False, True]).iloc[0]
        text += f"## {split}: pooled and station-macro performance\n\n" + metric_table(summary) + "\n\n"
        values = summary.set_index("model")
        weather_gain = values.loc["no_weather_delta", "mae_m"]-values.loc["model2_delta", "mae_m"]
        text += (f"The stage-change Model 2 versus no-weather control MAE difference (control minus weather) "
                 f"is **{weather_gain:+.6f} m**. Positive means weather helps; negative means weather worsens. "
                 f"Weather helps MAE at {int(station.weather_effect_mae.eq('helps').sum())} stations, worsens it "
                 f"at {int(station.weather_effect_mae.eq('worsens').sum())}, and ties at "
                 f"{int(station.weather_effect_mae.eq('ties').sum())}.\n\n"
                 f"- Lowest stage-change Model 2 MAE: **{easy.station_name}**, {easy.model2_delta_mae_m:.6f} m "
                 f"({int(easy.n_observations)} observations).\n"
                 f"- Highest stage-change Model 2 MAE: **{hard.station_name}**, {hard.model2_delta_mae_m:.6f} m "
                 f"({int(hard.n_observations)} observations).\n"
                 f"- Largest weather benefit: **{best_weather.station_name}**, control-minus-weather MAE "
                 f"{best_weather.weather_mae_reduction_m:+.6f} m.\n"
                 f"- Largest weather deterioration: **{worst_weather.station_name}**, control-minus-weather MAE "
                 f"{worst_weather.weather_mae_reduction_m:+.6f} m.\n\n")
        text += "### Rising, falling and unchanged observations\n\n"
        for direction in ("rising", "falling", "unchanged"):
            group = events[events.split.eq(split) & events.observed_direction.eq(direction)]
            text += f"**{direction.capitalize()}** ({int(group.n_observations.iloc[0])} station-days):\n\n" + metric_table(group) + "\n\n"
            indexed = group.set_index("model")
            change = indexed.loc["no_weather_delta", "mae_m"]-indexed.loc["model2_delta", "mae_m"]
            text += f"Control-minus-weather MAE difference: {change:+.6f} m.\n\n"
    text += ("## How to interpret the individual predictions\n\n"
             "Observed-versus-predicted station figures show both river-stage trajectories and absolute "
             "errors. Similar stage curves alone can hide meaningful differences in small daily changes, "
             "so the error panels and numerical tables accompany the plots. A good pooled stage result "
             "does not establish accurate timing or magnitude of individual rapid rises. Gauge datums "
             "also differ, so visual levels should not be compared as common-elevation measurements.\n\n"
             "Persistence uses the current observed stage and predicts zero next-day change. It is "
             "strong in this short dataset because many daily changes are small; its absolute error "
             "equals the magnitude of actual next-day change. Rising/falling groups use observed "
             "next-day change only for evaluation, never as a predictor. Unchanged means absolute "
             "change <= 1e-6 m, a numerical tolerance rather than a hydrological warning threshold.\n\n"
             "## Transparent station selection and rankings\n\n"
             "Each split selects lowest/highest stage-change MAE and greatest weather benefit/deterioration. "
             "Ties are resolved alphabetically by station ID. A station satisfying multiple rules gets "
             "one figure labelled with every applicable rule. selected_stations.csv retains all eight "
             "rule assignments. Both stage-change and absolute-stage Model 2 MAE rankings appear in "
             "station_error_comparison.csv; no favorable gauge is presented alone as representative.\n\n")
    for row in examples.itertuples():
        text += f"- {row.split}, {row.selection_rule}: [{row.station_name}](station_actual_vs_predicted/{row.figure}).\n"
    text += ("\n## Research conclusion and limitations\n\n"
             "The primary absolute-stage model is weaker than persistence. The selected secondary "
             "stage-change model has small pooled gains over persistence, but its matched no-weather "
             "control is slightly better overall in both splits. Thus the overall comparison does not "
             "demonstrate added predictive value from rainfall and soil moisture. Group-specific "
             "differences above describe this record only; small or mixed differences must not be "
             "generalized as a reliable weather benefit during floods.\n\n"
             "Macro MAE averages gauge MAEs equally; pooled MAE weights all observations equally. Mean "
             "station RMSE is not pooled RMSE. Event-group macro scores include only gauges represented "
             "in that group. Station ranks are unstable with only a few observations per gauge. "
             "The roughly 40-day study contains only seven validation and nine TEST input dates, and "
             "nearby/overlapping catchments are not independent. Weather windows and publication "
             "availability remain unverified, including the supplied extra-day rolling-rainfall lag. "
             "Candidate catchments, differing gauge datums and static land-cover vintage remain limitations. "
             "No statistical significance, seasonal generalization, unseen-station accuracy, physical "
             "causality or deployment readiness is established. No further model selection follows "
             "these analyses; longer records and a new independent evaluation period are needed.\n\n"
             "## Reproducibility\n\n"
             "Run this folder's analyze_predictions.py with the existing RP virtual-environment Python. "
             "It refuses to overwrite outputs. Source and pre-existing run-artifact hashes are captured "
             "in source_artifact_checksums.json and checked after generation. The script imports no "
             "training modules and contains no estimator loading or prediction calls.\n")
    return text


def main():
    planned = ["source_artifact_checksums.json", "prediction_comparison.csv", "pooled_and_macro_metrics.csv",
               "station_error_comparison.csv", "rising_falling_analysis.csv", "selected_stations.csv", "prediction_analysis_report.md"]
    if any((OUT / name).exists() for name in planned) or (OUT / "station_actual_vs_predicted").exists():
        raise FileExistsError("Analysis outputs already exist; no file will be overwritten.")
    preserved = [p for p in RUN.rglob("*") if p.is_file() and not p.is_relative_to(OUT)]
    preserved += [ROOT / f"results/predictions/model1_{split}_predictions.csv" for split in ("validation", "test")]
    checksum = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sorted(set(preserved))}
    comparison, evidence = load_comparisons()
    stations = analyze_stations(comparison)
    pooled = summarize(comparison, stations)
    events = rising_falling(comparison)
    examples = select_examples(stations)
    assert len(comparison) == 614 and len(stations) == 78 and len(pooled) == 10 and len(events) == 18
    write_csv(OUT / "prediction_comparison.csv", comparison)
    write_csv(OUT / "station_error_comparison.csv", stations)
    write_csv(OUT / "pooled_and_macro_metrics.csv", pooled)
    write_csv(OUT / "rising_falling_analysis.csv", events)
    write_csv(OUT / "selected_stations.csv", examples)
    make_figures(comparison, examples)
    for name, expected in checksum.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Pre-existing artifact changed: {name}")
    manifest = {"pre_existing_artifact_sha256": checksum, "analysis_script_sha256": digest(Path(__file__)),
                "metric_tolerance_m": NUMERIC_TOLERANCE_M, "observed_change_tolerance_m": OBSERVED_CHANGE_TOLERANCE_M,
                "checks_passed": evidence, "no_models_loaded_or_retrained": True,
                "pre_existing_artifacts_verified_unchanged": True}
    write_text(OUT / "source_artifact_checksums.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_text(OUT / "prediction_analysis_report.md", make_report(comparison, stations, pooled, events, examples, evidence))
    print(pooled.to_string(index=False))
    print("\nSELECTED STATIONS\n" + examples.to_string(index=False))
    print("\nRISING/FALLING\n" + events.to_string(index=False))
    print(f"\nCompleted {OUT}; verified {len(checksum)} pre-existing artifacts unchanged.")


if __name__ == "__main__":
    main()
