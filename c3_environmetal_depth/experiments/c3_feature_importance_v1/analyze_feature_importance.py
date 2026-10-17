"""Consolidate saved importance CSVs and metadata; no model loading or prediction.

Run from RP using .venv/Scripts/python.exe experiments/c3_feature_importance_v1/
analyze_feature_importance.py. Existing analysis outputs are never overwritten.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
M1 = ROOT / "experiments/model1_diagnostics_v2"
M2 = ROOT / "experiments/model2_dynamic_v1/runs/20261019T115140_557496Z"
GROUPS = ("River-stage history", "Static environment", "Rainfall", "Soil moisture")
COLORS = dict(zip(GROUPS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")))
STATIC = ("mean_elevation_m", "mean_slope_deg", "tree_fraction", "cropland_fraction", "builtup_fraction", "catchment_area_km2")
HISTORY = ("stage_at_issue_m", "stage_lag_1d_m", "stage_lag_2d_m", "stage_change_1d_m")
WEATHER = ("c2_rainfall_mm", "c2_rainfall_3day_mm", "c2_rainfall_7day_mm", "c2_rainfall_14day_mm",
           "c2_soil_moisture_0_7cm", "c2_soil_moisture_7_28cm")
DISPLAY = {
    "stage_at_issue_m": "Current river stage", "stage_lag_1d_m": "River stage: 1-day lag",
    "stage_lag_2d_m": "River stage: 2-day lag", "stage_change_1d_m": "Latest daily stage change",
    "mean_elevation_m": "Mean catchment elevation", "mean_slope_deg": "Mean catchment slope",
    "tree_fraction": "Tree-cover fraction", "cropland_fraction": "Cropland fraction",
    "builtup_fraction": "Built-up fraction", "catchment_area_km2": "Catchment area",
    "c2_rainfall_mm": "1-day rainfall (window unverified)", "c2_rainfall_3day_mm": "3-day rainfall total",
    "c2_rainfall_7day_mm": "7-day rainfall total", "c2_rainfall_14day_mm": "14-day rainfall total",
    "c2_soil_moisture_0_7cm": "Soil moisture: 0–7 cm", "c2_soil_moisture_7_28cm": "Soil moisture: 7–28 cm",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)


def feature_group(feature):
    if feature in HISTORY:
        return GROUPS[0]
    if feature in STATIC:
        return GROUPS[1]
    if feature in WEATHER[:4]:
        return GROUPS[2]
    if feature in WEATHER[4:]:
        return GROUPS[3]
    raise ValueError(f"Unknown predictor: {feature}")


def build_tables():
    metadata = [json.loads((M1 / "models/frozen_experiment.json").read_text(encoding="utf-8")),
                json.loads((M2 / "models/model2_metadata.json").read_text(encoding="utf-8"))]
    definitions = [
        ("revised_model1", "Revised Model 1", M1 / "metrics/feature_importance.csv",
         metadata[0]["selected_environmental"], metadata[0]["features"], metadata[0]["fitted_trees"], metadata[0]["parameters"]["objective"]),
        ("selected_model2", "Selected Model 2", M2 / "metrics/model2_feature_importance.csv",
         metadata[1]["selected_model2"], metadata[1]["features"], metadata[1]["selected_retained_trees"], metadata[1]["selected_parameters"]["objective"]),
    ]
    feature_tables, group_rows = [], []
    for model, label, path, selected, features, trees, objective in definitions:
        saved = pd.read_csv(path)
        chosen = saved[saved.model.eq(selected)].copy()
        expected = (*HISTORY, *STATIC) if model == "revised_model1" else (*HISTORY, *STATIC, *WEATHER)
        if tuple(features) != expected or len(chosen) != len(features) or chosen.feature.duplicated().any():
            raise ValueError(f"Feature contract does not match metadata for {model}.")
        chosen = chosen.set_index("feature").loc[features].reset_index()
        for column in ("gain", "split_count"):
            chosen[column] = pd.to_numeric(chosen[column], errors="raise")
            if not np.isfinite(chosen[column]).all() or chosen[column].lt(0).any():
                raise ValueError(f"Invalid saved importance in {model}.")
        if not np.equal(chosen.split_count, np.floor(chosen.split_count)).all():
            raise ValueError("Split counts must be integers.")
        gain_total, split_total = float(chosen.gain.sum()), int(chosen.split_count.sum())
        if gain_total <= 0 or split_total <= 0:
            raise ValueError("Cannot normalize an all-zero saved importance table.")
        chosen["model"] = model
        chosen["model_label"] = label
        chosen["selected_artifact_id"] = selected
        chosen["target_representation"] = "delta"
        chosen["objective"] = objective
        chosen["retained_trees"] = trees
        chosen["feature_order"] = np.arange(1, len(chosen)+1)
        chosen["feature_group"] = chosen.feature.map(feature_group)
        chosen["feature_label"] = chosen.feature.map(DISPLAY)
        chosen["gain_share_pct"] = 100 * chosen.gain / gain_total
        chosen["split_share_pct"] = 100 * chosen.split_count / split_total
        chosen["used_in_tree_splits"] = chosen.split_count.gt(0)
        chosen["explicit_reconstruction_input"] = chosen.feature.eq("stage_at_issue_m")
        chosen["source_importance_csv"] = path.relative_to(ROOT).as_posix()
        np.testing.assert_allclose(chosen.gain_share_pct.sum(), 100, atol=1e-10)
        np.testing.assert_allclose(chosen.split_share_pct.sum(), 100, atol=1e-10)
        feature_tables.append(chosen)
        for group in GROUPS:
            subset = chosen[chosen.feature_group.eq(group)]
            present = len(subset) > 0
            group_rows.append({"model": model, "model_label": label, "selected_artifact_id": selected,
                               "feature_group": group, "input_status": "included" if present else "not_an_input",
                               "n_input_features": len(subset), "n_features_used_in_splits": int(subset.used_in_tree_splits.sum()),
                               "gain": float(subset.gain.sum()) if present else np.nan,
                               "split_count": int(subset.split_count.sum()) if present else np.nan,
                               "gain_share_pct": float(subset.gain_share_pct.sum()) if present else np.nan,
                               "split_share_pct": float(subset.split_share_pct.sum()) if present else np.nan,
                               "retained_trees": trees, "total_tree_splits": split_total})
    return pd.concat(feature_tables, ignore_index=True), pd.DataFrame(group_rows), metadata


def plot_features(table, destination, title):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ordered = table.sort_values(["gain_share_pct", "feature_order"], ascending=[False, True], kind="stable")
    height = max(5.5, .34*len(ordered)+2)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, height), sharey=True, layout="constrained")
    y = np.arange(len(ordered))
    for ax, metric, label in zip(axes, ("gain_share_pct", "split_share_pct"), ("Gain share (%)", "Split-count share (%)")):
        values = ordered[metric].to_numpy()
        ax.barh(y, values, color=[COLORS[g] for g in ordered.feature_group], height=.7)
        ax.set_yticks(y, labels=ordered.feature_label)
        ax.set_xlim(0, max(float(values.max())*1.2, 5))
        for position, value in enumerate(values):
            ax.text(value + ax.get_xlim()[1]*.015, position, f"{value:.2f}", va="center", fontsize=8)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    handles = [Patch(color=COLORS[g], label=g) for g in GROUPS if g in set(ordered.feature_group)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title + f"\n{int(table.retained_trees.iloc[0])} retained tree(s), {int(table.split_count.sum())} splits; predicted-change component", fontsize=12)
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_groups(groups, destination):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    models = ["revised_model1", "selected_model2"]
    for ax, column, title in zip(axes, ("gain_share_pct", "split_share_pct"), ("Within-model gain share", "Within-model split-count share")):
        left = np.zeros(2)
        for group in GROUPS:
            values = np.array([groups[(groups.model.eq(model)) & groups.feature_group.eq(group)][column].fillna(0).iloc[0] for model in models])
            ax.barh([0, 1], values, left=left, color=COLORS[group], height=.5)
            for i, value in enumerate(values):
                if value >= 3:
                    ax.text(left[i]+value/2, i, f"{value:.1f}%", ha="center", va="center", fontsize=9,
                            color="white" if group in (GROUPS[0], GROUPS[2]) else "black")
            left += values
        ax.set(yticks=[0, 1], yticklabels=["Revised Model 1", "Selected Model 2"], xlim=(0, 100),
               xlabel="Share (%)", title=title, ylim=(-.6, 1.6))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=COLORS[g], label=g) for g in GROUPS], loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Feature groups in the fitted change-prediction trees\nModel 1 has no weather inputs; Model 2 includes static inputs but uses none in splits.", fontsize=11)
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def report(features, groups, metadata):
    lines = ["# C3 Environmental Depth: consolidated feature-importance analysis", "",
             "## Scope and existing work", "",
             "This step consolidates existing saved importance CSVs; it does not calculate new importance "
             "from observations, load models, generate predictions, retrain, tune settings, install packages "
             "or access remote services. Both selected models predict changes, with current stage added "
             "explicitly to reconstruct next-day stage. The earlier read-only model inspection verified "
             "that the saved importance values match the serialized estimators.", "",
             "The existing figures already contain gain plots. The new figures add paired split-versus-gain "
             "views of the selected models, readable group labels, zero-importance features and a consolidated "
             "group comparison. Existing figures and results are preserved.", "",
             "Existing sources:", "",
             "- [Model 1 importance CSV](../model1_diagnostics_v2/metrics/feature_importance.csv) and "
             "[existing figure](../model1_diagnostics_v2/figures/feature_importance_comparison.png).",
             "- [Model 2 importance CSV](../model2_dynamic_v1/runs/20261019T115140_557496Z/metrics/model2_feature_importance.csv) and "
             "[existing figure](../model2_dynamic_v1/runs/20261019T115140_557496Z/figures/model2_feature_importance.png).", "",
             "Selected artifacts: revised Model 1 `delta_small` (10 inputs, one retained L2-regression tree); "
             "Model 2 `model2_delta_mae_loss` (16 inputs, six retained MAE-loss trees). The original "
             "absolute-stage Model 1 is not substituted for the revised model in this analysis.", "",
             "## What the measures mean", "",
             "| Measure | Interpretation | What it does not establish |",
             "|---|---|---|",
             "| Split importance | Number of tree splits using a predictor | Size or direction of its effect, forecast accuracy or causation |",
             "| Gain importance | Accumulated improvement in the internal training split criterion for splits using the predictor | Held-out MAE reduction, explained variance or physical influence |",
             "| SHAP | Per-observation contributions to a model output relative to a reference expectation | A causal mechanism or proof of added forecasting value |", "",
             "The SHAP package and saved SHAP artifacts were absent at inspection. No SHAP computations "
             "or dependency installation are included. Native LightGBM can support contribution explanations "
             "in a separately scoped analysis, but none are represented as results here. For these wrappers, "
             "tree SHAP would explain the predicted change; the explicit current-stage addition would still "
             "need separate treatment in an explanation of final river stage.", "",
             "Shares below normalize each model's own total to 100%. They are not percentages of "
             "forecast skill or river level. Raw gains are not comparable across models because objectives, "
             "tree counts and model structures differ. Group shares sum feature attributions; they do not "
             "adjust for differing group sizes or provide an independent group-ablation estimate.", "",
             "## Feature-group comparison", "",
             "| Model | Group | Inputs | Features used | Gain share (%) | Split share (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for row in groups.itertuples():
        gain = f"{row.gain_share_pct:.2f}" if row.input_status == "included" else "Not an input"
        split = f"{row.split_share_pct:.2f}" if row.input_status == "included" else "Not an input"
        lines.append(f"| {row.model_label} | {row.feature_group} | {row.n_input_features} | {row.n_features_used_in_splits} | {gain} | {split} |")
    lines += ["", "River history leads by gain in both fitted models: 69.44% in revised Model 1 and "
              "58.47% in selected Model 2. Model 2 weather accounts for 41.53% of gain and 60% of split "
              "counts. Rainfall alone contributes 38.00% of gain and soil moisture 3.53%. By split count, "
              "rainfall exceeds river history (48.57% versus 40%). Thus 'most influential' depends on "
              "the measure; historical levels dominate by gain, not by every usage measure.", "",
              "## Revised Model 1: all six environmental features", "",
              "| Environmental feature | Gain share (%) | Split count | Interpretation |",
              "|---|---:|---:|---|"]
    for feature in STATIC:
        row = features[(features.model.eq("revised_model1")) & features.feature.eq(feature)].iloc[0]
        interpretation = "Used in one of two fitted splits" if row.split_count else "Included as an input; unused in retained tree"
        lines.append(f"| `{feature}` | {row.gain_share_pct:.2f} | {int(row.split_count)} | {interpretation} |")
    lines += ["", "Current stage contributes 69.44% of gain; catchment area contributes 30.56%. Each "
              "is used once, giving a 50/50 split-count share. The other eight inputs are unused by "
              "the retained tree. This is only a three-leaf, two-split tree, so treating the 30.56% "
              "area share as a stable environmental ranking would be unjustified. Catchment area may "
              "also encode station differences; its importance does not establish a hydrological cause.", "",
              "## Selected Model 2: individual feature use", "",
              "| Feature | Group | Gain share (%) | Splits | Split share (%) |",
              "|---|---|---:|---:|---:|"]
    second = features[features.model.eq("selected_model2")].sort_values(["gain_share_pct", "feature_order"], ascending=[False, True])
    for row in second.itertuples():
        lines.append(f"| `{row.feature}` | {row.feature_group} | {row.gain_share_pct:.2f} | {int(row.split_count)} | {row.split_share_pct:.2f} |")
    lines += ["", "Yesterday's stage is the leading individual feature (51.46% of gain), followed "
              "by 7-day rainfall (17.64%) and 14-day rainfall (16.97%). The six trees contain only 35 "
              "splits. All six static environmental inputs and the 1-day rainfall input receive zero "
              "split and gain importance in this particular fitted model. Zero importance means no "
              "retained split used the feature; it does not mean the variable is physically irrelevant "
              "or would never help with another independently evaluated dataset or model.", "",
              "## Current stage remains essential outside the trees", "",
              "`predicted_next_stage = stage_at_issue_m + predicted_delta`", "",
              "In Model 2, current stage has zero tree importance but is added directly to every final "
              "forecast. Holding the predicted change fixed, a one-metre change in this anchor shifts "
              "the reconstructed stage by one metre. That algebraic dependence is outside the tree "
              "importance calculation. The importance charts therefore describe only the change-prediction "
              "component, not the entire reliance of the forecasting pipeline on river observations. "
              "The same reconstruction principle applies to revised Model 1, where current stage is "
              "also used in a tree split.", "",
              "## Importance and forecasting accuracy are different questions", "",
              "Saved metrics, reused without new scoring:", "",
              "| Split | Selected Model 2 MAE (m) | Matched no-weather delta MAE (m) | Persistence MAE (m) | Weather minus no-weather MAE (m) |",
              "|---|---:|---:|---:|---:|"]
    for split in ("validation", "test"):
        table = pd.read_csv(M2 / "metrics" / f"model2_{split}_metrics.csv").set_index("model")
        weather = float(table.loc[metadata[1]["selected_model2"], "mae_m"])
        control = float(table.loc["control_delta_no_weather", "mae_m"])
        baseline = float(table.loc["persistence", "mae_m"])
        lines.append(f"| {split.upper()} | {weather:.6f} | {control:.6f} | {baseline:.6f} | {weather-control:+.6f} |")
    first_comparison = pd.read_csv(M1 / "metrics/model1_validation_comparison.csv").set_index("model")
    primary = float(first_comparison.loc[metadata[0]["selected_environmental"], "mae_m"])
    ablation = float(first_comparison.loc["history_only_ablation", "mae_m"])
    lines += ["", "Positive values in the last column mean the weather model is worse. Weather is used "
              "substantially inside the selected Model 2 trees, yet the matched no-weather control has "
              "slightly lower overall MAE on both splits. This is not a contradiction: training split "
              "gain measures how the fitted trees partitioned their training response, whereas held-out "
              "MAE measures predictive error. Features can be useful to the fitted partitioning without "
              "delivering an additional generalizable gain.", "",
              f"For revised Model 1, saved validation MAE is {primary:.6f} m with static inputs and "
              f"{ablation:.6f} m for its matched history-only ablation: a difference of "
              f"{ablation-primary:+.6f} m favouring the environmental version. This very small, "
              "selection-conditioned difference does not establish a robust static-environment benefit.", "",
              "The selected Model 2's small improvement over persistence cannot be attributed to weather "
              "because its matched no-weather delta model does slightly better overall. Comparisons "
              "with original absolute-stage Model 1 also change the target representation, loss and "
              "selected configuration. Do not label the whole improvement as an effect of environmental depth.", "",
              "## C3 research interpretation", "",
              "1. Historical gauge observations remain central: they dominate fitted gain and current "
              "stage anchors every reconstructed prediction.",
              "2. The revised static model uses catchment area but not the other five static descriptors; "
              "the selected weather model uses no static descriptors in its retained splits.",
              "3. The selected weather model uses multi-day rainfall and both soil-moisture features. "
              "Their internal importance establishes fitted use, not incremental predictive value.",
              "4. The matched accuracy comparisons do not demonstrate a useful overall weather gain "
              "under this short study and supplied alignment. This is a valid negative research finding, "
              "not proof that weather is generally unhelpful for hydrological forecasting.", "",
              "## Limitations and interpretation safeguards", "",
              "- Importance is a fitted association, not proof that a variable causes flooding. These "
              "models predict river stage, not flood-inundation or warning thresholds.",
              "- Importance is unsigned: these charts cannot establish whether more rainfall or larger "
              "catchments increase or decrease a particular prediction.",
              "- Overlapping rainfall windows and lagged stage measurements may act as substitutes. "
              "Correlated inputs can redistribute attribution; no new correlation analysis was performed here.",
              "- Static features can encode gauge identity, basin differences or gauge-datum effects. "
              "Candidate catchments remain unverified and WorldCover refers to 2021.",
              "- The models are small (one and six trees), the TRAIN split has only 700 records, and "
              "the approximately 40-day record does not establish seasonal or unseen-station generalization.",
              "- The supplied previous-day weather join adds an extra day of lag to rolling totals "
              "under the supplied C2 definition. One-day rainfall windows, soil-moisture aggregation "
              "and product publication availability remain unverified. No alignment or dataset was changed.",
              "- Weather-free controls are matched to selected weather-model settings and are not an "
              "unbiased search for the best possible weather-free model.",
              "- TEST outcomes were already examined. The comparisons and this interpretation are "
              "descriptive, not independent confirmation or evidence of statistical significance.", "",
              "## Files and reproducibility", "",
              "- `feature_importance_by_feature.csv`: 26 rows, preserving raw saved gain/split counts, "
              "feature order, groups, normalized shares and the explicit reconstruction-input flag.",
              "- `feature_importance_by_group.csv`: four groups for each model. Absent weather groups "
              "in Model 1 are `not_an_input` with blank numerical importance; Model 2's included but "
              "unused static group is explicitly zero. These are different situations.",
              "- `figures/model1_gain_vs_split.png` and `figures/model2_gain_vs_split.png`: all inputs "
              "retained on axes, including zeros, coloured by group.",
              "- `figures/feature_group_comparison.png`: within-model gain and split-share comparisons.",
              "- `source_artifact_checksums.json`: source hashes, preserved-experiment hashes, analysis "
              "script hash and integrity checks.", "",
              "Run from the RP root with `.\\.venv\\Scripts\\python.exe "
              "experiments/c3_feature_importance_v1/analyze_feature_importance.py`. The script refuses "
              "to overwrite any existing output. No estimators are loaded and no predictions are made. "
              "A completed analysis should be read, not regenerated over its existing files."]
    return "\n".join(lines) + "\n"


def main():
    planned = ["feature_importance_by_feature.csv", "feature_importance_by_group.csv",
               "feature_importance_report.md", "source_artifact_checksums.json"]
    if any((OUT / p).exists() for p in planned) or (OUT / "figures").exists():
        raise FileExistsError("Analysis outputs already exist; nothing will be overwritten.")
    existing = sorted({p for folder in (M1, M2) for p in folder.rglob("*") if p.is_file()})
    preservation = {p.relative_to(ROOT).as_posix(): sha256(p) for p in existing}
    features, groups, metadata = build_tables()
    assert len(features) == 26 and len(groups) == 8
    assert features.groupby("model").split_count.sum().to_dict() == {"revised_model1": 2, "selected_model2": 35}
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    figures = OUT / "figures"
    figures.mkdir(exist_ok=False)
    plot_features(features[features.model.eq("revised_model1")], figures / "model1_gain_vs_split.png", "Revised static-environment Model 1")
    plot_features(features[features.model.eq("selected_model2")], figures / "model2_gain_vs_split.png", "Selected stage-change Model 2")
    plot_groups(groups, figures / "feature_group_comparison.png")
    text_new(OUT / "feature_importance_by_feature.csv", features.to_csv(index=False))
    text_new(OUT / "feature_importance_by_group.csv", groups.to_csv(index=False))
    text_new(OUT / "feature_importance_report.md", report(features, groups, metadata))
    for name, expected in preservation.items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"Existing artifact changed: {name}")
    manifest = {"pre_existing_experiment_artifact_sha256": preservation,
                "analysis_script_sha256": sha256(Path(__file__)),
                "source_importance_csvs": [str((M1 / 'metrics/feature_importance.csv').relative_to(ROOT)).replace('\\', '/'),
                                          str((M2 / 'metrics/model2_feature_importance.csv').relative_to(ROOT)).replace('\\', '/')],
                "checks": {"feature_rows": 26, "group_rows": 8, "model1_splits": 2, "model2_splits": 35,
                           "within_model_shares_sum_to_100": True, "metadata_feature_order_verified": True,
                           "pre_existing_artifacts_unchanged": True, "models_loaded": False,
                           "new_predictions_or_fits": False, "packages_installed": False}}
    text_new(OUT / "source_artifact_checksums.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(groups.to_string(index=False))
    print(f"\nCompleted {OUT}; {len(preservation)} pre-existing experiment artifacts verified unchanged.")


if __name__ == "__main__":
    main()
