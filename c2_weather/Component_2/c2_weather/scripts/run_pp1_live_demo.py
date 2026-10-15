from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, project_path
from src.pp1_demo import VARIANTS, evaluate_record, load_demo_records


st.set_page_config(page_title="C2 PP1 Live Demo", page_icon="🌊", layout="wide")
config = load_config()
records = load_demo_records()

st.title("River Level Forecast")
st.warning(
    "RETROSPECTIVE INFERENCE on held-out historical test records. "
    "The actual next-day reading is shown only after prediction for evaluation; it is never an input."
)
st.caption("Saved Raw, Engineered, and Hybrid XGBoost models · no retraining · 24-hour horizon")

stations = sorted(records["station"].unique().tolist())
selected_station = st.selectbox("Station", stations)
station_records = records.loc[records["station"].eq(selected_station)].sort_values("date")
available_dates = station_records["date_label"].tolist()
selected_date_label = st.selectbox("Held-out forecast origin date", available_dates)
record = station_records.loc[station_records["date_label"].eq(selected_date_label)].iloc[0]
result = evaluate_record(record)

left, middle, right = st.columns(3)
left.metric("Forecast origin · 9:00 AM", result["forecast_origin"].strftime("%Y-%m-%d"))
middle.metric("Target date · next 9:00 AM", result["target_date"].strftime("%Y-%m-%d"))
right.metric("Current water level", f"{result['current_water_level_m']:.3f} m")

st.subheader("Retrospective predictions")
rows = []
for model_key, label in [
    ("persistence", "Persistence"),
    ("raw", "Raw XGBoost"),
    ("engineered", "Engineered XGBoost"),
    ("hybrid", "Hybrid XGBoost"),
]:
    prediction = result["predictions"][model_key]
    rows.append(
        {
            "Model": label,
            "Predicted next-day level (m)": prediction["predicted_water_level_m"],
            "Actual absolute error (m)": prediction["absolute_error_m"],
            "Difference from saved prediction (m)": prediction["difference_from_saved_m"],
            "Matches saved CSV": prediction["matches_saved_prediction"],
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    "Predicted next-day level (m)": st.column_config.NumberColumn(format="%.6f"),
    "Actual absolute error (m)": st.column_config.NumberColumn(format="%.6f"),
    "Difference from saved prediction (m)": st.column_config.NumberColumn(format="%.9f"),
})
st.metric("Actual next-day DMC water level", f"{result['actual_next_day_water_level_m']:.3f} m")

st.subheader("Station test-period series")
figure, axis = plt.subplots(figsize=(11, 4.8))
axis.plot(station_records["target_date"], station_records["actual_water_level_m"], marker="o", label="Actual DMC")
axis.plot(station_records["target_date"], station_records["persistence_prediction_m"], marker=".", label="Persistence")
for variant in VARIANTS:
    axis.plot(
        station_records["target_date"],
        station_records[f"saved_{variant}_prediction_m"],
        marker=".",
        label=f"{variant.title()} XGBoost",
    )
axis.axvline(result["target_date"], color="#555555", linestyle="--", linewidth=1, label="Selected target date")
axis.set_xlabel("Target observation date (09:00)")
axis.set_ylabel("Water level (m)")
axis.set_title(f"{selected_station} · held-out test period")
axis.grid(alpha=0.2)
axis.legend(ncol=3, frameon=False)
figure.tight_layout()
st.pyplot(figure, width="stretch")
plt.close(figure)

st.subheader("Original held-out test metrics")
metrics_path = project_path(config, config["outputs"]["metrics_dir"]) / "model_comparison.csv"
metrics = pd.read_csv(metrics_path)
metrics["model"] = metrics["model"].map(
    {"persistence": "Persistence", "raw": "Raw XGBoost", "engineered": "Engineered XGBoost", "hybrid": "Hybrid XGBoost"}
)
st.dataframe(metrics.rename(columns={"n": "Test samples", "mae_m": "MAE (m)", "rmse_m": "RMSE (m)", "r2": "R²", "model": "Model"}), hide_index=True, width="stretch")
st.caption("All rows above are from the original common held-out test set; persistence has the lowest overall MAE.")

st.subheader("Saved SHAP explanations")
figure_dir = project_path(config, config["outputs"]["figures_dir"])
for variant in VARIANTS:
    image = figure_dir / f"{variant}_shap_summary.png"
    if image.is_file():
        st.markdown(f"**{variant.title()} XGBoost**")
        st.image(str(image), width="stretch")
    else:
        st.info(f"Saved SHAP summary is unavailable for {variant}.")

st.caption("This is a historical evaluation interface, not a real-time forecast service. Weather timestamps and gauge datum remain unverified.")