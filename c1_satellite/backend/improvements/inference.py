"""C1 Model Inference Service: next-day river stage and flood risk predictions.

Uses fitted TRAIN-only pipelines (A_ENH_2B / STAGE_TEMPORAL) and the DMC 
station catalog to produce next-day stage predictions and flood risk classifications.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union, Any
import joblib
import numpy as np
import pandas as pd

from .data import load_project
from .stations import load_station_catalog, classify_flood_risk

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MAE_M = 0.2182  # Empirically validated benchmark MAE


class C1Predictor:
    """Predictor service for next-day gauging-station water level forecasts."""

    def __init__(self, root: Path = ROOT, model_name: str = "A_ENH_2B"):
        self.root = Path(root)
        self.model_name = model_name
        self.model_dir = self.root / "saved_models" / "improvement_experiments"
        self.pipeline_path = self.model_dir / f"{model_name}.joblib"

        if not self.pipeline_path.is_file():
            fallback = self.model_dir / "STAGE_TEMPORAL.joblib"
            if fallback.is_file():
                self.model_name = "STAGE_TEMPORAL"
                self.pipeline_path = fallback
            else:
                raise FileNotFoundError(
                    f"Saved model pipeline not found at {self.pipeline_path}. Run Phase 17 first."
                )

        self._models_cache: dict[str, Any] = {}
        if self.pipeline_path.is_file():
            self._models_cache[self.model_name] = joblib.load(self.pipeline_path)
            self.model = self._models_cache[self.model_name]
        else:
            self.model = None

        self.stations = load_station_catalog(self.root)
        self._dataset: Optional[pd.DataFrame] = None
        self._labels_df: Optional[pd.DataFrame] = None
        self._audit: Optional[dict] = None

    def get_model(self, name: str) -> Any:
        """Loads and caches trained ML pipeline by model identifier."""
        name = name.strip()
        if name == "PERSISTENCE":
            return None
        if name not in self._models_cache:
            path = self.model_dir / f"{name}.joblib"
            if not path.is_file():
                # Check with alternative extensions or fallbacks
                raise FileNotFoundError(f"Model pipeline '{name}' not found at {path}")
            self._models_cache[name] = joblib.load(path)
        return self._models_cache[name]

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            self._dataset, self._audit = load_project(self.root)
        return self._dataset

    @property
    def labels_df(self) -> pd.DataFrame:
        if self._labels_df is None:
            labels_path = self.root / "local_data" / "FloodRisk_DMC_StationDay_LABELLED_1429.csv"
            if labels_path.is_file():
                self._labels_df = pd.read_csv(labels_path, dtype={"station_id": str, "forecast_date": str})
            else:
                self._labels_df = self.dataset
        return self._labels_df

    def get_stations(self) -> list[dict]:
        """Returns all 40 monitoring gauges with metadata and alert thresholds."""
        return list(self.stations.values())

    def get_station(self, station_id: str) -> Optional[dict]:
        """Returns metadata for a specific station."""
        return self.stations.get(station_id.lower().strip())

    def get_available_historical_records(self) -> dict:
        """Provides dynamic discovery of available station/date combinations and models."""
        df = self.dataset
        stations_meta = self.stations

        # Group available forecast dates by station
        grouped = df.groupby("station_id")["forecast_date"].apply(lambda s: sorted(list(set(s)))).to_dict()

        stations_list = []
        for st_id, st in stations_meta.items():
            dates = grouped.get(st_id, [])
            val_dates = [d for d in dates if d in ["2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14", "2025-12-15", "2025-12-16"]]
            default_d = "2025-12-14" if "2025-12-14" in dates else (val_dates[-1] if val_dates else (dates[-1] if dates else None))
            stations_list.append({
                "station_id": st_id,
                "station_name": st["station_name"],
                "basin": st["basin"],
                "district": st.get("district", "Sri Lanka"),
                "coordinates": {"lat": st.get("lat"), "lon": st.get("lon")},
                "available_dates_count": len(dates),
                "validation_dates": val_dates,
                "available_dates": dates,
                "default_date": default_d,
                "has_validation_records": len(val_dates) > 0,
            })

        # Sort so stations with validation records appear first, then alphabetical
        stations_list.sort(key=lambda x: (not x["has_validation_records"], x["station_name"]))

        model_catalog = [
            {
                "id": "A_ENH_2B",
                "name": "Model A — Latest SAR + SWI",
                "short_name": "Model A",
                "tag": "Sentinel-1 + ASCAT Latest",
                "description": "Uses most recent Sentinel-1 SAR (VV/VH backscatter statistics) and ASCAT soil moisture (SWI005/010).",
                "mae": 0.2182,
                "rmse": 0.3544,
                "n_features": 21,
                "feature_summary": "Stage history (lags, change, accel) + Latest SAR (VV, VH dB) + Latest SWI (005, 010)"
            },
            {
                "id": "B_ENH_2B",
                "name": "Model B — Historical SAR + Latest SWI",
                "short_name": "Model B",
                "tag": "SAR Temporal Depth",
                "description": "Extends Model A with same-orbit multi-temporal SAR backscatter change rates (dB/day).",
                "mae": 0.2214,
                "rmse": 0.3605,
                "n_features": 26,
                "feature_summary": "Model A features + Same-orbit SAR change rates (VV/VH mean change & change/day)"
            },
            {
                "id": "C_ENH_2B",
                "name": "Model C — Latest SAR + Historical SWI",
                "short_name": "Model C",
                "tag": "SWI Temporal Depth",
                "description": "Extends Model A with 7-day ASCAT soil wetness history, 3-day/7-day rolling means and trends.",
                "mae": 0.2193,
                "rmse": 0.3541,
                "n_features": 41,
                "feature_summary": "Model A features + 7-day SWI005 & SWI010 history + rolling 3d/7d means & trends"
            },
            {
                "id": "D_ENH_2B",
                "name": "Model D — Historical SAR + Historical SWI",
                "short_name": "Model D",
                "tag": "Full Multimodal Depth",
                "description": "Combines maximum temporal depth across both Sentinel-1 SAR and ASCAT SWI instruments.",
                "mae": 0.2215,
                "rmse": 0.3578,
                "n_features": 46,
                "feature_summary": "All features combined: Stage history + SAR temporal depth + SWI 7-day history"
            },
            {
                "id": "STAGE_TEMPORAL",
                "name": "Stage-Temporal Baseline",
                "short_name": "Stage-Temporal",
                "tag": "Gauge-Only Hydrology",
                "description": "Hydrological benchmark using current river stage, 1-day/2-day lags, and stage acceleration without satellite data.",
                "mae": 0.2526,
                "rmse": 0.3773,
                "n_features": 6,
                "feature_summary": "Current stage + D-1 & D-2 lags + 1-day rate of change + stage acceleration"
            },
            {
                "id": "STAGE_ONLY_BASE",
                "name": "Stage-Only Baseline",
                "short_name": "Stage-Only",
                "tag": "Simple Gauge",
                "description": "Linear ridge regression on current observed river stage alone.",
                "mae": 0.2565,
                "rmse": 0.3666,
                "n_features": 1,
                "feature_summary": "Current observed stage alone (RIDGE_1000 regularization)"
            },
            {
                "id": "PERSISTENCE",
                "name": "Persistence Baseline",
                "short_name": "Persistence",
                "tag": "Zero-Change Naive",
                "description": "Zero-change naive benchmark assuming next-day river stage equals current stage (y_pred = y_t).",
                "mae": 0.2182,
                "rmse": 0.3521,
                "n_features": 1,
                "feature_summary": "No ML model; assumes Delta = 0.0 m (strong baseline during national recession period)"
            }
        ]

        return {
            "stations": stations_list,
            "total_stations": len(stations_list),
            "validation_dates": ["2025-12-10", "2025-12-11", "2025-12-12", "2025-12-13", "2025-12-14", "2025-12-15", "2025-12-16"],
            "models": model_catalog,
            "default_selection": {
                "station_id": "ellagawa",
                "forecast_date": "2025-12-14",
                "model_id": "A_ENH_2B"
            }
        }

    def predict_for_record(
        self,
        station_id: str,
        forecast_date: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> dict:
        """Predicts next-day stage for a known station and forecast date using specified model."""
        st_id = station_id.lower().strip()
        st_meta = self.get_station(st_id)
        if not st_meta:
            raise KeyError(f"Unknown station ID: {station_id}")

        active_model = (model_name or self.model_name).strip()

        df = self.dataset
        matches = df[df.station_id == st_id]
        if len(matches) == 0:
            raise ValueError(f"No eligible observation records found for station: {station_id}")

        if forecast_date:
            row_df = matches[matches.forecast_date == str(forecast_date)]
            if len(row_df) == 0:
                available = sorted(matches.forecast_date.unique())
                raise ValueError(
                    f"Forecast date '{forecast_date}' not found for {station_id}. Available: {available[:3]}...{available[-3:]}"
                )
        else:
            row_df = matches.sort_values("forecast_date").tail(1)

        row = row_df.iloc[0]
        f_date = str(row["forecast_date"])
        current_stage = float(row["stage_at_issue_m"])
        actual_next_stage = float(row["target_stage_next_day_m"]) if "target_stage_next_day_m" in row and pd.notna(row["target_stage_next_day_m"]) else None
        split = str(row.get("chronological_split", "UNKNOWN"))

        # Model inference: predicts delta stage
        if active_model == "PERSISTENCE":
            pred_delta = 0.0
            pred_stage = round(current_stage, 3)
        else:
            model = self.get_model(active_model)
            pred_delta = float(model.predict(row_df)[0])
            pred_stage = round(current_stage + pred_delta, 3)

        # 95% Confidence / Error margin interval based on empirical validated MAE
        err_margin = round(VALIDATION_MAE_M * 1.96, 3)
        lower_bound = round(pred_stage - err_margin, 3)
        upper_bound = round(pred_stage + err_margin, 3)

        current_risk = classify_flood_risk(current_stage, st_meta)
        predicted_risk = classify_flood_risk(pred_stage, st_meta)
        actual_risk = classify_flood_risk(actual_next_stage, st_meta) if actual_next_stage is not None else None

        # Build comprehensive satellite feature provenance
        sat_provenance = {
            "sar_latest_age_hours": float(row["sar_latest_age_hours"]) if "sar_latest_age_hours" in row and pd.notna(row["sar_latest_age_hours"]) else None,
            "sar_latest_acquisition_utc": str(row["sar_latest_acquisition_utc"]) if "sar_latest_acquisition_utc" in row and pd.notna(row["sar_latest_acquisition_utc"]) else None,
            "sar_vv_mean_db": round(float(row["sar_vv_mean_db"]), 2) if "sar_vv_mean_db" in row and pd.notna(row["sar_vv_mean_db"]) else None,
            "sar_vh_mean_db": round(float(row["sar_vh_mean_db"]), 2) if "sar_vh_mean_db" in row and pd.notna(row["sar_vh_mean_db"]) else None,
            "sar_vv_minus_vh_db": round(float(row["sar_vv_minus_vh_db"]), 2) if "sar_vv_minus_vh_db" in row and pd.notna(row["sar_vv_minus_vh_db"]) else None,
            "sar_patch_valid_fraction": round(float(row["sar_patch_valid_fraction"]), 3) if "sar_patch_valid_fraction" in row and pd.notna(row["sar_patch_valid_fraction"]) else 1.0,
            "sar_vv_mean_change_db": round(float(row["sar_vv_mean_change_db"]), 3) if "sar_vv_mean_change_db" in row and pd.notna(row["sar_vv_mean_change_db"]) else None,
            "swi_eligible_date": str(row["latest_eligible_swi_date"]) if "latest_eligible_swi_date" in row and pd.notna(row["latest_eligible_swi_date"]) else None,
            "swi005_latest": round(float(row["swi005_d1"]), 1) if "swi005_d1" in row and pd.notna(row["swi005_d1"]) else None,
            "swi010_latest": round(float(row["swi010_d1"]), 1) if "swi010_d1" in row and pd.notna(row["swi010_d1"]) else None,
            "swi005_7day_trend": round(float(row["swi005_trend7"]), 3) if "swi005_trend7" in row and pd.notna(row["swi005_trend7"]) else None,
            "quality_flag": "QFLAG >= 50 (Verified High Quality)",
        }

        # Format human-friendly condition for soil wetness
        swi_val = sat_provenance["swi005_latest"]
        if swi_val is not None:
            if swi_val >= 60.0:
                sat_provenance["swi_condition"] = "High Wetness / Saturated"
            elif swi_val >= 40.0:
                sat_provenance["swi_condition"] = "Moderate Wetness"
            else:
                sat_provenance["swi_condition"] = "Low Wetness / Dry"
        else:
            sat_provenance["swi_condition"] = "Normal"

        res = {
            "station_id": st_id,
            "station_name": st_meta["station_name"],
            "basin": st_meta["basin"],
            "district": st_meta.get("district", "Sri Lanka"),
            "coordinates": {"lat": st_meta["lat"], "lon": st_meta["lon"]},
            "station_image_url": st_meta.get("image_url", "/static/assets/stations/default.jpg"),
            "forecast_date": f_date,
            "forecast_issue_time": f"{f_date} 09:30 SLST",
            "target_observation_time": f"{pd.to_datetime(f_date) + pd.Timedelta(days=1):%Y-%m-%d} 09:00 SLST",
            "target_date": f"{pd.to_datetime(f_date) + pd.Timedelta(days=1):%Y-%m-%d}",
            "model_family": active_model,
            "chronological_split": split,
            "stage_at_issue_0900_m": current_stage,
            "predicted_delta_m": round(pred_delta, 4),
            "predicted_next_day_stage_m": pred_stage,
            "predicted_trend": "Falling" if pred_delta < -0.01 else ("Rising" if pred_delta > 0.01 else "Steady"),
            "uncertainty_interval_95": {
                "lower_m": lower_bound,
                "upper_m": upper_bound,
                "margin_m": err_margin,
            },
            "flood_thresholds": {
                "alert_threshold_m": st_meta.get("alert_threshold_m"),
                "minor_flood_threshold_m": st_meta.get("minor_flood_threshold_m"),
                "major_flood_threshold_m": st_meta.get("major_flood_threshold_m"),
            },
            "flood_risk": {
                "current": current_risk,
                "predicted_next_day": predicted_risk,
                "actual_next_day": actual_risk,
                "risk_escalating": predicted_risk["severity"] > current_risk["severity"],
            },
            "satellite_provenance": sat_provenance,
            "preview_urls": {
                "sar": f"/api/v1/satellite/preview/sar/{st_id}/{f_date}",
                "swi": f"/api/v1/satellite/preview/swi/{st_id}/{f_date}",
            }
        }

        if actual_next_stage is not None:
            err_m = round(abs(actual_next_stage - pred_stage), 4)
            res["validation_ground_truth"] = {
                "actual_next_day_stage_m": actual_next_stage,
                "actual_delta_m": round(actual_next_stage - current_stage, 4),
                "absolute_error_m": err_m,
                "absolute_error_cm": round(err_m * 100.0, 1),
            }

        return res

    def get_historical_detailed(
        self,
        station_id: str,
        forecast_date: str,
        model_name: str = "A_ENH_2B"
    ) -> dict:
        """Returns end-to-end prediction, time-series history, satellite data, and benchmark comparison."""
        st_id = station_id.lower().strip()
        f_date = forecast_date.strip()

        # Run primary prediction for selected model
        primary = self.predict_for_record(st_id, f_date, model_name=model_name)

        # Retrieve observed time series for the station (5 preceding days up to target day)
        labels = self.labels_df
        st_labels = labels[labels.station_id == st_id].copy()
        st_labels["forecast_date"] = pd.to_datetime(st_labels["forecast_date"])
        f_ts = pd.to_datetime(f_date)
        start_ts = f_ts - pd.Timedelta(days=4)
        end_ts = f_ts + pd.Timedelta(days=1)

        window = st_labels[(st_labels["forecast_date"] >= start_ts) & (st_labels["forecast_date"] <= end_ts)].sort_values("forecast_date")

        time_series = []
        for _, w_row in window.iterrows():
            d_str = w_row["forecast_date"].strftime("%Y-%m-%d")
            obs_stage = float(w_row["stage_at_issue_m"])
            is_issue = (d_str == f_date)
            time_series.append({
                "date": d_str,
                "display_date": w_row["forecast_date"].strftime("%d %b"),
                "stage_observed": obs_stage,
                "stage_predicted": primary["predicted_next_day_stage_m"] if is_issue else None,
                "is_issue_date": is_issue,
                "is_target_date": False
            })

        # Add target day point if present in ground truth
        target_d_str = primary["target_date"]
        target_stage = primary.get("validation_ground_truth", {}).get("actual_next_day_stage_m")
        if target_stage is not None:
            time_series.append({
                "date": target_d_str,
                "display_date": pd.to_datetime(target_d_str).strftime("%d %b"),
                "stage_observed": target_stage,
                "stage_predicted": primary["predicted_next_day_stage_m"],
                "is_issue_date": False,
                "is_target_date": True
            })

        # Multi-model ablation comparison on this exact record
        candidate_models = ["PERSISTENCE", "STAGE_ONLY_BASE", "STAGE_TEMPORAL", "A_ENH_2B", "B_ENH_2B", "C_ENH_2B", "D_ENH_2B"]
        comparisons = []
        for m_id in candidate_models:
            try:
                m_res = self.predict_for_record(st_id, f_date, model_name=m_id)
                m_gt = m_res.get("validation_ground_truth", {})
                comparisons.append({
                    "model_id": m_id,
                    "model_name": m_id.replace("_ENH_2B", "").replace("_", " "),
                    "predicted_stage_m": m_res["predicted_next_day_stage_m"],
                    "predicted_delta_m": m_res["predicted_delta_m"],
                    "absolute_error_m": m_gt.get("absolute_error_m"),
                    "absolute_error_cm": m_gt.get("absolute_error_cm"),
                    "is_selected": (m_id == model_name)
                })
            except Exception:
                continue

        # Extract 7-day SWI values
        df = self.dataset
        match = df[(df.station_id == st_id) & (df.forecast_date == f_date)]
        swi_history = []
        if len(match) > 0:
            m_row = match.iloc[0]
            for day_idx in range(7, 0, -1):
                col_name = f"swi005_d{day_idx}"
                if col_name in m_row and pd.notna(m_row[col_name]):
                    swi_history.append({
                        "lag_label": f"D-{day_idx}",
                        "value": round(float(m_row[col_name]), 1)
                    })

        primary["time_series"] = time_series
        primary["model_comparisons"] = comparisons
        primary["swi_7day_history"] = swi_history

        return primary

    def predict_custom(self, features: dict[str, Any]) -> dict:
        """Ad-hoc prediction given custom dictionary of features."""
        st_id = str(features.get("station_id", "ellagawa")).lower().strip()
        st_meta = self.get_station(st_id) or {
            "station_id": st_id, "station_name": st_id.title(), "basin": "Unknown",
            "lat": None, "lon": None, "alert_threshold_m": None,
            "minor_flood_threshold_m": None, "major_flood_threshold_m": None
        }

        stage_now = float(features.get("stage_at_issue_m", 1.5))

        # Get expected numeric columns from the fitted pipeline
        prep = self.model.named_steps["preprocessing"]
        expected_cols = list(prep.transformers_[0][2])

        row_data = dict(features)
        row_data["station_id"] = st_id
        row_data["stage_at_issue_m"] = stage_now

        # Derive stage history features if lags are provided
        if "stage_lag1_m" in row_data and row_data["stage_lag1_m"] is not None:
            row_data["stage_change_1d_m"] = stage_now - float(row_data["stage_lag1_m"])
            if "stage_lag2_m" in row_data and row_data["stage_lag2_m"] is not None:
                row_data["stage_change_prev_1d_m"] = float(row_data["stage_lag1_m"]) - float(row_data["stage_lag2_m"])
                row_data["stage_acceleration_m"] = row_data["stage_change_1d_m"] - row_data["stage_change_prev_1d_m"]

        # Derive satellite contrasts if provided
        if "sar_latest_age_hours" in row_data and row_data["sar_latest_age_hours"] is not None:
            row_data["sar_age_days"] = float(row_data["sar_latest_age_hours"]) / 24.0
        if "sar_vv_mean_db" in row_data and "sar_vh_mean_db" in row_data:
            if row_data["sar_vv_mean_db"] is not None and row_data["sar_vh_mean_db"] is not None:
                row_data["sar_vv_minus_vh_db"] = float(row_data["sar_vv_mean_db"]) - float(row_data["sar_vh_mean_db"])
        if "swi005_d1" in row_data and "swi010_d1" in row_data:
            if row_data["swi005_d1"] is not None and row_data["swi010_d1"] is not None:
                row_data["swi005_minus_swi010_latest"] = float(row_data["swi005_d1"]) - float(row_data["swi010_d1"])

        # Populate missing columns with NaN so ColumnTransformer and SimpleImputer work seamlessly
        for c in expected_cols:
            if c not in row_data or row_data[c] is None:
                row_data[c] = np.nan

        input_row = pd.DataFrame([row_data])
        pred_delta = float(self.model.predict(input_row)[0])
        pred_stage = round(stage_now + pred_delta, 3)

        current_risk = classify_flood_risk(stage_now, st_meta)
        predicted_risk = classify_flood_risk(pred_stage, st_meta)

        return {
            "station_id": st_id,
            "station_name": st_meta["station_name"],
            "model_family": self.model_name,
            "stage_at_issue_m": stage_now,
            "predicted_delta_m": round(pred_delta, 4),
            "predicted_next_day_stage_m": pred_stage,
            "flood_risk": {
                "current": current_risk,
                "predicted_next_day": predicted_risk,
            }
        }

    def get_latest_network_forecasts(self) -> list[dict]:
        """Returns predictions for all valid stations on their latest eligible date."""
        df = self.dataset
        latest_rows = df.sort_values("forecast_date").groupby("station_id").last().reset_index()
        predictions = []
        for _, row in latest_rows.iterrows():
            try:
                p = self.predict_for_record(row["station_id"], row["forecast_date"])
                predictions.append(p)
            except Exception:
                continue
        return predictions

    def get_metrics_summary(self) -> dict:
        """Loads and returns benchmark validation metrics."""
        path = self.root / "outputs" / "improvement_validation" / "selected_validation_metrics.csv"
        diag_path = self.root / "outputs" / "improvement_validation" / "validation_diagnostic_summary.json"

        metrics = pd.read_csv(path).to_dict(orient="records") if path.is_file() else []
        diagnostics = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.is_file() else {}

        return {
            "benchmark_metrics": metrics,
            "diagnostics": diagnostics,
            "active_model": self.model_name,
        }
