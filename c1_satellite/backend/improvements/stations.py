"""Station metadata and flood threshold catalog for Sri Lanka DMC river gauges."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


DISTRICT_MAP = {
    "badalgama": "Gampaha",
    "baddegama": "Galle",
    "deraniyagala": "Kegalle",
    "dunamale": "Gampaha",
    "ellagawa": "Ratnapura",
    "galgamuwa": "Kurunegala",
    "giriulla": "Kurunegala",
    "glencourse": "Colombo",
    "hanwella": "Colombo",
    "holombuwa": "Kegalle",
    "horowpothana": "Anuradhapura",
    "kalawellawa_millakanda": "Kalutara",
    "katharagama": "Monaragala",
    "kithulgala": "Kegalle",
    "kuda_oya": "Monaragala",
    "magura": "Kalutara",
    "manampitiya": "Polonnaruwa",
    "moragaswewa": "Kurunegala",
    "moraketiya": "Ratnapura",
    "nagalagam_street": "Colombo",
    "nakkala": "Monaragala",
    "nawalapitiya": "Kandy",
    "norwood": "Nuwara Eliya",
    "padiyathalawa": "Ampara",
    "panadugama": "Matara",
    "peradeniya": "Kandy",
    "pitabeddara": "Matara",
    "putupaula": "Kalutara",
    "rathnapura": "Ratnapura",
    "siyambalanduwa": "Ampara",
    "thaldena": "Badulla",
    "thalgahagoda": "Matara",
    "thambuththegama": "Anuradhapura",
    "thanamalwila": "Monaragala",
    "thanthirimale": "Anuradhapura",
    "thawalama": "Galle",
    "urawa": "Matara",
    "wellawaya": "Monaragala",
    "weraganthota": "Kandy",
    "yaka_wewa": "Vavuniya",
}


def load_station_catalog(root: Path = ROOT) -> dict[str, dict]:
    """Extracts station profiles, coordinates, and Irrigation Dept flood thresholds."""
    master_path = root / "local_data" / "FloodRisk_DMC_StationDay_MASTER_1600.csv"
    if not master_path.is_file():
        raise FileNotFoundError(f"Missing master data: {master_path}")

    df = pd.read_csv(master_path, dtype={"station_id": str})
    cols = [
        "station_id", "station_name", "basin", "tributary",
        "gauge_lat_wgs84_arcgis", "gauge_lon_wgs84_arcgis",
        "alert_threshold_m", "minor_flood_threshold_m", "major_flood_threshold_m"
    ]
    sub = df[cols].drop_duplicates("station_id").dropna(subset=["station_id"])

    catalog = {}
    for _, row in sub.iterrows():
        st_id = str(row["station_id"]).strip()
        alert = float(row["alert_threshold_m"]) if pd.notna(row["alert_threshold_m"]) else None
        minor = float(row["minor_flood_threshold_m"]) if pd.notna(row["minor_flood_threshold_m"]) else None
        major = float(row["major_flood_threshold_m"]) if pd.notna(row["major_flood_threshold_m"]) else None
        district = DISTRICT_MAP.get(st_id, "Sri Lanka")

        catalog[st_id] = {
            "station_id": st_id,
            "station_name": str(row["station_name"]),
            "basin": str(row["basin"]),
            "district": district,
            "tributary": str(row["tributary"]) if pd.notna(row["tributary"]) else "",
            "lat": float(row["gauge_lat_wgs84_arcgis"]) if pd.notna(row["gauge_lat_wgs84_arcgis"]) else None,
            "lon": float(row["gauge_lon_wgs84_arcgis"]) if pd.notna(row["gauge_lon_wgs84_arcgis"]) else None,
            "alert_threshold_m": alert,
            "minor_flood_threshold_m": minor,
            "major_flood_threshold_m": major,
            "station_type": "DMC Gauging Station",
            "image_url": f"/static/assets/stations/{st_id}.jpg" if (root / "src" / "api" / "static" / "assets" / "stations" / f"{st_id}.jpg").is_file() else "/static/assets/stations/default.jpg",
        }
    return catalog


def classify_flood_risk(stage_m: float, thresholds: dict) -> dict:
    """Classifies a given water level into official Irrigation Dept alert levels."""
    if stage_m is None or not np.isfinite(stage_m):
        return {"status": "UNKNOWN", "severity": 0, "color": "#9ca3af", "message": "No stage data available"}

    alert = thresholds.get("alert_threshold_m")
    minor = thresholds.get("minor_flood_threshold_m")
    major = thresholds.get("major_flood_threshold_m")

    # If thresholds are missing, return unclassified normal
    if alert is None or minor is None or major is None:
        return {
            "status": "NORMAL",
            "severity": 0,
            "color": "#22c55e",
            "message": "Normal stage (thresholds unverified)",
            "margin_to_alert_m": None,
            "margin_to_minor_m": None,
            "margin_to_major_m": None
        }

    margin_alert = round(float(alert - stage_m), 3)
    margin_minor = round(float(minor - stage_m), 3)
    margin_major = round(float(major - stage_m), 3)

    if stage_m >= major:
        return {
            "status": "MAJOR_FLOOD",
            "severity": 3,
            "color": "#ef4444",
            "message": f"EXCEEDS MAJOR FLOOD LEVEL by {-margin_major:.2f} m!",
            "margin_to_alert_m": margin_alert,
            "margin_to_minor_m": margin_minor,
            "margin_to_major_m": margin_major
        }
    elif stage_m >= minor:
        return {
            "status": "MINOR_FLOOD",
            "severity": 2,
            "color": "#f97316",
            "message": f"Exceeds Minor Flood level by {-margin_minor:.2f} m",
            "margin_to_alert_m": margin_alert,
            "margin_to_minor_m": margin_minor,
            "margin_to_major_m": margin_major
        }
    elif stage_m >= alert:
        return {
            "status": "ALERT",
            "severity": 1,
            "color": "#eab308",
            "message": f"Water level in Alert range ({-margin_alert:.2f} m above alert)",
            "margin_to_alert_m": margin_alert,
            "margin_to_minor_m": margin_minor,
            "margin_to_major_m": margin_major
        }
    else:
        return {
            "status": "NORMAL",
            "severity": 0,
            "color": "#22c55e",
            "message": f"Normal water level ({margin_alert:.2f} m below alert level)",
            "margin_to_alert_m": margin_alert,
            "margin_to_minor_m": margin_minor,
            "margin_to_major_m": margin_major
        }
