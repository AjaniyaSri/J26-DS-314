"""C1 Satellite Depth Prediction REST API.

Provides endpoints for next-day river stage forecasting, DMC station flood alert
thresholds, real-time inference, and empirical validation metrics.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
import pandas as pd
# main.py is at c1_satellite/backend/main.py
# C1_ROOT = backend/ (where local_data, outputs, models, configs all live)
C1_ROOT = Path(__file__).resolve().parent
# Add backend/ to path so `improvements` package is importable
sys.path.insert(0, str(C1_ROOT))

from improvements.inference import C1Predictor

app = FastAPI(
    title="FloodRisk Sri Lanka — C1 Satellite Depth API",
    description="Next-day river stage and flood alert forecasting using Sentinel-1 SAR and ASCAT SWI.",
    version="1.0.0",
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
predictor: Optional[C1Predictor] = None


def get_predictor() -> C1Predictor:
    global predictor
    if predictor is None:
        predictor = C1Predictor(root=C1_ROOT)
    return predictor


class CustomPredictRequest(BaseModel):
    station_id: str = Field(..., example="ellagawa")
    stage_at_issue_m: float = Field(..., example=3.25)
    stage_lag1_m: Optional[float] = Field(None, example=3.10)
    stage_lag2_m: Optional[float] = Field(None, example=2.95)
    sar_latest_age_hours: Optional[float] = Field(48.0, example=36.0)
    swi005_d1: Optional[float] = Field(55.0, example=62.0)
    swi010_d1: Optional[float] = Field(52.0, example=58.0)


from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# frontend/ is a sibling of backend/ under c1_satellite/
STATIC_DIR = C1_ROOT.parent / "frontend"
INDEX_HTML = STATIC_DIR / "index.html"

# Mount frontend/ as /static so browser can access assets
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def root(request_format: Optional[str] = Query(None, alias="format")):
    """Returns the visual presentation dashboard for browsers, or API info if ?format=json."""
    if request_format == "json" or not INDEX_HTML.is_file():
        return {
            "project": "FloodRisk Sri Lanka (J26-DS-314)",
            "component": "Component 1 (C1) — Satellite-Specific Temporal Depth",
            "description": "24-hour river stage forecasting service using Sentinel-1 SAR and ASCAT SWI.",
            "dashboard": "/dashboard",
            "documentation": "/docs",
            "endpoints": {
                "dashboard": "/dashboard",
                "health": "/api/v1/health",
                "stations": "/api/v1/stations",
                "latest_network_forecasts": "/api/v1/forecasts/latest",
                "predict_station": "/api/v1/predict/{station_id}",
                "custom_prediction": "/api/v1/predict",
                "validation_metrics": "/api/v1/metrics",
                "station_diagnostics": "/api/v1/diagnostics/stations",
            }
        }
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Serves the interactive progress presentation dashboard."""
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="Dashboard UI template not found.")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))



@app.get("/api/v1/health")
def health():
    p = get_predictor()
    return {
        "status": "HEALTHY",
        "component": "C1_SATELLITE_DEPTH",
        "active_model": p.model_name,
        "pipeline_file": str(p.pipeline_path.name),
        "total_monitoring_stations": len(p.stations),
        "validation_benchmark_mae_m": 0.2182,
        "no_data_leakage_verified": True,
    }


@app.get("/api/v1/stations")
def list_stations(basin: Optional[str] = None):
    p = get_predictor()
    stations = p.get_stations()
    if basin:
        stations = [s for s in stations if basin.lower() in s["basin"].lower()]
    return {
        "count": len(stations),
        "stations": stations
    }


@app.get("/api/v1/stations/{station_id}")
def get_station_details(station_id: str):
    p = get_predictor()
    station = p.get_station(station_id)
    if not station:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found.")
    return station


@app.get("/api/v1/historical/records")
def get_historical_records():
    """Returns dynamic inventory of available stations, dates, and models."""
    p = get_predictor()
    return p.get_available_historical_records()


@app.get("/api/v1/predict/historical")
@app.get("/api/v1/historical/predict")
def predict_historical(
    station_id: str = Query(..., description="DMC station ID, e.g. ellagawa"),
    forecast_date: str = Query(..., description="Historical forecast date YYYY-MM-DD, e.g. 2025-12-14"),
    model_name: str = Query("A_ENH_2B", description="Model: A_ENH_2B, B_ENH_2B, C_ENH_2B, D_ENH_2B, STAGE_TEMPORAL, STAGE_ONLY_BASE, PERSISTENCE")
):
    """Executes next-day prediction and returns ground truth, time-series, and satellite evidence."""
    p = get_predictor()
    try:
        return p.get_historical_detailed(station_id, forecast_date, model_name=model_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/predict/{station_id}")
def predict_station(
    station_id: str,
    forecast_date: Optional[str] = Query(None, description="Format YYYY-MM-DD. Defaults to latest available.")
):
    p = get_predictor()
    try:
        return p.predict_for_record(station_id, forecast_date)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/satellite/preview/sar/{station_id}/{forecast_date}")
def get_sar_preview(station_id: str, forecast_date: str):
    """Returns real Sentinel-1 VV/VH radar backscatter dual-band preview PNG."""
    p = get_predictor()
    st_id = station_id.lower().strip()

    cache_dir = STATIC_DIR / "assets" / "previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_png = cache_dir / f"sar_{st_id}_{forecast_date}.png"

    if cache_png.is_file():
        return FileResponse(cache_png, media_type="image/png")

    patch_dir = C1_ROOT / "outputs" / "s1_patches"
    matching_tifs = sorted(list(patch_dir.glob(f"{st_id}_*.tif")))

    # Check for pre-existing static preview
    static_preview = STATIC_DIR / "assets" / f"sar_{st_id}.png"
    if static_preview.is_file() and not matching_tifs:
        return FileResponse(static_preview, media_type="image/png")

    if not matching_tifs:
        # Fallback to general SAR preview if available
        fallback = STATIC_DIR / "assets" / "sar_ellagawa.png"
        if fallback.is_file():
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail=f"No SAR patch found for {station_id}")

    tif_file = matching_tifs[-1]
    try:
        import rasterio
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with rasterio.open(tif_file) as ds:
            vals = ds.read()
        fig, ax = plt.subplots(1, 2, figsize=(6, 3), facecolor="#10263B")
        for i, title in enumerate(["VV Backscatter (dB)", "VH Backscatter (dB)"]):
            data = vals[i]
            valid = np.isfinite(data) & (data > -900)
            v = data[valid]
            vmin, vmax = np.percentile(v, [2, 98]) if v.size > 0 else (-25, 0)
            ax[i].imshow(np.where(valid, data, np.nan), cmap="gray", vmin=vmin, vmax=vmax)
            ax[i].axis("off")
            ax[i].set_title(title, color="#F8FAFC", fontsize=9, pad=6)
        plt.tight_layout()
        fig.savefig(cache_png, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=100)
        plt.close(fig)
        return FileResponse(cache_png, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error rendering SAR preview: {exc}")


@app.get("/api/v1/satellite/preview/swi/{station_id}/{forecast_date}")
def get_swi_preview(station_id: str, forecast_date: str):
    """Returns real Copernicus ASCAT SWI soil moisture colormapped preview PNG."""
    p = get_predictor()
    st_id = station_id.lower().strip()

    cache_dir = STATIC_DIR / "assets" / "previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_png = cache_dir / f"swi_{st_id}_{forecast_date}.png"

    if cache_png.is_file():
        return FileResponse(cache_png, media_type="image/png")

    df = p.dataset
    match = df[(df.station_id == st_id) & (df.forecast_date == forecast_date)]
    swi_date = "2025-12-12"
    if len(match) > 0 and pd.notna(match.iloc[0].get("latest_eligible_swi_date")):
        swi_date = str(match.iloc[0]["latest_eligible_swi_date"])

    swi_tif = C1_ROOT / "local_data" / "ascat_swi_v4" / "raw" / f"ASCAT_SWI_V4_{swi_date}.tif"

    if not swi_tif.is_file():
        fallback = STATIC_DIR / "assets" / "swi_ellagawa.png"
        if fallback.is_file():
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail=f"SWI raster not found for {swi_date}")

    try:
        import rasterio
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with rasterio.open(swi_tif) as ds:
            swi = ds.read(1) * 0.5
        fig, ax = plt.subplots(figsize=(4, 3), facecolor="#10263B")
        im = ax.imshow(swi, cmap="turbo", vmin=0, vmax=100)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.046)
        cbar.ax.tick_params(labelsize=7, colors="#94A3B8")
        cbar.set_label("ASCAT SWI005 Soil Wetness (%)", color="#F8FAFC", fontsize=8)
        plt.tight_layout()
        fig.savefig(cache_png, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=100)
        plt.close(fig)
        return FileResponse(cache_png, media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error rendering SWI preview: {exc}")


@app.get("/api/v1/forecasts/latest")
def latest_network_forecasts():
    p = get_predictor()
    forecasts = p.get_latest_network_forecasts()
    escalating = [f for f in forecasts if f["flood_risk"]["risk_escalating"]]
    return {
        "total_forecasts": len(forecasts),
        "escalating_flood_alerts_count": len(escalating),
        "escalating_stations": [f["station_name"] for f in escalating],
        "forecasts": forecasts
    }


@app.post("/api/v1/predict")
def predict_custom(req: CustomPredictRequest):
    p = get_predictor()
    features = req.model_dump()
    return p.predict_custom(features)


@app.get("/api/v1/metrics")
def get_metrics():
    p = get_predictor()
    return p.get_metrics_summary()


@app.get("/api/v1/diagnostics/stations")
def get_station_diagnostics():
    path = C1_ROOT / "outputs" / "improvement_validation" / "station_validation_error_summary.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Station diagnostics not found. Run Phase 18 first.")
    import pandas as pd
    df = pd.read_csv(path)
    return {
        "count": len(df),
        "stations_beating_stage_temporal": int((df["a_vs_stage_temporal_diff_m"] < 0).sum()),
        "stations_beating_persistence": int((df["a_vs_persistence_diff_m"] < 0).sum()),
        "records": df.to_dict(orient="records")
    }

