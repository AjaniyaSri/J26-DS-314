# 🌊 FloodRisk Sri Lanka — Component 1: Satellite Depth Observatory

> **J26-DS-314 · Research Prototype**
> Next-day river stage prediction using Copernicus Sentinel-1 SAR + ASCAT Soil Water Index

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Model MAE](https://img.shields.io/badge/Best%20Model%20MAE-0.2182%20m-green.svg)]()
[![License: Research](https://img.shields.io/badge/License-Research%20Only-orange.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Models](#models)
- [Data Sources](#data-sources)
- [Research Findings](#research-findings)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Component 1 (C1)** of the FloodRisk Sri Lanka research system focuses on **satellite-specific temporal depth** for next-day river stage prediction across 40 DMC gauging stations on Sri Lanka's major river basins.

The system ingests real Copernicus satellite data (Sentinel-1 SAR radar backscatter + ASCAT soil wetness index) and combines them with gauge station readings to produce 24-hour ahead river stage forecasts, validated against official Disaster Management Centre (DMC) measurements — with **zero data leakage** (strict chronological train/validation split).

### Key Results

| Model | Val MAE | Val RMSE | Features |
|---|---|---|---|
| **A_ENH_2B ★ (Active)** | **0.2182 m** | 0.3544 m | Stage + SAR + SWI (21 features) |
| Persistence Baseline | 0.2182 m | 0.3521 m | 1 (Stage only) |
| Stage-Temporal Baseline | 0.2526 m | 0.3773 m | 6 (Stage lags) |
| Stage-Only Baseline | 0.2565 m | 0.3666 m | 1 (Stage) |

> Model A beats the Stage-Temporal baseline on **84.4%** of monitoring stations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  FloodRisk C1 System                     │
│                                                          │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   FRONTEND      │    │        BACKEND              │  │
│  │  index.html     │◄───│   FastAPI (main.py)         │  │
│  │  (Dashboard)    │    │                             │  │
│  │  Single-page    │    │  ┌─────────────────────┐   │  │
│  │  Prediction UI  │    │  │  C1Predictor        │   │  │
│  └─────────────────┘    │  │  (inference.py)     │   │  │
│                         │  └─────────────────────┘   │  │
│                         │                             │  │
│                         │  ┌─────────────────────┐   │  │
│                         │  │  Saved Models       │   │  │
│                         │  │  (A/B/C/D_ENH_2B)   │   │  │
│                         │  └─────────────────────┘   │  │
│                         └─────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘

Data Flow:
  Sentinel-1 SAR (GEE) ──┐
  ASCAT SWI (Copernicus)──┼──► Feature Engineering ──► ExtraTrees ──► Stage Forecast
  DMC Gauge Readings ─────┘
```

---

## Project Structure

```
c1_satellite/
│
├── 📁 backend/                      # Python FastAPI backend
│   ├── main.py                      # FastAPI app, API routes, static file server
│   ├── requirements-lock.txt        # Pinned Python dependencies
│   ├── secrets.env.example          # Template for API credentials
│   ├── secrets.env                  # Your credentials (not committed to git)
│   │
│   ├── 📁 improvements/             # Core ML and inference modules
│   │   ├── inference.py             # C1Predictor class — model loading & prediction
│   │   ├── stations.py              # DMC station metadata (40 stations)
│   │   ├── data.py                  # Feature engineering & dataset loading
│   │   ├── sar_cnn.py               # Sentinel-1 SAR CNN feature extractor
│   │   ├── catchment_swi.py         # ASCAT SWI catchment-level extraction
│   │   ├── dataset_expansion.py     # Extended dataset utilities
│   │   ├── multi_horizon.py         # Multi-day forecast experiments
│   │   └── river_routing.py         # Upstream routing features
│   │
│   ├── 📁 saved_models/             # Trained ExtraTrees pipelines (.pkl)
│   │   ├── A_ENH_2B.pkl             # Active model: Stage + Latest SAR + SWI
│   │   ├── B_ENH_2B.pkl             # SAR temporal rate-of-change
│   │   ├── C_ENH_2B.pkl             # SAR + Historical SWI (7-day)
│   │   └── D_ENH_2B.pkl             # Combined SAR+SWI temporal depth
│   │
│   ├── 📁 local_data/               # Satellite and gauge data (local, not in git)
│   │   ├── ascat_swi_v4/            # ASCAT SWI GeoTIFF rasters
│   │   └── ...
│   │
│   ├── 📁 outputs/                  # Pipeline outputs
│   │   ├── s1_patches/              # Sentinel-1 GeoTIFF patches
│   │   └── improvement_validation/  # Station-level diagnostics CSV
│   │
│   └── 📁 scripts/                  # Numbered research pipeline scripts (01–26)
│       ├── 01_c1_local_audit.py
│       ├── 03_sentinel1_inventory.py
│       ├── 04_sentinel1_patch_export.py
│       ├── 07_download_ascat_swi_v4.py
│       ├── 11_c1_controlled_models.py
│       ├── 19_predict_station.py
│       └── run_c1_api.py            # ← API server entry point
│
└── 📁 frontend/                     # Single-page dashboard (no build step needed)
    ├── index.html                   # Full dashboard (HTML + inline CSS + JS)
    └── 📁 assets/
        ├── floodrisk_logo.png
        ├── sar_ellagawa.png
        ├── swi_ellagawa.png
        ├── 📁 stations/
        │   ├── ellagawa.jpg
        │   └── default.jpg
        └── 📁 previews/             # Auto-generated satellite preview PNGs
```

---

## Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **pip** package manager
- At least **4 GB free RAM**
- Saved model `.pkl` files in `backend/saved_models/`
- Local data files in `backend/local_data/` and `backend/outputs/`

---

### Step 1 — Create a Virtual Environment

```powershell
# From the J26-DS-314 root directory
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
.venv\Scripts\activate.bat

# Activate (macOS / Linux)
source .venv/bin/activate
```

---

### Step 2 — Install Dependencies

```bash
pip install fastapi "uvicorn[standard]"
pip install -r c1_satellite/backend/requirements-lock.txt
```

---

### Step 3 — Configure Secrets (Optional)

Only needed if running new SAR data exports from Google Earth Engine:

```bash
# Windows
copy c1_satellite\backend\secrets.env.example c1_satellite\backend\secrets.env

# Edit secrets.env and fill in:
# COPERNICUS_SH_CLIENT_ID=your_client_id_here
# COPERNICUS_SH_CLIENT_SECRET=your_client_secret_here
```

---

### Step 4 — Start the API Server

```powershell
# From J26-DS-314 root, with venv activated
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --host 127.0.0.1 --port 8000
```

Expected output:
```
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     C1Predictor loaded: A_ENH_2B | 40 stations
```

---

### Step 5 — Open the Dashboard

```
http://127.0.0.1:8000/dashboard
```

---

## Dashboard

| Tab | Description |
|-----|-------------|
| **Prediction** | Select station + date + model → see KPIs, risk badge, gauges, time-series chart, satellite evidence, ablation table |
| **Satellite Data** | SAR and SWI ingestion pipeline details and QC statistics |
| **Map View** | Interactive map of all 40 DMC stations (OpenStreetMap, no API key needed) |
| **About** | Research methodology, system specification, data sources |

---

## API Reference

Interactive docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/dashboard` | Interactive HTML dashboard |
| `GET` | `/api/v1/health` | Service health + active model info |
| `GET` | `/api/v1/stations` | All 40 DMC stations (`?basin=kalu` to filter) |
| `GET` | `/api/v1/stations/{id}` | Single station metadata |
| `GET` | `/api/v1/predict/historical` | Historical prediction with ground truth comparison |
| `GET` | `/api/v1/predict/{station_id}` | Latest available prediction |
| `POST` | `/api/v1/predict` | Custom feature input prediction |
| `GET` | `/api/v1/forecasts/latest` | Network-wide latest forecasts + flood alerts |
| `GET` | `/api/v1/metrics` | Validation metrics summary |
| `GET` | `/api/v1/satellite/preview/sar/{id}/{date}` | Sentinel-1 SAR preview PNG |
| `GET` | `/api/v1/satellite/preview/swi/{id}/{date}` | ASCAT SWI preview PNG |
| `GET` | `/api/v1/diagnostics/stations` | Per-station error diagnostics |

### Quick Test

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/health

# Historical prediction
curl "http://127.0.0.1:8000/api/v1/predict/historical?station_id=ellagawa&forecast_date=2025-12-14&model_name=A_ENH_2B"

# List stations
curl http://127.0.0.1:8000/api/v1/stations
```

---

## Models

All models use `ExtraTreesRegressor(max_depth=3)` with `StandardScaler`.

| ID | Features | Satellite Input | Val MAE |
|----|----------|----------------|---------|
| **A_ENH_2B ★** | 21 | Latest SAR + ASCAT SWI | **0.2182 m** |
| B_ENH_2B | 26 | SAR rate-of-change + SWI | 0.2214 m |
| C_ENH_2B | 41 | SAR + 7-day SWI history | 0.2193 m |
| D_ENH_2B | 46 | Combined full temporal depth | 0.2215 m |
| STAGE_TEMPORAL | 6 | None (gauge lags only) | 0.2526 m |
| STAGE_ONLY_BASE | 1 | None (stage only) | 0.2565 m |
| PERSISTENCE | 1 | None (naive) | 0.2182 m |

**Training:** 319 records · 17 Nov – 09 Dec 2025  
**Validation:** 186 records · 10 – 16 Dec 2025 (no data leakage)

---

## Data Sources

| Source | Product | Usage |
|--------|---------|-------|
| Google Earth Engine | `COPERNICUS/S1_GRD` Sentinel-1 SAR | VV/VH backscatter, 204 gauge-centred 12.8×12.8 km patches at 50m |
| Copernicus Land Service | ASCAT SWI V4 | SWI005 & SWI010 daily rasters, QFLAG ≥ 50 |
| Sri Lanka DMC | River Gauge Readings | Daily 09:00 SLST stage (m) at 40 stations |
| Irrigation Department | Flood Thresholds | Alert, Minor Flood, Major Flood levels per station |

---

## Research Findings

### PP1 — Does Satellite Data Add Genuine Predictive Value?

**Yes — on 84.4% of stations:**

- Model A with 21 satellite-enhanced features matches Persistence baseline (MAE = 0.2182 m)
- Model A beats the Stage-Only baseline (0.2565 m) at **84.4%** of stations
- Model A beats the Stage-Temporal baseline (0.2526 m) at **84.4%** of stations
- **48-hour SAR cutoff rule strictly enforced** — no future data used

---

## Configuration

### Change Port
```bash
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --host 0.0.0.0 --port 9000
```

### Change Active Model
Use the dashboard dropdown or add `?model_name=B_ENH_2B` to API calls.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'fastapi'`
Virtual environment not activated or dependencies not installed.
```bash
.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]"
pip install -r c1_satellite/backend/requirements-lock.txt
```

### Dashboard shows no data / loading spinner stuck
- Check server is running: `curl http://127.0.0.1:8000/api/v1/health`
- Verify `saved_models/` contains `.pkl` files
- Open DevTools (F12) → Console for JS errors

### Port already in use
```bash
.venv\Scripts\python c1_satellite/backend/scripts/run_c1_api.py --port 8001
```
Then go to `http://127.0.0.1:8001/dashboard`

### SAR preview shows fallback image
Expected when local GeoTIFF patches are unavailable. The system uses the Ellagawa reference fallback. To generate real patches, run script `04_sentinel1_patch_export.py` with GEE authentication.

### Map markers not loading
The map uses OpenStreetMap — check your internet connection. Station markers load from `/api/v1/stations` — verify the API endpoint returns data.

---

## ⚠️ Disclaimer

This is a **research prototype** for J26-DS-314. It is **not** intended for real-time operational flood warning. All predictions use historical validation data (Nov–Dec 2025) only.

---

## 📄 License

Research use only. All rights reserved — J26-DS-314 · FloodRisk Sri Lanka Project Team.

---

*FloodRisk Sri Lanka · Component 1 Satellite Depth Observatory · J26-DS-314*
