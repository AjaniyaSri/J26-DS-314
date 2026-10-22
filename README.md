# FloodRisk Sri Lanka 🌊

**J26-DS-314 · Final-Year Research Project · PP1 research prototype**

Research on **next-day (09:00) river-stage prediction** at Sri Lankan monitoring stations, comparing satellite, weather, environmental and multimodal information.

> **Status:** Retrospective hindcast research prototype. **Not** an operational flood-warning system. The output is a predicted gauge reading in metres, not an inundation map or validated emergency alert.

## Research objective

For each station with an observed 09:00 river level on day D, estimate the river level at 09:00 on day D+1:

```text
predicted_change_m = model(eligible historical features)
predicted_next_day_stage_m = stage_at_issue_m + predicted_change_m
```

The baseline is **persistence** (predict no stage change).

## Four research components

| Component | Focus | Repository status |
| --- | --- | --- |
| **C1 — Satellite** | Sentinel-1 SAR observations, ASCAT Soil Water Index, single/multiple date comparison | Application source prepared for this branch; local datasets and models excluded |
| **C2 — Weather** | Raw, engineered and hybrid CHIRPS/ERA5 histories | Team workspace; no implementation included here yet |
| **C3 — Environment** | Static versus dynamic catchment factors | Team workspace; no implementation included here yet |
| **C4 — Fusion** | Independent S/W/E feature branches and seven ablations | Team workspace and experiment plan; not an operational integrated system |

## Shared dataset and evaluation

The team contract covers **40 research gauges × 40 dates** (17 Nov–26 Dec 2025), with **1,600 station-day rows** and **1,429 verified consecutive-day numerical labels** from **39 gauges**. These are **not** 1,429 feature-complete multimodal examples.

| Split | Forecast dates | Label pairs |
| --- | --- | ---: |
| TRAIN | 17 Nov–09 Dec 2025 | 809 |
| VALIDATION | 10–16 Dec 2025 | 269 |
| TEST | 17–25 Dec 2025 | 351 |

Stage-only persistence on those **351 TEST rows**: **MAE 0.2172 m, RMSE 0.4066 m**. This is *not* a trained ML-model result. Sensor models must be compared fairly on matched eligible rows. The short observation window does not demonstrate generalization to unseen major floods.

See [the research contract](docs/RESEARCH_CONTRACT.md) and the team's shared original data guide for definitions and limitations.

## C1 source and local demonstration

The sanitized source-code package for C1 contains:

- `c1_satellite/backend/main.py` — FastAPI application
- `c1_satellite/backend/improvements/` — feature and inference utilities
- `c1_satellite/backend/scripts/` — ingestion, model studies and API runner
- `c1_satellite/backend/tests/` — tests
- `c1_satellite/frontend/` — dashboard interface

Once the C1 source is added to this branch, run from the repository root with Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r c1_satellite/backend/requirements-lock.txt
python -m pip install -r c1_satellite/backend/requirements-api.txt
python c1_satellite/backend/scripts/run_c1_api.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard` or `http://127.0.0.1:8000/docs`. Inference **requires approved local data, extracted features and serialized models**, intentionally omitted from public source control.

## Data policy and contributing

The repository is public. **Never commit secrets, raw gauge observations, downloaded satellite products, private geospatial assets or trained model binaries** before checking redistribution rights. See [Data Policy](docs/DATA_POLICY.md) and [Contributing](CONTRIBUTING.md). Open feature branches and pull requests; keep the shared station/date keys and chronological split unchanged.

**Group:** J26-DS-314 · License pending team confirmation (no open-source license is implied).
