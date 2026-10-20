# C4 — Multimodal River-Stage Hindcasting: Implementation Blueprint

**FloodRisk Sri Lanka | Research component C4 | PP1 methodology**

## Objective and study dataset

In C4, I will investigate whether **combining satellite, weather and environmental information** improves next-day river-stage prediction compared with using fewer modalities. My emphasis is **multimodal breadth**, not reproducing C1–C3's specialized feature-engineering comparisons. I will construct C4's inputs and learn its representations independently; I will share only the common station identities, catchment definitions, observation targets and evaluation dates with the other components.

The shared DMC dataset covers **17 November–26 December 2025**: 40 geographic stations, 1,600 station-day slots and **1,429 valid next-day numerical stage pairs across 39 gauges**. The 40th station, Thambuththegama, has no labels in these reports. My fixed chronological evaluation uses **809 training pairs (17 Nov–9 Dec), 269 validation pairs (10–16 Dec) and 351 test pairs (17–25 Dec)**. These are correlated observations from a short period, not independent flood events.

## Forecast definition

My retrospective prediction origin is **09:00 Sri Lanka time on Day T** and the target is the observed stage at **09:00 on Day T+1**, a 24-hour observation-to-observation horizon. The DMC source's nominal public report is **09:30**, so immediate availability of the 09:00 reading is an **explicit hindcast assumption**, not an operational fact. I will not use observations occurring after the simulated 09:00 origin, but I will assume historical reanalysis products are available for this retrospective experiment without recreating their real publication delays.

I will train the model to predict stage change in metres, `ΔH = H(T+1) − H(T)`, then reconstruct `predicted stage(T+1) = H(T) + predicted ΔH`. The last neuron is **linear (one continuous number)**, rather than the original sigmoid flood-probability output. All seven models receive the same latest observed stage as their common persistence reference.

## Geography, sources and inputs

I preserve the **original EPSG:4326 gauge coordinates** and existing MERIT-derived candidate upstream polygons. I will use WGS84/UTM 44N (EPSG:32644) only where metre-based patch geometry or lengths are needed; I will not resample every product to the same artificial pixel grid. Catchment QA flags remain attached to the data.

| C4 branch | Historical inputs and extraction | Fixed model input | Learned representation |
|---|---|---|---|
| **Satellite** | Sentinel-1 VV/VH from the latest eligible acquisition **before T 09:00**, cropped to a fixed **12.8 × 12.8 km** gauge-centred region and resampled to **256 × 256 × 2**. Also retrieve **ASCAT SWI001/005/010** over the irregular upstream catchment at the product's coarse native grid, with valid fraction, flags, acquisition age and missingness. | SAR image + numeric SWI auxiliary vector | EfficientNet-B0 → global pooling → Dense(64); SWI MLP → 16; concatenate → Dense(**128**) |
| **Weather** | **ERA5-Land Hourly (`ECMWF/ERA5_LAND/HOURLY`)**, aggregated over each upstream polygon into 14 successive trailing 24-hour periods ending no later than the simulated origin. Sum `total_precipitation_hourly` (m → mm) and average `temperature_2m` (K → °C), `dewpoint_temperature_2m` (K → °C), `surface_pressure` (Pa → hPa). Use actual UTC hourly boundaries: 09:00 SLST = 03:30 UTC, so the latest hourly sample is no later than 03:00 UTC. | **14 × 4** | Small BiLSTM → **16** |
| **Environment** | Once per upstream polygon: log area; mean elevation and slope from **Copernicus DEM GLO-30**; tree, cropland, built-up and water fractions from **ESA WorldCover v200**; historical persistent-water fraction from **JRC GSW transition classes 1, 2 and 7**; river-network density from **HydroRIVERS**. Use original valid pixel areas / intersecting line lengths; JRC `seasonality = 12` is a diagnostic cross-check, not an additional temporal predictor. | **9 static descriptors** | Dense(32) → Dense(**16**) |

Each input is a **fixed-size array or vector despite different catchment shapes**. The 12.8 km SAR window is an initial controlled local context, **not an optimized catchment-wide satellite representation**. ASCAT and ERA5-Land remain spatially coarse; upsampling would not create new information. For SAR, image ages and repeat acquisitions are recorded and a maximum acceptable age will be chosen after observing real scene coverage. Missing source values remain flagged rather than fabricated. Static layers are pinned by source version and do not supply daily flood observations.

## Neural fusion and seven ablations

The full fusion combines **128 satellite + 16 weather + 16 environmental = 160 learned features**. A compact modality-aware fusion head (Dense(64) → dropout → Dense(32) → Linear(1)) estimates the next-day stage change. I will first freeze the EfficientNet-B0 backbone because the available training period is small.

| Configuration | Active branch(es) |
|---|---|
| M1 | Satellite |
| M2 | Weather |
| M3 | Environment |
| M4 | Satellite + Weather |
| M5 | Satellite + Environment |
| M6 | Weather + Environment |
| **M7** | **Satellite + Weather + Environment** |

The central comparison is **M7 versus M6**, which tests whether the satellite branch improves weather-plus-environment performance. I will compare models on **identical held-out station-days**, using the same data split, stage reference, missing-input policy, training controls, and MAE/RMSE in metres. I will compare against **persistence** (`Ĥ(T+1) = H(T)`). Static environmental features may mainly identify catchment differences; a gain is not guaranteed.

## Implementation and reproducibility

I will implement eight independent Jupyter notebooks inside `c4_fusion/notebooks/`:

1. **01 — Common contract:** validate 1,429 labels, private 40-station registry, 39 labelled polygons, splits and source integrity.
2. **05 — Environment:** derive and cache nine static features once per catchment.
3. **04 — Weather:** generate and cache ERA5-Land Hourly sequences for each station-day.
4. **06 — Preliminary alignment:** verify one real station-day loads as an environmental vector and a 14×4 weather tensor.
5. **02 and 03 — Satellite:** inventory/export unique SAR patches, retrieve native-grid SWI summaries and their QA metadata.
6. **06 — Full alignment:** join all modalities by `station_id + forecast_date`, report common coverage and missingness.
7. **07 — Training:** implement the three encoders and M1–M7 using one configurable neural-network notebook.
8. **08 — Evaluation:** compare the untouched chronological test records, persistence and seven ablations.

The preprocessing notebooks will read stable files, cache reusable outputs and record **asset IDs, acquisition dates, preprocessing parameters, checksums, QA flags and train-only normalization**. The supervised future labels will be stored separately from the sample index used for retrieving input data. Large arrays and private station coordinates will not be committed publicly. Jupyter notebooks are acceptable for the prototype; reusable loader functions can later move to `src/` without changing the scientific definitions.

## Boundaries of our claims

This is a **retrospective proof of concept**, not a validated operational flood-warning system. The current data cover a short correlated interval and one principal flood episode, reanalysis products were not necessarily available at the simulated origin, and several catchments retain spatial QA limitations. Operational adaptation would require timely gauge access, suitable live weather and satellite feeds, retraining/recalibration on archived operational inputs, and testing on multiple unseen events before any commercial use. I will discuss commercialization as a feasible next development stage, not a feature already demonstrated by PP1.
