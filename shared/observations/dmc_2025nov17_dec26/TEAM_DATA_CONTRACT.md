# FloodRisk Sri Lanka — Shared Prediction & Data Guide
**C1–C4 team agreement | PP1 retrospective prototype | 18 October 2026**

**READ THIS FIRST:** We use one common set of observed gauge water levels, dates, station IDs, catchment references and evaluation splits. Each component independently adds its own modality data. **The supplied labels are real; satellite, weather and environmental features are not yet included.**

## 1. Exactly what we predict

**Goal:** For each station and report date, predict the **next calendar day's 09:00 river-stage reading in metres**, using only eligible information from the current day or earlier. This is a numerical regression target, **not** a flood/no-flood label, flood map or water depth across land.

**Nominal forecast setup:** Current observation at **09:00** on day D; nominal DMC report/forecast issue at **09:30** on day D; target observation at **09:00** on day D+1. Thus the readings are **24 hours apart**, but the nominal issue-to-target lead is **23.5 hours**. The actual historical publication times of the reports are **not independently verified** by the spreadsheet.

**Preferred model target (same for all four components):**

`true_change_m = next_day_stage_m − current_stage_m`

`predicted_next_day_stage_m = current_stage_m + predicted_change_m`

The final output remains a **decimal river stage in metres**. All models also compare against **persistence**, which predicts no change: `predicted_next_day_stage_m = current_stage_m`. Because gauge reference zeros differ, do not compare absolute stage numbers between stations as though they share a common height datum.

## 2. What the shared ZIP actually contains

| Verified measure | Actual count / detail |
| --- | --- |
| Canonical station-date index | **1,600 rows** = 40 research gauges × 40 dates |
| Research dates | **17 Nov–26 Dec 2025** |
| Gauges present in the DMC numerical workbook | **39 of 40**; Thambuththegama absent |
| Original DMC station-date rows | **1,558** |
| Valid current-day 09:00 numerical readings | **1,489** |
| Actual same-station, consecutive-day numerical targets | **1,429** |
| Non-numerical stage entries | **69**; retained as missing, not zero |
| Missing source station-date rows | Galgamuwa (24 Nov); Giriulla (16 Dec) |

**Unit rule:** Nagalagam Street is given in **feet** in the source and is converted to metres (`feet × 0.3048`) in the standard target columns. Other listed gauges are in metres. Original numbers, original units, negative stage readings, source worksheet and Excel row numbers are retained. Negative readings are not automatically errors.

**Use these files from `FloodRisk_DMC_StationDay_TEAM_SHARE.zip`:**

| File / worksheet | Purpose |
| --- | --- |
| `FloodRisk_DMC_StationDay_1600_VERIFIED.xlsx` → `Station_Day_1600` | Complete station-day master, including blanks and QA flags |
| Workbook → `Labelled_Pairs_1429` | Actual usable current/next-day numeric labels |
| `FloodRisk_DMC_StationDay_LABELLED_1429.csv` | Primary supervised-label join table for C1–C4 |
| `FloodRisk_DMC_StationDay_MASTER_1600.csv` | Preserve all dates/stations and missingness; coverage audit |
| `FloodRisk_DMC_SourceObservations_1558.csv` | Source-level traceability; original rows and values |
| `FloodRisk_DMC_StationCoverage_40.csv` | Station coverage summary |
| `FloodRisk_DMC_StationDay_README.md` | Source methodology, exact field explanations and caveats |

**Important:** The 1,429 rows are valid **water-level labels**, not 1,429 feature-complete multimodal samples. Modality coverage must be measured after data extraction. Do not invent readings or replace missing stages with zero.

## 3. Common station-day contract for all components

**Join key:** `station_id` + `forecast_date` (one row per station and forecast date). Keep all common identifiers unchanged. Add features in separate component-owned tables keyed by these same two fields, rather than editing or re-extracting the targets separately.

| Exact shared field(s) in labelled CSV | Use |
| --- | --- |
| `record_id`, `station_id`, `station_name`, `basin`, `forecast_date` | Unique record and station/date identity |
| `stage_at_issue_m` | Current observed 09:00 stage; common baseline for all models |
| `target_stage_next_day_m` | **Future target**: do not use as an input |
| `target_delta_stage_m` | **Training target**: next-day stage minus current stage; never an input |
| `chronological_split` | Fixed `TRAIN` / `VALIDATION` / `TEST` assignment |
| `stage_label_valid`, `pair_quality_status` | Whether current and next-day numerical stages are valid |
| `issue_time_assumed_local`, `stage_0900_obs_time_local`, `target_0900_obs_time_local` | Observation/nominal issuance timing |
| `gauge_lon_wgs84_arcgis`, `gauge_lat_wgs84_arcgis` | Original dashboard gauge location (not the shifted MERIT outlet) |
| `catchment_polygon_path`, `catchment_qa_status` | Shared provisional upstream AOI; carry its uncertainty |
| `current_source_excel_row`, `next_day_source_excel_row`, `source_unit` | Audit provenance and original measurement unit |
| `rain_24h_at_0830_mm` | Optional *station-level* DMC rainfall; not equivalent to catchment-average rainfall |
| `sar_feature_ref`, `swi_feature_ref`, `weather_feature_ref`, `environment_feature_ref` | Empty placeholders for independently extracted features |

**Shared scientific definition:** Use the same 40-station registry, provisional MERIT catchment polygons and QA flags, source target values, time fields, chronological splits and evaluation protocol. A flagged catchment is not a verified official watershed boundary.

## 4. What each component must add (independently)

| Component | Keep the research focus | Add per station-day record | Output and PP1 experiment |
| --- | --- | --- | --- |
| **C1 — Satellite depth** | Compare satellite representations, including single-date vs richer/multi-date inputs where feasible | Eligible pre-cutoff Sentinel-1 VV/VH scenes or patches, acquisition time, age, availability/quality flags; satellite soil-wetness indicators where used | Predict next-day stage **change** / stage. Report sensor coverage and comparison between satellite representations; no flood-mask sigmoid target |
| **C2 — Weather depth** | Compare **raw / engineered / hybrid** meteorological representations | Catchment-aggregated CHIRPS/ERA5 past histories, rainfall accumulations, antecedent wetness and time-availability flags | Test appropriate tabular/temporal regressors (XGBoost first; LSTM only if time-series coverage supports it), with the same target/split |
| **C3 — Environmental depth** | Compare **static vs dynamic** environmental information | Catchment elevation, slope, drainage, land cover, soil/terrain features, plus explicitly documented dynamic conditions if used | Gauge-level regression. Note that static geography alone cannot represent daily river-stage changes without changing conditions and/or current stage |
| **C4 — Multimodal breadth** | Keep **three independently constructed branches** and **seven ablations** | Its own pre-cutoff Sentinel-1 + ASCAT SWI, catchment weather histories and catchment environmental vectors; age, quality and missing indicators | Compact intermediate fusion → **linear regression output**, predicting stage change; reconstruct numerical stage using today's stage |

**C4 independence rule:** We may share raw sources, polygons, target labels, dates, station IDs, quality definitions and split files. C4 does **not** reuse trained C1–C3 weights, embeddings, fitted scalers or predictions as its fusion inputs.

### C4's seven ablation models

| Model | Satellite (S) | Weather (W) | Environment (E) | Question |
| --- | :---: | :---: | :---: | --- |
| M1 | Yes | — | — | What can the satellite branch add? |
| M2 | — | Yes | — | What can weather alone add? |
| M3 | — | — | Yes | What can environment alone add? |
| M4 | Yes | Yes | — | Does S improve W? |
| M5 | Yes | — | Yes | Does S improve E? |
| M6 | — | Yes | Yes | Weather + environment baseline |
| **M7** | **Yes** | **Yes** | **Yes** | **Does satellite improve W+E? Compare M7 vs M6** |

**Fairness:** Today's verified stage is the **same common reference** for M1–M7; each model predicts a correction to the persistence estimate. Evaluate ablations on the **same held-out rows** when comparing scores; if sensor coverage forces different samples, report their separate counts rather than making an unfair direct comparison. Each branch must be inactive when it is not in that model.

## 5. Our fixed train / validation / test split — use exactly this

| Split | Forecast dates (2025) | Actual labelled pairs | Used for |
| --- | --- | ---: | --- |
| **TRAIN** | 17 Nov – 09 Dec | **809** | Fit model parameters and preprocessing |
| **VALIDATION** | 10 Dec – 16 Dec | **269** | Tune models and early stopping |
| **TEST** | 17 Dec – 25 Dec | **351** | Final evaluation after choices are fixed |
| 26 Dec | No in-window next-day target | **0** | Retain in 1,600-row master only |

**No random row split.** All stations use the same chronological date boundaries. Fit scalers, imputers and encoders on TRAIN only; use VALIDATION for model selection. Never train with the TEST labels or features computed using future observations. Verify that each feature existed by the forecast cutoff, or label the experiment as retrospective hindcasting.

**Shared evaluation:** Report MAE and RMSE of the **reconstructed next-day stage in metres**, per-station error when possible, number of evaluated rows and improvement relative to persistence. The verified stage-only **TEST persistence benchmark** is **MAE 0.2172 m, RMSE 0.4066 m**; this is not a trained ML model result. The 351-row test set has **no rows marked `Major Flood`** in the audited source-day remarks, so it cannot validate generalization to major floods.

## 6. Timing, leakage and missing data — non-negotiable

1. **No future target as input.** Never feed `target_stage_next_day_m` or `target_delta_stage_m` into feature columns, including inside preprocessing transforms.
2. **Observation time is not publication time.** Final ERA5 and CHIRPS archives, Sentinel-1 scenes or ASCAT SWI may have become available after the nominal DMC issuance. Log acquisition/observation time, product version, temporal age and assumed availability lag. If true historical availability is unverified, call PP1 a **retrospective hindcast prototype**, not an operational 09:30 forecast service.
3. **Never fabricate a missing modality or label.** A station-day can have a valid stage target but no qualifying SAR scene. Preserve it in the master; report actual sample coverage for each experiment. If you use imputation/missing masks, fit only on TRAIN and document the approach.
4. **Keep raw-source meaning.** The report's `rain_24h_at_0830_mm` is a gauge-level reported rainfall reading, not automatically the same as catchment-mean CHIRPS/ERA5 rainfall. Flood-alert remarks are reported source context, not independent proof of forecast quality.
5. **Keep spatial QA flags.** MERIT catchments are provisional. Compare high-confidence and all-candidate subsets later, especially for flagged stations.
6. **No claims of operational validation.** Forty dates cover a short, correlated period largely associated with one event; 1,429 labelled pairs are **not 1,429 independent flood events**. Long historical gauge series and independent floods are still needed.

## 7. What we should do now — shortest shared workflow

1. **Everyone:** Download the team ZIP; open `FloodRisk_DMC_StationDay_1600_VERIFIED.xlsx`. Read `Read_Me` and `Field_Dictionary`; do not edit the original shared label values.
2. **Everyone:** Use `FloodRisk_DMC_StationDay_LABELLED_1429.csv` as the supervised label table; join by `station_id` + `forecast_date`. Keep the 1,600-row master for missingness and completeness reporting.
3. **C1 / C2 / C3:** Create one component-specific feature table with the same join keys, actual timestamps, QA and source version. Train/test under the fixed split.
4. **C4:** Build its **own** S/W/E features; train persistence/XGBoost first, then M1–M7 as coverage allows. Share the **number of feature-complete samples per model** before reporting ablation results.
5. **Everyone:** Return test predictions and metrics keyed by `record_id`, including `model_id`, `predicted_next_day_stage_m`, true next-day stage, `split`, feature-availability status and experiment version. Do not merge datasets by row order.

**For PP1, our claim is:** We built a traceable **station-day water-level prediction prototype** with verified targets and an honest chronological evaluation. We are **not** claiming independently validated real-time flood warnings, verified inundation maps, or generalization to unseen major flood events.

**Data handling:** Share the source-derived measurements with the four-person project team only as permitted. Check the original source terms before placing numerical observations or bundled source data in a public GitHub repository. Code, schemas, methodology and aggregate counts can be version-controlled separately.