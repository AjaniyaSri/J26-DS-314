# FloodRisk Sri Lanka — DMC daily gauge stage contract (PP1 pilot)

**Source workbook:** `2025-11-17 to 2025-12-26 DMC water level reports.xlsx`  
**Source SHA-256:** `f103816f6d6a7758496cd2c9972c7ae81cb9ecda944ef556a34e7536049bc93e`  
**Source worksheet:** `Sheet1`, Excel rows 3–1560 (source row numbers retained per observation).  
**Coverage:** 17 November–26 December 2025 (Sri Lanka local time, 40 calendar dates).

## Files and intended use

| File | Rows | Meaning |
| --- | ---: | --- |
| `FloodRisk_DMC_StationDay_1600_VERIFIED.xlsx` → `Station_Day_1600` | 1,600 | Canonical 40-gauge × 40-day *complete index*, with measured data and explicit missingness. |
| Workbook → `Labelled_Pairs_1429` | 1,429 | Source-verified current-next-day numerical stage pairs; eligible for **stage-only/persistence** baselines. |
| Workbook → `Source_Observations_1558` | 1,558 | All raw station-date source rows plus source Excel row numbers. |
| Workbook → `Station_Audit_40` | 40 | Station-level source coverage, missing dates and split counts. |
| Workbook → `Date_Audit_40` | 40 | Date-level source/label coverage and reported warning statuses. |
| Workbook → `Field_Dictionary`, `Read_Me`, `Split_Protocol` | — | Column semantics, provenance and evaluation rules. |
| `FloodRisk_DMC_StationDay_MASTER_1600.csv` | 1,600 | Machine-readable full index and QA. |
| `FloodRisk_DMC_StationDay_LABELLED_1429.csv` | 1,429 | Direct input for supervised-stage baseline; multimodal features still need joining. |
| `FloodRisk_DMC_SourceObservations_1558.csv` | 1,558 | Independently auditable normalized source extract. |
| `FloodRisk_DMC_StationCoverage_40.csv` | 40 | Compact station coverage audit. |

**Do not use the earlier values-pending contract or the teammate's station-count assessment as the source of the counts in this release.** Every observation and pair count in this release was recomputed from the attached DMC workbook.

## What the workbook contains (direct audit)

| Metric | Count |
| --- | ---: |
| Research stations in our shared spatial registry | **40** |
| Stations actually listed in the DMC workbook | **39** |
| DMC station-date rows present | **1,558** |
| Numerical 09:00 stage values | **1,489** |
| Numerical 08:00 stage values | **1,489** |
| Numerical station-level 24-hour rainfall fields, recorded at 08:30 | **1,450** |
| Missing 09:00 values in existing rows | **69** (67 `NA`, 2 `-`) |
| Absent source rows among the 39 stations | **2** |
| Complete next-calendar-day 09:00 → 09:00 pairs | **1,429** |
| Canonical 40 × 40 station-day slots | **1,600** |

**Thambuththegama** does not occur in the DMC workbook. Its 40 rows are preserved in the master index with missing-source status, *not* labelled. **Galgamuwa (24 Nov)** and **Giriulla (16 Dec)** each have a missing source row. No station or missing date was quietly removed from the 1,600-row index.

## Exact derivation procedure

1. Read `Sheet1` from the DMC workbook without relying on external station-count sheets. Rows 1–2 are headers. The first station row for each day carries the Excel date serial; the following station rows inherit the same report date within that block.
2. Normalize station labels by trimming surrounding whitespace. Match the 39 listed stations to the previously established **40-station ArcGIS dashboard-based spatial registry**, including original point geometry and the matching candidate catchment path. No stations are invented.
3. Preserve **both 08:00 and 09:00 source stage readings**, source station-specific units, alert/minor/major thresholds, DMC remarks/direction, 24-hour rainfall fields, and original source Excel row numbers.
4. Treat 09:00 as the current observed stage and match only the **same station's numeric 09:00 value on the next calendar date** as the target. Do not bridge missing days or interpolate missing readings.
5. Convert all standardized levels to **metres**. Nagalagam Street is in **feet** in the source; multiply by exactly `0.3048`. All other 38 stations present in the DMC workbook are in metres. Original values/units are retained. Negative gauge-stage values are retained rather than treated as missing, because river *stage* can be below a local reference zero.
6. For each valid pair compute `target_delta_stage_m = target_stage_next_day_m - stage_at_issue_m`. The **persistence baseline** predicts next-day level equal to current level, so its error equals the true daily stage change. The 1,429 labelled rows require no guessed target values.
7. Expand to all 40 stations × all 40 source dates to create 1,600 canonical rows. Empty or absent source readings remain blank with explicit quality flags. The 26 Dec rows have no in-window next-day target, but their current observations are retained.
8. Allocate splits using **forecast dates** (not random station-day rows) and put all stations on the same chronological side of each boundary.

## Fixed chronological splits (09:00 → next-day 09:00)

| Split | Forecast issue dates | Valid pairs | Persistence MAE (m) | Persistence RMSE (m) |
| --- | --- | ---: | ---: | ---: |
| TRAIN | 2025-11-17 – 2025-12-09 | **809** | **0.6654** | **1.1319** |
| VALIDATION | 2025-12-10 – 2025-12-16 | **269** | **0.2349** | **0.4036** |
| TEST | 2025-12-17 – 2025-12-25 | **351** | **0.2172** | **0.4066** |
| No in-window target | 2025-12-26 | **0** | — | — |

**Important scientific limitation:** Among valid rows, the test split has **343 `Normal`, 7 `Alert`, 1 `Minor Flood`, and 0 `Major Flood`** source-day remarks. Most stronger flood stages are in training, while validation/test are predominantly non-flood. The test results therefore **cannot demonstrate generalization to unseen major floods**. This is only a reproducible within-period chronological pilot; results from independent historical flood events and longer gauge histories are required.

The 1,429 station-day pairs are highly correlated and cover mostly **one flood episode**. They are not 1,429 independent hydrological events. Fit scalers/encoders solely on TRAIN; tune solely on VALIDATION; hold TEST untouched. All components must use the same date split.

## Timing/availability caveats

The source worksheet labels the measurements `Water Level at 9:00 am`; the **09:30 report issue time is a project-level nominal assumption, not proven actual historical PDF publication time by this workbook**. Thus 09:00 to next-day 09:00 is a **24-hour observation-to-observation forecast horizon**, while 09:30 issue to next-day 09:00 is nominally **23.5 hours**.

Similarly, daily CHIRPS/ERA5, ASCAT SWI, and Sentinel-1 features must be joined only under an explicit observation-time / historical-publication-availability rule. Using final reanalysis available only afterwards must be labelled *retrospective hindcasting*, not claimed as a real-time operational warning.

The `rain_24h_at_0830_mm` field is taken from the DMC **station-level report** and **does not equal upstream-catchment mean rainfall**. It is retained as an optional observational covariate, not substituted for C2/C4 catchment-wide precipitation.

## C1–C4 joining and non-leakage contract

- Join keys: `station_id`, `forecast_date` (one master row per key); retain canonical ArcGIS coordinates and provisional MERIT Hydro catchment IDs.
- Shared labels: `stage_at_issue_m`, `target_stage_next_day_m`, `target_delta_stage_m`. Baseline-ready `stage_label_valid=True` means **numerical target validity only**.
- All multimodal reference fields (`sar_feature_ref`, `swi_feature_ref`, `weather_feature_ref`, `environment_feature_ref`) are intentionally blank. `multimodal_inputs_populated=False` for every row; do **not** claim 1,429 fully feature-complete C4 samples yet.
- Keep `catchment_qa_status` and exclude or sensitivity-test flagged catchments as agreed across components.
- Use per-station MAE/RMSE and persistence-relative scores; also evaluate on a flagged-catchment-excluded subset and report the actual sample count in every experiment.
- Do not treat stage absolute values of different local gauge datums as interchangeable; modelling **stage change** and supplying current stage as a common reference is one defensible approach.
- Missing levels are **not** zeros. No forward-filling or other label imputation is performed.

## Source provenance and distribution

Stage observations, thresholds and rainfall are transcribed from the user-supplied workbook; original DMC PDF URLs/pages are **not** present and cannot be reconstructed from this workbook alone. Canonical gauge point locations and polygons are taken from our separately documented public Irrigation Department dashboard / MERIT Hydro catchment workflow. This contract does not certify those polygons as official.

Before publicly committing or redistributing the source-derived numerical readings, confirm source licensing/usage terms. Share privately with the research team if permitted; a public repository can retain only the schema, reproducibility instructions and aggregate audit if redistribution is uncertain.