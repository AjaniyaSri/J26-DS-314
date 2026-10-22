# Shared Prediction & Data Contract

Summary of the team's **FloodRisk Sri Lanka — Shared Prediction & Data Guide** (18 Oct 2026).

## Scientific target

Predict each station's **next-day 09:00 river stage in metres**, using the current day's 09:00 observation and eligible preceding information.

- `target_delta_stage_m = target_stage_next_day_m - stage_at_issue_m`
- `predicted_next_day_stage_m = stage_at_issue_m + predicted_change_m`
- Join tables by **`station_id + forecast_date`**. Do not join by row order.
- The 09:30 report issuance is nominal, and actual historic publication time has not been independently verified; observed readings are 24 hours apart, nominal issue-to-target is 23.5 hours.
- Respect station-specific zero datums and missing entries. The Nagalagam Street source is in feet and is converted using 0.3048.

## Verified audit counts

| Measure | Count |
| --- | ---: |
| 40 stations × 40 dates (17 Nov–26 Dec 2025) | 1,600 station-days |
| Gauges present in source workbook | 39 |
| Valid current-day observations | 1,489 |
| Valid consecutive-day target pairs | 1,429 |
| Non-numeric stage entries (missing, not zero) | 69 |

The number of valid *labels* is not the number of multimodal feature-complete samples.

## Fixed chronological split

| Split | Dates | Pairs |
| --- | --- | ---: |
| TRAIN | 17 Nov–09 Dec 2025 | 809 |
| VALIDATION | 10–16 Dec 2025 | 269 |
| TEST | 17–25 Dec 2025 | 351 |

Fit scalers and imputers only on TRAIN. Tune on VALIDATION; report held-out TEST after model selection. Never provide future labels as features.

**Stage-only persistence TEST baseline:** MAE **0.2172 m** and RMSE **0.4066 m** over 351 rows. This is not a trained-model result. Use the same eligible rows for satellite/weather/fusion comparisons.

## Component responsibilities

- **C1:** Sentinel-1 SAR VV/VH and ASCAT SWI, timestamps, ages, missingness, single-date versus multi-date information.
- **C2:** CHIRPS/ERA5 raw, engineered and hybrid catchment weather.
- **C3:** Static and dynamic environmental and catchment context.
- **C4:** Independently train S/W/E branches, ablations M1(S), M2(W), M3(E), M4(S+W), M5(S+E), M6(W+E), M7(S+W+E); do not consume fitted C1–C3 models as fusion inputs.

## Limitations

Archived satellite and weather products may have arrived after the assumed forecast issuance. Track source date, version, acquisition time and historical availability. Where availability is not verified, call the work **retrospective hindcasting**. Provisional MERIT polygons are not official watershed boundaries. This short period does not independently validate unseen major floods; the team reported no TEST labels marked Major Flood in the source-day remarks.

See the approved original team guide for the complete field dictionary, source references and QA instructions.
