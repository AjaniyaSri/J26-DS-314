# C3 PP1 offline inference demo — 23 October

This demo loads existing trained models and calculates predictions on demand. It does not train, tune, save results, change datasets or access the network. No new dependencies are needed. It uses the existing RP virtual environment. Execution policy changes and virtual-environment activation are unnecessary when using the executable directly.

## Start the demo

Open PowerShell and run these as two separate lines:

```powershell
cd C:\Users\LENOVO\Desktop\RP
.\.venv\Scripts\python.exe -B scripts/demo_c3_inference.py
```

Choose a station by its listed number or ID. Choose a forecast-input date by its listed number or `YYYY-MM-DD`. Enter `y` to reveal the historical actual after predictions. Enter `y` to select another station/date, or press Enter to finish. Enter `q` at a selection prompt to quit. Invalid menu entries can be corrected without restarting.

For a predictable two-minute demonstration:

```powershell
.\.venv\Scripts\python.exe -B scripts/demo_c3_inference.py --station badalgama --date 2025-12-17 --model1 revised
```

The default is revised Model 1. To demonstrate the original absolute-stage Model 1 instead:

```powershell
.\.venv\Scripts\python.exe -B scripts/demo_c3_inference.py --station badalgama --date 2025-12-17 --model1 original
```

With a fixed station and date the program exits after the actual-reveal question. To reveal automatically after predictions, append `--reveal-actual`. Without that flag, an unavailable input stream leaves the actual hidden. To see available selections without loading models:

```powershell
.\.venv\Scripts\python.exe -B scripts/demo_c3_inference.py --list-stations
.\.venv\Scripts\python.exe -B scripts/demo_c3_inference.py --station badalgama --list-dates
```

Station names are accepted with `--station`; quote names containing spaces. The historical datasets have 39 stations and 1,314 common station/date rows, with input dates spanning 19 November–25 December 2025. Some stations have fewer dates. The target date is the following calendar day. There is no automatic substitution of a nearby station or date.

## Which models are shown

| Display | Exact artifact | Interpretation |
|---|---|---|
| Model 1, default `revised` | `experiments/model1_diagnostics_v2/models/selected_environmental_model.joblib` | `delta_small`, ten inputs, stage-change wrapper |
| Model 1, optional `original` | `models/model1_static/lightgbm_model1.joblib` | Ten inputs, direct absolute-stage estimator |
| Model 2 | `experiments/model2_dynamic_v1/runs/20261019T115140_557496Z/models/model2_selected.joblib` | Selected secondary `model2_delta_mae_loss`, sixteen inputs, stage-change wrapper |
| Persistence | Current observed stage | Tomorrow equals today; no fitted estimator |

Stage-change wrappers add current stage exactly once to the predicted change. Original Model 1 directly outputs absolute stage and receives no additional offset. Displayed values are metres at the station's own gauge datum, not flood depth or common-elevation measurements. The terminal prints the model identities and artifact paths for each forecast.

Model 1 features are parsed from `C3_Static_Model_Lag2_Ready_PRELIMINARY.csv`; Model 2 features are parsed from `C3_Model2_Weather_PreviousDay_1314_PRELIMINARY (1).csv`. Their own CSV values are preserved because small floating-point differences can affect tree thresholds. Only identifiers, split labels and the feature allowlists are parsed before inference. The target is parsed separately only after predictions are displayed and reveal is requested. Hash verification reads file bytes for integrity, without supplying target values to either estimator.

## Two-minute narration

1. “This is live inference from trained LightGBM models on a historical station/date. No training is happening.”
2. Select Badalgama and 17 December 2025. Explain that the forecast concerns 18 December.
3. Identify revised static Model 1 and selected stage-change Model 2; point to persistence as the baseline. Current observations plus predicted changes produce final stages.
4. Reveal the historical actual and explain that signed error is prediction minus observation. Negative errors are underpredictions.
5. “This example demonstrates functioning inference. Our overall matched evaluation does not show an MAE gain from adding weather, and the previously examined TEST results are descriptive.”

The reproducibility example yields:

| Method | Predicted stage (m) |
|---|---:|
| Revised Model 1 | 2.418655 |
| Selected Model 2 | 2.277824 |
| Persistence | 2.330000 |
| Original Model 1, when chosen | 2.726328 |

Historical actual is 2.330000 m. This is an example, not a representative accuracy claim. Consult the [final report](../reports/final_c3_research_report_v1/final_c3_research_report.md) for full comparisons, rising-stage weaknesses and limitations.

## Checks and error handling

The demo verifies frozen dataset/model SHA-256 hashes, station/date uniqueness and correspondence, feature metadata, wrapper representation, estimator feature order/count, numeric/finite inputs and installed core dependency versions. It screens fractions/soil moisture within [0,1], slope within [0,90], positive catchment area and nonnegative rainfall. Negative gauge stages are allowed. It does not assume one-day rainfall is nested inside the rolling totals.

Stage/elevation units are documented as metres, slope as degrees, area as km², rainfall as accumulated mm and cover as fractions. Soil moisture is presumed m³/m³. These checks cannot independently verify physical units, gauge datums, source publication times or weather windows. The interface labels those uncertainties rather than claiming they are resolved.

An unknown station, malformed/unavailable date, changed frozen file, missing value, incompatible model or mismatched environment produces a clear `Demo error` and exit code 2. Use the listing commands to correct station/date selections. Use the existing RP `.venv` for compatibility. Do not edit datasets or retrain to bypass a compatibility error. Missing dependencies should be reported for investigation; this demo installs nothing.

## New forecasts and scope

The models can evaluate compatible new feature rows, but this interface intentionally accepts only the supplied historical rows. It cannot produce a forecast for October 2026 from data ending in December 2025. New forecasts require updated current/lagged stages, the same static descriptors, and correctly aligned rainfall and soil moisture for Model 2. Their availability at assumed issuance is unverified. The saved previous-day weather convention adds an extra lag to rolling rainfall under the supplied definition; one-day rainfall and soil-moisture windows remain uncertain.

This is retrospective next-day river-stage inference at known gauges, with nominal 09:00 readings and assumed 09:30 Asia/Colombo issuance. It is not a verified real-time warning service or flood map. No thresholds, warning classes, seasonal generalization or operational readiness are established by the demonstration. TRAIN examples show in-sample behaviour; VALIDATION/TEST labels are printed so their roles remain visible.

## Check the demo locally

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_c3_inference_demo.py -q -p no:cacheprovider
```

These tests load frozen estimators and compare both Model 1 variants and selected Model 2 with all saved TEST predictions, verify stage reconstruction, prohibit fitting during prediction, test target exclusion and reveal ordering, and check invalid inputs and compatibility failures. They do not retrain or write experiment outputs.
