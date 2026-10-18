# C2 Weather-Specific Depth Study

This project compares a persistence baseline with three conservative XGBoost regressors for next-day 9:00 AM river water-level prediction. The target is the DMC field `water_level_9am_m`, expressed in metres as declared by its column name. Source units and vertical datum should still be confirmed against authoritative metadata.

LSTM is future work only. It is not implemented or trained here.

## Verified Inputs

The original CSV files remain in the parent `Component_2` directory and are never copied, renamed, or overwritten:

- `C2_Weather_RawData_9AM_Water_Level_2025- .csv`
- `C2_Weather_EngineeredData_9AM_Water_Level_2025 - Copy.csv`
- `C2_Weather_HybridData_9AM_Water_Level_2025.csv`

The checked files each have 1,521 rows, 39 station identifiers, and daily coverage from 2025-11-17 through 2025-12-25. Station-date keys and water-level values agree across files, with no duplicate keys. There are 71 missing water-level cells; rows are excluded when the current or exact-next-day target is missing. No target interpolation is performed. Full generated validation results are written to `outputs/metrics/data_quality_report.json`.

The source columns are:

- Raw: `station`, `date`, `rainfall_mm`, `soil_moisture_0_7cm`, `soil_moisture_7_28cm`, `temperature_C`, `water_level_9am_m`.
- Engineered: station/date and water level, soil moisture and temperature, plus `rainfall_3day_mm`, `rainfall_7day_mm`, and `rainfall_14day_mm`.
- Hybrid: the raw daily weather fields plus the three rainfall windows.

The daily weather data do not carry observation/publication time or timezone metadata. To avoid using same-day full-day values at a 9 AM forecast cutoff, daily weather features and supplied rainfall windows are shifted by one day. Rainfall windows are audited against complete prior-day sums where the source coverage allows verification; small differences are reported rather than silently rewritten.

## Forecast and Split Contract

For station `s` and forecast origin `t`, the target is the station's recorded 9 AM level at `t + 1 day`. It is retained only when the next observation is exactly one calendar day later. Features include current level, 1/3/7-day level lags, and weather values available through the prior date. Missing historical features remain missing for XGBoost's native missing-value handling; targets are never filled.

All variants use a common set of eligible station-date rows and these chronological forecast-origin splits:

| Split | Forecast origins |
| --- | --- |
| Train | 2025-11-17 to 2025-12-07 |
| Validation | 2025-12-08 to 2025-12-15 |
| Test | 2025-12-16 to 2025-12-24 |

Train labels extend through Dec 8, available at the validation period's first 9 AM cutoff; validation labels extend through Dec 16, available at the test period's first 9 AM cutoff. The target-date ranges themselves do not overlap. Validation selects among only three small, predefined parameter candidates. The selected iteration count is refit on train plus validation; test is used only for final comparison.

## Feature Experiments

All variants include the same historical-level features and one-hot station indicators:

- **Raw:** lagged daily rainfall, soil moisture, and temperature.
- **Engineered:** lagged soil moisture and temperature plus lagged 3/7/14-day rainfall windows.
- **Hybrid:** raw daily weather and the engineered rainfall windows, all lagged.
- **Persistence:** predicts the target using the current 9 AM level.

Exact feature names are saved to each model's `metadata.json` and summarized in `outputs/metrics/training_dataset_summary.json`.

## Run in VS Code / PowerShell

Open `D:\Component_2\c2_weather` as the workspace, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_validation.py
python scripts/run_training.py
python scripts/run_evaluation.py
python scripts/generate_pp1_evidence.py
```

For a next-day forecast after models are trained:

```powershell
python -c "from src.config import load_config; from src.predict import predict_latest; predict_latest('raw', load_config())"
```

Change `raw` to `engineered` or `hybrid` to use another trained model. Predictions are written under `outputs/predictions/`.

## Outputs

- `models/xgboost/{raw,engineered,hybrid}/`: XGBoost JSON model, fitted station preprocessor, and feature/training metadata.
- `outputs/predictions/`: test predictions and next-day forecasts.
- `outputs/metrics/`: data quality, split summary, overall metrics, per-station MAE, and feature importance.
- `outputs/figures/`: actual-versus-predicted, residual, and SHAP figures.
- `outputs/pp1_evidence/`: comparison tables, model architecture, reproducibility note, progress summary, and limitations.
- `data/processed/`: aligned feature samples for each variant.

The baseline and all three models are evaluated on identical test station-date keys. Metrics are MAE/RMSE in metres, R-squared when mathematically defined, and per-station MAE. The report explicitly identifies whether each XGBoost model beats persistence on test MAE. No result should be interpreted as evidence of strong generalization from approximately 39 days of data.

## Tests

Run `python -m pytest -q`. Tests cover station-local target construction, exact-next-day alignment, missing dates, weather and engineered-window availability lagging, complete rainfall windows, identical evaluation samples, and prediction-shape validation.

## Limitations / Future Work

The short record limits generalization conclusions. Confirm the water-level unit/datum, weather product timestamps and timezone, source-specific rainfall aggregation method, and delivery latency before operational use. LSTM is future work only; no recurrent model is included in this implementation.