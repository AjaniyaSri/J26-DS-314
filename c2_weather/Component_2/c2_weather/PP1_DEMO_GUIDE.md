# PP1 Viva Live Demo Guide

## Purpose and Safety Label

This is a retrospective inference demonstration on verified held-out historical test records. It loads the saved Raw, Engineered, and Hybrid XGBoost models and their saved preprocessing pipelines. It does not retrain, tune, modify, or overwrite model files. The next-day actual level is shown for evaluation only; it is never passed to a model as an input.

The demonstration is not a real-time forecasting service. The available weather CSVs do not document publication timestamps/timezone, and gauge unit/datum metadata still require confirmation.

## Actual Artifact Locations

- Saved models and metadata: `models/xgboost/{raw,engineered,hybrid}/model.json` and `metadata.json`.
- Fitted preprocessors: `models/xgboost/{raw,engineered,hybrid}/preprocessor.joblib`.
- Verified model feature schemas: each model's `metadata.json` `feature_columns` field.
- Held-out processed rows: `data/processed/{raw,engineered,hybrid}_samples.csv`.
- Saved predictions: `outputs/predictions/xgboost_{raw,engineered,hybrid}_test_predictions.csv` and `persistence_test_predictions.csv`.
- Original test metrics: `outputs/metrics/model_comparison.csv`.
- SHAP plots: `outputs/figures/{raw,engineered,hybrid}_shap_summary.png` and `*_shap_importance.png`.
- PP1 evidence figures: `outputs/pp1_final_package/figures/`.

The live app discovers test records by filtering the configured test-origin date interval and checks that each processed row matches its saved prediction record by station, origin, and target date.

## Start the Demo

From the `c2_weather` directory in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run scripts/run_pp1_live_demo.py
```

The terminal prints the local URL, normally `http://localhost:8501`. Select a station, then one of its available held-out forecast origin dates. The app displays the origin, next-day target date, current 9 AM level, actual next-day DMC reading, all four predictions and absolute errors, the station's test-period series, original test metrics, and saved SHAP explanations.

## Reproducibility Check

Run this before the viva to recompute three real held-out examples and compare each inference result against the corresponding saved prediction CSV:

```powershell
.\.venv\Scripts\python.exe -m src.pp1_demo --verify 3
```

The checker prints station/date, recomputed value, saved value, absolute difference, and whether the difference is within `1e-6` m. A mismatch returns a failing exit code; results are never silently replaced. To check every held-out record, run `--verify 351`.

## What Each Model Does

- **Persistence:** predicts the target as the current 9 AM level.
- **Raw XGBoost:** current level and 1/3/7-day exact water-level lags plus prior-day raw weather features.
- **Engineered XGBoost:** the same level history plus prior-day soil/temperature and 3/7/14-day rainfall-window features.
- **Hybrid XGBoost:** raw and engineered weather features with the same water-level history.

The saved preprocessing pipeline one-hot encodes station and passes the numeric feature columns in the recorded order. The app constructs model input only from `station` and the exact `feature_columns` in saved model metadata. `target_water_level_m`, `actual_water_level_m`, and `target_date` are excluded from inference input.

## Results to Explain

On the original untouched 351-record test set, persistence achieved MAE 0.2159 m and RMSE 0.4086 m. Raw, Engineered, and Hybrid XGBoost achieved MAE 0.2771, 0.3206, and 0.2963 m. Do not claim that XGBoost beat persistence.

The app's selected-record prediction should agree with the saved CSV to floating-point tolerance. The selected observation is one point in the held-out test; the headline model metrics summarize all 351 common test records. Saved SHAP plots describe feature influence in each fitted model and are not causal explanations.

The separate Delta experiment is validation-only. Its best MAE was 0.2902 m versus validation persistence at 0.2574 m; it did not improve validation performance. Do not combine Delta validation metrics with the original test table.

## Suggested Viva Walkthrough

1. State the forecast definition: inputs available at 9 AM on t, target DMC reading at 9 AM on t+1.
2. Choose a station and test-origin date; identify the current level and target date.
3. Explain persistence first, then Raw/Engineered/Hybrid feature differences.
4. Show the saved-model prediction table and errors. Point out the saved-prediction difference column as a reproducibility check.
5. Show the station time-series plot and saved SHAP summaries.
6. End with overall held-out test metrics and the honest finding that persistence wins overall.
7. Mention rapid-rise underprediction, limited 39-day coverage, unverified weather release times, and gauge metadata as limitations.

## Common Viva Questions

**Why predict 24 hours ahead?**  
The operational target is the next day's 9 AM gauge reading; the forecast origin and target are exactly one calendar day apart.

**How do you prevent target leakage in this demonstration?**  
The inference helper reads only the saved metadata feature schema and excludes target/actual columns. It transforms those features and predicts first; actuals are used afterward only to calculate retrospective error.

**Why are weather features shifted by a day?**  
The input files lack observation/publication timestamps. Lagging daily weather to the prior calendar date avoids relying on a full date-t weather total at a 9 AM cutoff, although source availability still needs verification.

**Did XGBoost outperform the baseline?**  
No. On the held-out test set persistence MAE was 0.2159 m, lower than every XGBoost variant. The result is reported as-is.

**What does SHAP tell you?**  
It describes how features contributed to individual model predictions within the fitted model. It does not establish hydrological causality.

**What is the main rapid-rise limitation?**  
Every XGBoost variant underpredicted all 12 top-decile rise events in the test set. The small record cannot establish reliable flood-peak behavior.

**Is Delta part of the main test comparison?**  
No. Delta is a separate exploratory experiment selected and evaluated on the validation period only; it did not beat validation persistence.

**What would you do next?**  
Expand data across seasons and flood events, verify gauge units/datums and weather delivery times, and use multiple rolling-origin holdouts. LSTM and multimodal integration remain future work.