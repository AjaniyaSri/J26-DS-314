# C3 Model 1: next-day river-stage forecasting

This standalone research project predicts `target_stage_next_day_m` using exactly
four gauge-history and six static catchment predictors. LightGBM is a boosted-tree
regressor. This is a preliminary, known-gauge chronological experiment, not an
operational flood warning system or an inundation model.

The original CSV stays at the RP root and is opened for reading only. No script
performs Git operations, uploads data, downloads weather, or creates Model 2.

## Run on Windows PowerShell

Open a terminal at `C:\Users\LENOVO\Desktop\RP`. The supplied local environment
uses Python 3.11; no activation or global PATH changes are necessary:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe scripts/audit_dataset.py
.\.venv\Scripts\python.exe -m pytest -q -o cache_dir=.cache/pytest --basetemp=.tools/test-tmp
.\.venv\Scripts\python.exe scripts/train_model1.py
.\.venv\Scripts\python.exe scripts/evaluate_model1.py
```

Training already performs the final TEST evaluation. `evaluate_model1.py` displays
saved results when evaluation is complete; it does not rescore TEST. If a frozen
run has not started TEST evaluation, it can perform that evaluation once. A failed
evaluation retains its start marker and artifacts for diagnosis; it cannot be
silently rerun. Inspect the failure before undertaking another experiment.

If you activate the environment with `.\.venv\Scripts\Activate.ps1`, the commands
`python scripts/audit_dataset.py` and `python scripts/train_model1.py` also work.
Activation is optional; changing PowerShell execution policy is unnecessary.

For a fresh machine with an already installed Python 3.11, local setup is:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This downloads dependencies and should only be run with the user's approval.
On this machine, Python itself is kept in `.python/`, the portable uv bootstrap
tool in `.tools/uv/`, and downloads/caches inside RP. Reuse the local uv tool to
install dependencies if pip is absent from this uv-created environment:

```powershell
$env:TEMP = Join-Path (Get-Location) '.tools\tmp'
$env:TMP = $env:TEMP
.\.tools\uv\uv.exe pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv -r requirements.txt
```

Do not recreate or overwrite an existing `.venv`. Full installed dependency
versions are captured in `reports/environment_lock.txt` and each model's metadata.
To reproduce the exact transitive environment, install from that lock file.

## What each directory contains

- `configs/`: exact ordered feature contract, seed, six predefined candidates.
- `src/`: loading, validation, persistence, fitting, metrics and plots.
- `scripts/`: audit, complete experiment, and saved/frozen evaluation entry points.
- `tests/`: actual schema checks plus malformed-data, calendar-lag, split and leakage checks.
- `models/model1_static/`: joblib estimator, native LightGBM text model, frozen
  configuration, metadata and final evaluation markers.
- `results/metrics/`: validation search, pooled, station and importance tables.
- `results/predictions/`: validation/TEST records, observed and predicted stages,
  persistence predictions and errors (positive means overprediction).
- `results/figures/`: actual versus predicted TEST stages, station MAE, gain
  importance and comparison against persistence.
- `reports/`: data audit, environment lock and experiment interpretation.
- `logs/`: local execution logs.

The first experiment uses the requested output paths directly. Subsequent
training invocations preserve previous outputs and use `runs/<UTC timestamp>/`.
Use `--run-dir runs/<timestamp>` with `evaluate_model1.py` to inspect such a run.
Repeated audit commands create timestamped reports. Existing outputs are never
silently overwritten. The CSV is excluded by `.gitignore`; no repository is initialized.

## Scientific procedure

Validation checks the 1,314-row, 21-column dataset, 39 stations, exact split counts
and date windows, uniqueness, numerical ranges, static-feature consistency and
calendar-based station-local lags/labels. It reports uncheckable references rather
than claiming complete verification. All stations and original rows stay included.
Failures stop training and show affected stations, dates and columns in the audit.

TRAIN contains 700 rows (2025-11-19–2025-12-09), VALIDATION 265
(2025-12-10–2025-12-16), TEST 349 (2025-12-17–2025-12-25).
All candidate fits use TRAIN alone. Early stopping monitors validation RMSE with
40 rounds' patience and at most 500 trees. Validation MAE selects among six
predefined candidates; exact ties retain candidate order. The selected fitted
model is frozen without a TRAIN+VALIDATION refit. Only then are TEST predictions
and persistence metrics calculated, once. There is no random split, scaling,
station encoding, target-derived input or TEST-guided tuning.

MAE is the average absolute error. RMSE is the square root of average squared
error. Both are in metres, computed on identical rows for the model and baseline,
with sample counts. Persistence predicts tomorrow equals today's stage. Pooled
and per-station results include both methods. A model that loses to persistence
is reported as such. Deterministic CPU settings use one thread and a fixed seed;
cross-platform numerical identity is not guaranteed.

Source data, selected model, source code and dependency versions are fingerprinted
in metadata. A repeated experiment creates a new run; it must not become iterative
TEST-driven model development after final TEST results are known. Future research
choices require a new independent holdout for an unbiased final assessment.

## Interpretation and limitations

The dataset has 37 input dates within a roughly 40-day study, with uneven gauge
coverage. Catchment boundaries are candidates with unresolved QA flags, and
WorldCover is from 2021. Gauge datums may differ and related catchments may overlap.
Static environmental features can encode gauge identity; importance is an
association, not causation. This initial experiment cannot establish that static
features add predictive value without a separate gauge-history-only ablation.

The data describe 09:00 observations but assume 09:30 forecast issuance. All source
publication verification flags are False. The next-day 09:00 horizon is one day
from observation, about 23.5 hours from assumed issuance. Availability before
issuance is unverified, including observations that provide the labels at split
boundaries. Existing splits are retained with this limitation documented.
Historical inputs on later TEST dates may include earlier observed TEST readings:
this represents daily rolling information, not a single forecast of the full block.

Units and original source provenance cannot be independently certified from this
CSV alone. Negative gauge stages are retained. This short temporal study cannot
establish seasonal generalization, new-gauge generalization or flood-inundation
accuracy. Formal group approval of the revised river-stage objective remains a
governance item requiring confirmation. The generated summary lists all limitations.

No rainfall, soil moisture, weather names or timestamps are fabricated. Model 2
awaits a verified C2 file and a separately agreed comparison on identical rows.

Package setup uses the [official uv command reference](https://docs.astral.sh/uv/reference/cli/).
