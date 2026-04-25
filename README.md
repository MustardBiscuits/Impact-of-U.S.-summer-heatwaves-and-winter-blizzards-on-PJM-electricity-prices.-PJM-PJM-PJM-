# PJM Electricity Price Forecasting with Weather Events

## Project Description

This project forecasts PJM Dominion (`DOM`) day-ahead locational marginal price
(`LMP`) using hourly electricity load and weather variables. The main task is a
24-hour-ahead forecasting problem: for each forecast origin, the pipeline
predicts `target_lmp_h01` through `target_lmp_h24`.

The project also studies weather-driven price behavior, especially:

- Summer high-price periods and sudden heat events.
- Winter cold waves and winter storm periods.
- Major winter storm event studies comparing pre-event, event, and post-event
  market behavior.
- High-price detection using ROC curves and AUC.

The model comparison includes:

- `seasonal_naive_24h`: same hour from the previous day.
- `seasonal_naive_168h`: same hour from the previous week.
- `dmdc`: Dynamic Mode Decomposition with control variables.
- `dmdc_residual_24h`: previous-day seasonal naive forecast plus a DMDc residual
  correction.
- `SARIMAX`: optional baseline. 

## Project Structure

```text
PJM_Weather_24h/
  README.md
  main.py
  data/
    readme_data.txt
    raw/
      PJM/
        pjm_dom_da_lmp_hrl_fixed.csv
        pjm_dom_load_hourly_2018_2025UTC.csv
      weather/
        weather_model_ready.csv
    processed/
      pjm_weather_hourly_clean.csv
      pjm_weather_supervised_24h.csv
      train_24h.csv
      validation_24h.csv
      test_24h.csv
  scripts/
    build_model_dataset.py
    train_24h_models.py
    analyze_winter_summer_patterns.py
    generate_report_figures.py
  reports/
    forecast_metrics_summary.csv
    forecast_metrics_by_horizon.csv
    time_series_cv_metrics_summary.csv
    roc_auc_summary.csv
    event_forecast_metrics.csv
    pattern_summary.md
    fig_*.png
```

`main.py` stays in the project root as the main entry point. All other Python
source files are stored in the `scripts/` folder. The `reports/` folder is kept
as the analysis output folder.

## Data Sources

Raw data are stored under `data/raw/`. See `data/readme_data.txt` for a short
data placement guide.

### PJM day-ahead LMP data

- File: `data/raw/PJM/pjm_dom_da_lmp_hrl_fixed.csv`
- Main variable: hourly day-ahead `LMP` for PJM Dominion (`DOM`)
- Timestamp: UTC
- Approximate coverage used here: 2018-01-01 to 2025-08-27

### PJM load data

- File: `data/raw/PJM/pjm_dom_load_hourly_2018_2025UTC.csv`
- Main variable: hourly `load_mw`
- Timestamp: UTC

### Weather data

- File: `data/raw/weather/weather_model_ready.csv`
- Variables include temperature, dew point, relative humidity, wind speed,
  sea-level pressure, precipitation, cooling degree days (`CDD`), heating degree
  days (`HDD`), and station count.
- Timestamp: UTC

### Excluded data

The current pipeline uses PJM LMP, PJM load, and observed weather features only.

## Required Packages

There is requirements

numpy
pandas
pyarrow
scikit-learn
statsmodels
matplotlib
scipy

Use Python 3.10 or newer. Required packages:

```text
numpy
pandas
pyarrow
scikit-learn
statsmodels
matplotlib
scipy
```

Install with:

```powershell
cd /path/to/DataSet/PJM_Weather_24h
python -m pip install numpy pandas pyarrow scikit-learn statsmodels matplotlib scipy
```

Optional virtual environment setup:

```powershell
cd /path/to/DataSet/PJM_Weather_24h
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas pyarrow scikit-learn statsmodels matplotlib scipy
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Main Entry Point

The main entry point is:

```text
main.py
```

`main.py` runs the full project workflow in order:

1. `scripts/build_model_dataset.py`
   Builds the cleaned hourly table, creates lag features, creates 24-hour
   forecast targets, and writes train/validation/test files to `data/processed/`.
2. `scripts/train_24h_models.py`
   Trains/evaluates the seasonal naive baselines, DMDc, and DMDc residual model.
   It also runs expanding-window cross-validation by default.
3. `scripts/analyze_winter_summer_patterns.py`
   Detects summer sudden-heat events, winter cold waves, winter storms, and
   creates event-based forecast metrics.
4. `scripts/generate_report_figures.py`
   Generates the main PNG figures for the report.

Run the full pipeline from the project root:

```powershell
cd /path/to/DataSet/PJM_Weather_24h
python main.py
```

Useful options:

```powershell
python main.py --skip-cv
python main.py --origin-step-hours 1
python main.py --run-sarimax
python main.py --skip-figures
```

## Running Individual Scripts

Run these commands from the project root:

```powershell
cd /path/to/DataSet/PJM_Weather_24h
python scripts/build_model_dataset.py
python scripts/train_24h_models.py
python scripts/analyze_winter_summer_patterns.py
python scripts/generate_report_figures.py
```

The default time split is:

- Training: 2018 through the end of 2022
- Validation: 2023
- Test: 2024 through 2025-08-26

## Data Leakage Control

The pipeline uses several safeguards to reduce data leakage:

- Chronological split only: data are split by time, not by random sampling.
- Holdout test period is kept after the training and validation periods.
- Lag features are generated with `shift`, so they use past values only.
- Rolling features use shifted series, so the current or future target is not
  included in the rolling statistic.
- Forecast targets `target_lmp_h01` through `target_lmp_h24` are future LMP
  values and are never used as input features.
- Rows with missing future target values are removed from supervised modeling.
- DMDc standardization is fitted only on the training period or the training
  portion of each cross-validation fold.
- Expanding-window cross-validation trains on past years and validates on later
  years.
- The high-price ROC/AUC threshold is computed from the training period only.

Important note: the current weather features are observed historical weather
values. For a fully deployable real-time forecasting system, these exogenous
weather inputs should be replaced by weather forecasts available at the forecast
origin.

## Outputs

Key processed data files:

- `data/processed/pjm_weather_hourly_clean.csv`
- `data/processed/pjm_weather_supervised_24h.csv`
- `data/processed/train_24h.csv`
- `data/processed/validation_24h.csv`
- `data/processed/test_24h.csv`

Key model reports:

- `reports/forecast_predictions_24h.csv`
- `reports/forecast_metrics_summary.csv`
- `reports/forecast_metrics_by_horizon.csv`
- `reports/time_series_cv_metrics_summary.csv`
- `reports/roc_auc_summary.csv`
- `reports/event_forecast_metrics.csv`
- `reports/loss_function_analysis.md`
- `reports/pattern_summary.md`

Key figures:

- `reports/fig_train_validation_test_split.png`
- `reports/fig_lmp_load_temperature_timeseries.png`
- `reports/fig_overall_model_comparison_mae_rmse_r2.png`
- `reports/fig_error_by_forecast_horizon_1_24h.png`
- `reports/fig_actual_vs_predicted_lmp_test_daily.png`
- `reports/fig_actual_vs_predicted_lmp_summer_2025_hourly.png`
- `reports/fig_roc_curve_high_price_detection.png`
- `reports/fig_event_based_mae_rmse_comparison.png`

## Evaluation Metrics

The regression forecast metrics are:

- `MAE`: mean absolute error.
- `MSE`: mean squared error.
- `RMSE`: root mean squared error.
- `R2`: coefficient of determination.
- `Bias`: mean signed forecast error.

For high-price detection, ROC/AUC is used as a classification diagnostic. The
observed high-price class is defined using a training-set LMP quantile, and the
forecasted LMP is used as the model score.

## Current Main Finding

The best default model in the current run is `dmdc_residual_24h`, which combines
the previous-day seasonal naive forecast with a DMDc residual correction.

Test-period performance:

- `dmdc_residual_24h`: `MAE = 12.50`, `RMSE = 28.32`, `R2 = 0.536`
- `seasonal_naive_24h`: `MAE = 13.07`, `RMSE = 29.15`, `R2 = 0.509`
- `dmdc`: `MAE = 18.89`, `RMSE = 39.12`, `R2 = 0.114`
- `seasonal_naive_168h`: `MAE = 23.32`, `RMSE = 50.14`, `R2 = -0.455`

Validation-period performance for the best model:

- `dmdc_residual_24h`: `MAE = 7.50`, `RMSE = 17.51`, `R2 = 0.236`

The residual DMDc model improves over the simple previous-day seasonal naive
baseline in the test period. Extreme price periods, especially summer
heat-price events and winter storms, remain the most difficult cases and are
therefore analyzed separately in the event-study outputs.
