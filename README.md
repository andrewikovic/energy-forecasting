# AESO Load Forecasting & Market Analytics

Reproducible analytics engineering and machine learning MVP for forecasting hourly Alberta Internal Load and analyzing AESO-style market data. The project is built as an internal analyst tool: raw data lands in PostgreSQL, dbt creates governed marts, Python trains and evaluates forecasting models, and Streamlit reads warehouse outputs for dashboarding.

## Architecture

```text
sample AESO-style fixtures or AESO historical CSV backfill
        |
        v
Python ingestion -> PostgreSQL raw schema
        |
        v
dbt staging -> dbt intermediate features -> dbt marts
        |
        v
Python training/evaluation -> PostgreSQL ml schema + local model artifacts
        |
        v
Streamlit analytics dashboard
```

Schemas:

- `raw`: ingested load and pool price tables.
- `staging`: deduplicated and typed source models.
- `intermediate`: hourly load, price, calendar, lag, and rolling features.
- `marts`: training, market overview, and peak demand marts.
- `ml`: model registry, forecast results, model performance, and feature importance.

## Setup

```bash
make setup
```

`make setup` creates `.venv`, installs dependencies, copies `.env.example` to `.env` if needed, and starts local Postgres with Docker when Docker is available. Python 3.10-3.12 is recommended; the Makefile prefers `python3.12` when it is installed.

On macOS, XGBoost may require the OpenMP runtime:

```bash
brew install libomp
```

If Docker is unavailable, start PostgreSQL manually and set these values in `.env`:

```text
POSTGRES_HOST=localhost
POSTGRES_PORT=55432
POSTGRES_DB=aeso_analytics
POSTGRES_USER=aeso
POSTGRES_PASSWORD=aeso
```

## Command Workflow

```bash
make ingest
make dbt-run
make dbt-test
make train
make evaluate
make app
```

What each command does:

- `make ingest`: loads the configured AESO data source and writes `raw.raw_aeso_load` and `raw.raw_aeso_pool_price`.
- `make dbt-run`: builds staging, intermediate, and marts models in PostgreSQL.
- `make dbt-test`: runs dbt schema tests plus a no-future-leakage assertion.
- `make train`: trains baselines and XGBoost for 1-hour and 24-hour load forecasts.
- `make evaluate`: prints model performance from `ml.model_performance`.
- `make app`: starts the Streamlit dashboard.

## dbt Models

Staging models:

- `stg_aeso_load`: typed and deduplicated hourly Alberta Internal Load.
- `stg_aeso_pool_price`: typed and deduplicated hourly pool price.
- `stg_calendar`: hourly calendar spine with hour, day of week, month, and weekend flag.

Intermediate models:

- `int_load_features_hourly`: load lags and rolling load averages.
- `int_price_features_hourly`: pool price lag and rolling price average.
- `int_energy_features_hourly`: joined hourly load, price, and calendar features.

Marts:

- `mart_forecast_training_set`: horizon-aware training rows for 1-hour and 24-hour forecasts.
- `mart_market_overview`: hourly dashboard-ready market data.
- `mart_peak_demand_events`: top 5% load hours for peak monitoring.

## Forecasting Approach

The training mart includes the required forecasting features:

- hour, day of week, month, weekend flag
- load lag 1h, 24h, 168h
- rolling 24h and 168h load averages
- price lag 24h and rolling 24h price average

The mart is horizon-aware to avoid future leakage. Calendar features describe the target hour because they are known in advance. Lag features are only populated when the source timestamp is known at or before the forecast issue time. For example, `load_lag_1h` is valid for the 1-hour-ahead model but intentionally null for the 24-hour-ahead model. Rolling features are calculated as of the issue timestamp.

Models trained:

- Baseline: same hour yesterday
- Baseline: same hour last week
- Baseline: rolling 24-hour average
- XGBoost regressor

Splits are time-based: approximately 70% train, 15% validation, and 15% test by target timestamp.

## Model Comparison

`make train` writes:

- `ml.model_registry`: model metadata and artifact locations.
- `ml.forecast_results`: timestamp-level actuals, predictions, errors, and peak flags.
- `ml.model_performance`: MAE, RMSE, MAPE, mean error, median absolute error, underprediction rate, and peak-period MAE.
- `ml.feature_importance`: XGBoost feature importances.

Use `make evaluate` for a terminal summary or the Streamlit Model Comparison page for charts.

## Streamlit App

The dashboard includes:

1. Market Overview: hourly load, daily average load, pool price, load vs price, KPIs.
2. Load Forecast: actual vs predicted, forecast errors, residuals, and forecast table.
3. Model Comparison: baseline and XGBoost MAE/RMSE/MAPE comparisons.
4. Error Analysis: errors by hour, day of week, month, and peak periods.
5. Peak Demand Monitor: top load hours and underprediction during peaks.
6. Data Quality: record counts, date ranges, missing hours, duplicates, nulls, latest ingestion.

## Data Ingestion

The project uses deterministic sample fixtures by default so it can run without network access or manual AESO data collection:

```text
AESO_DATA_SOURCE=sample
SAMPLE_START_UTC=2024-01-01T00:00:00Z
SAMPLE_DAYS=240
SAMPLE_SEED=42
```

Sample ingestion writes generated CSVs to `data/raw/` and loads the same PostgreSQL raw tables consumed by dbt.

To load real AESO backfill data, set:

```bash
AESO_DATA_SOURCE=historical_csv make ingest
make dbt-run
make dbt-test
```

Historical CSV mode downloads the AESO historical backfill files from the [AESO data request page](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-generation-metered-volumes-and-pool-price-and-ail-data-2001-to-july-2025/) for hourly generation metered volumes, pool price, and AIL data from 2001 to July 2025. It reads only `Date_Begin_GMT`, `ACTUAL_AIL`, and `ACTUAL_POOL_PRICE`, parses the timestamp as UTC, coerces load and price to numeric values, drops incomplete rows, and writes these existing raw table contracts:

- `raw.raw_aeso_load`: `interval_start_utc`, `alberta_internal_load_mw`, `source`, `ingested_at`
- `raw.raw_aeso_pool_price`: `interval_start_utc`, `pool_price_cad_mwh`, `source`, `ingested_at`

No AESO API key is required for historical CSV ingestion. `AESO_USE_SAMPLE_DATA=false` remains supported as a legacy fallback to `historical_csv` only when `AESO_DATA_SOURCE` is not set.

## Tests

```bash
pytest
make dbt-test
```

Pytest coverage focuses on metric calculations, horizon-aware lag behavior, and no-future-source timestamp checks. dbt tests cover source/model integrity and a warehouse-level no-leakage assertion.

## Limitations

- AESO APIM/API-key live ingestion is not implemented in the MVP.
- Sample data is realistic but synthetic, so sample-mode model scores are demonstration metrics.
- Weather, outage, generation stack, imports/exports, and holiday features are not included.
- Forecasting uses batch training only; there is no scheduler or model serving endpoint.

## Future Improvements

- Add AESO APIM ingestion and incremental raw loads.
- Add weather forecasts, holidays, generation availability, and intertie features.
- Add backtesting windows and champion/challenger promotion rules.
- Add dbt exposures and source freshness checks.
- Add scheduled retraining and forecast publishing jobs.
