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

- `raw`: ingested load, pool price, and optional known-as-of weather, generation availability, and intertie tables.
- `staging`: deduplicated and typed source models.
- `intermediate`: hourly load, price, calendar, holiday, lag, rolling, weather forecast, generation availability, and intertie features.
- `marts`: training, market overview, and peak demand marts.
- `ml`: model registry, forecast results, model performance, feature importance, backtest results, and promotion decisions.

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

- `make ingest`: loads the configured AESO data source and writes `raw.raw_aeso_load`, `raw.raw_aeso_pool_price`, and optional known-as-of feature raw tables when sample fixtures or CSV paths are configured.
- `make dbt-run`: builds staging, intermediate, and marts models in PostgreSQL.
- `make dbt-test`: runs dbt schema tests plus a no-future-leakage assertion.
- `make train`: trains baselines and XGBoost for 1-hour and 24-hour load forecasts, runs historical backtests, and applies champion/challenger promotion rules.
- `make evaluate`: prints model performance, latest backtest summaries, and latest promotion decisions.
- `make app`: starts the Streamlit dashboard.

Pass dbt vars through the Makefile with `DBT_VARS='{"key": value}'`, for example to disable optional exogenous feature groups on a real load/price-only run.

## Usage

### Sample data

Use sample data when you want a fully local smoke test, demo, or development workflow. Sample mode creates synthetic AESO-style load, pool price, weather forecast, generation availability, and intertie schedule fixtures.

```bash
make setup
AESO_DATA_SOURCE=sample make ingest
make dbt-run
make dbt-test
make train
make evaluate
make app
```

After `make app`, open the Streamlit URL printed in the terminal. The dashboard reads the warehouse tables built by dbt and the forecast outputs written by `make train`.

### Real AESO historical data

Use real AESO historical data when you want the forecasting pipeline trained and evaluated on actual historical Alberta Internal Load and pool price data. Historical mode downloads the public AESO historical CSV backfill and writes real load/price raw tables.

If you do not have real weather forecast, generation availability, or intertie CSVs, use this load/price-only workflow. It keeps deterministic calendar and holiday features enabled, but disables optional exogenous groups so old sample optional feature tables in a reused local database cannot be joined to real historical rows.

```bash
make setup
AESO_DATA_SOURCE=historical_csv make ingest
make dbt-run DBT_VARS='{"enable_weather_features": false, "enable_generation_availability_features": false, "enable_intertie_features": false}'
make dbt-test DBT_VARS='{"enable_weather_features": false, "enable_generation_availability_features": false, "enable_intertie_features": false}'
make train
make evaluate
make app
```

### Real data with exogenous CSVs

If you do have real known-as-of exogenous CSVs, set the CSV paths before ingestion and use the normal dbt run:

```bash
WEATHER_FORECAST_CSV_PATH=/path/to/weather_forecasts.csv \
GENERATION_AVAILABILITY_CSV_PATH=/path/to/generation_availability.csv \
INTERTIE_SCHEDULE_CSV_PATH=/path/to/intertie_schedules.csv \
AESO_DATA_SOURCE=historical_csv make ingest

make dbt-run
make dbt-test
make train
make evaluate
make app
```

The single-run forecast, performance, and feature-importance outputs are overwritten on each training run. The model registry preserves historical model records, while backtest results and promotion decisions are appended by run for auditability. If you switch between sample and real data, rerun the full sequence from ingestion through training so the marts, ML tables, and dashboard are all using the same data source.

## dbt Models

Staging models:

- `stg_aeso_load`: typed and deduplicated hourly Alberta Internal Load.
- `stg_aeso_pool_price`: typed and deduplicated hourly pool price.
- `stg_calendar`: hourly calendar spine with hour, day of week, month, weekend, Alberta-local holiday, long-weekend, and workday flags.
- `stg_weather_forecasts`: optional weather forecast snapshots by issue time, target time, and Alberta region.
- `stg_generation_availability`: optional known-as-of generation availability or outage snapshots by fuel type and unit.
- `stg_intertie_schedules`: optional known-as-of intertie schedules, transfer limits, and constraints.

Intermediate models:

- `int_load_features_hourly`: load lags and rolling load averages.
- `int_price_features_hourly`: pool price lag and rolling price average.
- `int_energy_features_hourly`: joined hourly load, price, calendar, and holiday features.
- `int_weather_forecast_features_hourly`: Alberta aggregate weather forecast features by issue and target hour.
- `int_generation_availability_features_hourly`: available capacity and outage features by issue and target hour.
- `int_intertie_features_hourly`: scheduled import/export, transfer limit, constraint, and headroom features by issue and target hour.

Marts:

- `mart_forecast_training_set`: horizon-aware training rows for 1-hour and 24-hour forecasts.
- `mart_market_overview`: hourly dashboard-ready market data.
- `mart_peak_demand_events`: top 5% load hours for peak monitoring.

## Forecasting Approach

The training mart includes the required forecasting features:

- hour, day of week, month, weekend flag
- Alberta/Canada statutory holiday, long-weekend, workday, and non-workday flags
- load lag 1h, 24h, 168h
- rolling 24h and 168h load averages
- price lag 24h and rolling 24h price average
- forecast weather: temperature, wind, humidity, precipitation, heating degree, cooling degree, forecast age, and contributing region count
- generation availability: total available, outage, derated, and available capacity by major fuel type
- intertie schedules: scheduled imports/exports, net scheduled imports, transfer limits, constraints, and headroom

The mart is horizon-aware to avoid future leakage. Calendar features describe the target hour because they are known in advance. Lag features are only populated when the source timestamp is known at or before the forecast issue time. For example, `load_lag_1h` is valid for the 1-hour-ahead model but intentionally null for the 24-hour-ahead model. Rolling features are calculated as of the issue timestamp.

Weather, generation availability, and intertie features are joined as known-as-of snapshots. For each target hour and forecast horizon, dbt selects the latest row whose issue timestamp is less than or equal to `feature_as_of_timestamp_utc` and whose target timestamp equals `target_timestamp_utc`. These columns intentionally use weather forecasts rather than future observed weather, availability/outage information known by the issue time, and published intertie schedules or limits rather than finalized future actual flows.

Models trained:

- Baseline: same hour yesterday
- Baseline: same hour last week
- Baseline: rolling 24-hour average
- XGBoost regressor

The backward-compatible single-split outputs still use time-based splits: approximately 70% train, 15% validation, and 15% test by target timestamp.

Model governance uses additional month-sized historical backtest windows. By default, `make train` evaluates the latest six eligible monthly windows with an expanding training history, requiring at least 90 days of training history and at least 7 days of test data per window. Set `MODEL_BACKTEST_MODE=rolling` to use a fixed rolling training history instead.

## Model Comparison

`make train` writes:

- `ml.model_registry`: model metadata, artifact locations, and preferred champion status.
- `ml.forecast_results`: timestamp-level actuals, predictions, errors, and peak flags.
- `ml.model_performance`: MAE, RMSE, MAPE, mean error, median absolute error, underprediction rate, and peak-period MAE.
- `ml.feature_importance`: XGBoost feature importances.
- `ml.model_backtest_results`: per-model, per-window backtest metrics with train/test window bounds and row counts.
- `ml.model_promotion_decisions`: deterministic promotion/rejection decisions with rule settings and comparison details.

Use `make evaluate` for a terminal summary or the Streamlit Model Comparison page for charts.

Champion/challenger promotion is controlled by environment variables:

- `MODEL_BACKTEST_ENABLED`: defaults to `true`.
- `MODEL_BACKTEST_MODE`: `expanding` or `rolling`; defaults to `expanding`.
- `MODEL_BACKTEST_MIN_TRAIN_DAYS`: defaults to `90`.
- `MODEL_BACKTEST_ROLLING_TRAIN_DAYS`: defaults to `180`.
- `MODEL_BACKTEST_MIN_TEST_DAYS`: defaults to `7`.
- `MODEL_BACKTEST_MAX_WINDOWS`: defaults to `6`; set `0` for no cap.
- `MODEL_PROMOTION_PRIMARY_METRIC`: `mae` or `rmse`; defaults to `mae`.
- `MODEL_PROMOTION_MIN_RELATIVE_IMPROVEMENT`: defaults to `0.0`.
- `MODEL_PROMOTION_MAX_PEAK_MAE_REGRESSION`: defaults to `0.05`.
- `MODEL_PROMOTION_MAX_UNDERPREDICTION_RATE_REGRESSION`: defaults to `0.02`.
- `MODEL_PROMOTION_MIN_WIN_RATE`: defaults to `0.5`.
- `MODEL_PROMOTION_MIN_WINDOWS`: defaults to `3`.

A challenger is promoted only when it improves the configured average primary metric versus the current preferred champion, does not exceed the peak-period and underprediction tolerances, wins or ties enough comparable windows, and has the minimum number of valid windows. If no champion exists yet, the initial champion is bootstrapped only after the minimum-window requirement is met.

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

Sample mode also writes deterministic known-as-of fixture tables by default:

- `raw.raw_weather_forecast_hourly`
- `raw.raw_generation_availability_hourly`
- `raw.raw_intertie_schedule_hourly`

Set `AESO_WRITE_SAMPLE_FEATURE_SOURCES=false` to skip those optional sample feature fixtures.

To load real AESO backfill data without real exogenous weather, availability, or intertie files, use the load/price-only dbt build:

```bash
AESO_DATA_SOURCE=historical_csv make ingest
make dbt-run DBT_VARS='{"enable_weather_features": false, "enable_generation_availability_features": false, "enable_intertie_features": false}'
make dbt-test DBT_VARS='{"enable_weather_features": false, "enable_generation_availability_features": false, "enable_intertie_features": false}'
make train
make evaluate
make app
```

Historical CSV mode downloads the AESO historical backfill files from the [AESO data request page](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-generation-metered-volumes-and-pool-price-and-ail-data-2001-to-july-2025/) for hourly generation metered volumes, pool price, and AIL data from 2001 to July 2025. It reads only `Date_Begin_GMT`, `ACTUAL_AIL`, and `ACTUAL_POOL_PRICE`, parses the timestamp as UTC, coerces load and price to numeric values, drops incomplete rows, and writes these existing raw table contracts:

- `raw.raw_aeso_load`: `interval_start_utc`, `alberta_internal_load_mw`, `source`, `ingested_at`
- `raw.raw_aeso_pool_price`: `interval_start_utc`, `pool_price_cad_mwh`, `source`, `ingested_at`

No AESO API key is required for historical CSV ingestion. `AESO_USE_SAMPLE_DATA=false` remains supported as a legacy fallback to `historical_csv` only when `AESO_DATA_SOURCE` is not set.

Optional real feature inputs can be loaded from CSVs during `make ingest`:

```text
WEATHER_FORECAST_CSV_PATH=/path/to/weather_forecasts.csv
GENERATION_AVAILABILITY_CSV_PATH=/path/to/generation_availability.csv
INTERTIE_SCHEDULE_CSV_PATH=/path/to/intertie_schedules.csv
```

Weather CSV contract:

- `forecast_issue_utc`, `forecast_target_utc`, `region`
- optional numeric columns: `temperature_c`, `wind_speed_mps`, `relative_humidity_pct`, `precipitation_mm`

Generation availability CSV contract:

- `availability_issue_utc`, `availability_target_utc`, `fuel_type`
- optional columns: `unit_id`, `available_capacity_mw`, `derated_capacity_mw`, `outage_capacity_mw`

Intertie schedule CSV contract:

- `schedule_issue_utc`, `schedule_target_utc`, `intertie_id`
- optional numeric columns: `scheduled_import_mw`, `scheduled_export_mw`, `transfer_limit_import_mw`, `transfer_limit_export_mw`, `constraint_mw`

All optional CSV rows may include `source`; ingestion stamps `ingested_at`. If optional raw tables are absent, the dbt staging models return empty typed relations and the final mart keeps the new optional feature columns null. The optional dbt feature groups can also be disabled with `--vars '{"enable_weather_features": false}'`, `enable_generation_availability_features`, or `enable_intertie_features`.

## Tests

```bash
pytest
make dbt-test
```

Pytest coverage focuses on metric calculations, horizon-aware lag behavior, deterministic holiday logic, optional source transforms, and no-future-source timestamp checks. dbt tests cover source/model integrity, holiday correctness, optional feature grain, target alignment, and warehouse-level no-leakage assertions.

## Limitations

- AESO APIM/API-key live ingestion is not implemented in the MVP.
- Sample data is realistic but synthetic, so sample-mode model scores are demonstration metrics.
- Real weather, outage, generation availability, and intertie feeds must be supplied through the documented raw tables or optional CSV contracts.
- Forecasting uses batch training only; there is no scheduler or model serving endpoint.

## Future Improvements

- Add AESO APIM ingestion and incremental raw loads.
- Add live weather, outage, generation availability, and intertie connectors.
- Add dbt exposures and source freshness checks.
- Add scheduled retraining and forecast publishing jobs.
