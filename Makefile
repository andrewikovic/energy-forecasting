SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3)
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DBT := $(VENV)/bin/dbt
STREAMLIT := $(VENV)/bin/streamlit
DBT_VARS ?=
DBT_VAR_ARGS := $(if $(strip $(DBT_VARS)),--vars '$(DBT_VARS)',)

-include .env

export PYTHONPATH := src
export DBT_PROFILES_DIR ?= dbt
export POSTGRES_HOST
export POSTGRES_PORT
export POSTGRES_DB
export POSTGRES_USER
export POSTGRES_PASSWORD
export DBT_TARGET
export AESO_DATA_SOURCE
export AESO_USE_SAMPLE_DATA
export SAMPLE_START_UTC
export SAMPLE_DAYS
export SAMPLE_SEED
export AESO_WRITE_SAMPLE_FEATURE_SOURCES
export WEATHER_FORECAST_CSV_PATH
export GENERATION_AVAILABILITY_CSV_PATH
export INTERTIE_SCHEDULE_CSV_PATH
export MODEL_BACKTEST_ENABLED
export MODEL_BACKTEST_MODE
export MODEL_BACKTEST_MIN_TRAIN_DAYS
export MODEL_BACKTEST_ROLLING_TRAIN_DAYS
export MODEL_BACKTEST_MIN_TEST_DAYS
export MODEL_BACKTEST_MIN_TRAIN_ROWS
export MODEL_BACKTEST_MIN_TEST_ROWS
export MODEL_BACKTEST_MAX_WINDOWS
export MODEL_PROMOTION_PRIMARY_METRIC
export MODEL_PROMOTION_MIN_RELATIVE_IMPROVEMENT
export MODEL_PROMOTION_MAX_PEAK_MAE_REGRESSION
export MODEL_PROMOTION_MAX_UNDERPREDICTION_RATE_REGRESSION
export MODEL_PROMOTION_MIN_WIN_RATE
export MODEL_PROMOTION_MIN_WINDOWS

.PHONY: setup postgres-up ingest dbt-run dbt-test train evaluate app test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then docker compose up -d postgres; else echo "Docker is not available; start PostgreSQL manually before running make ingest."; fi

postgres-up:
	docker compose up -d postgres

ingest:
	$(PY) -m aeso_analytics.ingest

dbt-run:
	$(DBT) run --project-dir dbt --profiles-dir dbt $(DBT_VAR_ARGS)

dbt-test:
	$(DBT) test --project-dir dbt --profiles-dir dbt $(DBT_VAR_ARGS)

train:
	$(PY) -m aeso_analytics.train

evaluate:
	$(PY) -m aeso_analytics.evaluate

app:
	$(STREAMLIT) run app/streamlit_app.py

test:
	$(PY) -m pytest

clean:
	rm -rf dbt/target .pytest_cache
