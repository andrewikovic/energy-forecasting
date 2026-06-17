SHELL := /bin/bash

VENV ?= .venv
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3)
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
DBT := $(VENV)/bin/dbt
STREAMLIT := $(VENV)/bin/streamlit

-include .env

export PYTHONPATH := src
export DBT_PROFILES_DIR ?= dbt
export POSTGRES_HOST
export POSTGRES_PORT
export POSTGRES_DB
export POSTGRES_USER
export POSTGRES_PASSWORD
export DBT_TARGET

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
	$(DBT) run --project-dir dbt --profiles-dir dbt

dbt-test:
	$(DBT) test --project-dir dbt --profiles-dir dbt

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
