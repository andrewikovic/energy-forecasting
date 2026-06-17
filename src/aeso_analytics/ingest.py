from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from aeso_analytics.config import get_ingestion_config
from aeso_analytics.database import ensure_schemas, get_engine, write_dataframe
from aeso_analytics.sample_data import (
    generate_sample_aeso_data,
    generate_sample_known_as_of_feature_data,
    write_sample_csvs,
)


HISTORICAL_CSV_URLS = (
    "https://www.aeso.ca/assets/Uploads/data-requests/"
    "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2001-2009.csv",
    "https://www.aeso.ca/assets/Uploads/data-requests/"
    "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2010-2019.csv",
    "https://www.aeso.ca/assets/Uploads/data-requests/"
    "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv",
)
HISTORICAL_CSV_COLUMNS = ("Date_Begin_GMT", "ACTUAL_AIL", "ACTUAL_POOL_PRICE")
HISTORICAL_CSV_SOURCE = "aeso_historical_csv"

LOAD_RAW_COLUMNS = (
    "interval_start_utc",
    "alberta_internal_load_mw",
    "source",
    "ingested_at",
)
PRICE_RAW_COLUMNS = (
    "interval_start_utc",
    "pool_price_cad_mwh",
    "source",
    "ingested_at",
)
WEATHER_FORECAST_RAW_COLUMNS = (
    "forecast_issue_utc",
    "forecast_target_utc",
    "region",
    "temperature_c",
    "wind_speed_mps",
    "relative_humidity_pct",
    "precipitation_mm",
    "source",
    "ingested_at",
)
GENERATION_AVAILABILITY_RAW_COLUMNS = (
    "availability_issue_utc",
    "availability_target_utc",
    "fuel_type",
    "unit_id",
    "available_capacity_mw",
    "derated_capacity_mw",
    "outage_capacity_mw",
    "source",
    "ingested_at",
)
INTERTIE_SCHEDULE_RAW_COLUMNS = (
    "schedule_issue_utc",
    "schedule_target_utc",
    "intertie_id",
    "scheduled_import_mw",
    "scheduled_export_mw",
    "transfer_limit_import_mw",
    "transfer_limit_export_mw",
    "constraint_mw",
    "source",
    "ingested_at",
)
SUPPORTED_DATA_SOURCES = {"sample", "historical_csv"}


@dataclass(frozen=True)
class RawAesoTables:
    load: pd.DataFrame
    price: pd.DataFrame


@dataclass(frozen=True)
class RawKnownAsOfFeatureTables:
    weather_forecasts: pd.DataFrame
    generation_availability: pd.DataFrame
    intertie_schedules: pd.DataFrame


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        series = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def transform_historical_csv_data(
    frame: pd.DataFrame,
    source: str = HISTORICAL_CSV_SOURCE,
    ingested_at: pd.Timestamp | None = None,
) -> RawAesoTables:
    missing_columns = set(HISTORICAL_CSV_COLUMNS) - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Historical AESO CSV data is missing required columns: {missing}")

    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    normalized = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(
                frame["Date_Begin_GMT"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "alberta_internal_load_mw": _coerce_numeric(frame["ACTUAL_AIL"]),
            "pool_price_cad_mwh": _coerce_numeric(frame["ACTUAL_POOL_PRICE"]),
        }
    )
    normalized = normalized.dropna(
        subset=[
            "interval_start_utc",
            "alberta_internal_load_mw",
            "pool_price_cad_mwh",
        ]
    ).copy()
    normalized["source"] = source
    normalized["ingested_at"] = ingested_at

    return RawAesoTables(
        load=normalized.loc[:, LOAD_RAW_COLUMNS].copy(),
        price=normalized.loc[:, PRICE_RAW_COLUMNS].copy(),
    )


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str], label: str) -> None:
    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{label} data is missing required columns: {missing}")


def _optional_text_column(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype("string").str.strip().replace("", pd.NA).fillna(default)
    return pd.Series(default, index=frame.index, dtype="string")


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return _coerce_numeric(frame[column])
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _normalized_required_text(frame: pd.DataFrame, column: str, default: str) -> pd.Series:
    return (
        frame[column]
        .astype("string")
        .str.lower()
        .str.strip()
        .replace("", pd.NA)
        .fillna(default)
    )


def transform_weather_forecast_data(
    frame: pd.DataFrame,
    source: str = "external_weather_forecast_csv",
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    _require_columns(
        frame,
        ("forecast_issue_utc", "forecast_target_utc", "region"),
        "Weather forecast",
    )
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    normalized = pd.DataFrame(
        {
            "forecast_issue_utc": pd.to_datetime(
                frame["forecast_issue_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "forecast_target_utc": pd.to_datetime(
                frame["forecast_target_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "region": _normalized_required_text(frame, "region", "alberta"),
            "temperature_c": _optional_numeric_column(frame, "temperature_c"),
            "wind_speed_mps": _optional_numeric_column(frame, "wind_speed_mps"),
            "relative_humidity_pct": _optional_numeric_column(frame, "relative_humidity_pct"),
            "precipitation_mm": _optional_numeric_column(frame, "precipitation_mm"),
            "source": _optional_text_column(frame, "source", source),
            "ingested_at": ingested_at,
        }
    )
    normalized = normalized.dropna(
        subset=["forecast_issue_utc", "forecast_target_utc", "region"]
    ).copy()
    return normalized.loc[:, WEATHER_FORECAST_RAW_COLUMNS].copy()


def transform_generation_availability_data(
    frame: pd.DataFrame,
    source: str = "external_generation_availability_csv",
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    _require_columns(
        frame,
        ("availability_issue_utc", "availability_target_utc", "fuel_type"),
        "Generation availability",
    )
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    normalized = pd.DataFrame(
        {
            "availability_issue_utc": pd.to_datetime(
                frame["availability_issue_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "availability_target_utc": pd.to_datetime(
                frame["availability_target_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "fuel_type": _normalized_required_text(frame, "fuel_type", "unknown"),
            "unit_id": _optional_text_column(frame, "unit_id", "unreported"),
            "available_capacity_mw": _optional_numeric_column(frame, "available_capacity_mw"),
            "derated_capacity_mw": _optional_numeric_column(frame, "derated_capacity_mw"),
            "outage_capacity_mw": _optional_numeric_column(frame, "outage_capacity_mw"),
            "source": _optional_text_column(frame, "source", source),
            "ingested_at": ingested_at,
        }
    )
    normalized = normalized.dropna(
        subset=["availability_issue_utc", "availability_target_utc", "fuel_type"]
    ).copy()
    normalized["unit_id"] = normalized["unit_id"].fillna("unreported")
    return normalized.loc[:, GENERATION_AVAILABILITY_RAW_COLUMNS].copy()


def transform_intertie_schedule_data(
    frame: pd.DataFrame,
    source: str = "external_intertie_schedule_csv",
    ingested_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    _require_columns(
        frame,
        ("schedule_issue_utc", "schedule_target_utc", "intertie_id"),
        "Intertie schedule",
    )
    ingested_at = ingested_at or pd.Timestamp.now(tz="UTC")
    normalized = pd.DataFrame(
        {
            "schedule_issue_utc": pd.to_datetime(
                frame["schedule_issue_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "schedule_target_utc": pd.to_datetime(
                frame["schedule_target_utc"],
                errors="coerce",
                format="mixed",
                utc=True,
            ),
            "intertie_id": _normalized_required_text(frame, "intertie_id", "unknown"),
            "scheduled_import_mw": _optional_numeric_column(frame, "scheduled_import_mw"),
            "scheduled_export_mw": _optional_numeric_column(frame, "scheduled_export_mw"),
            "transfer_limit_import_mw": _optional_numeric_column(
                frame, "transfer_limit_import_mw"
            ),
            "transfer_limit_export_mw": _optional_numeric_column(
                frame, "transfer_limit_export_mw"
            ),
            "constraint_mw": _optional_numeric_column(frame, "constraint_mw"),
            "source": _optional_text_column(frame, "source", source),
            "ingested_at": ingested_at,
        }
    )
    normalized = normalized.dropna(
        subset=["schedule_issue_utc", "schedule_target_utc", "intertie_id"]
    ).copy()
    return normalized.loc[:, INTERTIE_SCHEDULE_RAW_COLUMNS].copy()


def read_historical_csv_tables(urls: Iterable[str] = HISTORICAL_CSV_URLS) -> RawAesoTables:
    load_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []

    for url in urls:
        csv_frame = pd.read_csv(url, usecols=list(HISTORICAL_CSV_COLUMNS))
        tables = transform_historical_csv_data(csv_frame)
        load_parts.append(tables.load)
        price_parts.append(tables.price)

    if not load_parts or not price_parts:
        raise ValueError("No AESO historical CSV data sources were provided.")

    load_df = (
        pd.concat(load_parts, ignore_index=True)
        .sort_values("interval_start_utc")
        .reset_index(drop=True)
    )
    price_df = (
        pd.concat(price_parts, ignore_index=True)
        .sort_values("interval_start_utc")
        .reset_index(drop=True)
    )
    if load_df.empty or price_df.empty:
        raise ValueError("AESO historical CSV ingestion produced no valid rows.")

    return RawAesoTables(load=load_df, price=price_df)


def _write_raw_tables(engine, tables: RawAesoTables) -> None:
    write_dataframe(engine, tables.load, "raw", "raw_aeso_load", if_exists="replace")
    write_dataframe(engine, tables.price, "raw", "raw_aeso_pool_price", if_exists="replace")


def _write_known_as_of_feature_tables(engine, tables: RawKnownAsOfFeatureTables) -> None:
    write_dataframe(
        engine,
        tables.weather_forecasts,
        "raw",
        "raw_weather_forecast_hourly",
        if_exists="replace",
    )
    write_dataframe(
        engine,
        tables.generation_availability,
        "raw",
        "raw_generation_availability_hourly",
        if_exists="replace",
    )
    write_dataframe(
        engine,
        tables.intertie_schedules,
        "raw",
        "raw_intertie_schedule_hourly",
        if_exists="replace",
    )


def _write_configured_optional_feature_csvs(engine, feature_cfg) -> None:
    if feature_cfg.weather_forecast_csv_path:
        weather_df = transform_weather_forecast_data(
            pd.read_csv(feature_cfg.weather_forecast_csv_path)
        )
        write_dataframe(
            engine,
            weather_df,
            "raw",
            "raw_weather_forecast_hourly",
            if_exists="replace",
        )
        print(f"Wrote {len(weather_df):,} configured weather forecast rows")

    if feature_cfg.generation_availability_csv_path:
        generation_df = transform_generation_availability_data(
            pd.read_csv(feature_cfg.generation_availability_csv_path)
        )
        write_dataframe(
            engine,
            generation_df,
            "raw",
            "raw_generation_availability_hourly",
            if_exists="replace",
        )
        print(f"Wrote {len(generation_df):,} configured generation availability rows")

    if feature_cfg.intertie_schedule_csv_path:
        intertie_df = transform_intertie_schedule_data(
            pd.read_csv(feature_cfg.intertie_schedule_csv_path)
        )
        write_dataframe(
            engine,
            intertie_df,
            "raw",
            "raw_intertie_schedule_hourly",
            if_exists="replace",
        )
        print(f"Wrote {len(intertie_df):,} configured intertie schedule rows")


def _raw_table_exists(engine, table: str) -> bool:
    return inspect(engine).has_table(table, schema="raw")


def _run_sample_ingestion(engine, sample_cfg, feature_cfg) -> None:
    load_df, price_df = generate_sample_aeso_data(
        start_utc=sample_cfg.start_utc,
        days=sample_cfg.days,
        seed=sample_cfg.seed,
    )
    load_path, price_path = write_sample_csvs(load_df, price_df)

    _write_raw_tables(engine, RawAesoTables(load=load_df, price=price_df))

    print(f"Wrote {len(load_df):,} load rows to raw.raw_aeso_load")
    print(f"Wrote {len(price_df):,} pool price rows to raw.raw_aeso_pool_price")
    print(f"Sample CSVs: {load_path} and {price_path}")

    if feature_cfg.write_sample_feature_sources:
        weather_df, generation_df, intertie_df = generate_sample_known_as_of_feature_data(
            load_df,
            seed=sample_cfg.seed,
        )
        _write_known_as_of_feature_tables(
            engine,
            RawKnownAsOfFeatureTables(
                weather_forecasts=weather_df,
                generation_availability=generation_df,
                intertie_schedules=intertie_df,
            ),
        )
        print(f"Wrote {len(weather_df):,} weather forecast rows")
        print(f"Wrote {len(generation_df):,} generation availability rows")
        print(f"Wrote {len(intertie_df):,} intertie schedule rows")

    _write_configured_optional_feature_csvs(engine, feature_cfg)


def _run_historical_csv_ingestion(engine, feature_cfg) -> None:
    tables = read_historical_csv_tables()
    _write_raw_tables(engine, tables)

    start_utc = tables.load["interval_start_utc"].min()
    end_utc = tables.load["interval_start_utc"].max()
    print(f"Wrote {len(tables.load):,} load rows to raw.raw_aeso_load")
    print(f"Wrote {len(tables.price):,} pool price rows to raw.raw_aeso_pool_price")
    print(f"Historical AESO CSV range: {start_utc} to {end_utc}")
    _write_configured_optional_feature_csvs(engine, feature_cfg)

    optional_raw_tables = (
        "raw_weather_forecast_hourly",
        "raw_generation_availability_hourly",
        "raw_intertie_schedule_hourly",
    )
    missing_tables = [table for table in optional_raw_tables if not _raw_table_exists(engine, table)]
    if missing_tables:
        missing = ", ".join(missing_tables)
        print(f"Optional known-as-of raw tables are not loaded: {missing}")


def run_ingestion() -> None:
    ingest_cfg = get_ingestion_config()
    if ingest_cfg.data_source not in SUPPORTED_DATA_SOURCES:
        supported = ", ".join(sorted(SUPPORTED_DATA_SOURCES))
        raise SystemExit(
            f"Unsupported AESO_DATA_SOURCE={ingest_cfg.data_source!r}. "
            f"Use one of: {supported}."
        )

    engine = get_engine()
    ensure_schemas(engine)

    if ingest_cfg.data_source == "sample":
        _run_sample_ingestion(engine, ingest_cfg.sample_data, ingest_cfg.external_features)
    elif ingest_cfg.data_source == "historical_csv":
        _run_historical_csv_ingestion(engine, ingest_cfg.external_features)


def main() -> None:
    try:
        run_ingestion()
    except OperationalError as exc:
        raise SystemExit(
            "Could not connect to PostgreSQL. Run `make setup` or `make postgres-up`, "
            "then retry `make ingest`."
        ) from exc


if __name__ == "__main__":
    main()
