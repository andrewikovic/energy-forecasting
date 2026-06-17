from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sqlalchemy.exc import OperationalError

from aeso_analytics.config import get_ingestion_config
from aeso_analytics.database import ensure_schemas, get_engine, write_dataframe
from aeso_analytics.sample_data import generate_sample_aeso_data, write_sample_csvs


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
SUPPORTED_DATA_SOURCES = {"sample", "historical_csv"}


@dataclass(frozen=True)
class RawAesoTables:
    load: pd.DataFrame
    price: pd.DataFrame


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


def _run_sample_ingestion(engine, sample_cfg) -> None:
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


def _run_historical_csv_ingestion(engine) -> None:
    tables = read_historical_csv_tables()
    _write_raw_tables(engine, tables)

    start_utc = tables.load["interval_start_utc"].min()
    end_utc = tables.load["interval_start_utc"].max()
    print(f"Wrote {len(tables.load):,} load rows to raw.raw_aeso_load")
    print(f"Wrote {len(tables.price):,} pool price rows to raw.raw_aeso_pool_price")
    print(f"Historical AESO CSV range: {start_utc} to {end_utc}")


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
        _run_sample_ingestion(engine, ingest_cfg.sample_data)
    elif ingest_cfg.data_source == "historical_csv":
        _run_historical_csv_ingestion(engine)


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
