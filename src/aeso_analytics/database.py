from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import Engine, create_engine, inspect, text

from aeso_analytics.config import WarehouseConfig, get_warehouse_config


WAREHOUSE_SCHEMAS = ("raw", "staging", "intermediate", "marts", "ml")


def get_engine(config: WarehouseConfig | None = None) -> Engine:
    cfg = config or get_warehouse_config()
    return create_engine(cfg.sqlalchemy_url, pool_pre_ping=True, future=True)


def ensure_schemas(engine: Engine, schemas: Iterable[str] = WAREHOUSE_SCHEMAS) -> None:
    with engine.begin() as conn:
        for schema in schemas:
            conn.execute(text(f'create schema if not exists "{schema}"'))


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def write_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    if_exists: str = "replace",
) -> None:
    if if_exists == "replace" and inspect(engine).has_table(table, schema=schema):
        qualified_table = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
        with engine.begin() as conn:
            conn.execute(text(f"truncate table {qualified_table}"))
        if_exists = "append"

    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )
