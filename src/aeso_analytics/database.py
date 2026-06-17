from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from aeso_analytics.config import WarehouseConfig, get_warehouse_config


WAREHOUSE_SCHEMAS = ("raw", "staging", "intermediate", "marts", "ml")


def get_engine(config: WarehouseConfig | None = None) -> Engine:
    cfg = config or get_warehouse_config()
    return create_engine(cfg.sqlalchemy_url, pool_pre_ping=True, future=True)


def ensure_schemas(engine: Engine, schemas: Iterable[str] = WAREHOUSE_SCHEMAS) -> None:
    with engine.begin() as conn:
        for schema in schemas:
            conn.execute(text(f'create schema if not exists "{schema}"'))


def write_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    if_exists: str = "replace",
) -> None:
    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )
