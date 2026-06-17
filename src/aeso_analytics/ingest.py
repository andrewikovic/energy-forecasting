from __future__ import annotations

import sys

from sqlalchemy.exc import OperationalError

from aeso_analytics.config import get_sample_data_config
from aeso_analytics.database import ensure_schemas, get_engine, write_dataframe
from aeso_analytics.sample_data import generate_sample_aeso_data, write_sample_csvs


def run_ingestion() -> None:
    sample_cfg = get_sample_data_config()
    if not sample_cfg.use_sample_data:
        print(
            "Live AESO ingestion is not configured in this MVP. "
            "Set AESO_USE_SAMPLE_DATA=true or extend aeso_analytics.ingest."
        )
        sys.exit(2)

    engine = get_engine()
    ensure_schemas(engine)

    load_df, price_df = generate_sample_aeso_data(
        start_utc=sample_cfg.start_utc,
        days=sample_cfg.days,
        seed=sample_cfg.seed,
    )
    load_path, price_path = write_sample_csvs(load_df, price_df)

    write_dataframe(engine, load_df, "raw", "raw_aeso_load", if_exists="replace")
    write_dataframe(engine, price_df, "raw", "raw_aeso_pool_price", if_exists="replace")

    print(f"Wrote {len(load_df):,} load rows to raw.raw_aeso_load")
    print(f"Wrote {len(price_df):,} pool price rows to raw.raw_aeso_pool_price")
    print(f"Sample CSVs: {load_path} and {price_path}")


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
