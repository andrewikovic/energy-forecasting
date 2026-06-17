from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class WarehouseConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "aeso_analytics")
    user: str = os.getenv("POSTGRES_USER", "aeso")
    password: str = os.getenv("POSTGRES_PASSWORD", "aeso")

    @property
    def sqlalchemy_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg2",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )


@dataclass(frozen=True)
class SampleDataConfig:
    use_sample_data: bool = os.getenv("AESO_USE_SAMPLE_DATA", "true").lower() == "true"
    start_utc: str = os.getenv("SAMPLE_START_UTC", "2024-01-01T00:00:00Z")
    days: int = int(os.getenv("SAMPLE_DAYS", "240"))
    seed: int = int(os.getenv("SAMPLE_SEED", "42"))


def get_warehouse_config() -> WarehouseConfig:
    return WarehouseConfig()


def get_sample_data_config() -> SampleDataConfig:
    return SampleDataConfig()
