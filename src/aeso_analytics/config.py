from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SAMPLE_START_UTC = "2024-01-01T00:00:00Z"
DEFAULT_SAMPLE_DAYS = 240
DEFAULT_SAMPLE_SEED = 42


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class WarehouseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

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
    use_sample_data: bool
    start_utc: str
    days: int
    seed: int


@dataclass(frozen=True)
class ExternalFeatureConfig:
    weather_forecast_csv_path: str | None
    generation_availability_csv_path: str | None
    intertie_schedule_csv_path: str | None
    write_sample_feature_sources: bool


@dataclass(frozen=True)
class IngestionConfig:
    data_source: str
    sample_data: SampleDataConfig
    external_features: ExternalFeatureConfig


def _default_sample_data_config() -> SampleDataConfig:
    return SampleDataConfig(
        use_sample_data=True,
        start_utc=DEFAULT_SAMPLE_START_UTC,
        days=DEFAULT_SAMPLE_DAYS,
        seed=DEFAULT_SAMPLE_SEED,
    )


def get_warehouse_config() -> WarehouseConfig:
    return WarehouseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=_env_int("POSTGRES_PORT", 5432),
        database=os.getenv("POSTGRES_DB", "aeso_analytics"),
        user=os.getenv("POSTGRES_USER", "aeso"),
        password=os.getenv("POSTGRES_PASSWORD", "aeso"),
    )


def get_sample_data_config() -> SampleDataConfig:
    return SampleDataConfig(
        use_sample_data=_env_bool("AESO_USE_SAMPLE_DATA", True),
        start_utc=os.getenv("SAMPLE_START_UTC", DEFAULT_SAMPLE_START_UTC),
        days=_env_int("SAMPLE_DAYS", DEFAULT_SAMPLE_DAYS),
        seed=_env_int("SAMPLE_SEED", DEFAULT_SAMPLE_SEED),
    )


def _optional_path_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def get_external_feature_config() -> ExternalFeatureConfig:
    return ExternalFeatureConfig(
        weather_forecast_csv_path=_optional_path_env("WEATHER_FORECAST_CSV_PATH"),
        generation_availability_csv_path=_optional_path_env("GENERATION_AVAILABILITY_CSV_PATH"),
        intertie_schedule_csv_path=_optional_path_env("INTERTIE_SCHEDULE_CSV_PATH"),
        write_sample_feature_sources=_env_bool("AESO_WRITE_SAMPLE_FEATURE_SOURCES", True),
    )


def get_ingestion_config() -> IngestionConfig:
    data_source = os.getenv("AESO_DATA_SOURCE")
    external_features = get_external_feature_config()
    if data_source is not None and data_source.strip():
        normalized_source = data_source.strip().lower()
        sample_cfg = (
            get_sample_data_config()
            if normalized_source == "sample"
            else _default_sample_data_config()
        )
        return IngestionConfig(
            data_source=normalized_source,
            sample_data=sample_cfg,
            external_features=external_features,
        )

    sample_cfg = get_sample_data_config()
    return IngestionConfig(
        data_source="sample" if sample_cfg.use_sample_data else "historical_csv",
        sample_data=sample_cfg,
        external_features=external_features,
    )
