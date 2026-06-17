from aeso_analytics.config import get_ingestion_config


def test_get_ingestion_config_defaults_to_sample(monkeypatch):
    monkeypatch.delenv("AESO_DATA_SOURCE", raising=False)
    monkeypatch.delenv("AESO_USE_SAMPLE_DATA", raising=False)
    monkeypatch.delenv("SAMPLE_DAYS", raising=False)
    monkeypatch.delenv("SAMPLE_SEED", raising=False)

    cfg = get_ingestion_config()

    assert cfg.data_source == "sample"
    assert cfg.sample_data.days == 240
    assert cfg.sample_data.seed == 42


def test_get_ingestion_config_supports_legacy_sample_flag(monkeypatch):
    monkeypatch.delenv("AESO_DATA_SOURCE", raising=False)
    monkeypatch.setenv("AESO_USE_SAMPLE_DATA", "false")

    cfg = get_ingestion_config()

    assert cfg.data_source == "historical_csv"


def test_get_ingestion_config_historical_csv_skips_sample_env_parsing(monkeypatch):
    monkeypatch.setenv("AESO_DATA_SOURCE", "historical_csv")
    monkeypatch.setenv("SAMPLE_DAYS", "not-an-int")
    monkeypatch.setenv("SAMPLE_SEED", "not-an-int")

    cfg = get_ingestion_config()

    assert cfg.data_source == "historical_csv"
    assert cfg.sample_data.days == 240
    assert cfg.sample_data.seed == 42


def test_get_ingestion_config_reads_optional_feature_paths(monkeypatch):
    monkeypatch.setenv("AESO_DATA_SOURCE", "sample")
    monkeypatch.setenv("WEATHER_FORECAST_CSV_PATH", "/tmp/weather.csv")
    monkeypatch.setenv("GENERATION_AVAILABILITY_CSV_PATH", "/tmp/availability.csv")
    monkeypatch.setenv("INTERTIE_SCHEDULE_CSV_PATH", "/tmp/intertie.csv")
    monkeypatch.setenv("AESO_WRITE_SAMPLE_FEATURE_SOURCES", "false")

    cfg = get_ingestion_config()

    assert cfg.external_features.weather_forecast_csv_path == "/tmp/weather.csv"
    assert cfg.external_features.generation_availability_csv_path == "/tmp/availability.csv"
    assert cfg.external_features.intertie_schedule_csv_path == "/tmp/intertie.csv"
    assert not cfg.external_features.write_sample_feature_sources


def test_get_ingestion_config_treats_empty_optional_bool_as_default(monkeypatch):
    monkeypatch.setenv("AESO_DATA_SOURCE", "sample")
    monkeypatch.setenv("AESO_WRITE_SAMPLE_FEATURE_SOURCES", "")

    cfg = get_ingestion_config()

    assert cfg.external_features.write_sample_feature_sources
