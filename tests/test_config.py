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
