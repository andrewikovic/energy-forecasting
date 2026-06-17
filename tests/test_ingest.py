import pandas as pd

from aeso_analytics.ingest import (
    GENERATION_AVAILABILITY_RAW_COLUMNS,
    INTERTIE_SCHEDULE_RAW_COLUMNS,
    LOAD_RAW_COLUMNS,
    PRICE_RAW_COLUMNS,
    WEATHER_FORECAST_RAW_COLUMNS,
    transform_generation_availability_data,
    transform_historical_csv_data,
    transform_intertie_schedule_data,
    transform_weather_forecast_data,
)


def test_transform_historical_csv_data_produces_raw_tables_and_drops_bad_rows():
    ingested_at = pd.Timestamp("2026-01-02T03:04:05Z")
    frame = pd.DataFrame(
        {
            "Date_Begin_GMT": [
                "2020-01-01 7:00",
                "1/1/2020 8:00",
                "not-a-date",
                "2020-01-01 10:00",
                "2020-01-01 11:00",
            ],
            "ACTUAL_AIL": ["9,467", "9361", "9200", None, "bad-load"],
            "ACTUAL_POOL_PRICE": ["30.24", "29.43", "28.19", "30.00", None],
        }
    )

    tables = transform_historical_csv_data(
        frame,
        source="test_historical_csv",
        ingested_at=ingested_at,
    )

    assert tuple(tables.load.columns) == LOAD_RAW_COLUMNS
    assert tuple(tables.price.columns) == PRICE_RAW_COLUMNS
    assert len(tables.load) == 2
    assert len(tables.price) == 2
    assert tables.load["interval_start_utc"].tolist() == [
        pd.Timestamp("2020-01-01T07:00:00Z"),
        pd.Timestamp("2020-01-01T08:00:00Z"),
    ]
    assert str(tables.load["interval_start_utc"].dt.tz) == "UTC"
    assert tables.load["alberta_internal_load_mw"].tolist() == [9467.0, 9361.0]
    assert tables.price["pool_price_cad_mwh"].tolist() == [30.24, 29.43]
    assert tables.load["source"].tolist() == ["test_historical_csv", "test_historical_csv"]
    assert tables.price["ingested_at"].tolist() == [ingested_at, ingested_at]


def test_transform_historical_csv_data_requires_expected_columns():
    frame = pd.DataFrame({"Date_Begin_GMT": ["2020-01-01 7:00"]})

    try:
        transform_historical_csv_data(frame)
    except ValueError as exc:
        assert "ACTUAL_AIL" in str(exc)
        assert "ACTUAL_POOL_PRICE" in str(exc)
    else:
        raise AssertionError("Expected missing historical CSV columns to raise ValueError")


def test_transform_weather_forecast_data_preserves_known_as_of_contract_and_nulls():
    ingested_at = pd.Timestamp("2026-01-02T03:04:05Z")
    frame = pd.DataFrame(
        {
            "forecast_issue_utc": ["2024-01-01 00:00", "bad-date"],
            "forecast_target_utc": ["2024-01-02 00:00", "2024-01-02 01:00"],
            "region": ["Calgary", "Edmonton"],
            "temperature_c": ["-12.5", "not-a-number"],
            "wind_speed_mps": ["5.2", "6.1"],
        }
    )

    weather = transform_weather_forecast_data(frame, ingested_at=ingested_at)

    assert tuple(weather.columns) == WEATHER_FORECAST_RAW_COLUMNS
    assert len(weather) == 1
    assert weather["forecast_issue_utc"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert weather["forecast_target_utc"].iloc[0] == pd.Timestamp("2024-01-02T00:00:00Z")
    assert weather["region"].iloc[0] == "calgary"
    assert weather["temperature_c"].iloc[0] == -12.5
    assert pd.isna(weather["relative_humidity_pct"].iloc[0])
    assert weather["ingested_at"].iloc[0] == ingested_at


def test_transform_generation_availability_data_supports_unit_or_fuel_grain():
    frame = pd.DataFrame(
        {
            "availability_issue_utc": ["2024-01-01T00:00:00Z"],
            "availability_target_utc": ["2024-01-02T00:00:00Z"],
            "fuel_type": ["Gas"],
            "available_capacity_mw": ["1,234.5"],
            "outage_capacity_mw": ["50"],
        }
    )

    availability = transform_generation_availability_data(frame)

    assert tuple(availability.columns) == GENERATION_AVAILABILITY_RAW_COLUMNS
    assert availability["fuel_type"].tolist() == ["gas"]
    assert availability["unit_id"].tolist() == ["unreported"]
    assert availability["available_capacity_mw"].tolist() == [1234.5]
    assert availability["outage_capacity_mw"].tolist() == [50.0]


def test_transform_intertie_schedule_data_normalizes_schedule_columns():
    frame = pd.DataFrame(
        {
            "schedule_issue_utc": ["2024-01-01T00:00:00Z"],
            "schedule_target_utc": ["2024-01-02T00:00:00Z"],
            "intertie_id": ["BC"],
            "scheduled_import_mw": ["100"],
            "scheduled_export_mw": ["25"],
            "transfer_limit_import_mw": ["1,200"],
            "transfer_limit_export_mw": ["900"],
            "constraint_mw": [None],
        }
    )

    intertie = transform_intertie_schedule_data(frame)

    assert tuple(intertie.columns) == INTERTIE_SCHEDULE_RAW_COLUMNS
    assert intertie["intertie_id"].tolist() == ["bc"]
    assert intertie["scheduled_import_mw"].tolist() == [100.0]
    assert intertie["transfer_limit_import_mw"].tolist() == [1200.0]
    assert pd.isna(intertie["constraint_mw"].iloc[0])
