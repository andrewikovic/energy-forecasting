import pandas as pd

from aeso_analytics.ingest import (
    LOAD_RAW_COLUMNS,
    PRICE_RAW_COLUMNS,
    transform_historical_csv_data,
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
