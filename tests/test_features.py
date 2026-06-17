import pandas as pd

from aeso_analytics.features import build_horizon_training_frame


def hourly_fixture(rows: int = 240) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "load_mw": range(rows),
            "pool_price_cad_mwh": [100 + i for i in range(rows)],
        }
    )


def test_one_hour_horizon_uses_latest_safe_lags_and_rolling_window():
    frame = build_horizon_training_frame(hourly_fixture(), horizon_hours=1)
    row = frame.loc[frame["target_timestamp_utc"] == pd.Timestamp("2024-01-09 12:00:00Z")].iloc[0]
    target_index = 8 * 24 + 12

    assert row["feature_as_of_timestamp_utc"] == pd.Timestamp("2024-01-09 11:00:00Z")
    assert row["load_lag_1h"] == target_index - 1
    assert row["load_lag_24h"] == target_index - 24
    assert row["load_lag_168h"] == target_index - 168
    assert row["load_rolling_24h_avg"] == sum(range(target_index - 24, target_index)) / 24
    assert row["price_lag_24h"] == 100 + target_index - 24


def test_twenty_four_hour_horizon_nulls_lag_1h_to_prevent_leakage():
    frame = build_horizon_training_frame(hourly_fixture(), horizon_hours=24)
    row = frame.loc[frame["target_timestamp_utc"] == pd.Timestamp("2024-01-09 12:00:00Z")].iloc[0]
    target_index = 8 * 24 + 12

    assert row["feature_as_of_timestamp_utc"] == pd.Timestamp("2024-01-08 12:00:00Z")
    assert pd.isna(row["load_lag_1h"])
    assert row["load_lag_24h"] == target_index - 24
    assert row["load_lag_168h"] == target_index - 168
    assert row["load_rolling_24h_avg"] == sum(range(target_index - 47, target_index - 23)) / 24
    assert row["price_lag_24h"] == 100 + target_index - 24


def test_horizon_features_have_no_future_source_timestamp():
    for horizon in (1, 24):
        frame = build_horizon_training_frame(hourly_fixture(), horizon_hours=horizon)
        valid = frame.dropna(subset=["load_rolling_24h_avg"])
        assert (
            valid["feature_max_source_timestamp_utc"] <= valid["feature_as_of_timestamp_utc"]
        ).all()
