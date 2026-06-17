from __future__ import annotations

import numpy as np
import pandas as pd

from aeso_analytics.holidays import holiday_flags

BASE_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
    "load_rolling_24h_avg",
    "load_rolling_168h_avg",
    "price_lag_24h",
    "price_rolling_24h_avg",
]

HOLIDAY_FEATURE_COLUMNS = [
    "is_stat_holiday",
    "is_alberta_stat_holiday",
    "is_canada_stat_holiday",
    "is_long_weekend",
    "is_workday",
    "is_non_workday",
]

WEATHER_FORECAST_FEATURE_COLUMNS = [
    "weather_forecast_age_hours",
    "weather_forecast_region_count",
    "forecast_temperature_c",
    "forecast_wind_speed_mps",
    "forecast_relative_humidity_pct",
    "forecast_precipitation_mm",
    "forecast_heating_degree_c",
    "forecast_cooling_degree_c",
]

GENERATION_AVAILABILITY_FEATURE_COLUMNS = [
    "generation_availability_age_hours",
    "generation_availability_fuel_type_count",
    "available_capacity_total_mw",
    "outage_capacity_total_mw",
    "derated_capacity_total_mw",
    "available_capacity_gas_mw",
    "available_capacity_coal_mw",
    "available_capacity_hydro_mw",
    "available_capacity_wind_mw",
    "available_capacity_solar_mw",
    "available_capacity_storage_mw",
    "available_capacity_other_mw",
]

INTERTIE_FEATURE_COLUMNS = [
    "intertie_schedule_age_hours",
    "intertie_count",
    "scheduled_import_mw",
    "scheduled_export_mw",
    "net_scheduled_import_mw",
    "import_transfer_limit_mw",
    "export_transfer_limit_mw",
    "intertie_constraint_mw",
    "import_headroom_mw",
    "export_headroom_mw",
]

OPTIONAL_KNOWN_AS_OF_FEATURE_COLUMNS = [
    *WEATHER_FORECAST_FEATURE_COLUMNS,
    *GENERATION_AVAILABILITY_FEATURE_COLUMNS,
    *INTERTIE_FEATURE_COLUMNS,
]

FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *HOLIDAY_FEATURE_COLUMNS,
    *OPTIONAL_KNOWN_AS_OF_FEATURE_COLUMNS,
]


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    result = df.copy()
    ts = pd.to_datetime(result[timestamp_col], utc=True)
    local_ts = ts.dt.tz_convert("America/Edmonton")
    local_dates = list(local_ts.dt.date)
    result["hour"] = ts.dt.hour
    result["day_of_week"] = ts.dt.dayofweek
    result["month"] = ts.dt.month
    result["is_weekend"] = ts.dt.dayofweek >= 5
    for column, values in holiday_flags(local_dates).items():
        result[column] = values
    return result


def build_horizon_training_frame(
    hourly_df: pd.DataFrame,
    horizon_hours: int,
    timestamp_col: str = "timestamp_utc",
    load_col: str = "load_mw",
    price_col: str = "pool_price_cad_mwh",
) -> pd.DataFrame:
    """Build target-time rows using only features known by prediction issue time.

    Calendar features describe the target hour because those are known in advance.
    Load and price lags are populated only when the source timestamp is at or
    before the issue time. Rolling features are calculated as of the issue time.
    """
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be positive")

    required = {timestamp_col, load_col, price_col}
    missing = required.difference(hourly_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base = hourly_df[[timestamp_col, load_col, price_col]].copy()
    base[timestamp_col] = pd.to_datetime(base[timestamp_col], utc=True)
    base = base.sort_values(timestamp_col).drop_duplicates(timestamp_col).reset_index(drop=True)
    base = base.rename(
        columns={
            timestamp_col: "target_timestamp_utc",
            load_col: "target_load_mw",
            price_col: "actual_pool_price_cad_mwh",
        }
    )

    asof = base[["target_timestamp_utc", "target_load_mw", "actual_pool_price_cad_mwh"]].copy()
    asof["load_rolling_24h_avg"] = asof["target_load_mw"].rolling(window=24, min_periods=1).mean()
    asof["load_rolling_168h_avg"] = asof["target_load_mw"].rolling(window=168, min_periods=1).mean()
    asof["price_rolling_24h_avg"] = asof["actual_pool_price_cad_mwh"].rolling(
        window=24, min_periods=1
    ).mean()
    asof = asof.set_index("target_timestamp_utc")

    result = add_calendar_features(base, "target_timestamp_utc")
    result["horizon_hours"] = horizon_hours
    result["feature_as_of_timestamp_utc"] = result["target_timestamp_utc"] - pd.to_timedelta(
        horizon_hours, unit="h"
    )

    load_by_ts = asof["target_load_mw"]
    price_by_ts = asof["actual_pool_price_cad_mwh"]

    for lag in (1, 24, 168):
        col = f"load_lag_{lag}h"
        if lag >= horizon_hours:
            source_ts = result["target_timestamp_utc"] - pd.to_timedelta(lag, unit="h")
            result[col] = source_ts.map(load_by_ts)
        else:
            result[col] = np.nan

    if 24 >= horizon_hours:
        price_lag_source = result["target_timestamp_utc"] - pd.to_timedelta(24, unit="h")
        result["price_lag_24h"] = price_lag_source.map(price_by_ts)
    else:
        result["price_lag_24h"] = np.nan

    result["load_rolling_24h_avg"] = result["feature_as_of_timestamp_utc"].map(
        asof["load_rolling_24h_avg"]
    )
    result["load_rolling_168h_avg"] = result["feature_as_of_timestamp_utc"].map(
        asof["load_rolling_168h_avg"]
    )
    result["price_rolling_24h_avg"] = result["feature_as_of_timestamp_utc"].map(
        asof["price_rolling_24h_avg"]
    )
    for column in OPTIONAL_KNOWN_AS_OF_FEATURE_COLUMNS:
        result[column] = np.nan
    result["feature_max_source_timestamp_utc"] = result["feature_as_of_timestamp_utc"]

    ordered = [
        "horizon_hours",
        "feature_as_of_timestamp_utc",
        "feature_max_source_timestamp_utc",
        "target_timestamp_utc",
        "target_load_mw",
        "actual_pool_price_cad_mwh",
        *FEATURE_COLUMNS,
    ]
    return result[ordered]
