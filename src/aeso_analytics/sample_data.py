from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aeso_analytics.config import PROJECT_ROOT


def generate_sample_aeso_data(
    start_utc: str = "2024-01-01T00:00:00Z",
    days: int = 240,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic AESO-style hourly load and pool price fixtures."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start_utc, periods=days * 24, freq="h", tz="UTC")
    n = len(timestamps)

    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    is_weekend = day_of_week >= 5

    morning_peak = 520 * np.exp(-((hour - 8) / 3.0) ** 2)
    evening_peak = 980 * np.exp(-((hour - 18) / 4.2) ** 2)
    overnight_trough = -430 * np.exp(-((hour - 3) / 3.8) ** 2)
    winter_shape = 780 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    shoulder_shape = 220 * np.sin(4 * np.pi * day_of_year / 365.25)
    weekend_effect = np.where(is_weekend, -480, 0)
    slow_growth = np.linspace(0, 180, n)
    noise = rng.normal(0, 165, n)

    load_mw = (
        9650
        + morning_peak
        + evening_peak
        + overnight_trough
        + winter_shape
        + shoulder_shape
        + weekend_effect
        + slow_growth
        + noise
    )
    load_mw = np.maximum(load_mw, 6500).round(1)

    load_pressure = (load_mw - np.median(load_mw)) / np.std(load_mw)
    scarcity = np.clip(load_pressure - 1.2, 0, None)
    random_spikes = rng.binomial(1, 0.015, n) * rng.gamma(shape=3.0, scale=35.0, size=n)
    price_noise = rng.normal(0, 11, n)
    pool_price = 72 + 18 * load_pressure + 45 * scarcity + random_spikes + price_noise
    pool_price = np.clip(pool_price, 5, None).round(2)

    ingested_at = pd.Timestamp.now(tz="UTC")
    load_df = pd.DataFrame(
        {
            "interval_start_utc": timestamps,
            "alberta_internal_load_mw": load_mw,
            "source": "sample_fixture",
            "ingested_at": ingested_at,
        }
    )
    price_df = pd.DataFrame(
        {
            "interval_start_utc": timestamps,
            "pool_price_cad_mwh": pool_price,
            "source": "sample_fixture",
            "ingested_at": ingested_at,
        }
    )
    return load_df, price_df


def write_sample_csvs(
    load_df: pd.DataFrame,
    price_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    output_dir = output_dir or PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    load_path = output_dir / "aeso_load_sample.csv"
    price_path = output_dir / "aeso_pool_price_sample.csv"
    load_df.to_csv(load_path, index=False)
    price_df.to_csv(price_path, index=False)
    return load_path, price_path


def generate_sample_known_as_of_feature_data(
    load_df: pd.DataFrame,
    seed: int = 42,
    horizons: tuple[int, ...] = (1, 24),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate deterministic known-as-of optional feature fixtures.

    Rows are forecast/schedule snapshots keyed by issue time and target time.
    They intentionally do not include future observed weather, final outage
    actuals, or finalized intertie actual flows.
    """
    rng = np.random.default_rng(seed + 10_000)
    timestamps = pd.to_datetime(load_df["interval_start_utc"], utc=True).drop_duplicates()
    timestamps = timestamps.sort_values().reset_index(drop=True)
    timestamp_set = set(timestamps)
    ingested_at = pd.Timestamp.now(tz="UTC")

    weather_regions = {
        "calgary": 1.5,
        "edmonton": -0.5,
        "north": -3.0,
        "south": 2.0,
    }
    weather_rows = []
    for horizon in horizons:
        for target_ts in timestamps:
            issue_ts = target_ts - pd.Timedelta(hours=horizon)
            if issue_ts not in timestamp_set:
                continue
            day_of_year = target_ts.dayofyear
            hour = target_ts.hour
            seasonal_temp = 3 + 18 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
            diurnal_temp = 5 * np.sin(2 * np.pi * (hour - 8) / 24)
            wind_base = 5 + 2 * np.sin(2 * np.pi * (hour + 3) / 24)
            humidity_base = 58 - 0.7 * seasonal_temp
            precip_signal = max(0.0, 1.2 * np.sin(2 * np.pi * day_of_year / 17))
            for region, temp_offset in weather_regions.items():
                forecast_error = rng.normal(0, 0.8 + horizon * 0.03)
                temperature_c = seasonal_temp + diurnal_temp + temp_offset + forecast_error
                wind_speed_mps = max(0, wind_base + rng.normal(0, 0.8))
                humidity_pct = np.clip(humidity_base + rng.normal(0, 8), 15, 100)
                precipitation_mm = max(0, precip_signal + rng.normal(0, 0.25))
                weather_rows.append(
                    {
                        "forecast_issue_utc": issue_ts,
                        "forecast_target_utc": target_ts,
                        "region": region,
                        "temperature_c": round(float(temperature_c), 2),
                        "wind_speed_mps": round(float(wind_speed_mps), 2),
                        "relative_humidity_pct": round(float(humidity_pct), 2),
                        "precipitation_mm": round(float(precipitation_mm), 3),
                        "source": "sample_weather_forecast_fixture",
                        "ingested_at": ingested_at,
                    }
                )

    installed_capacity = {
        "gas": 10_200.0,
        "coal": 2_500.0,
        "hydro": 900.0,
        "wind": 4_500.0,
        "solar": 1_800.0,
        "storage": 250.0,
        "other": 700.0,
    }
    generation_rows = []
    for horizon in horizons:
        for target_ts in timestamps:
            issue_ts = target_ts - pd.Timedelta(hours=horizon)
            if issue_ts not in timestamp_set:
                continue
            day_of_year = target_ts.dayofyear
            hour = target_ts.hour
            for fuel_type, capacity in installed_capacity.items():
                outage_rate = 0.04 + 0.025 * np.sin(2 * np.pi * (day_of_year + horizon) / 91)
                if fuel_type == "wind":
                    availability_factor = 0.45 + 0.15 * np.sin(2 * np.pi * (hour + 4) / 24)
                elif fuel_type == "solar":
                    availability_factor = max(0, np.sin(np.pi * (hour - 6) / 14))
                elif fuel_type == "hydro":
                    availability_factor = 0.78
                elif fuel_type == "storage":
                    availability_factor = 0.86
                else:
                    availability_factor = 1 - outage_rate
                outage_capacity = max(0, capacity * outage_rate + rng.normal(0, capacity * 0.01))
                derated_capacity = max(0, capacity * (1 - availability_factor))
                available_capacity = max(0, capacity - outage_capacity - derated_capacity)
                generation_rows.append(
                    {
                        "availability_issue_utc": issue_ts,
                        "availability_target_utc": target_ts,
                        "fuel_type": fuel_type,
                        "unit_id": f"sample_{fuel_type}_fleet",
                        "available_capacity_mw": round(float(available_capacity), 2),
                        "derated_capacity_mw": round(float(derated_capacity), 2),
                        "outage_capacity_mw": round(float(outage_capacity), 2),
                        "source": "sample_generation_availability_fixture",
                        "ingested_at": ingested_at,
                    }
                )

    intertie_limits = {
        "bc": (1_200.0, 1_000.0),
        "sk": (150.0, 150.0),
        "mt": (300.0, 300.0),
    }
    intertie_rows = []
    for horizon in horizons:
        for target_ts in timestamps:
            issue_ts = target_ts - pd.Timedelta(hours=horizon)
            if issue_ts not in timestamp_set:
                continue
            hour = target_ts.hour
            for intertie_id, (import_limit, export_limit) in intertie_limits.items():
                schedule_shape = np.sin(2 * np.pi * (hour + horizon) / 24)
                scheduled_import = max(0, import_limit * (0.28 + 0.12 * schedule_shape))
                scheduled_export = max(0, export_limit * (0.14 - 0.07 * schedule_shape))
                constraint_mw = max(0, rng.normal(20, 35))
                intertie_rows.append(
                    {
                        "schedule_issue_utc": issue_ts,
                        "schedule_target_utc": target_ts,
                        "intertie_id": intertie_id,
                        "scheduled_import_mw": round(float(scheduled_import), 2),
                        "scheduled_export_mw": round(float(scheduled_export), 2),
                        "transfer_limit_import_mw": round(float(import_limit), 2),
                        "transfer_limit_export_mw": round(float(export_limit), 2),
                        "constraint_mw": round(float(constraint_mw), 2),
                        "source": "sample_intertie_schedule_fixture",
                        "ingested_at": ingested_at,
                    }
                )

    return (
        pd.DataFrame(weather_rows),
        pd.DataFrame(generation_rows),
        pd.DataFrame(intertie_rows),
    )
