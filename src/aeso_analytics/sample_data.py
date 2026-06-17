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
