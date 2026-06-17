from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    peak_threshold: float | None = None,
) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    if not mask.any():
        raise ValueError("No finite actual/predicted pairs available for metric calculation")

    actual_arr = actual_arr[mask]
    predicted_arr = predicted_arr[mask]
    error = predicted_arr - actual_arr
    abs_error = np.abs(error)

    non_zero_actual = actual_arr != 0
    if non_zero_actual.any():
        mape = float(np.mean(abs_error[non_zero_actual] / actual_arr[non_zero_actual]) * 100)
    else:
        mape = float("nan")

    if peak_threshold is None:
        peak_threshold = float(np.quantile(actual_arr, 0.9))
    peak_mask = actual_arr >= peak_threshold
    peak_mae = float(np.mean(abs_error[peak_mask])) if peak_mask.any() else float("nan")

    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "mean_error": float(np.mean(error)),
        "median_absolute_error": float(np.median(abs_error)),
        "underprediction_rate": float(np.mean(predicted_arr < actual_arr)),
        "peak_period_mae": peak_mae,
    }
