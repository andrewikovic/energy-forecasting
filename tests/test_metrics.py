import math

import pandas as pd

from aeso_analytics.metrics import calculate_metrics


def test_calculate_metrics_expected_values():
    actual = pd.Series([100.0, 200.0, 300.0, 400.0])
    predicted = pd.Series([90.0, 210.0, 330.0, 360.0])

    metrics = calculate_metrics(actual, predicted, peak_threshold=300.0)

    assert metrics["mae"] == 22.5
    assert math.isclose(metrics["rmse"], math.sqrt((100 + 100 + 900 + 1600) / 4))
    assert math.isclose(metrics["mape"], (0.10 + 0.05 + 0.10 + 0.10) / 4 * 100)
    assert metrics["mean_error"] == -2.5
    assert metrics["median_absolute_error"] == 20.0
    assert metrics["underprediction_rate"] == 0.5
    assert metrics["peak_period_mae"] == 35.0


def test_calculate_metrics_rejects_all_missing_pairs():
    actual = pd.Series([float("nan")])
    predicted = pd.Series([float("nan")])

    try:
        calculate_metrics(actual, predicted)
    except ValueError as exc:
        assert "No finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for all-missing metric inputs")
