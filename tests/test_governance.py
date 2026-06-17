from datetime import datetime, timezone

import pandas as pd

from aeso_analytics.config import BacktestConfig, PromotionRuleConfig
from aeso_analytics.governance import (
    aggregate_backtest_metrics,
    apply_promotion_decisions,
    decide_promotions,
    evaluate_challenger_promotion,
    generate_backtest_windows,
)


def promotion_rules(minimum_valid_windows: int = 3) -> PromotionRuleConfig:
    return PromotionRuleConfig(
        primary_metric="mae",
        minimum_relative_improvement=0.0,
        max_peak_mae_regression=0.05,
        max_underprediction_rate_regression=0.02,
        minimum_win_rate=0.5,
        minimum_valid_windows=minimum_valid_windows,
    )


def registry_row(model_id: str, model_name: str, is_preferred: bool = False) -> dict:
    return {
        "model_id": model_id,
        "model_name": model_name,
        "horizon_hours": 1,
        "model_type": "baseline",
        "artifact_path": None,
        "feature_names": "[]",
        "parameters": "{}",
        "trained_at": pd.Timestamp("2024-01-01T00:00:00Z"),
        "train_start_timestamp_utc": pd.Timestamp("2024-01-01T00:00:00Z"),
        "train_end_timestamp_utc": pd.Timestamp("2024-03-31T23:00:00Z"),
        "is_preferred": is_preferred,
        "preferred_at": pd.Timestamp("2024-04-01T00:00:00Z") if is_preferred else pd.NaT,
    }


def metric_rows(
    model_id: str,
    model_name: str,
    mae_values: list[float],
    peak_values: list[float] | None = None,
    underprediction_values: list[float] | None = None,
) -> pd.DataFrame:
    peak_values = peak_values or [100.0] * len(mae_values)
    underprediction_values = underprediction_values or [0.20] * len(mae_values)
    return pd.DataFrame(
        {
            "model_id": model_id,
            "model_name": model_name,
            "horizon_hours": 1,
            "window_id": [f"w{i}" for i in range(1, len(mae_values) + 1)],
            "mae": mae_values,
            "rmse": [value * 1.2 for value in mae_values],
            "mape": [1.0] * len(mae_values),
            "mean_error": [0.0] * len(mae_values),
            "median_absolute_error": mae_values,
            "peak_period_mae": peak_values,
            "underprediction_rate": underprediction_values,
        }
    )


def test_generate_backtest_windows_supports_expanding_and_rolling_months():
    timestamps = pd.date_range("2024-01-01", "2024-06-30 23:00", freq="h", tz="UTC")
    df = pd.DataFrame({"target_timestamp_utc": timestamps})

    expanding = BacktestConfig(
        enabled=True,
        mode="expanding",
        min_train_days=60,
        rolling_train_days=90,
        min_test_days=25,
        min_train_rows=24,
        min_test_rows=24,
        max_windows=None,
    )
    expanding_windows = generate_backtest_windows(df, expanding)

    assert expanding_windows[0].train_start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert expanding_windows[0].train_end == pd.Timestamp("2024-02-29T23:00:00Z")
    assert expanding_windows[0].test_start == pd.Timestamp("2024-03-01T00:00:00Z")
    assert expanding_windows[1].train_start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert expanding_windows[1].test_start == pd.Timestamp("2024-04-01T00:00:00Z")

    rolling = BacktestConfig(
        enabled=True,
        mode="rolling",
        min_train_days=30,
        rolling_train_days=31,
        min_test_days=25,
        min_train_rows=24,
        min_test_rows=24,
        max_windows=None,
    )
    rolling_windows = generate_backtest_windows(df, rolling)

    assert rolling_windows[0].test_start == pd.Timestamp("2024-02-01T00:00:00Z")
    assert rolling_windows[1].test_start == pd.Timestamp("2024-03-01T00:00:00Z")
    assert rolling_windows[1].train_start == pd.Timestamp("2024-01-30T00:00:00Z")


def test_aggregate_backtest_metrics_averages_per_window_results():
    results = pd.concat(
        [
            metric_rows("candidate", "candidate_model", [10.0, 14.0, 18.0]),
            metric_rows("other", "other_model", [20.0, 22.0]),
        ],
        ignore_index=True,
    )

    summary = aggregate_backtest_metrics(results)
    candidate = summary[summary["model_id"] == "candidate"].iloc[0]

    assert candidate["valid_window_count"] == 3
    assert candidate["average_mae"] == 14.0
    assert candidate["average_rmse"] == 16.8


def test_challenger_promoted_when_rules_pass():
    existing = pd.DataFrame([registry_row("champion", "champion_model", True)])
    candidate = pd.DataFrame([registry_row("challenger", "challenger_model", False)])
    backtests = pd.concat(
        [
            metric_rows("champion", "champion_model", [100.0, 110.0, 120.0]),
            metric_rows(
                "challenger",
                "challenger_model",
                [90.0, 105.0, 119.0],
                peak_values=[101.0, 100.0, 99.0],
                underprediction_values=[0.20, 0.21, 0.20],
            ),
        ],
        ignore_index=True,
    )

    decisions, promoted = decide_promotions(
        existing,
        candidate,
        backtests,
        promotion_rules(),
        "run-1",
        datetime.now(timezone.utc),
    )
    updated = apply_promotion_decisions(
        pd.concat([existing, candidate], ignore_index=True),
        decisions,
        datetime.now(timezone.utc),
    )

    assert promoted == {1: "challenger"}
    assert decisions.iloc[0]["decision"] == "promoted"
    assert updated.loc[updated["model_id"] == "challenger", "is_preferred"].iloc[0]
    assert not updated.loc[updated["model_id"] == "champion", "is_preferred"].iloc[0]


def test_challenger_rejected_when_promotion_rules_fail():
    detail = evaluate_challenger_promotion(
        metric_rows("champion", "champion_model", [100.0, 100.0, 100.0]),
        metric_rows(
            "challenger",
            "challenger_model",
            [90.0, 90.0, 90.0],
            peak_values=[130.0, 130.0, 130.0],
        ),
        promotion_rules(),
    )

    assert not detail["passed"]
    assert "peak-period MAE regression exceeded tolerance" in detail["reason"]


def test_no_registry_update_when_backtest_windows_are_insufficient():
    existing = pd.DataFrame([registry_row("champion", "champion_model", True)])
    candidate = pd.DataFrame([registry_row("challenger", "challenger_model", False)])
    backtests = pd.concat(
        [
            metric_rows("champion", "champion_model", [100.0, 100.0]),
            metric_rows("challenger", "challenger_model", [80.0, 80.0]),
        ],
        ignore_index=True,
    )

    decisions, promoted = decide_promotions(
        existing,
        candidate,
        backtests,
        promotion_rules(minimum_valid_windows=3),
        "run-1",
        datetime.now(timezone.utc),
    )
    updated = apply_promotion_decisions(
        pd.concat([existing, candidate], ignore_index=True),
        decisions,
        datetime.now(timezone.utc),
    )

    assert promoted == {}
    assert decisions.iloc[0]["decision"] == "rejected"
    assert "comparable backtest windows" in decisions.iloc[0]["reason"]
    assert updated.loc[updated["model_id"] == "champion", "is_preferred"].iloc[0]
    assert not updated.loc[updated["model_id"] == "challenger", "is_preferred"].iloc[0]
