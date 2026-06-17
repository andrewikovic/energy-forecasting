import pandas as pd

from aeso_analytics.dashboard_governance import (
    classify_feature_name,
    current_champions,
    holiday_error_summary_from_rows,
    tag_feature_importance,
)


def test_current_champions_selects_latest_preferred_model_by_horizon():
    registry = pd.DataFrame(
        [
            {
                "model_id": "old-h1",
                "model_name": "baseline_same_hour_last_week",
                "horizon_hours": 1,
                "is_preferred": "true",
                "preferred_at": "2024-01-01T00:00:00Z",
            },
            {
                "model_id": "new-h1",
                "model_name": "xgboost_regressor",
                "horizon_hours": 1,
                "is_preferred": True,
                "preferred_at": "2024-02-01T00:00:00Z",
            },
            {
                "model_id": "h24",
                "model_name": "baseline_rolling_24h_average",
                "horizon_hours": 24,
                "is_preferred": "1",
                "preferred_at": "2024-01-15T00:00:00Z",
            },
        ]
    )

    champions = current_champions(registry).sort_values("horizon_hours")

    assert champions["model_id"].tolist() == ["new-h1", "h24"]


def test_feature_importance_tags_weekly_baseline_and_holiday_features():
    importance = tag_feature_importance(
        pd.DataFrame(
            {
                "feature_name": [
                    "load_lag_168h",
                    "is_stat_holiday",
                    "forecast_temperature_c",
                ],
                "importance": [0.4, 0.2, 0.1],
            }
        )
    )

    assert classify_feature_name("baseline_same_hour_last_week") == "Weekly baseline"
    assert importance["feature_group"].tolist() == [
        "Weekly baseline",
        "Holiday / calendar",
        "Other",
    ]


def test_holiday_error_summary_uses_existing_absolute_error():
    forecasts = pd.DataFrame(
        {
            "horizon_hours": [1, 1, 1],
            "model_name": ["xgboost", "xgboost", "xgboost"],
            "evaluation_split": ["test", "test", "test"],
            "absolute_error_mw": [10.0, 30.0, 20.0],
            "is_stat_holiday": [True, False, False],
        }
    )

    summary = holiday_error_summary_from_rows(forecasts, "is_stat_holiday")

    holiday = summary[summary["holiday_group"] == "Holiday"].iloc[0]
    non_holiday = summary[summary["holiday_group"] == "Non-holiday"].iloc[0]
    assert holiday["mae"] == 10.0
    assert holiday["row_count"] == 1
    assert non_holiday["mae"] == 25.0
    assert non_holiday["row_count"] == 2


def test_holiday_error_summary_can_compute_from_actual_and_predicted():
    forecasts = pd.DataFrame(
        {
            "target_load_mw": [100.0, 100.0],
            "predicted_load_mw": [90.0, 130.0],
            "is_holiday": ["yes", "no"],
        }
    )

    summary = holiday_error_summary_from_rows(forecasts, "is_holiday")

    assert summary.set_index("holiday_group")["mae"].to_dict() == {
        "Holiday": 10.0,
        "Non-holiday": 30.0,
    }
