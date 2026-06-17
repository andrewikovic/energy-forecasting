from __future__ import annotations

import pandas as pd


ACCURACY_METRICS = ("mae", "rmse", "mape")
HOLIDAY_FLAG_COLUMNS = (
    "is_holiday",
    "is_stat_holiday",
    "is_alberta_stat_holiday",
    "is_canada_stat_holiday",
    "is_long_weekend",
    "is_non_workday",
)
WEEKLY_BASELINE_FEATURE_NAMES = {
    "baseline_same_hour_last_week",
    "load_lag_168h",
    "load_rolling_168h_avg",
}


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def current_champions(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty or "is_preferred" not in registry.columns:
        return registry.iloc[0:0].copy()

    result = registry.copy()
    result["is_preferred"] = result["is_preferred"].map(truthy)
    preferred = result[result["is_preferred"]].copy()
    if preferred.empty:
        return preferred

    for column in ("preferred_at", "trained_at", "created_at"):
        if column in preferred.columns:
            preferred[column] = pd.to_datetime(preferred[column], utc=True, errors="coerce")

    sort_columns = [
        column
        for column in ("preferred_at", "trained_at", "created_at", "model_id")
        if column in preferred.columns
    ]
    if sort_columns:
        preferred = preferred.sort_values(sort_columns, na_position="first")

    if "horizon_hours" not in preferred.columns:
        return preferred.tail(1)
    return preferred.groupby("horizon_hours", as_index=False).tail(1)


def available_accuracy_metrics(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in ACCURACY_METRICS if metric in frame.columns]


def classify_feature_name(feature_name: object) -> str:
    name = str(feature_name).strip().lower()
    if name in WEEKLY_BASELINE_FEATURE_NAMES or "last_week" in name or "168h" in name:
        return "Weekly baseline"
    if any(token in name for token in ("holiday", "workday", "long_weekend")):
        return "Holiday / calendar"
    return "Other"


def tag_feature_importance(importance: pd.DataFrame) -> pd.DataFrame:
    result = importance.copy()
    if "feature_name" not in result.columns:
        result["feature_group"] = "Other"
        return result
    result["feature_group"] = result["feature_name"].map(classify_feature_name)
    return result


def holiday_error_summary_from_rows(
    forecasts: pd.DataFrame,
    holiday_column: str,
) -> pd.DataFrame:
    output_columns = [
        "horizon_hours",
        "model_name",
        "evaluation_split",
        "holiday_group",
        "mae",
        "row_count",
    ]
    if forecasts.empty or holiday_column not in forecasts.columns:
        return pd.DataFrame(columns=output_columns)

    result = forecasts.copy()
    if "absolute_error_mw" in result.columns:
        result["_absolute_error_mw"] = pd.to_numeric(
            result["absolute_error_mw"],
            errors="coerce",
        )
    elif {"target_load_mw", "predicted_load_mw"}.issubset(result.columns):
        result["_absolute_error_mw"] = (
            pd.to_numeric(result["predicted_load_mw"], errors="coerce")
            - pd.to_numeric(result["target_load_mw"], errors="coerce")
        ).abs()
    else:
        return pd.DataFrame(columns=output_columns)

    group_columns = [
        column
        for column in ("horizon_hours", "model_name", "evaluation_split")
        if column in result.columns
    ]
    result["holiday_group"] = result[holiday_column].map(
        lambda value: "Holiday" if truthy(value) else "Non-holiday"
    )
    summary = (
        result.dropna(subset=["_absolute_error_mw"])
        .groupby([*group_columns, "holiday_group"], dropna=False)
        .agg(mae=("_absolute_error_mw", "mean"), row_count=("_absolute_error_mw", "size"))
        .reset_index()
    )
    for column in output_columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary[output_columns]
