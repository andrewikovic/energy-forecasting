from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from aeso_analytics.config import BacktestConfig, PromotionRuleConfig


METRIC_COLUMNS = (
    "mae",
    "rmse",
    "mape",
    "mean_error",
    "median_absolute_error",
    "peak_period_mae",
    "underprediction_rate",
)

REGISTRY_GOVERNANCE_DEFAULTS = {
    "run_id": None,
    "is_preferred": False,
    "preferred_at": pd.NaT,
    "retired_at": pd.NaT,
    "promotion_decision_id": None,
}


@dataclass(frozen=True)
class BacktestWindow:
    window_id: str
    mode: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_row_count: int
    test_row_count: int


def _month_floor(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(
        year=timestamp.year,
        month=timestamp.month,
        day=1,
        tz=timestamp.tz,
    )


def _timestamp_series(df: pd.DataFrame, timestamp_col: str) -> pd.Series:
    timestamps = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    return timestamps.dropna().drop_duplicates().sort_values().reset_index(drop=True)


def _enough_calendar_span(
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_days: int,
) -> bool:
    return end >= start + pd.Timedelta(days=minimum_days) - pd.Timedelta(hours=1)


def generate_backtest_windows(
    df: pd.DataFrame,
    config: BacktestConfig,
    timestamp_col: str = "target_timestamp_utc",
) -> list[BacktestWindow]:
    """Generate month-sized test windows with expanding or rolling train history."""
    timestamps = _timestamp_series(df, timestamp_col)
    if timestamps.empty or not config.enabled:
        return []

    first_timestamp = timestamps.iloc[0]
    last_timestamp = timestamps.iloc[-1]
    month_starts = pd.date_range(
        start=_month_floor(first_timestamp),
        end=_month_floor(last_timestamp),
        freq="MS",
        tz=first_timestamp.tz,
    )

    windows: list[BacktestWindow] = []
    for calendar_test_start in month_starts:
        next_month_start = calendar_test_start + pd.DateOffset(months=1)
        test_timestamps = timestamps[
            (timestamps >= calendar_test_start) & (timestamps < next_month_start)
        ]
        if test_timestamps.empty:
            continue

        actual_test_start = test_timestamps.iloc[0]
        actual_test_end = test_timestamps.iloc[-1]
        if len(test_timestamps) < config.min_test_rows:
            continue
        if not _enough_calendar_span(
            actual_test_start,
            actual_test_end,
            config.min_test_days,
        ):
            continue

        if config.mode == "rolling":
            train_floor = calendar_test_start - pd.Timedelta(days=config.rolling_train_days)
        else:
            train_floor = first_timestamp

        train_timestamps = timestamps[
            (timestamps >= train_floor) & (timestamps < actual_test_start)
        ]
        if train_timestamps.empty:
            continue

        actual_train_start = train_timestamps.iloc[0]
        actual_train_end = train_timestamps.iloc[-1]
        if len(train_timestamps) < config.min_train_rows:
            continue
        if not _enough_calendar_span(
            actual_train_start,
            actual_train_end,
            config.min_train_days,
        ):
            continue

        window_id = (
            f"{config.mode}_"
            f"{actual_test_start.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{actual_test_end.strftime('%Y%m%dT%H%M%SZ')}"
        )
        windows.append(
            BacktestWindow(
                window_id=window_id,
                mode=config.mode,
                train_start=actual_train_start,
                train_end=actual_train_end,
                test_start=actual_test_start,
                test_end=actual_test_end,
                train_row_count=int(len(train_timestamps)),
                test_row_count=int(len(test_timestamps)),
            )
        )

    if config.max_windows is not None:
        return windows[-config.max_windows :]
    return windows


def aggregate_backtest_metrics(
    backtest_results: pd.DataFrame,
    primary_metric: str = "mae",
) -> pd.DataFrame:
    columns = [
        "model_id",
        "model_name",
        "horizon_hours",
        "valid_window_count",
        *[f"average_{metric}" for metric in METRIC_COLUMNS],
    ]
    if backtest_results.empty:
        return pd.DataFrame(columns=columns)

    results = backtest_results.copy()
    if primary_metric not in results.columns:
        return pd.DataFrame(columns=columns)
    if "window_id" not in results.columns:
        results["window_id"] = range(len(results))

    results[primary_metric] = pd.to_numeric(results[primary_metric], errors="coerce")
    valid = results[np.isfinite(results[primary_metric])].copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    aggregations = {"window_id": "nunique"}
    for metric in METRIC_COLUMNS:
        if metric in valid.columns:
            valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
            aggregations[metric] = "mean"

    summary = (
        valid.groupby(["model_id", "model_name", "horizon_hours"], dropna=False)
        .agg(aggregations)
        .rename(columns={"window_id": "valid_window_count"})
        .reset_index()
    )
    for metric in METRIC_COLUMNS:
        if metric in summary.columns:
            summary = summary.rename(columns={metric: f"average_{metric}"})
        else:
            summary[f"average_{metric}"] = np.nan
    return summary[columns]


def normalize_model_registry(registry: pd.DataFrame) -> pd.DataFrame:
    result = registry.copy()
    for column, default in REGISTRY_GOVERNANCE_DEFAULTS.items():
        if column not in result.columns:
            result[column] = default

    if "is_preferred" in result.columns:
        result["is_preferred"] = result["is_preferred"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "t", "yes", "y"}
            if not isinstance(value, bool)
            else value
        )
    for column in ("trained_at", "preferred_at", "retired_at"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    return result


def current_champion_rows(registry: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return registry.copy()

    normalized = normalize_model_registry(registry)
    preferred = normalized[normalized["is_preferred"]].copy()
    if preferred.empty:
        return preferred

    preferred["_preferred_sort"] = pd.to_datetime(
        preferred["preferred_at"],
        utc=True,
        errors="coerce",
    )
    preferred["_trained_sort"] = pd.to_datetime(
        preferred["trained_at"],
        utc=True,
        errors="coerce",
    )
    preferred = preferred.sort_values(
        ["horizon_hours", "_preferred_sort", "_trained_sort", "model_id"],
        na_position="first",
    )
    champions = preferred.groupby("horizon_hours", as_index=False).tail(1)
    return champions.drop(columns=["_preferred_sort", "_trained_sort"])


def _clean_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _mean_metric(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return None
    return float(values.mean())


def _candidate_backtest_summary(
    candidate_results: pd.DataFrame,
    rules: PromotionRuleConfig,
) -> dict[str, object]:
    if rules.primary_metric not in candidate_results.columns:
        candidate_results = candidate_results.copy()
        candidate_results[rules.primary_metric] = np.nan
    if "window_id" not in candidate_results.columns:
        candidate_results = candidate_results.copy()
        candidate_results["window_id"] = pd.Series(dtype=object)
    primary_values = pd.to_numeric(
        candidate_results[rules.primary_metric],
        errors="coerce",
    )
    valid_results = candidate_results[np.isfinite(primary_values)].copy()
    detail: dict[str, object] = {
        "valid_window_count": int(valid_results["window_id"].nunique()) if not valid_results.empty else 0,
        "required_window_count": int(rules.minimum_valid_windows),
        "win_count": None,
        "win_rate": None,
    }
    for metric in METRIC_COLUMNS:
        detail[f"challenger_average_{metric}"] = _mean_metric(valid_results, metric)
        detail[f"champion_average_{metric}"] = None
    return detail


def evaluate_challenger_promotion(
    champion_results: pd.DataFrame,
    challenger_results: pd.DataFrame,
    rules: PromotionRuleConfig,
) -> dict[str, object]:
    champion = champion_results.copy()
    challenger = challenger_results.copy()
    if "window_id" not in champion.columns:
        champion["window_id"] = pd.Series(dtype=object)
    if "window_id" not in challenger.columns:
        challenger["window_id"] = pd.Series(dtype=object)
    for metric in METRIC_COLUMNS:
        if metric not in champion.columns:
            champion[metric] = np.nan
        if metric not in challenger.columns:
            challenger[metric] = np.nan

    merged = champion[["window_id", *METRIC_COLUMNS]].merge(
        challenger[["window_id", *METRIC_COLUMNS]],
        on="window_id",
        suffixes=("_champion", "_challenger"),
    )
    for metric in METRIC_COLUMNS:
        merged[f"{metric}_champion"] = pd.to_numeric(
            merged[f"{metric}_champion"],
            errors="coerce",
        )
        merged[f"{metric}_challenger"] = pd.to_numeric(
            merged[f"{metric}_challenger"],
            errors="coerce",
        )

    primary_champion = f"{rules.primary_metric}_champion"
    primary_challenger = f"{rules.primary_metric}_challenger"
    valid = merged[
        np.isfinite(merged[primary_champion]) & np.isfinite(merged[primary_challenger])
    ].copy()

    detail: dict[str, object] = {
        "valid_window_count": int(len(valid)),
        "required_window_count": int(rules.minimum_valid_windows),
        "win_count": 0,
        "win_rate": 0.0,
    }
    for metric in METRIC_COLUMNS:
        detail[f"champion_average_{metric}"] = _mean_metric(valid, f"{metric}_champion")
        detail[f"challenger_average_{metric}"] = _mean_metric(valid, f"{metric}_challenger")

    failures: list[str] = []
    if len(valid) < rules.minimum_valid_windows:
        failures.append(
            f"only {len(valid)} comparable backtest windows; "
            f"{rules.minimum_valid_windows} required"
        )

    if not failures:
        champion_primary = _clean_number(detail[f"champion_average_{rules.primary_metric}"])
        challenger_primary = _clean_number(detail[f"challenger_average_{rules.primary_metric}"])
        if champion_primary is None or challenger_primary is None:
            failures.append(f"{rules.primary_metric} comparison is unavailable")
        else:
            required_delta = abs(champion_primary) * rules.minimum_relative_improvement
            if challenger_primary >= champion_primary - required_delta:
                failures.append(
                    f"average {rules.primary_metric} did not improve enough "
                    f"({challenger_primary:.6f} vs {champion_primary:.6f})"
                )

        win_count = int((valid[primary_challenger] <= valid[primary_champion]).sum())
        required_wins = int(math.ceil(len(valid) * rules.minimum_win_rate))
        detail["win_count"] = win_count
        detail["win_rate"] = float(win_count / len(valid)) if len(valid) else 0.0
        detail["required_window_count"] = required_wins
        if win_count < required_wins:
            failures.append(
                f"won or tied {win_count} windows; {required_wins} required"
            )

        champion_peak = _clean_number(detail["champion_average_peak_period_mae"])
        challenger_peak = _clean_number(detail["challenger_average_peak_period_mae"])
        if champion_peak is not None and challenger_peak is not None:
            peak_limit = champion_peak * (1 + rules.max_peak_mae_regression)
            if challenger_peak > peak_limit:
                failures.append(
                    "peak-period MAE regression exceeded tolerance "
                    f"({challenger_peak:.6f} > {peak_limit:.6f})"
                )

        champion_under = _clean_number(detail["champion_average_underprediction_rate"])
        challenger_under = _clean_number(detail["challenger_average_underprediction_rate"])
        if champion_under is not None and challenger_under is not None:
            under_limit = champion_under + rules.max_underprediction_rate_regression
            if challenger_under > under_limit:
                failures.append(
                    "underprediction-rate regression exceeded tolerance "
                    f"({challenger_under:.6f} > {under_limit:.6f})"
                )

    passed = not failures
    detail["passed"] = passed
    detail["reason"] = "passed all promotion rules" if passed else "; ".join(failures)
    return detail


def _rules_json(rules: PromotionRuleConfig) -> str:
    return json.dumps(asdict(rules), sort_keys=True)


def _decision_row(
    *,
    run_id: str,
    horizon_hours: int,
    candidate: pd.Series,
    champion: pd.Series | None,
    decision: str,
    reason: str,
    detail: dict[str, object],
    rules: PromotionRuleConfig,
    evaluated_at: datetime,
    promoted_model_id: str | None,
) -> dict[str, object]:
    return {
        "decision_id": str(uuid.uuid4()),
        "run_id": run_id,
        "horizon_hours": int(horizon_hours),
        "champion_model_id": None if champion is None else champion.get("model_id"),
        "champion_model_name": None if champion is None else champion.get("model_name"),
        "challenger_model_id": candidate["model_id"],
        "challenger_model_name": candidate["model_name"],
        "promoted_model_id": promoted_model_id,
        "previous_champion_model_id": None if champion is None else champion.get("model_id"),
        "decision": decision,
        "reason": reason,
        "valid_window_count": detail.get("valid_window_count"),
        "required_window_count": detail.get("required_window_count"),
        "win_count": detail.get("win_count"),
        "win_rate": detail.get("win_rate"),
        "champion_average_mae": detail.get("champion_average_mae"),
        "challenger_average_mae": detail.get("challenger_average_mae"),
        "champion_average_rmse": detail.get("champion_average_rmse"),
        "challenger_average_rmse": detail.get("challenger_average_rmse"),
        "champion_average_peak_period_mae": detail.get("champion_average_peak_period_mae"),
        "challenger_average_peak_period_mae": detail.get("challenger_average_peak_period_mae"),
        "champion_average_underprediction_rate": detail.get(
            "champion_average_underprediction_rate"
        ),
        "challenger_average_underprediction_rate": detail.get(
            "challenger_average_underprediction_rate"
        ),
        "rules": _rules_json(rules),
        "comparison_details": json.dumps(_json_safe(detail), sort_keys=True),
        "evaluated_at": evaluated_at,
    }


def _candidate_sort_key(detail: dict[str, object], candidate: pd.Series, rules: PromotionRuleConfig) -> tuple:
    primary = detail.get(f"challenger_average_{rules.primary_metric}")
    rmse = detail.get("challenger_average_rmse")
    return (
        float("inf") if primary is None else float(primary),
        float("inf") if rmse is None else float(rmse),
        str(candidate.get("model_name", "")),
        str(candidate.get("model_id", "")),
    )


def decide_promotions(
    existing_registry: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    backtest_results: pd.DataFrame,
    rules: PromotionRuleConfig,
    run_id: str,
    evaluated_at: datetime,
) -> tuple[pd.DataFrame, dict[int, str]]:
    if candidate_registry.empty:
        return pd.DataFrame(), {}

    candidates = normalize_model_registry(candidate_registry)
    existing = normalize_model_registry(existing_registry)
    backtest_results = backtest_results.copy()
    if "model_id" not in backtest_results.columns:
        backtest_results["model_id"] = pd.Series(dtype=object)
    if "window_id" not in backtest_results.columns:
        backtest_results["window_id"] = pd.Series(dtype=object)
    for metric in METRIC_COLUMNS:
        if metric not in backtest_results.columns:
            backtest_results[metric] = pd.Series(dtype=float)
    champions = current_champion_rows(existing)
    champion_by_horizon = {
        int(row["horizon_hours"]): row for _, row in champions.iterrows()
    }

    decisions: list[dict[str, object]] = []
    promoted_by_horizon: dict[int, str] = {}
    for horizon_hours in sorted(candidates["horizon_hours"].dropna().astype(int).unique()):
        horizon_candidates = candidates[candidates["horizon_hours"].astype(int) == horizon_hours]
        champion = champion_by_horizon.get(horizon_hours)
        evaluated_candidates: list[tuple[pd.Series, dict[str, object]]] = []

        if champion is None:
            for _, candidate in horizon_candidates.iterrows():
                candidate_results = backtest_results[
                    backtest_results["model_id"] == candidate["model_id"]
                ]
                detail = _candidate_backtest_summary(candidate_results, rules)
                if int(detail["valid_window_count"]) >= rules.minimum_valid_windows:
                    detail["passed"] = True
                    detail["reason"] = "eligible for initial champion bootstrap"
                    evaluated_candidates.append((candidate, detail))
                else:
                    detail["passed"] = False
                    detail["reason"] = (
                        f"only {detail['valid_window_count']} backtest windows; "
                        f"{rules.minimum_valid_windows} required"
                    )
                    decisions.append(
                        _decision_row(
                            run_id=run_id,
                            horizon_hours=horizon_hours,
                            candidate=candidate,
                            champion=None,
                            decision="rejected",
                            reason=str(detail["reason"]),
                            detail=detail,
                            rules=rules,
                            evaluated_at=evaluated_at,
                            promoted_model_id=None,
                        )
                    )

            if evaluated_candidates:
                selected_candidate, selected_detail = sorted(
                    evaluated_candidates,
                    key=lambda item: _candidate_sort_key(item[1], item[0], rules),
                )[0]
                promoted_by_horizon[horizon_hours] = str(selected_candidate["model_id"])
                for candidate, detail in evaluated_candidates:
                    selected = candidate["model_id"] == selected_candidate["model_id"]
                    reason = (
                        "no existing champion; selected as initial champion after backtests"
                        if selected
                        else "initial champion bootstrap selected another challenger"
                    )
                    if selected:
                        detail = {**detail, **selected_detail}
                    decisions.append(
                        _decision_row(
                            run_id=run_id,
                            horizon_hours=horizon_hours,
                            candidate=candidate,
                            champion=None,
                            decision="promoted" if selected else "rejected",
                            reason=reason,
                            detail={**detail, "reason": reason},
                            rules=rules,
                            evaluated_at=evaluated_at,
                            promoted_model_id=str(candidate["model_id"]) if selected else None,
                        )
                    )
            continue

        champion_results = backtest_results[backtest_results["model_id"] == champion["model_id"]]
        passed_candidates: list[tuple[pd.Series, dict[str, object]]] = []
        failed_rows: list[dict[str, object]] = []
        for _, candidate in horizon_candidates.iterrows():
            challenger_results = backtest_results[
                backtest_results["model_id"] == candidate["model_id"]
            ]
            detail = evaluate_challenger_promotion(
                champion_results,
                challenger_results,
                rules,
            )
            if detail["passed"]:
                passed_candidates.append((candidate, detail))
            else:
                failed_rows.append(
                    _decision_row(
                        run_id=run_id,
                        horizon_hours=horizon_hours,
                        candidate=candidate,
                        champion=champion,
                        decision="rejected",
                        reason=str(detail["reason"]),
                        detail=detail,
                        rules=rules,
                        evaluated_at=evaluated_at,
                        promoted_model_id=None,
                    )
                )

        decisions.extend(failed_rows)
        if not passed_candidates:
            continue

        selected_candidate, _selected_detail = sorted(
            passed_candidates,
            key=lambda item: _candidate_sort_key(item[1], item[0], rules),
        )[0]
        promoted_by_horizon[horizon_hours] = str(selected_candidate["model_id"])
        for candidate, detail in passed_candidates:
            selected = candidate["model_id"] == selected_candidate["model_id"]
            reason = (
                "challenger passed promotion rules"
                if selected
                else "passed rules but another challenger had better average "
                f"{rules.primary_metric}"
            )
            decisions.append(
                _decision_row(
                    run_id=run_id,
                    horizon_hours=horizon_hours,
                    candidate=candidate,
                    champion=champion,
                    decision="promoted" if selected else "rejected",
                    reason=reason,
                    detail={**detail, "reason": reason},
                    rules=rules,
                    evaluated_at=evaluated_at,
                    promoted_model_id=str(candidate["model_id"]) if selected else None,
                )
            )

    return pd.DataFrame(decisions), promoted_by_horizon


def apply_promotion_decisions(
    registry: pd.DataFrame,
    decisions: pd.DataFrame,
    decided_at: datetime,
) -> pd.DataFrame:
    result = normalize_model_registry(registry)
    if result.empty or decisions.empty:
        return result

    promoted = decisions[
        (decisions["decision"] == "promoted") & decisions["promoted_model_id"].notna()
    ]
    for _, decision in promoted.iterrows():
        horizon_hours = int(decision["horizon_hours"])
        model_id = str(decision["promoted_model_id"])
        horizon_mask = result["horizon_hours"].astype(int) == horizon_hours
        previous_mask = horizon_mask & result["is_preferred"]
        target_mask = result["model_id"] == model_id

        result.loc[previous_mask & ~target_mask, "is_preferred"] = False
        result.loc[previous_mask & ~target_mask, "retired_at"] = decided_at
        result.loc[target_mask, "is_preferred"] = True
        result.loc[target_mask, "preferred_at"] = decided_at
        result.loc[target_mask, "retired_at"] = pd.NaT
        result.loc[target_mask, "promotion_decision_id"] = decision["decision_id"]

    return result
