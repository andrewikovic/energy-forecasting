from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from aeso_analytics.database import get_engine


def _read_optional_query(engine, query: str, params: dict | None = None) -> pd.DataFrame:
    try:
        return pd.read_sql_query(text(query), engine, params=params)
    except ProgrammingError:
        return pd.DataFrame()


def print_backtest_summary(engine) -> None:
    latest_run = _read_optional_query(
        engine,
        """
        select run_id
        from ml.model_backtest_results
        order by evaluated_at desc
        limit 1
        """,
    )
    if latest_run.empty:
        return

    run_id = latest_run["run_id"].iloc[0]
    summary = _read_optional_query(
        engine,
        """
        select
            horizon_hours,
            model_name,
            count(*) as valid_windows,
            avg(mae) as average_mae,
            avg(rmse) as average_rmse,
            avg(peak_period_mae) as average_peak_period_mae,
            avg(underprediction_rate) as average_underprediction_rate
        from ml.model_backtest_results
        where run_id = :run_id
        group by horizon_hours, model_name
        order by horizon_hours, average_mae
        """,
        {"run_id": run_id},
    )
    if summary.empty:
        return

    print(f"\nLatest backtest summary by model (run_id={run_id}):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    best = (
        summary.sort_values(["horizon_hours", "average_mae"])
        .groupby("horizon_hours")
        .head(1)
    )
    print("\nBest backtested models by average MAE:")
    print(
        best[["horizon_hours", "model_name", "valid_windows", "average_mae", "average_rmse"]]
        .to_string(index=False, float_format=lambda value: f"{value:,.3f}")
    )


def print_promotion_decisions(engine) -> None:
    latest_run = _read_optional_query(
        engine,
        """
        select run_id
        from ml.model_promotion_decisions
        order by evaluated_at desc
        limit 1
        """,
    )
    if latest_run.empty:
        return

    run_id = latest_run["run_id"].iloc[0]
    decisions = _read_optional_query(
        engine,
        """
        select
            horizon_hours,
            champion_model_name,
            challenger_model_name,
            decision,
            reason,
            valid_window_count,
            win_rate,
            champion_average_mae,
            challenger_average_mae,
            champion_average_rmse,
            challenger_average_rmse
        from ml.model_promotion_decisions
        where run_id = :run_id
        order by horizon_hours, decision desc, challenger_average_mae nulls last
        """,
        {"run_id": run_id},
    )
    if decisions.empty:
        return

    print(f"\nLatest champion/challenger promotion decisions (run_id={run_id}):")
    print(decisions.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


def main() -> None:
    engine = get_engine()
    try:
        performance = pd.read_sql_query(
            text(
                """
                select
                    horizon_hours,
                    evaluation_split,
                    model_name,
                    mae,
                    rmse,
                    mape,
                    mean_error,
                    median_absolute_error,
                    underprediction_rate,
                    peak_period_mae
                from ml.model_performance
                order by horizon_hours, evaluation_split, mae
                """
            ),
            engine,
        )
    except ProgrammingError as exc:
        raise SystemExit("No model performance table found. Run `make train` first.") from exc

    if performance.empty:
        raise SystemExit("ml.model_performance is empty. Run `make train` first.")

    print(performance.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    best = (
        performance[performance["evaluation_split"] == "test"]
        .sort_values(["horizon_hours", "mae"])
        .groupby("horizon_hours")
        .head(1)
    )
    print("\nBest test models by MAE:")
    print(best[["horizon_hours", "model_name", "mae", "rmse", "mape"]].to_string(index=False))
    print_backtest_summary(engine)
    print_promotion_decisions(engine)


if __name__ == "__main__":
    main()
